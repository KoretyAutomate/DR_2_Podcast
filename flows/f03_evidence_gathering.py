"""
flows/f03_evidence_gathering.py — Evidence Gathering (Phase 0 + Deep Research + Crew 1).

Covers:
  Phase 0:  Research Framing Specialist
  Pre-Scan: Dual-model deep research (Orchestrator)
  Phase 1:  Lead Researcher (Principal Investigator)
  Phase 2:  Auditor gate (PASS/FAIL)
  Phase 2b: Gap-fill loop (conditional, up to 3 rounds)
"""

import os
import re
import json
import asyncio
from pathlib import Path

from crewai import Agent, Task, Crew

from shared.models import PipelineParams, PipelineApproach, EvidenceGatheringResult
from shared.config import (
    SUPPORTED_LANGUAGES,
    ACCESSIBILITY_INSTRUCTIONS,
    build_llm_instances,
    assign_roles,
    summarize_report_with_fast_model,
    check_fast_model_available,
)
from shared.tools import (
    FlowContext,
    list_research_sources,
    read_research_source,
    read_full_report,
    request_search,
    search_tool,
    deep_search_tool,
    get_pending_search_requests,
    clear_pending_search_requests,
    execute_gap_fill_searches,
    append_sources_to_library,
)
from deep_research_agent import run_deep_research


def run_evidence_gathering(
    params: PipelineParams,
    approach: PipelineApproach,
) -> EvidenceGatheringResult:
    """Execute Phase 0, Deep Research Pre-Scan, and Crew 1 (Phases 1-2b)."""

    output_dir = approach.output_dir
    result = EvidenceGatheringResult(output_dir=output_dir)

    # Set tool context
    FlowContext.set(output_dir)

    language = params.language
    language_config = SUPPORTED_LANGUAGES[language]
    english_instruction = "Write all content in English."
    accessibility_instruction = ACCESSIBILITY_INSTRUCTIONS.get(
        params.accessibility_level, ACCESSIBILITY_INSTRUCTIONS["simple"]
    )
    session_roles = assign_roles(params.host_order)

    # Build LLMs
    dgx_llm_strict, dgx_llm_creative = build_llm_instances()

    # ================================================================
    # PHASE 0: Research Framing
    # ================================================================
    print(f"\n{'='*70}")
    print("PHASE 0: RESEARCH FRAMING & HYPOTHESIS")
    print(f"{'='*70}")

    research_framer = Agent(
        role='Research Framing Specialist',
        goal=(
            f'Define the research scope, core questions, and evidence criteria '
            f'for investigating {params.topic}. {english_instruction}'
        ),
        backstory=(
            'You are a senior research methodologist who designs investigation frameworks. '
            'Before any evidence is gathered, you establish:\n'
            '  1. Core research questions that must be answered\n'
            '  2. Scope boundaries (what is in/out of scope)\n'
            '  3. Evidence criteria (what counts as strong/weak evidence)\n'
            '  4. Suggested search directions and keywords\n'
            '  5. Hypotheses to test\n\n'
            'Your framing document guides all downstream research.'
        ),
        llm=dgx_llm_strict,
        verbose=True,
    )

    framing_task = Task(
        description=(
            f"Define the research framework for investigating: {params.topic}\n\n"
            f"Produce a structured framing document with:\n\n"
            f"## 1. Core Research Questions\nList 5-8 specific questions.\n\n"
            f"## 2. Scope Boundaries\nDefine IN SCOPE and OUT OF SCOPE.\n\n"
            f"## 3. Evidence Criteria\nDefine strong vs weak evidence.\n\n"
            f"## 4. Suggested Search Directions\n8-12 specific search queries.\n\n"
            f"## 5. Hypotheses\n3-5 testable hypotheses.\n\n"
            f"Do NOT search for evidence. Only define the framework. {english_instruction}"
        ),
        expected_output=(
            f"Structured research framing document with core questions, scope, "
            f"evidence criteria, search directions, and hypotheses. {english_instruction}"
        ),
        agent=research_framer,
    )

    crew_0 = Crew(
        agents=[research_framer],
        tasks=[framing_task],
        verbose=True,
        process='sequential',
    )

    try:
        crew_0.kickoff()
        framing_output = framing_task.output.raw if hasattr(framing_task, 'output') and framing_task.output else ""
        print(f"Phase 0 complete: Research framing ({len(framing_output)} chars)")
    except Exception as e:
        print(f"Phase 0 (Research Framing) failed: {e}")
        framing_output = ""

    result.framing_output = framing_output

    # ================================================================
    # DEEP RESEARCH PRE-SCAN (Dual-Model Map-Reduce)
    # ================================================================
    print(f"\n{'='*70}")
    print("DEEP RESEARCH PRE-SCAN")
    print(f"{'='*70}")

    brave_key = os.getenv("BRAVE_API_KEY", "")
    fast_model_available = check_fast_model_available()

    deep_reports = None
    try:
        deep_reports = asyncio.run(run_deep_research(
            topic=params.topic,
            brave_api_key=brave_key,
            results_per_query=15,
            fast_model_available=fast_model_available,
            framing_context=framing_output,
        ))

        # Save reports
        for role_name, report in deep_reports.items():
            report_file = output_dir / f"deep_research_{role_name}.md"
            with open(report_file, 'w') as f:
                f.write(report.report)
            print(f"{role_name.capitalize()} report saved: {report_file} ({report.total_summaries} sources)")

        # Save sources JSON
        sources_json = {}
        for role_name in ("lead", "counter"):
            report = deep_reports[role_name]
            role_sources = []
            for idx, src in enumerate(report.sources):
                if src.error or not src.summary or src.summary.strip().upper() == "NO RELEVANT DATA":
                    continue
                entry = {
                    "index": idx,
                    "url": src.url,
                    "title": src.title,
                    "query": src.query,
                    "goal": src.goal,
                    "summary": src.summary,
                    "metadata": src.metadata.to_dict() if src.metadata else None,
                }
                if src.cited_references:
                    entry["cited_references"] = src.cited_references
                role_sources.append(entry)
            sources_json[role_name] = role_sources

        sources_file = output_dir / "deep_research_sources.json"
        with open(sources_file, 'w') as f:
            json.dump(sources_json, f, indent=2, ensure_ascii=False)
        print(f"Research library saved: {sources_file} "
              f"(lead={len(sources_json['lead'])}, counter={len(sources_json['counter'])})")

        result.deep_reports = deep_reports
        result.deep_sources_json = sources_json

    except Exception as e:
        print(f"Deep research pre-scan failed: {e}")
        print("Continuing with standard agent research...")

    # ================================================================
    # CREW 1: Phases 1-2 (Research + Gate)
    # ================================================================
    print(f"\n{'='*70}")
    print("CREW 1: PHASES 1-2 (RESEARCH + GATE)")
    print(f"{'='*70}")

    # --- Agents ---
    researcher = Agent(
        role='Principal Investigator (Lead Researcher)',
        goal=(
            f'Find and document credible scientific signals about {params.topic}, '
            f'organized by mechanism of action. {english_instruction}'
        ),
        backstory=(
            f'You are a desperate scientist looking for signals in the noise. '
            f'CONSTRAINT: If Human RCTs are unavailable, you are AUTHORIZED to use Animal Models, '
            f'but you MUST label them as "Early Signal" or "Animal Model".\n\n'
            f'OUTPUT REQUIREMENT: Group findings by:\n'
            f'  1. "Mechanism of Action" (HOW it works biologically)\n'
            f'  2. "Clinical Evidence" (WHAT human studies show)\n\n'
            f'Evidence hierarchy: (1) Human RCTs/meta-analyses, (2) Observational, (3) Animal/in vitro.\n'
            f'In this podcast, you will be portrayed by "{session_roles["presenter"]["character"]}" '
            f'who has a {session_roles["presenter"]["personality"]} approach.\n\n'
            f'Use ListResearchSources to browse, ReadResearchSource to read specific sources. '
            f'If you find a CRITICAL gap, use RequestSearch to queue targeted searches. '
            f'{english_instruction}'
        ),
        tools=[list_research_sources, read_research_source, read_full_report, request_search],
        llm=dgx_llm_strict,
        verbose=True,
        max_iter=10,
    )

    auditor = Agent(
        role='Scientific Auditor (The Grader)',
        goal=f'Grade the research quality with a Reliability Scorecard. {english_instruction}',
        backstory=(
            f'You are a harsh peer reviewer. You do not write content; you GRADE it.\n\n'
            f'YOUR TASKS:\n'
            f'  1. Link Check: If a claim has no URL, REJECT it.\n'
            f'  2. Strength Rating: Score (1-10) for main claims.\n'
            f'  3. The Caveat Box: List why findings might be wrong.\n'
            f'  4. Consensus Check from Research Library.\n'
            f'  5. Source Validation via ReadResearchSource.\n\n'
            f'OUTPUT: Structured Markdown with "Reliability Scorecard". {english_instruction}'
        ),
        tools=[list_research_sources, read_research_source, read_full_report],
        llm=dgx_llm_strict,
        verbose=True,
        max_iter=15,
    )

    # --- Build task descriptions with injections ---
    research_desc = (
        f"Conduct exhaustive deep dive into {params.topic}, guided by the Research Framing document. "
        f"Draft condensed scientific paper (Nature style).\n\n"
        f"RESEARCH LIBRARY: Use ListResearchSources('lead') and ReadResearchSource('lead:N') "
        f"as your PRIMARY evidence source. If a critical gap exists, use RequestSearch.\n\n"
        f"Every citation MUST include a URL. CONCLUDE with available evidence. "
        f"Include: Abstract, Introduction, 3 Biochemical Mechanisms, Bibliography with URLs. "
        f"{english_instruction}"
    )

    # Inject deep research context
    if deep_reports:
        lead_report = deep_reports["lead"]
        counter_report = deep_reports["counter"]
        lead_summary = summarize_report_with_fast_model(lead_report.report, "lead", params.topic)

        research_desc += (
            f"\n\nIMPORTANT: A deep research pre-scan analyzed {lead_report.total_summaries} supporting sources.\n"
            f"YOUR PRIMARY TASK: Synthesize this pre-collected evidence.\n"
            f"SEARCH POLICY: Do NOT use RequestSearch unless CRITICAL gap with ZERO coverage.\n\n"
            f"PRE-COLLECTED SUPPORTING EVIDENCE (condensed):\n{lead_summary}"
        )

    # Inject framing context
    if framing_output:
        research_desc += (
            f"\n\nRESEARCH FRAMING CONTEXT (from Phase 0):\n{framing_output}\n"
            f"--- END FRAMING CONTEXT ---\n"
        )

    research_task = Task(
        description=research_desc,
        expected_output=(
            f"Scientific paper with health mechanisms, citations with URLs. "
            f"{english_instruction}"
        ),
        agent=researcher,
        context=[framing_task],
    )

    gap_analysis_task = Task(
        description=(
            f"RESEARCH GATE: Evaluate whether the initial research on {params.topic} "
            f"adequately addresses the core research questions.\n\n"
            f"For EACH core question, assess: ADDRESSED / PARTIALLY / NOT ADDRESSED\n\n"
            f"OUTPUT: ## Research Gate Assessment\n"
            f"### Question Coverage\n### Identified Gaps\n"
            f"### Weak Points for Adversarial Review\n### VERDICT: [PASS or FAIL]\n"
            f"{english_instruction}"
        ),
        expected_output=(
            f"Research gate assessment with question coverage, gaps, weak points, "
            f"and VERDICT: PASS or VERDICT: FAIL. {english_instruction}"
        ),
        agent=auditor,
        context=[framing_task, research_task],
    )

    crew_1 = Crew(
        agents=[researcher, auditor],
        tasks=[research_task, gap_analysis_task],
        verbose=True,
        process='sequential',
    )

    try:
        crew_1.kickoff()
    except TimeoutError as e:
        print(f"CREW 1: AGENT TIMED OUT \u2014 using partial results: {str(e)[:200]}")
    except Exception as e:
        print(f"CREW 1 FAILED: {e}")
        raise

    # Capture outputs
    research_output = research_task.output.raw if hasattr(research_task, 'output') and research_task.output else ""
    gate_output = gap_analysis_task.output.raw if hasattr(gap_analysis_task, 'output') and gap_analysis_task.output else ""
    result.supporting_research = research_output
    result.gap_analysis = gate_output

    # ================================================================
    # GATE CHECK
    # ================================================================
    print(f"\n{'='*70}")
    print("RESEARCH GATE CHECK")
    print(f"{'='*70}")

    verdict_match = re.search(r'VERDICT:\s*(PASS|FAIL)', gate_output, re.IGNORECASE)
    if verdict_match:
        gate_passed = verdict_match.group(1).upper() == "PASS"
        print(f"Gate verdict: {verdict_match.group(1).upper()}")
    else:
        print("No clear VERDICT found. Defaulting to PASS.")
        gate_passed = True

    result.gate_passed = gate_passed

    # ================================================================
    # PHASE 2b: GAP-FILL (conditional)
    # ================================================================
    gap_fill_output = ""
    if not gate_passed:
        print(f"\n{'='*70}")
        print("PHASE 2b: GAP-FILL RESEARCH (Gate FAILED)")
        print(f"{'='*70}")

        gap_fill_task = Task(
            description=(
                f"Conduct TARGETED supplementary research to fill gaps on {params.topic}.\n\n"
                f"GAP ANALYSIS RESULTS:\n{gate_output}\n--- END ---\n\n"
                f"Use RequestSearch for missing evidence. {english_instruction}"
            ),
            expected_output=f"Targeted supplementary research with verifiable sources. {english_instruction}",
            agent=researcher,
            tools=[request_search, list_research_sources, read_research_source],
            context=[research_task, gap_analysis_task],
        )

        MAX_SEARCH_ROUNDS = 3
        for search_round in range(MAX_SEARCH_ROUNDS):
            print(f"\n  --- Gap-Fill Round {search_round + 1}/{MAX_SEARCH_ROUNDS} ---")
            clear_pending_search_requests()

            gap_fill_crew = Crew(
                agents=[researcher],
                tasks=[gap_fill_task],
                verbose=True,
                process='sequential',
            )

            try:
                gap_fill_crew.kickoff()
                gap_fill_output = (
                    gap_fill_task.output.raw
                    if hasattr(gap_fill_task, 'output') and gap_fill_task.output
                    else ""
                )
                print(f"  Round {search_round + 1}: {len(gap_fill_output)} chars")
            except Exception as e:
                print(f"  Round {search_round + 1}: Gap-fill failed: {e}")
                break

            pending = get_pending_search_requests()
            if not pending:
                print("  No search requests queued \u2014 gap-fill complete")
                break

            print(f"  Executing {len(pending)} queued searches...")
            new_sources = asyncio.run(execute_gap_fill_searches(
                pending_requests=list(pending),
                role="lead",
                brave_api_key=brave_key,
                fast_model_available=fast_model_available,
            ))

            if new_sources:
                append_sources_to_library(new_sources, "lead", output_dir)
                print(f"  Added {len(new_sources)} new sources")
            else:
                print("  No new sources found \u2014 gap-fill complete")
                break
    else:
        print("Gate PASSED \u2014 skipping gap-fill research")

    result.gap_fill_output = gap_fill_output

    # --- Save markdown outputs ---
    from shared.pdf_utils import save_markdown
    save_markdown("Research Framing", framing_output, "research_framing.md", output_dir)
    save_markdown("Supporting Research", research_task, "supporting_research.md", output_dir)
    save_markdown("Gap Analysis", gap_analysis_task, "gap_analysis.md", output_dir)
    if gap_fill_output:
        save_markdown("Gap-Fill Research", gap_fill_output, "gap_fill_research.md", output_dir)

    return result


if __name__ == "__main__":
    print("Usage: This module is called by the orchestrator.")
    print("  from flows.f03_evidence_gathering import run_evidence_gathering")
