"""
flows/f04_evidence_validation.py — Evidence Validation (Crew 2: Phases 3-4b).

Covers:
  Phase 3:  Adversarial Researcher (counter-evidence)
  Phase 4a: Source Verifier (URL validation)
  Phase 4b: Scientific Auditor (synthesis -> source_of_truth.md)
"""

import os
import re
import json
from pathlib import Path

from crewai import Agent, Task, Crew

from shared.models import PipelineParams, EvidenceGatheringResult, ValidationResult
from shared.config import (
    SUPPORTED_LANGUAGES,
    build_llm_instances,
    assign_roles,
    summarize_report_with_fast_model,
)
from shared.tools import (
    FlowContext,
    link_validator,
    list_research_sources,
    read_research_source,
    read_full_report,
    request_search,
    read_validation_results,
)
from shared.progress import CrewMonitor, ProgressTracker
from link_validator_tool import validate_multiple_urls_parallel


def run_evidence_validation(
    params: PipelineParams,
    evidence: EvidenceGatheringResult,
) -> ValidationResult:
    """Execute Crew 2: Phases 3-4b (adversarial research + validation + synthesis)."""

    output_dir = evidence.output_dir
    result = ValidationResult(output_dir=output_dir)

    FlowContext.set(output_dir, evidence.deep_reports)

    language = params.language
    english_instruction = "Write all content in English."
    session_roles = assign_roles(params.host_order)

    dgx_llm_strict, _ = build_llm_instances()

    # ================================================================
    # BATCH URL VALIDATION (parallel, before agents)
    # ================================================================
    print(f"\n{'='*70}")
    print("BATCH URL VALIDATION (parallel)")
    print(f"{'='*70}")

    all_urls = set()
    url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')
    if evidence.supporting_research:
        all_urls.update(url_pattern.findall(evidence.supporting_research))
    if evidence.gap_fill_output:
        all_urls.update(url_pattern.findall(evidence.gap_fill_output))

    sources_file = output_dir / "deep_research_sources.json"
    if sources_file.exists():
        try:
            src_data = json.loads(sources_file.read_text())
            for role_sources in src_data.values():
                if isinstance(role_sources, list):
                    for src in role_sources:
                        if src.get("url"):
                            all_urls.add(src["url"])
        except Exception:
            pass

    print(f"  Found {len(all_urls)} unique URLs to validate")

    validation_results = {}
    validation_summary = ""
    if all_urls:
        validation_results = validate_multiple_urls_parallel(list(all_urls), max_workers=15)
        valid_count = sum(1 for v in validation_results.values() if "Valid" in v)
        broken_count = sum(1 for v in validation_results.values() if "Broken" in v or "Invalid" in v)
        print(f"  Results: {valid_count} valid, {broken_count} broken, "
              f"{len(validation_results) - valid_count - broken_count} other")

        validation_file = output_dir / "url_validation_results.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_results, f, indent=2, ensure_ascii=False)
        print(f"  Saved to {validation_file}")

        result.url_validation = validation_results

        validation_summary = "\n".join(
            f"  {url}: {status}" for url, status in sorted(validation_results.items())
        )

    # ================================================================
    # AGENTS
    # ================================================================
    counter_researcher = Agent(
        role='Adversarial Researcher (The Skeptic)',
        goal=(
            f'Systematically challenge and debunk specific claims about {params.topic}. '
            f'{english_instruction}'
        ),
        backstory=(
            f'Skeptical meta-analyst who hunts for contradictory evidence.\n\n'
            f'COUNTER-EVIDENCE HIERARCHY:\n'
            f'  1. Contradictory RCTs, systematic reviews with null/negative effects\n'
            f'  2. Observatory/cohort studies with null findings\n'
            f'  3. Animal studies contradicting proposed mechanisms\n\n'
            f'In this podcast, you will be portrayed by "{session_roles["questioner"]["character"]}" '
            f'who has a {session_roles["questioner"]["personality"]} approach.\n\n'
            f'Use ListResearchSources("counter") and ReadResearchSource("counter:N"). '
            f'{english_instruction}'
        ),
        tools=[list_research_sources, read_research_source, read_full_report, request_search],
        llm=dgx_llm_strict,
        verbose=True,
        max_iter=10,
    )

    source_verifier = Agent(
        role='Scientific Source Verifier',
        goal='Extract, validate, and categorize all scientific sources from research papers.',
        backstory=(
            'Librarian and bibliometrics expert specializing in source verification. '
            'Uses LinkValidatorTool to check every URL. '
            'Ensures citations come from reputable peer-reviewed journals.\n'
            'COMMERCIAL BIAS RULE: Flag sources from commercial entities that sell the product '
            'being researched (marked with [COMMERCIAL_BIAS] in funding_source). '
            'These receive trust_level: low. Only retain them if no peer-reviewed alternative exists.'
        ),
        tools=[link_validator, read_validation_results],
        llm=dgx_llm_strict,
        verbose=True,
    )

    auditor = Agent(
        role='Scientific Auditor (The Grader)',
        goal=f'Synthesize ALL research into a single authoritative Research Question document. {english_instruction}',
        backstory=(
            f'You create the SINGLE authoritative reference document.\n'
            f'This is NOT a grade \u2014 it is a SYNTHESIS.\n'
            f'OUTPUT: Source-of-Truth with Executive Summary, Key Claims with Confidence Levels, '
            f'Reliability Scorecard, Caveat Box, Evidence Table, Bibliography.\n'
            f'{english_instruction}'
        ),
        tools=[list_research_sources, read_research_source, read_full_report],
        llm=dgx_llm_strict,
        verbose=True,
        max_iter=15,
    )

    # ================================================================
    # TASKS (with cross-crew context injection)
    # ================================================================
    # Build adversarial task description with injections
    adversarial_desc = (
        f"Based on 'Supporting Paper' and 'Gap Analysis', draft 'Anti-Thesis' paper on {params.topic}.\n\n"
        f"COUNTER-EVIDENCE HIERARCHY:\n"
        f"1. Contradictory RCTs\n2. Observatory with null findings\n3. Animal contradictions\n\n"
        f"Use ListResearchSources('counter') and ReadResearchSource('counter:N').\n"
        f"Every citation MUST include a URL. {english_instruction}"
    )

    # Inject deep research counter context
    if evidence.deep_reports and "counter" in evidence.deep_reports:
        counter_report = evidence.deep_reports["counter"]
        counter_summary = summarize_report_with_fast_model(counter_report.report, "counter", params.topic)
        adversarial_desc += (
            f"\n\nIMPORTANT: Pre-scan analyzed {counter_report.total_summaries} opposing sources.\n"
            f"PRE-COLLECTED OPPOSING EVIDENCE (condensed):\n{counter_summary}"
        )

    # Inject prior research context
    adversarial_desc += (
        f"\n\nPRIOR RESEARCH CONTEXT (from Phases 1-2):\n"
        f"=== SUPPORTING RESEARCH (summary) ===\n{evidence.supporting_research[:4000]}\n\n"
        f"=== GAP ANALYSIS ===\n{evidence.gap_analysis[:2000]}\n"
    )
    if evidence.gap_fill_output:
        adversarial_desc += f"\n=== GAP-FILL ===\n{evidence.gap_fill_output[:2000]}\n"
    adversarial_desc += "--- END PRIOR CONTEXT ---\n"

    adversarial_task = Task(
        description=adversarial_desc,
        expected_output=(
            f"Scientific paper challenging health claims with contradictory evidence. "
            f"Bibliography with URLs. {english_instruction}"
        ),
        agent=counter_researcher,
    )

    # Source verification task
    sv_desc = (
        f"Extract ALL sources from Supporting and Anti-Thesis papers.\n"
        f"For each source verify URL, source type, trust level.\n"
        f"CLAIM-TO-SOURCE VERIFICATION: check claims match sources.\n"
        f"REJECT non-scientific sources. {english_instruction}"
    )
    if validation_summary:
        sv_desc += (
            f"\n\nPRE-VALIDATED URL RESULTS ({len(validation_results)} URLs):\n"
            f"{validation_summary}\n--- END PRE-VALIDATION ---\n"
            f"Use these instead of checking one by one."
        )
    if evidence.framing_output:
        sv_desc += f"\n\nRESEARCH FRAMING:\n{evidence.framing_output[:2000]}\n--- END ---\n"

    source_verification_task = Task(
        description=sv_desc,
        expected_output=f"JSON bibliography with verified sources. {english_instruction}",
        agent=source_verifier,
        context=[adversarial_task],
    )

    # Audit (source-of-truth) task
    audit_desc = (
        f"Synthesize ALL research on {params.topic} into a single Source-of-Truth document.\n\n"
        f"OUTPUT FORMAT:\n"
        f"# Research Question: {params.topic}\n\n"
        f"## Executive Summary\n## Key Claims with Confidence Levels\n"
        f"## Settled Science vs Active Debate\n## Reliability Scorecard\n"
        f"## The Caveat Box\n## Evidence Table\n## Complete Bibliography\n\n"
        f"The output MUST contain concrete health information.\n\n"
        f"COMMERCIAL BIAS RULE: Sources flagged with [COMMERCIAL_BIAS] (industry/commercial entities "
        f"selling the product under research) MUST be deprioritized. Only cite them when no higher-quality "
        f"peer-reviewed evidence exists for that specific claim. Always flag potential conflicts of interest "
        f"in the Reliability Scorecard and Evidence Table.\n"
        f"{english_instruction}"
    )

    # Inject prior research context into audit
    audit_desc += (
        f"\n\nPRIOR RESEARCH CONTEXT:\n"
        f"=== SUPPORTING RESEARCH ===\n{evidence.supporting_research[:4000]}\n\n"
        f"=== GAP ANALYSIS ===\n{evidence.gap_analysis[:2000]}\n"
    )
    if evidence.gap_fill_output:
        audit_desc += f"\n=== GAP-FILL ===\n{evidence.gap_fill_output[:2000]}\n"
    audit_desc += "--- END PRIOR CONTEXT ---\n"

    audit_task = Task(
        description=audit_desc,
        expected_output=(
            f"Structured Research Question document with Executive Summary, "
            f"Key Claims, Reliability Scorecard, Bibliography. {english_instruction}"
        ),
        agent=auditor,
        context=[adversarial_task, source_verification_task],
    )

    # ================================================================
    # CREW 2 EXECUTION
    # ================================================================
    print(f"\n{'='*70}")
    print("CREW 2: PHASES 3-4b (EVIDENCE VALIDATION)")
    print(f"{'='*70}")

    crew_2 = Crew(
        agents=[counter_researcher, source_verifier, auditor],
        tasks=[adversarial_task, source_verification_task, audit_task],
        verbose=True,
        process='sequential',
    )

    try:
        crew_2.kickoff()
    except Exception as e:
        print(f"CREW 2 FAILED: {e}")
        raise

    # Capture outputs
    result.adversarial_research = (
        adversarial_task.output.raw
        if hasattr(adversarial_task, 'output') and adversarial_task.output
        else ""
    )
    result.source_verification = (
        source_verification_task.output.raw
        if hasattr(source_verification_task, 'output') and source_verification_task.output
        else ""
    )
    result.source_of_truth = (
        audit_task.output.raw
        if hasattr(audit_task, 'output') and audit_task.output
        else ""
    )

    # --- Save markdown + PDF outputs ---
    from shared.pdf_utils import save_markdown, save_pdf_safe
    save_markdown("Adversarial Research", adversarial_task, "adversarial_research.md", output_dir)
    save_markdown("Source Verification", source_verification_task, "source_verification.md", output_dir)
    save_markdown("Research Question", audit_task, "source_of_truth.md", output_dir)

    save_pdf_safe("Adversarial Anti-Thesis Paper", adversarial_task, "adversarial_paper.pdf", output_dir, language)
    save_pdf_safe("Verified Source Bibliography", source_verification_task, "verified_sources_bibliography.pdf", output_dir, language)
    save_pdf_safe("Research Question", audit_task, "source_of_truth.pdf", output_dir, language)

    return result


if __name__ == "__main__":
    print("Usage: This module is called by the orchestrator.")
    print("  from flows.f04_evidence_validation import run_evidence_validation")
