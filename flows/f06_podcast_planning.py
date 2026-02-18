"""
flows/f06_podcast_planning.py — Podcast Production (Crew 3: Phases 5-8).

Covers:
  Phase 5: Show notes (Podcast Producer)
  Phase 6: Script generation (Podcast Producer)
  Phase 7: Script polishing (Personality Editor)
  Phase 8: Accuracy check (Auditor, advisory)
"""

import os
from pathlib import Path

from crewai import Agent, Task, Crew

from shared.models import PipelineParams, PodcastPlanningResult
from shared.config import (
    SUPPORTED_LANGUAGES,
    ACCESSIBILITY_INSTRUCTIONS,
    build_llm_instances,
    assign_roles,
    get_length_targets,
)
from shared.progress import CrewMonitor, ProgressTracker


def run_podcast_planning(
    params: PipelineParams,
    source_of_truth: str,
    supporting_research: str,
    adversarial_research: str,
    output_dir: Path,
) -> PodcastPlanningResult:
    """Execute Crew 3: Phases 5-8 (show notes, script, polish, accuracy check)."""

    result = PodcastPlanningResult(output_dir=output_dir)

    language = params.language
    language_config = SUPPORTED_LANGUAGES[language]
    english_instruction = "Write all content in English."
    target_instruction = language_config['instruction']

    session_roles = assign_roles(params.host_order)
    presenter = session_roles['presenter']['character']
    questioner = session_roles['questioner']['character']

    accessibility_instruction = ACCESSIBILITY_INSTRUCTIONS.get(
        params.accessibility_level, ACCESSIBILITY_INSTRUCTIONS["simple"]
    )

    length_targets = get_length_targets(language, params.podcast_length)
    length_mode = params.podcast_length
    if language != 'ja':
        target_words = {"short": "1,500", "medium": "3,000", "long": "4,500"}.get(length_mode, "4,500")
        target_label = f"{target_words}-word"
    else:
        target_chars = {"short": "5,000", "medium": "10,000", "long": "15,000"}.get(length_mode, "15,000")
        target_label = f"{target_chars}-character"

    dgx_llm_strict, dgx_llm_creative = build_llm_instances()

    # ================================================================
    # AGENTS
    # ================================================================
    scriptwriter = Agent(
        role='Podcast Producer (The Showrunner)',
        goal=(
            f'Transform research into an engaging, in-depth teaching conversation on "{params.topic}". '
            f'Target: Intellectual, curious professionals. {english_instruction}'
        ),
        backstory=(
            f'Science Communicator targeting Post-Graduate Professionals.\n'
            f'Tone: "Huberman Lab" / "Lex Fridman" \u2014 intellectual, deep-diving.\n\n'
            f'CRITICAL RULES:\n'
            f'  1. NO BASICS: Do NOT define DNA, inflation, peer review, RCT, etc.\n'
            f'  2. LENGTH: Generate exactly {target_label}.\n'
            f'  3. FORMAT: Use "{presenter}:" and "{questioner}:".\n'
            f'  4. TEACHING STYLE: Presenter explains, Questioner bridges gaps.\n'
            f'  5. DEPTH: Cover 3-4 main aspects thoroughly.\n'
            f'{english_instruction}'
            + (f'\nCRITICAL LANGUAGE: Output MUST be in Japanese (\u65e5\u672c\u8a9e) ONLY. '
               f'Do NOT output Chinese (\u4e2d\u6587). Use katakana for host names: \u30ab\u30ba and \u30a8\u30ea\u30ab '
               f'(NOT \u5361\u5179/\u57c3\u91cc\u5361). '
               f'Avoid Kanji only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50).'
               if language == 'ja' else '')
        ),
        llm=dgx_llm_creative,
        verbose=True,
    )

    personality = Agent(
        role='Podcast Personality (The Editor)',
        goal=(
            f'Polish the "{params.topic}" script for natural verbal delivery. '
            f'Target: Exactly {target_label}. {target_instruction}'
        ),
        backstory=(
            f'Editor for high-end intellectual podcasts.\n'
            f'EDITING RULES:\n'
            f'  - Remove definitions of basic scientific concepts\n'
            f'  - Ensure questioner questions feel natural\n'
            f'  - Target exactly {target_label}\n'
            f'  - If too short, ADD DEPTH AND EXAMPLES\n'
            f'{target_instruction}'
            + (f'\nCRITICAL LANGUAGE: Output MUST be in Japanese (\u65e5\u672c\u8a9e) ONLY. '
               f'Do NOT output Chinese (\u4e2d\u6587). Use katakana for host names: \u30ab\u30ba and \u30a8\u30ea\u30ab '
               f'(NOT \u5361\u5179/\u57c3\u91cc\u5361). '
               f'Avoid Kanji only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50).'
               if language == 'ja' else '')
        ),
        llm=dgx_llm_creative,
        verbose=True,
    )

    auditor = Agent(
        role='Scientific Auditor (The Grader)',
        goal=f'Check script accuracy against Source-of-Truth. {target_instruction}',
        backstory=(
            f'Post-polish accuracy checker.\n'
            f'CHECK FOR: Correlation\u2192Causation drift, hedge removal, '
            f'confidence inflation, cherry-picking, contested-as-settled.\n'
            f'{target_instruction}'
            + (f'\nCRITICAL LANGUAGE: Output MUST be in Japanese (\u65e5\u672c\u8a9e) ONLY. '
               f'Do NOT output Chinese (\u4e2d\u6587).'
               if language == 'ja' else '')
        ),
        llm=dgx_llm_strict,
        verbose=True,
    )

    # ================================================================
    # TASKS
    # ================================================================

    # Inject source-of-truth into context
    sot_injection = (
        f"\n\nSOURCE OF TRUTH (authoritative reference):\n"
        f"{source_of_truth[:8000]}\n"
        f"--- END SOURCE OF TRUTH ---\n"
    )

    # Phase 6: Script generation
    recording_task = Task(
        description=(
            f"Using the audit report, write a comprehensive {target_label} podcast dialogue "
            f"about \"{params.topic}\" featuring {presenter} (presenter) and {questioner} (questioner).\n\n"
            f"STRUCTURE:\n"
            f"  1. OPENING: welcome \u2192 hook question \u2192 topic transition\n"
            f"  2. BODY: 3-4 main aspects with mechanisms, evidence, implications\n"
            f"  3. CLOSING: takeaways + practical advice + sign off\n\n"
            f"CHARACTER ROLES:\n"
            f"  - {presenter} (Presenter): {session_roles['presenter']['personality']}\n"
            f"  - {questioner} (Questioner): {session_roles['questioner']['personality']}\n\n"
            f"Format: {presenter}: [dialogue]\\n{questioner}: [dialogue]\n\n"
            f"TARGET LENGTH: {target_label}. CRITICAL \u2014 do not write less.\n"
            f"{english_instruction}"
            f"{sot_injection}"
        ),
        expected_output=(
            f"A {target_label} teaching-style dialogue between {presenter} and {questioner}. "
            f"{english_instruction}"
        ),
        agent=scriptwriter,
    )

    # Phase 5: Show notes
    show_notes_task = Task(
        description=(
            f"Generate comprehensive show notes (SHOW_NOTES.md) for the episode on {params.topic}.\n\n"
            f"Using the Source-of-Truth, create:\n"
            f"1. Episode title and topic\n"
            f"2. Key takeaways (3-5 bullets)\n"
            f"3. Full citation list with validity ratings\n"
            f"{target_instruction}"
            f"{sot_injection}"
        ),
        expected_output=(
            f"Markdown show notes with title, takeaways, citations with validity ratings. "
            f"{target_instruction}"
        ),
        agent=scriptwriter,
    )

    # Phase 7: Polish
    polish_task = Task(
        description=(
            f"Polish the \"{params.topic}\" dialogue for natural spoken delivery.\n\n"
            f"RULES:\n"
            f"- Remove ALL definitions of basic concepts\n"
            f"- Target exactly {target_label}\n"
            f"- Maintain {presenter}: / {questioner}: format\n"
            f"- Remove meta-tags, markdown, stage directions. Dialogue only.\n"
            + (f"\nCRITICAL: Output in Japanese (\u65e5\u672c\u8a9e). "
               f"Use katakana: \u30ab\u30ba / \u30a8\u30ea\u30ab."
               if language == 'ja' else '')
            + f"\n{target_instruction}"
        ),
        expected_output=(
            f"Final {target_label} dialogue. No basic definitions. "
            f"Teaching style with engaging opening. {target_instruction}"
        ),
        agent=personality,
        context=[recording_task],
    )

    # Phase 8: Accuracy check
    accuracy_task = Task(
        description=(
            f"Compare the POLISHED SCRIPT against the Source-of-Truth for {params.topic}.\n\n"
            f"CHECK FOR:\n"
            f"1. Correlation \u2192 Causation drift\n"
            f"2. Hedge removal ('may' \u2192 'does')\n"
            f"3. Confidence inflation\n"
            f"4. Cherry-picking\n"
            f"5. Contested-as-settled\n\n"
            f"OUTPUT: # Accuracy Check\\n## Overall Assessment\\n## Drift Instances\\n## Recommendations\n\n"
            f"NOTE: Advisory only \u2014 does NOT block audio generation. {target_instruction}"
            f"{sot_injection}"
        ),
        expected_output=(
            f"Accuracy check report with drift instances and severity ratings. "
            f"{target_instruction}"
        ),
        agent=auditor,
        context=[polish_task],
    )

    # Renamed aliases
    planning_task = show_notes_task
    post_process_task = polish_task

    # ================================================================
    # TRANSLATION CONTEXT (if applicable)
    # ================================================================
    # When translating, polish reads from translated script
    # (Translation is handled externally by f05 before we get source_of_truth)

    # ================================================================
    # CREW 3 EXECUTION
    # ================================================================
    print(f"\n{'='*70}")
    print("CREW 3: PHASES 5-8 (PODCAST PRODUCTION)")
    print(f"{'='*70}")

    crew_3_tasks = [
        planning_task,       # Phase 5
        recording_task,      # Phase 6
        post_process_task,   # Phase 7
        accuracy_task,       # Phase 8
    ]

    crew_3 = Crew(
        agents=[scriptwriter, personality, auditor],
        tasks=crew_3_tasks,
        verbose=True,
        process='sequential',
    )

    try:
        crew_result = crew_3.kickoff()
    except Exception as e:
        print(f"CREW 3 FAILED: {e}")
        raise

    # Capture outputs
    result.show_notes = (
        planning_task.output.raw
        if hasattr(planning_task, 'output') and planning_task.output
        else ""
    )
    result.script_raw = (
        recording_task.output.raw
        if hasattr(recording_task, 'output') and recording_task.output
        else ""
    )
    result.script_polished = (
        post_process_task.output.raw
        if hasattr(post_process_task, 'output') and post_process_task.output
        else crew_result.raw if crew_result else ""
    )
    result.accuracy_check = (
        accuracy_task.output.raw
        if hasattr(accuracy_task, 'output') and accuracy_task.output
        else ""
    )

    # --- Save outputs ---
    from shared.pdf_utils import save_markdown, save_pdf_safe
    save_markdown("Show Notes", planning_task, "show_notes.md", output_dir)
    save_markdown("Podcast Script (Raw)", recording_task, "podcast_script_raw.md", output_dir)
    save_markdown("Podcast Script (Polished)", post_process_task, "podcast_script_polished.md", output_dir)
    save_markdown("Accuracy Check", accuracy_task, "accuracy_check.md", output_dir)

    save_pdf_safe("Accuracy Check", accuracy_task, "accuracy_check.pdf", output_dir, language)
    save_pdf_safe("Research Framing", result.show_notes, "research_framing.pdf", output_dir, language)

    return result


if __name__ == "__main__":
    print("Usage: This module is called by the orchestrator.")
    print("  from flows.f06_podcast_planning import run_podcast_planning")
