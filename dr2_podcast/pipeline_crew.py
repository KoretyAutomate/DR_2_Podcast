"""
CrewAI agent/task construction and kickoff utilities.

Extracted from pipeline.py (T4.1).
Contains: Agent/Task definitions, _estimate_task_tokens, _build_sot_injection_for_stage,
_crew_kickoff_guarded, _SOT_BLOCK_RE, PHASE_MARKERS, TASK_METADATA,
display_workflow_plan, ProgressTracker.
"""

import logging
import os
import re
import time
from pathlib import Path

from crewai import Agent, Task
from dr2_podcast.prompt_strings import get_prompt

logger = logging.getLogger(__name__)


_SOT_BLOCK_RE = re.compile(
    r'\n\nSOURCE OF TRUTH SUMMARY[^\n]*\n.*?--- END SOT ---\n'
    r'|\n\n\[SOT Stage \d[^\n]*\n.*?--- END SOT ---\n',
    re.DOTALL
)


def _estimate_task_tokens(task, translation_task_obj=None, language='en'):
    """Rough estimate of input tokens for a CrewAI task (description + context chain outputs).

    Japanese/Chinese: ~2 chars/token. Other languages: ~4 chars/token.
    Adds 2000-token buffer for agent system prompt overhead.
    """
    chars_per_tok = 2 if language in ('ja', 'zh') else 4
    total_chars = len(task.description or '')
    for ctx_task in (task.context or []):
        raw = getattr(getattr(ctx_task, 'output', None), 'raw', '') or ''
        total_chars += len(raw)
    return total_chars // chars_per_tok + 2000


def _build_sot_injection_for_stage(stage, sot_file, translated_sot_file,
                                    sot_summary, translated_sot_summary,
                                    grade_numbers_text, language_config):
    """Return SOT injection text for a context-degradation stage.

    Stage 1: Full target-language fast-model summary + file path    (~3K tokens)
    Stage 2: IMRaD Abstract + GRADE section from file + path         (~1.5K tokens)
    Stage 3: File path + pre-extracted GRADE/clinical numbers only   (~300 tokens)
    """
    target_file = str(translated_sot_file or sot_file or '')
    lang_name = language_config.get('name', 'target language') if isinstance(language_config, dict) else str(language_config)

    if stage == 1:
        summary = (translated_sot_summary or sot_summary or '')
        return (
            f"\n\nSOURCE OF TRUTH SUMMARY ({lang_name}):\n"
            f"Use this as your primary research reference.\n\n"
            f"{summary}\n\n"
            f"Full research file: {target_file}\n"
            f"--- END SOT ---\n"
        )
    elif stage == 2:
        abstract_text = ''
        grade_text = ''
        if target_file and Path(target_file).exists():
            try:
                raw = Path(target_file).read_text(encoding='utf-8')
                m = re.search(r'(?:## 1\.|##\s*Abstract)(.*?)(?=\n## |\Z)', raw, re.DOTALL | re.IGNORECASE)
                if m:
                    abstract_text = m.group(1).strip()[:2000]
                # Try GRADE (clinical) or Evidence Quality Synthesis (social science)
                m = re.search(r'(?:### 4\.3|###\s*GRADE|##\s*GRADE|###\s*Evidence\s+Quality\s+Synthesis)(.*?)(?=\n### |\n## |\Z)', raw, re.DOTALL | re.IGNORECASE)
                if m:
                    grade_text = m.group(1).strip()[:1000]
            except Exception:
                pass
        evidence_label = "EVIDENCE ASSESSMENT"
        return (
            f"\n\n[SOT Stage 2 \u2014 reduced for context budget]\n"
            f"RESEARCH ABSTRACT:\n{abstract_text or '(not available)'}\n\n"
            f"{evidence_label}:\n{grade_text or '(not available)'}\n\n"
            f"Full research file: {target_file}\n"
            f"--- END SOT ---\n"
        )
    else:  # stage 3
        return (
            f"\n\n[SOT Stage 3 \u2014 minimal context; use research file for details]\n"
            f"Full research file: {target_file}\n"
            f"{grade_numbers_text or ''}\n"
            f"--- END SOT ---\n"
        )


def _crew_kickoff_guarded(crew_factory_fn, task, translation_task_obj, language,
                           sot_file, translated_sot_file, sot_summary, translated_sot_summary,
                           grade_numbers_text, language_config, crew_name,
                           ctx_window=32768, max_tokens=16000):
    """Run a crew kickoff with pre-emptive 3-stage context-budget check.

    Before kickoff, estimates input tokens. If over budget, degrades the SOT
    injection to the next stage (summary -> abstract+GRADE+path -> path only).
    Selects the lowest stage that fits; runs the crew exactly once.

    Stages:
      1 -- Full target-language summary inline        (~3K tokens, default)
      2 -- Abstract + GRADE sections + file path      (~1.5K tokens)
      3 -- File path + clinical numbers only           (~300 tokens)
    """
    budget = ctx_window - max_tokens - 2000  # 2000-token system-prompt buffer

    for stage in range(1, 4):
        est = _estimate_task_tokens(task, translation_task_obj, language)
        if est <= budget or stage == 3:
            if stage > 1:
                logger.warning("  %s: SOT stage %d selected (est %s tokens, budget %s)",
                      crew_name, stage, f"{est:,}", f"{budget:,}")
            try:
                crew_factory_fn().kickoff()
            except SystemExit as e:
                # CrewAI raises SystemExit when context window is exhausted
                # and respect_context_window=False. Convert to RuntimeError
                # so the pipeline can handle it gracefully.
                raise RuntimeError(
                    f"{crew_name}: context window exhausted after summarization attempts. "
                    f"Budget={budget:,}, est={est:,}. Original: {e}"
                ) from e
            return
        # Over budget -- degrade to next stage
        logger.warning("  %s: Stage %d est %s tokens > budget %s. Degrading to stage %d...",
              crew_name, stage, f"{est:,}", f"{budget:,}", stage + 1)
        base_desc = _SOT_BLOCK_RE.sub('', task.description)
        task.description = base_desc + _build_sot_injection_for_stage(
            stage + 1, sot_file, translated_sot_file,
            sot_summary, translated_sot_summary, grade_numbers_text, language_config
        )


# ---------------------------------------------------------------------------
# Agent & Task definitions
# ---------------------------------------------------------------------------

def create_agents_and_tasks(
    *,
    topic_name,
    language,
    language_config,
    english_instruction,
    target_instruction,
    target_script,
    target_unit_singular,
    target_unit_plural,
    _target_min,
    target_length_int,
    SESSION_ROLES,
    channel_intro,
    core_target,
    channel_mission,
    dgx_llm_strict,
    dgx_llm_creative,
    SCRIPT_TOLERANCE,
    output_dir,
    output_path_fn,
    list_research_sources,
    read_research_source,
    read_full_report,
    link_validator,
):
    """Construct all CrewAI Agents and Tasks.

    Called from __main__ after all runtime variables are initialized.
    Returns a dict of agent and task objects.

    Parameters are keyword-only to make call sites self-documenting.
    """
    auditor_agent = Agent(
        role='Scientific Auditor',
        goal=f'Grade the research quality with a Reliability Scorecard. Do NOT write content - GRADE it. {english_instruction}',
        backstory=(
            f'You are a harsh peer reviewer. You do not write content; you GRADE it.\n\n'
            f'YOUR TASKS:\n'
            f'  1. Link Check: If a claim has no URL or a broken URL, REJECT it.\n'
            f'  2. Strength Rating: Assign a score (1-10) to the main claims:\n'
            f'       10 = Meta-analysis from top journal\n'
            f'       7-9 = Human RCT with good sample size\n'
            f'       4-6 = Observational/cohort study\n'
            f'       1-3 = Animal model or speculation\n'
            f'  3. The Caveat Box: Explicitly list why the findings might be wrong:\n'
            f'       (e.g., "Mouse study only", "Sample size n=12", "Conflicts of interest")\n'
            f'  4. Consensus Check: Verify consensus from pre-scanned sources in the Research Library.\n'
            f'  5. Source Validation: Use ReadResearchSource to read source content. Verify claims match sources. REJECT misrepresented sources.\n'
            f'\n'
            f'You have access to a Research Library containing all sources from the deep research pre-scan. '
            f'Use ListResearchSources to browse available sources, then ReadResearchSource to read specific ones in detail.\n\n'
            f'OUTPUT: A structured Markdown report with a "Reliability Scorecard". '
            f'{english_instruction}'
        ),
        tools=[list_research_sources, read_research_source, read_full_report, link_validator],
        llm=dgx_llm_strict,
        verbose=True,
        max_iter=15,
    )

    producer_agent = Agent(
        role='Podcast Producer',
        goal=(
            f'Transform research into an engaging, in-depth teaching conversation on "{topic_name}". '
            f'Target: Intellectual, curious professionals who want to learn. {english_instruction}'
        ),
        backstory=(
            f'Science Communicator targeting Post-Graduate Professionals (Masters/PhD level). '
            f'Tone: Think "Huberman Lab" or "Lex Fridman" - intellectual, curious, deep-diving.\n\n'
            f'CRITICAL RULES:\n'
            f'  1. NO BASICS: Do NOT define basic terms like "DNA", "inflation", "supply chain", '
            f'     "peer review", "RCT", or "meta-analysis". Assume the listener knows them.\n'
            f'  2. LENGTH: Generate AT LEAST {target_script} {target_unit_plural} (approx {_target_min} min). '
            f'Aim for {int(target_length_int * 1.2):,} {target_unit_plural} --- more content is better than less.\n'
            f'  3. FORMAT: Script MUST use "{SESSION_ROLES["presenter"]["label"]}:" (Presenter) '
            f'     and "{SESSION_ROLES["questioner"]["label"]}:" (Questioner).\n'
            f'  4. TEACHING STYLE: The Presenter explains the topic systematically. '
            f'     The Questioner asks bridging questions on behalf of the audience:\n'
            f'     - Clarify jargon or uncommon terms\n'
            f'     - Request real-world examples and analogies\n'
            f'     - Occasionally push back on weak or debated evidence\n'
            f'  5. DEPTH: Cover 3-4 main aspects of the topic thoroughly with mechanisms, evidence, and implications.\n'
            f'\n'
            f'Your dialogue should dive into nuance, trade-offs, and practical implications. '
            f'The questioner keeps it accessible without dumbing it down. '
            f'{english_instruction}\n\n'
            f'DIALOGUE RULE: Hosts must NEVER address each other by name inside dialogue --- '
            f'no personal names, no "Host 1", no "Host 2" spoken aloud. '
            f'Names are only used as speaker LABELS before the colon, never within the dialogue itself.'
            + (f'\n\nLANGUAGE WARNING: When generating Japanese output, you MUST stay in Japanese throughout. '
               f'Do NOT switch to Chinese. '
               f'Avoid Kanji that is only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50).'
               if language == 'ja' else '')
        ),
        llm=dgx_llm_creative,
        verbose=True,
        tools=[read_full_report],
    )

    editor_agent = Agent(
        role='Podcast Editor',
        goal=(
            f'Polish the "{topic_name}" script for natural verbal delivery at Masters-level. '
            f'Target: Exactly {target_script} {target_unit_plural} ({_target_min} minutes). '
            f'{target_instruction}'
        ),
        backstory=(
            f'Editor for high-end intellectual podcasts (Huberman Lab, Lex Fridman). '
            f'Your audience has advanced degrees - they want depth, not hand-holding.\n\n'
            f'EDITING RULES:\n'
            f'  - Remove any definitions of basic scientific concepts\n'
            f'  - Ensure the questioner\'s questions feel natural and audience-aligned\n'
            f'  - Keep technical language intact (no dumbing down)\n'
            f'  - Target exactly {target_script} {target_unit_plural} for {_target_min}-minute runtime.\n'
            f'  - Ensure the opening follows the 3-part structure: welcome -> hook question -> topic shift\n'
            f'  - Teaching flow: presenter explains, questioner bridges gaps for listeners\n'
            f'\n'
            f'If script is at or near target: refine for natural delivery without changing length significantly.\n'
            f'If script is over target: trim repetition and redundant examples to hit target. DO NOT trim factual content.\n'
            f'{target_instruction}'
        ),
        llm=dgx_llm_creative,
        verbose=True
    )

    framing_agent = Agent(
        role='Research Framing Specialist',
        goal=f'Define the research scope, core questions, and evidence criteria for investigating {topic_name}. {english_instruction}',
        backstory=(
            'You are a senior research methodologist who designs investigation frameworks. '
            'Before any evidence is gathered, you establish:\n'
            '  1. Core research questions that must be answered\n'
            '  2. Scope boundaries (what is in/out of scope)\n'
            '  3. Evidence criteria (what counts as strong/weak evidence)\n'
            '  4. Suggested search directions and keywords\n'
            '  5. Hypotheses to test\n\n'
            'Your framing document guides all downstream research, ensuring systematic '
            'coverage rather than ad-hoc searching. You do NOT search for evidence yourself --- '
            'you define WHAT to look for and HOW to evaluate it.'
        ),
        llm=dgx_llm_strict,
        verbose=True
    )

    # --- TASKS ---
    framing_task = Task(
        description=(
            f"Define the research framework for investigating: {topic_name}\n\n"
            f"Produce a structured framing document with:\n\n"
            f"## 1. Core Research Questions\n"
            f"List 5-8 specific questions that this research MUST answer. "
            f"These should cover mechanisms, clinical evidence, population effects, and limitations.\n\n"
            f"## 2. Scope Boundaries\n"
            f"Define what is IN SCOPE and OUT OF SCOPE. "
            f"E.g., 'In scope: human health effects. Out of scope: economic impact, agricultural methods.'\n\n"
            f"## 3. Evidence Criteria\n"
            f"Define what constitutes strong vs weak evidence for this topic:\n"
            f"  - What study types are most relevant? (RCT, cohort, meta-analysis, etc.)\n"
            f"  - What sample sizes would be convincing?\n"
            f"  - What confounders should researchers watch for?\n\n"
            f"## 4. Suggested Search Directions\n"
            f"Provide 8-12 specific search queries or keyword combinations that would "
            f"systematically cover the topic. Group by: supporting evidence, opposing evidence, "
            f"mechanistic evidence, and population-level data.\n\n"
            f"## 5. Hypotheses\n"
            f"State 3-5 testable hypotheses that the research should evaluate.\n\n"
            f"Do NOT search for evidence. Only define the framework. "
            f"{english_instruction}"
        ),
        expected_output=(
            f"Structured research framing document with core questions, scope boundaries, "
            f"evidence criteria, search directions, and hypotheses. {english_instruction}"
        ),
        agent=framing_agent,
        output_file=os.path.relpath(output_path_fn(output_dir, "research_framing.md"))
    )

    # Build channel intro directive for script
    if channel_intro:
        _channel_intro_directive = (
            f"  1. CHANNEL INTRO (~25 {target_unit_plural}, ~10 seconds):\n"
            f"     {SESSION_ROLES['presenter']['label']}: {channel_intro}\n"
            f"     CRITICAL: Use this text EXACTLY as written above. Do NOT rephrase, summarize, or modify it.\n\n"
        )
    else:
        _channel_intro_directive = (
            f"  1. CHANNEL INTRO (~25 {target_unit_plural}, ~10 seconds):\n"
            f"     Both hosts briefly introduce the show and today's topic.\n"
            f"     {SESSION_ROLES['presenter']['label']}: Welcome to Deep Research Podcast. Today we're diving deep into {topic_name}.\n\n"
        )

    # Only Act 2 has a numeric target (above 50%); other acts fill as needed
    _act2_min = int(target_length_int * 0.50)
    _core_target_or_default = core_target or "our listeners"

    # One Action tailoring text (avoid backslash inside f-string on Python 3.11)
    _one_action_tail = "tailored to " + core_target if core_target else "to try this week"

    script_task = Task(
        description=(
            f"Using the Episode Blueprint, write a comprehensive {target_script}-{target_unit_singular} podcast dialogue about \"{topic_name}\" "
            f"featuring {SESSION_ROLES['presenter']['label']} (presenter) and {SESSION_ROLES['questioner']['label']} (questioner).\n\n"
            f"SCRIPT STRUCTURE (follow this EXACTLY):\n\n"
            + _channel_intro_directive
            + get_prompt("script", "hook", language,
                         target_unit_plural=target_unit_plural,
                         presenter=SESSION_ROLES['presenter']['label'],
                         questioner=SESSION_ROLES['questioner']['label'])
            + get_prompt("script", "act1", language)
            + get_prompt("script", "act2", language,
                         act2_min=f"{_act2_min:,}",
                         target_unit_plural=target_unit_plural)
            + get_prompt("script", "act3", language,
                         core_target_or_default=_core_target_or_default)
            + get_prompt("script", "act4", language,
                         core_target_or_default=_core_target_or_default)
            + get_prompt("script", "length_note", language,
                         target_script=target_script,
                         target_unit_plural=target_unit_plural)
            + get_prompt("script", "wrapup", language,
                         target_unit_plural=target_unit_plural)
            + get_prompt("script", "one_action", language,
                         target_unit_plural=target_unit_plural,
                         presenter=SESSION_ROLES['presenter']['label'],
                         questioner=SESSION_ROLES['questioner']['label'],
                         one_action_tail=_one_action_tail)
            + get_prompt("script", "personality", language)
            + get_prompt("script", "character_roles", language,
                         presenter=SESSION_ROLES['presenter']['label'],
                         questioner=SESSION_ROLES['questioner']['label'],
                         presenter_personality=SESSION_ROLES['presenter']['personality'],
                         questioner_personality=SESSION_ROLES['questioner']['personality']) +
            f"Format STRICTLY as:\n"
            f"{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
            f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
            + get_prompt("script", "target_length", language,
                         target_script=target_script,
                         target_unit_plural=target_unit_plural,
                         target_min=str(_target_min),
                         aim_target=f"{int(target_length_int * 1.2):,}")
            + (f"\nCRITICAL LANGUAGE RULE: You are writing in Japanese. "
               f"Do NOT use Chinese at any point. Every sentence must be in Japanese. "
               f"Use standard Japanese kanji only (\u6c17 not \u6c14, \u697d not \u4e50).\n"
               if language == 'ja' else '')
            + f"{target_instruction}"
        ),
        expected_output=(
            f"A {target_script}-{target_unit_singular} podcast dialogue about {topic_name} between "
            f"{SESSION_ROLES['presenter']['label']} (presents and explains) "
            f"and {SESSION_ROLES['questioner']['label']} (asks bridging questions). "
            f"Follows 8-part structure: Hook, Channel Intro, 4 Acts (Claim, Evidence & Nuance, Holistic Conclusion, Protocol), Wrap-up, One Action. "
            f"{target_instruction}"
        ),
        agent=producer_agent,
        context=[]
    )

    # --- SOT TRANSLATION TASK (only when language != 'en') ---
    translation_task = None
    if language != 'en':
        translation_task = Task(
            description=(
                (f"ABSOLUTE RULE: Output MUST be in Japanese ONLY. NEVER use Chinese at any point.\n"
                 f"WRONG: \u6267\u884c\u529f\u80fd -> CORRECT: \u5b9f\u884c\u6a5f\u80fd; WRONG: \u8865\u5145 -> CORRECT: \u88dc\u5145; WRONG: \u8ba4\u77e5 -> CORRECT: \u8a8d\u77e5\n"
                 f"If unsure of the Japanese term, keep the English term --- NEVER use Chinese.\n\n"
                 if language == 'ja' else '')
                + f"Translate the entire Source-of-Truth document about {topic_name} into {language_config['name']}.\n\n"
                f"TRANSLATION RULES:\n"
                f"- Translate ALL sections faithfully: Executive Summary, Key Claims, Evidence, Bibliography\n"
                f"- Preserve scientific accuracy --- translate meaning, not word-for-word\n"
                f"- Keep confidence labels (HIGH/MEDIUM/LOW/CONTESTED) intact\n"
                f"- Keep study names, journal names, and URLs in English\n"
                f"- Keep clinical abbreviations in English: ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR\n"
                f"- Maintain all markdown formatting (headers, tables, bullet points)\n"
                f"- Preserve ALL numerical values exactly (percentages, CI ranges, p-values, sample sizes) --- do NOT convert or round\n"
                + f"{target_instruction}"
            ),
            expected_output=(
                f"Complete {language_config['name']} translation of the Source-of-Truth document, "
                f"preserving all sections, claims, confidence levels, and evidence citations."
            ),
            agent=producer_agent,
            context=[],
        )

    polish_task = Task(
        description=(
            f"Polish the \"{topic_name}\" dialogue for natural spoken delivery at Masters-level.\n\n"
            + get_prompt("polish", "masters_level", language)
            + f"- Target length: {target_script} {target_unit_plural} (acceptable range: "
            f"{int(target_length_int * (1 - SCRIPT_TOLERANCE)):,}–{int(target_length_int * (1 + SCRIPT_TOLERANCE)):,})\n\n"
            f"MAINTAIN ROLES:\n"
            f"  - {SESSION_ROLES['presenter']['label']} (Presenter): explains and teaches the topic\n"
            f"  - {SESSION_ROLES['questioner']['label']} (Questioner): asks bridging questions, occasionally pushes back\n\n"
            f"VERIFY 8-PART STRUCTURE (all must be present):\n"
            f"  1. Channel Intro\n"
            f"  2. Hook (provocative question)\n"
            + get_prompt("polish", "structure_acts", language) +
            f"  7. Wrap-up\n"
            f"  8. One Action Ending\n\n"
            + (f"CHANNEL INTRO VERIFICATION:\n"
               f"The Channel Intro MUST contain this EXACT text: \"{channel_intro}\"\n"
               f"Do NOT rephrase, modify, or remove it.\n\n"
               if channel_intro else '')
            + f"TRANSITION MARKERS:\n"
            f"Insert [TRANSITION] on its own line between major sections:\n"
            f"  - After Channel Intro, before Act 1\n"
            f"  - Between Act 1 and Act 2\n"
            f"  - Between Act 2 and Act 3\n"
            f"  - Between Act 3 and Act 4\n"
            f"  - After Act 4, before Wrap-up\n"
            f"These markers create musical transition moments in the final audio. Do NOT speak them.\n"
            f"Format: place [TRANSITION] on a line by itself between the last line of one act and the first line of the next.\n\n"
            f"ONE ACTION ENDING CHECK:\n"
            f"Verify the script ends with a single, specific, actionable recommendation.\n"
            f"If missing, add one based on the Protocol section (Act 4).\n\n"
            f"GRADE FRAMING CHECK:\n"
            f"Verify that claims use appropriate hedging language per confidence level.\n"
            f"Do NOT present LOW-confidence claims as settled fact.\n\n"
            f"Format:\n{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
            f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
            f"Remove meta-tags, markdown, stage directions. Dialogue only (plus [TRANSITION] markers).\n"
            + get_prompt("polish", "length_section", language,
                         target_script=target_script,
                         target_unit_plural=target_unit_plural,
                         range_low=f"{int(target_length_int * (1 - SCRIPT_TOLERANCE)):,}",
                         range_high=f"{int(target_length_int * (1 + SCRIPT_TOLERANCE)):,}")
            + (f"\nCRITICAL: Output MUST be in Japanese only. Do NOT switch to Chinese. "
               f"Keep speaker labels exactly as 'Host 1:' and 'Host 2:' --- do NOT replace them with Japanese names. "
               f"Avoid Kanji that is only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50). "
               if language == 'ja' else '')
            + f"{target_instruction}"
        ),
        expected_output=(
            f"Final Masters-level dialogue about {topic_name}, approximately {target_script} {target_unit_plural} "
            f"(at least {int(target_length_int * (1 - SCRIPT_TOLERANCE)):,}). "
            f"8-part structure with [TRANSITION] markers between acts. One Action ending present. "
            f"{target_instruction}"
        ),
        agent=editor_agent,
        context=[script_task]
    )

    audit_task = Task(
        description=(
            f"Compare the POLISHED SCRIPT against the Source-of-Truth document for {topic_name}.\n\n"
            f"This is a POST-POLISH accuracy check. The script has been edited for natural delivery, "
            f"and scientific drift may have been introduced during polishing.\n\n"
            f"CHECK FOR THESE SPECIFIC DRIFT PATTERNS:\n"
            f"1. **Correlation -> Causation drift**: Script says 'X causes Y' when source says 'X is associated with Y'\n"
            f"2. **Hedge removal**: Source says 'may' or 'suggests', script says 'does' or 'proves'\n"
            f"3. **Confidence inflation**: LOW confidence claims presented as settled fact\n"
            f"4. **Cherry-picking**: Only one side of CONTESTED claims presented\n"
            f"5. **Contested-as-settled**: Claims marked CONTESTED in source-of-truth presented as consensus\n"
            + (f"6. **Language consistency**: Flag any non-{language_config['name']} sentences that should be in {language_config['name']}. "
               f"(Exclude scientific abbreviations: ARR, NNT, GRADE, RCT, CI, HR, OR)\n\n"
               if language != 'en' else '\n')
            + f"OUTPUT FORMAT:\n"
            f"# Accuracy Check: {topic_name}\n\n"
            f"## Overall Assessment\n"
            f"[PASS / PASS WITH WARNINGS / FAIL]\n\n"
            f"## Drift Instances Found\n"
            f"For each issue:\n"
            f"- **Script says**: [exact quote from script]\n"
            f"- **Source-of-truth says**: [what the evidence actually supports]\n"
            f"- **Drift type**: [one of the 5 patterns above]\n"
            f"- **Severity**: HIGH / MEDIUM / LOW\n\n"
            f"## Recommendations\n"
            f"[Specific line-level fixes if needed]\n\n"
            f"NOTE: HIGH-severity drift will trigger a script correction pass before audio generation. "
            f"{target_instruction}"
        ),
        expected_output=(
            f"Accuracy check report comparing polished script against source-of-truth, "
            f"listing any scientific drift with severity ratings. {target_instruction}"
        ),
        agent=auditor_agent,
        context=[polish_task],
        output_file=os.path.relpath(output_path_fn(output_dir, "accuracy_audit.md"))
    )

    # --- Audience context for blueprint & script prompts ---
    _audience_context = ""
    if core_target:
        _audience_context += f"TARGET AUDIENCE: {core_target}\n"
    if channel_mission:
        _audience_context += f"CHANNEL MISSION: {channel_mission}\n"

    _audience_context_block = (
        _audience_context + "Tailor the value proposition to this specific audience.\n"
    ) if _audience_context else ""

    # Content framework hint based on channel mission
    _framework_hint = ""
    if channel_mission and any(kw in channel_mission.lower() for kw in ("actionable", "practical", "protocol", "how-to", "how to")):
        _framework_hint = "Note: The channel mission suggests PPP (Problem-Proof-Protocol) may be a good fit.\n"

    blueprint_task = Task(
        description=(
            f"Create an Episode Blueprint for the podcast episode on \"{topic_name}\".\n\n"
            f"RESEARCH WORKFLOW — Follow these two passes:\n\n"
            f"Pass 1 (Full Read): Start by reading the complete Source of Truth:\n"
            f"  ReadFullReport('sot')\n"
            f"This automatically gives you the SOT in the target language. Use it to understand\n"
            f"the full evidence landscape and draft the high-level Episode Blueprint structure:\n"
            f"narrative arc, segment order, key claims, debate tension.\n\n"
            f"Pass 2 (Deep Dive): Based on your blueprint, read 2-3 specific SOT sections\n"
            f"to enrich areas that need more depth:\n"
            f"  ReadFullReport('sot:discussion')  <- §5: affirmative case, falsification, GRADE\n"
            f"  ReadFullReport('sot:results')     <- §4: study characteristics, effect sizes\n"
            f"  ReadFullReport('sot:abstract')    <- §1: high-level findings summary\n"
            f"Use these details to add specific data points, study findings, and nuance\n"
            f"to your blueprint. Your blueprint MUST reflect BOTH supporting (§5.1) and\n"
            f"contradicting (§5.2) evidence.\n\n"
            f"Do NOT use more than 4 total ReadFullReport calls to stay within context limits.\n\n"
            f"This is a CONTENT STRATEGY document that guides the script writer. It defines what the episode "
            f"will say, why listeners should care, and how to structure the narrative.\n\n"
            f"OUTPUT FORMAT --- produce ALL 7 sections:\n\n"
            f"# Episode Blueprint: {topic_name}\n\n"
            f"## 1. Episode Thesis\n"
            f"One sentence: what this episode will prove or explore.\n\n"
            f"## 2. Listener Value Proposition\n"
            f"{_audience_context_block}"
            f"- What will the listener GAIN from this episode?\n"
            f"- Why should they listen to THIS episode instead of reading an article?\n"
            f"- What will they be able to DO differently after listening?\n\n"
            f"## 3. Hook\n"
            f"A provocative QUESTION for listeners based on the most surprising finding from the research.\n"
            f"The question should make listeners want to know the answer and feel personally relevant.\n"
            f"BAD: 'Have you ever wondered about coffee?' (too vague)\n"
            f"GOOD: 'What if your morning coffee habit was actually adding years to your life --- but only if you drink exactly the right amount?'\n\n"
            f"## 4. Content Framework\n"
            f"{_framework_hint}"
            f"Choose ONE:\n"
            f"- [PPP] Problem-Proof-Protocol --- if the topic has a clear actionable outcome\n"
            f"- [QEI] Question-Evidence-Insight --- if the topic is exploratory with no single recommendation\n\n"
            + get_prompt("blueprint", "section5_intro", language,
                        core_target_or_default=core_target or "curious listener")
            + get_prompt("blueprint", "act1_header", language)
            + get_prompt("blueprint", "act1_description", language)
            + "\n"
            + get_prompt("blueprint", "act1_discussion", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + get_prompt("blueprint", "act2_header", language)
            + get_prompt("blueprint", "act2_description", language)
            + "\n"
            + get_prompt("blueprint", "act2_bad_example", language) + "\n"
            + get_prompt("blueprint", "act2_good_example", language) + "\n\n"
            + get_prompt("blueprint", "act2_sub_structure", language)
            + "\n"
            + get_prompt("blueprint", "act2_discussion", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + get_prompt("blueprint", "act3_header", language)
            + get_prompt("blueprint", "act3_description", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + get_prompt("blueprint", "act3_discussion", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + get_prompt("blueprint", "act4_header", language)
            + get_prompt("blueprint", "act4_description", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + get_prompt("blueprint", "act4_bad_example", language) + "\n"
            + get_prompt("blueprint", "act4_good_example", language) + "\n\n"
            + get_prompt("blueprint", "act4_discussion", language,
                         core_target_or_default=core_target or "curious listener")
            + "\n"
            + f"## 6. GRADE-Informed Framing Guide\n"
            f"For each major claim in the episode, specify the appropriate framing language.\n"
            f"Use this mapping based on the evidence confidence:\n"
            f"- HIGH confidence -> 'Research clearly demonstrates...'\n"
            f"- MODERATE confidence -> 'Evidence suggests...'\n"
            f"- LOW confidence -> 'Emerging research indicates...'\n"
            f"- VERY LOW confidence -> 'Preliminary findings hint at...'\n"
            f"List each major claim with its recommended framing.\n\n"
            f"## 7. Citations\n"
            f"### Supporting Evidence\n"
            f"- [Study Title] (Journal, Year) - [URL] - **Validity: V High/Medium/Low**\n"
            f"  - Evidence Type: [RCT/Observational/Animal Model]\n"
            f"  - Key Finding: [One sentence summary]\n\n"
            f"### Contradicting Evidence\n"
            f"- [Study Title] (Journal, Year) - [URL] - **Validity: V High/Medium/Low**\n"
            f"  - Evidence Type: [RCT/Observational/Animal Model]\n"
            f"  - Key Finding: [One sentence summary]\n\n"
            f"Include validity ratings from the Reliability Scorecard. "
            f"Mark broken links as 'X Broken Link'.\n\n"
            f"{target_instruction}"
        ),
        expected_output=(
            f"Episode Blueprint with all 7 sections: thesis, listener value proposition, hook, "
            f"content framework (PPP or QEI), 4-act narrative arc with 5-8 inline discussion points per act "
            f"(3-5 for Act 3), GRADE framing guide, and citations. "
            f"{target_instruction}"
        ),
        agent=producer_agent,
        context=[],
        output_file=os.path.relpath(output_path_fn(output_dir, "EPISODE_BLUEPRINT.md"))
    )

    # --- CONTEXT CHAIN: script_task always depends on blueprint_task ---
    script_task.context = [blueprint_task]

    # --- SOT TRANSLATION PIPELINE: Update contexts when translating ---
    if translation_task is not None:
        script_task.context = [blueprint_task, translation_task]
        blueprint_task.context = [translation_task]
        polish_task.context = [script_task, translation_task]
        audit_task.context = [polish_task, translation_task]

    return {
        'auditor_agent': auditor_agent,
        'producer_agent': producer_agent,
        'editor_agent': editor_agent,
        'framing_agent': framing_agent,
        'framing_task': framing_task,
        'script_task': script_task,
        'translation_task': translation_task,
        'polish_task': polish_task,
        'audit_task': audit_task,
        'blueprint_task': blueprint_task,
    }


# ---------------------------------------------------------------------------
# Phase/task metadata and progress tracking
# ---------------------------------------------------------------------------

PHASE_MARKERS = [
    ("PHASE 0: RESEARCH FRAMING", "Research Framing", 5),
    ("PHASE 1: CLINICAL RESEARCH", "Clinical Research", 10),
    ("PHASE 2: SOURCE VALIDATION", "Source Validation", 50),
    ("PHASE 3: REPORT TRANSLATION", "Report Translation", 55),
    ("PHASE 4: SHOW OUTLINE", "Show Outline", 60),
    ("PHASE 5: SCRIPT WRITING", "Script Writing", 75),
    ("PHASE 6: SCRIPT POLISH", "Script Polish", 90),
    ("PHASE 7: ACCURACY AUDIT", "Accuracy Audit", 95),
    ("PHASE 8: AUDIO PRODUCTION", "Audio Production", 98),
]

TASK_METADATA = {
    'framing_task': {
        'name': 'Research Framing',
        'phase': '0',
        'estimated_duration_min': 2,
        'description': 'Defining scope, questions, and evidence criteria',
        'agent': 'Research Framing Specialist',
        'dependencies': [],
        'crew': 1
    },
    'clinical_research': {
        'name': 'Clinical Research (7-Step Pipeline)',
        'phase': '1',
        'estimated_duration_min': 6,
        'description': 'PICO strategy, wide net, screening, extraction, cases, math, GRADE synthesis',
        'agent': 'Dual-Model Pipeline',
        'dependencies': ['framing_task'],
        'crew': 'procedural'
    },
    'source_validation': {
        'name': 'Source Validation',
        'phase': '2',
        'estimated_duration_min': 1,
        'description': 'Batch HEAD requests to validate all cited URLs',
        'agent': 'Automated',
        'dependencies': ['clinical_research'],
        'crew': 'procedural'
    },
    'translation_task': {
        'name': 'Report Translation',
        'phase': '3',
        'estimated_duration_min': 3,
        'description': 'Translate SOT to target language (conditional)',
        'agent': 'Podcast Producer',
        'dependencies': ['source_validation'],
        'crew': 2,
        'conditional': True
    },
    'blueprint_task': {
        'name': 'Show Outline',
        'phase': '4',
        'estimated_duration_min': 3,
        'description': 'Developing show outline, citations, and narrative arc',
        'agent': 'Podcast Producer',
        'dependencies': ['translation_task'],
        'crew': 3
    },
    'script_task': {
        'name': 'Script Writing',
        'phase': '5',
        'estimated_duration_min': 6,
        'description': 'Script writing and conversation generation',
        'agent': 'Podcast Producer',
        'dependencies': ['blueprint_task'],
        'crew': 3
    },
    'polish_task': {
        'name': 'Script Polish',
        'phase': '6',
        'estimated_duration_min': 5,
        'description': 'Script polishing for natural verbal delivery',
        'agent': 'Podcast Editor',
        'dependencies': ['script_task'],
        'crew': 3
    },
    'audit_task': {
        'name': 'Accuracy Audit',
        'phase': '7',
        'estimated_duration_min': 3,
        'description': 'Advisory drift detection against Source-of-Truth',
        'agent': 'Scientific Auditor',
        'dependencies': ['polish_task'],
        'crew': 3
    },
}


def display_workflow_plan(topic_name, language_config, output_dir):
    """
    Display detailed workflow plan before execution.
    Shows Phases 0-8 with durations, dependencies, and total time estimate.
    Phase 2b is marked as conditional.
    """
    logger.info("\n" + "="*70)
    logger.info(" "*20 + "PODCAST GENERATION WORKFLOW")
    logger.info("="*70)
    logger.info("\nTopic: %s", topic_name)
    logger.info("Language: %s", language_config['name'])
    logger.info("Output Directory: %s", output_dir)
    logger.info("\n" + "-"*70)
    logger.info("%-6s %-40s %-12s %-25s", "PHASE", "TASK NAME", "EST TIME", "AGENT")
    logger.info("-"*70)

    total_duration = 0
    for task_name, metadata in TASK_METADATA.items():
        phase = metadata['phase']
        name = metadata['name']
        duration = metadata['estimated_duration_min']
        agent = metadata['agent']
        is_conditional = metadata.get('conditional', False)

        if not is_conditional:
            total_duration += duration

        conditional_marker = " [CONDITIONAL]" if is_conditional else ""
        logger.info("%-6s %-40s %3d min       %-25s%s", phase, name, duration, agent, conditional_marker)
        logger.info("       |-- %s", metadata['description'])
        if metadata['dependencies']:
            deps_str = ', '.join(["Phase %s" % TASK_METADATA[d]['phase'] for d in metadata['dependencies'] if d in TASK_METADATA])
            logger.info("          Dependencies: %s", deps_str)
        logger.info("")

    logger.info("-"*70)
    logger.info("TOTAL ESTIMATED TIME: %d minutes (~%dh %dm)", total_duration, total_duration // 60, total_duration % 60)
    logger.info("  (+ up to 4 min if gap-fill triggers)")
    logger.info("="*70 + "\n")


class ProgressTracker:
    """
    Real-time progress tracking for CrewAI task execution.
    Tracks current phase, elapsed time, and estimated remaining time.
    """
    def __init__(self, task_metadata: dict):
        self.task_metadata = task_metadata
        self.task_names = list(task_metadata.keys())
        self.current_task_index = 0
        self.total_phases = len([m for m in task_metadata.values() if not m.get('conditional', False)])
        self.start_time = None
        self.task_start_time = None
        self.completed_tasks = []

    def start_workflow(self):
        """Mark workflow start time"""
        self.start_time = time.time()
        logger.info("\n" + "="*70)
        logger.info("WORKFLOW EXECUTION STARTED")
        logger.info("="*70 + "\n")

    def task_started(self, task_index: int):
        """Called when a task begins"""
        if task_index >= len(self.task_names):
            return

        task_name = self.task_names[task_index]
        self.current_task_index = task_index
        self.task_start_time = time.time()

        metadata = self.task_metadata[task_name]

        logger.info("\n" + "="*70)
        logger.info("PHASE %s/%d: %s", metadata['phase'], self.total_phases, metadata['name'].upper())
        logger.info("="*70)
        logger.info("Agent: %s", metadata['agent'])
        logger.info("Description: %s", metadata['description'])
        logger.info("Estimated Duration: %d minutes", metadata['estimated_duration_min'])
        if metadata['dependencies']:
            deps_str = ', '.join([self.task_metadata[d]['name'] for d in metadata['dependencies'] if d in self.task_metadata])
            logger.info("Dependencies: %s", deps_str)
        logger.info("-"*70)

    def task_completed(self, task_index: int):
        """Called when a task completes"""
        if self.start_time is None or self.task_start_time is None:
            return
        if task_index >= len(self.task_names):
            return

        task_name = self.task_names[task_index]
        elapsed_task = time.time() - self.task_start_time
        self.completed_tasks.append({
            'name': task_name,
            'duration': elapsed_task
        })

        # Calculate progress
        progress_pct = (len(self.completed_tasks) / self.total_phases) * 100

        # Calculate time estimates
        elapsed_total = time.time() - self.start_time
        avg_time_per_task = elapsed_total / len(self.completed_tasks)
        remaining_tasks = self.total_phases - len(self.completed_tasks)
        estimated_remaining = avg_time_per_task * remaining_tasks

        metadata = self.task_metadata[task_name]

        logger.info("\n" + "="*70)
        logger.info("PHASE %s/%d COMPLETED", metadata['phase'], self.total_phases)
        logger.info("="*70)
        logger.info("Task Duration: %.1f minutes (%.0f seconds)", elapsed_task / 60, elapsed_task)
        logger.info("Total Elapsed: %.1f minutes", elapsed_total / 60)
        logger.info("Progress: %.1f%% complete (%d/%d tasks)", progress_pct, len(self.completed_tasks), self.total_phases)
        logger.info("Estimated Remaining: %.1f minutes", estimated_remaining / 60)
        logger.info("="*70 + "\n")

    def workflow_completed(self):
        """Called when entire workflow finishes"""
        total_time = time.time() - self.start_time

        logger.info("\n" + "="*70)
        logger.info(" "*22 + "WORKFLOW COMPLETED")
        logger.info("="*70)
        logger.info("\nTotal Execution Time: %.1f minutes (%.2f hours)", total_time / 60, total_time / 3600)
        logger.info("Tasks Completed: %d/%d", len(self.completed_tasks), self.total_phases)

        logger.info("\n%s", "Task Performance Summary".center(70))
        logger.info("-"*70)
        for i, task_info in enumerate(self.completed_tasks, 1):
            task_name = task_info['name']
            duration = task_info['duration']
            estimated = self.task_metadata[task_name]['estimated_duration_min'] * 60
            variance = ((duration - estimated) / estimated) * 100 if estimated > 0 else 0

            logger.info("%d. %-40s %6.1f min (est: %.1f min, %+.0f%%)",
                  i, self.task_metadata[task_name]['name'],
                  duration / 60, estimated / 60, variance)

        logger.info("="*70 + "\n")
