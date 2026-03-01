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
            crew_factory_fn().kickoff()
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
        verbose=True
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
        output_file=os.path.relpath(output_path_fn(output_dir, "RESEARCH_FRAMING.md"))
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

    # Compute approximate word allocations per act
    _act1_target = int(target_length_int * 0.20)
    _act2_target = int(target_length_int * 0.35)
    _act3_target = int(target_length_int * 0.25)
    _act4_target = int(target_length_int * 0.20)

    # Build core_target line (avoid backslash inside f-string on Python 3.11)
    _newline = chr(10)
    _act4_audience_line = (
        f'     - Tailor recommendations specifically to {core_target}{_newline}'
        if core_target
        else f'     - Who should pay attention vs. who can safely ignore this{_newline}'
    )

    script_task = Task(
        description=(
            f"Using the Episode Blueprint, write a comprehensive {target_script}-{target_unit_singular} podcast dialogue about \"{topic_name}\" "
            f"featuring {SESSION_ROLES['presenter']['label']} (presenter) and {SESSION_ROLES['questioner']['label']} (questioner).\n\n"
            f"SCRIPT STRUCTURE (follow this EXACTLY):\n\n"
            + _channel_intro_directive +
            f"  2. THE HOOK (~40 {target_unit_plural}, ~15 seconds):\n"
            f"     Based on the hook question from the Episode Blueprint.\n"
            f"     {SESSION_ROLES['presenter']['label']}: [Provocative question from Blueprint --- must be a question, NOT a statement]\n"
            f"     {SESSION_ROLES['questioner']['label']}: [Engaged reaction: 'Oh, that's a great question!' or 'Hmm, I actually have no idea...']\n\n"
            f"  3. ACT 1 --- THE CLAIM (~{_act1_target:,} {target_unit_plural}):\n"
            f"     What people believe. The folk wisdom. Why this matters personally.\n"
            f"     - Presenter sets up the common belief or question\n"
            f"     - Questioner validates: 'Right, I've heard that too' / 'That's what everyone says'\n"
            f"     - Establish emotional stakes: why should the listener care?\n\n"
            f"  4. ACT 2 --- THE EVIDENCE (~{_act2_target:,} {target_unit_plural}):\n"
            f"     What science actually says. Use BOTH supporting and contradicting evidence from the Blueprint.\n"
            f"     - Present key studies with GRADE-informed framing from the Blueprint's Section 6\n"
            f"     - Include specific numbers (NNT, ARR, sample sizes) where available\n"
            f"     - Questioner challenges: 'But how strong is that evidence?' / 'What about the studies that say otherwise?'\n"
            f"     - Address contradicting evidence honestly --- do NOT cherry-pick\n\n"
            f"  5. ACT 3 --- THE NUANCE (~{_act3_target:,} {target_unit_plural}):\n"
            f"     Where it gets complicated.\n"
            f"     - GRADE confidence level and what it means for the listener\n"
            f"     - Population differences, dose-response relationships, timing factors\n"
            f"     - Questioner pushes: 'So it's not as simple as people think?'\n"
            f"     - Acknowledge what we DON'T know --- science is honest about its limits\n\n"
            f"  6. ACT 4 --- THE PROTOCOL (~{_act4_target:,} {target_unit_plural}):\n"
            f"     Translate science into daily life.\n"
            f"     - Specific, practical recommendations\n"
            f"     - 'In practical terms, this means...'\n"
            f"{_act4_audience_line}"
            f"     - Questioner: 'So what should our listeners actually DO with this?'\n\n"
            f"  7. WRAP-UP (~60 {target_unit_plural}, ~25 seconds):\n"
            f"     Three-sentence summary of the most important takeaways.\n\n"
            f"  8. THE 'ONE ACTION' ENDING (~40 {target_unit_plural}, ~15 seconds):\n"
            f"     {SESSION_ROLES['presenter']['label']}: 'If you take ONE thing from today --- [action{'tailored to ' + core_target if core_target else 'to try this week'}].'\n"
            f"     {SESSION_ROLES['questioner']['label']}: [Brief agreement + sign-off]\n\n"
            f"PERSONALITY DIRECTIVES:\n"
            f"- ENERGY: Vary vocal energy --- excited for surprising findings, thoughtful pauses for nuance, urgency for practical advice\n"
            f"- REACTIONS: Questioner reacts authentically --- genuine surprise ('Wait, seriously?!'), skepticism ('Hmm, that sounds too good to be true...'), humor ('Okay, so basically I've been doing this all wrong')\n"
            f"- BANTER: Include brief moments of friendly banter between hosts --- a shared laugh, a playful jab, a relatable personal admission\n"
            f"- FILLERS: Natural conversational fillers: 'Hm, that's interesting', 'Right, right', 'Oh wow', 'Okay so let me get this straight...'\n"
            f"- EMPHASIS: Dramatic pauses via ellipses: 'And here's where it gets interesting...'\n"
            f"- STORYTELLING: After each key finding, paint a picture: 'Imagine you're...' or 'Think about your morning routine...'\n"
            f"- PERSONAL: Brief personal connections: 'I actually tried this myself and...' or 'My partner always says...'\n"
            f"- MOMENTUM: Each act builds energy --- start curious, peak at the most surprising finding, resolve with practical clarity\n\n"
            f"CHARACTER ROLES:\n"
            f"  - {SESSION_ROLES['presenter']['label']} (Presenter): presents evidence and explains the topic, "
            f"{SESSION_ROLES['presenter']['personality']}\n"
            f"  - {SESSION_ROLES['questioner']['label']} (Questioner): asks questions the audience would ask, bridges gaps, "
            f"{SESSION_ROLES['questioner']['personality']}\n\n"
            f"Format STRICTLY as:\n"
            f"{SESSION_ROLES['presenter']['label']}: [dialogue]\n"
            f"{SESSION_ROLES['questioner']['label']}: [dialogue]\n\n"
            f"TARGET LENGTH: AT LEAST {target_script} {target_unit_plural} (= {_target_min} minutes). "
            f"Aim for {int(target_length_int * 1.2):,} {target_unit_plural}. "
            f"Writing more than the target is fine --- it will be trimmed during polish. "
            f"Writing less will cause the production to FAIL. Cover ALL items in the Coverage Checklist above.\n"
            f"ACT CHECKLIST: You must write all 4 acts plus Hook, Channel Intro, Wrap-up, and One Action. Count them as you write.\n"
            f"TO REACH THIS LENGTH: You must be extremely detailed and conversational. For every single claim or mechanism, you MUST provide:\n"
            f"  1. A deep-dive explanation of the specific scientific mechanism\n"
            f"  2. A real-world analogy or metaphor that lasts several lines\n"
            f"  3. A practical, relatable example or case study\n"
            f"  4. A counter-argument or nuance followed by a rebuttal\n"
            f"  5. Interactive host dialogue (e.g., 'Wait, let me make sure I've got this right...', 'That's fascinating, tell me more about...')\n"
            f"Expand the conversation. Do not just list facts. Have the hosts explore the 'So what?' and 'What now?' for the audience.\n"
            f"Maintain consistent roles throughout. NO role switching mid-conversation. "
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
            f"Follows 8-part structure: Hook, Channel Intro, 4 Acts (Claim, Evidence, Nuance, Protocol), Wrap-up, One Action. "
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
            f"MASTERS-LEVEL REQUIREMENTS:\n"
            f"- Remove ALL definitions of basic scientific concepts (DNA, peer review, RCT, meta-analysis)\n"
            f"- Ensure the questioner's questions feel natural and audience-aligned\n"
            f"- Keep technical language intact - NO dumbing down\n"
            f"- Target exactly {target_script} {target_unit_plural}\n\n"
            f"MAINTAIN ROLES:\n"
            f"  - {SESSION_ROLES['presenter']['label']} (Presenter): explains and teaches the topic\n"
            f"  - {SESSION_ROLES['questioner']['label']} (Questioner): asks bridging questions, occasionally pushes back\n\n"
            f"VERIFY 8-PART STRUCTURE (all must be present):\n"
            f"  1. Channel Intro\n"
            f"  2. Hook (provocative question)\n"
            f"  3. Act 1 --- The Claim\n"
            f"  4. Act 2 --- The Evidence\n"
            f"  5. Act 3 --- The Nuance\n"
            f"  6. Act 4 --- The Protocol\n"
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
            f"- LENGTH: Target exactly {target_script} {target_unit_plural}.\n"
            f"  - If input is OVER target: trim by cutting repetition, redundant examples, and filler --- "
            f"preserve all factual claims and the 8-part structure.\n"
            f"  - If input is AT or SLIGHTLY UNDER target: do NOT shorten further. Add minor depth where natural.\n"
            f"  - Do NOT trim below {int(target_length_int * (1 - SCRIPT_TOLERANCE)):,} {target_unit_plural}.\n"
            + (f"\nCRITICAL: Output MUST be in Japanese only. Do NOT switch to Chinese. "
               f"Keep speaker labels exactly as 'Host 1:' and 'Host 2:' --- do NOT replace them with Japanese names. "
               f"Avoid Kanji that is only used in Chinese (e.g., use \u6c17 instead of \u6c14, \u697d instead of \u4e50). "
               if language == 'ja' else '')
            + f"{target_instruction}"
        ),
        expected_output=(
            f"Final Masters-level dialogue about {topic_name}, exactly {target_script} {target_unit_plural}. "
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
        output_file=os.path.relpath(output_path_fn(output_dir, "ACCURACY_AUDIT.md"))
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
            f"## 5. Narrative Arc (4 Acts)\n"
            f"### Act 1 --- The Claim (~20% of episode)\n"
            f"What people believe. The folk wisdom or common assumption. Why this matters personally.\n"
            f"Key points to cover: [3-4 bullets]\n\n"
            f"### Act 2 --- The Evidence (~35% of episode)\n"
            f"What science actually says. Key studies from BOTH supporting and contradicting evidence.\n"
            f"Supporting evidence: [2-3 key studies with how to frame them]\n"
            f"Contradicting evidence: [1-2 key studies]\n"
            f"Key numbers to cite: [NNT, ARR, sample sizes if available]\n\n"
            f"### Act 3 --- The Nuance (~25% of episode)\n"
            f"Where it gets complicated. Contested findings, population differences, dose-response, limitations.\n"
            f"Key nuance points: [2-3 bullets]\n\n"
            f"### Act 4 --- The Protocol (~20% of episode)\n"
            f"Actionable translation to daily life.\n"
            f"'One Action' for the ending: [specific, memorable, doable this week]\n\n"
            f"## 6. GRADE-Informed Framing Guide\n"
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
            f"## 8. Discussion Inventory\n"
            f"For each of the 4 Acts, list 3-4 discussion items the script writer can elaborate on.\n"
            f"Classify each item's complexity:\n"
            f"  - Basic: core concept --- most listeners need this to follow the episode\n"
            f"  - Context: helpful background --- enriches understanding, skippable for experts\n"
            f"  - Deep-dive: specialist detail --- for curious listeners, optional depth\n\n"
            f"Format STRICTLY as:\n"
            f"### Act 1 --- The Claim\n"
            f"- [Basic] Q: [question a curious listener would ask about this act]\n"
            f"  A: [50-100 word answer with specific details from the research]\n"
            f"- [Context] Q: ...\n"
            f"  A: ...\n\n"
            f"### Act 2 --- The Evidence\n"
            f"[same format]\n\n"
            f"### Act 3 --- The Nuance\n"
            f"[same format]\n\n"
            f"### Act 4 --- The Protocol\n"
            f"[same format]\n\n"
            f"{target_instruction}"
        ),
        expected_output=(
            f"Episode Blueprint with all 8 sections: thesis, listener value proposition, hook, "
            f"content framework (PPP or QEI), 4-act narrative arc, GRADE framing guide, citations, "
            f"and Discussion Inventory (Section 8) with 3-4 Basic/Context/Deep-dive Q&A pairs per act. "
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
        'crew': 2
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
