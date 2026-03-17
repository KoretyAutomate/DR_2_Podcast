"""
Script validation, condensing, and deduplication utilities.

Extracted from pipeline.py (T4.1).
Contains: _validate_script, _count_words, _deduplicate_script,
_parse_blueprint_inventory, _run_condense_pass (_run_trim_pass alias),
_add_reaction_guidance, _quick_content_audit.
"""

import logging
import re
from dr2_podcast.utils import strip_think_blocks

logger = logging.getLogger(__name__)

# Script tolerance constant (shared with pipeline.py)
SCRIPT_TOLERANCE = 0.10  # +/-10% around target length is acceptable


def _add_reaction_guidance(script_text: str, language_config: dict,
                           *, _call_smart_model) -> str:
    """Insert ## [emotion, delivery cue] lines before key Host dialogue lines.

    Uses a two-phase approach:
    1. Send numbered Host lines to the LLM and ask it to output ONLY annotations
       as "LINE_NUM: [cue]" pairs (no script reproduction required).
    2. Programmatically insert ## [cue] lines before the matched Host lines.

    This avoids the model's tendency to reproduce the script verbatim without
    inserting annotations when asked to return the modified script.

    Annotations are always in English (metadata, not dialogue).
    ~40-60% of Host lines are annotated -- routine connectors are skipped.
    Lines starting with ## are stripped by clean_script_for_tts() before TTS.
    """
    system = (
        "You are a podcast delivery coach. Analyze the numbered Host dialogue "
        "lines below and generate emotion/delivery cues for vocal performance.\n\n"
        "OUTPUT FORMAT \u2014 one annotation per line, exactly:\n"
        "LINE_NUMBER: [cue1, cue2]\n\n"
        "Example output:\n"
        "2: [intrigued, building suspense]\n"
        "3: [genuinely curious, leaning in]\n"
        "5: [authoritative, measured pace]\n"
        "8: [surprised, delighted]\n\n"
        "RULES:\n"
        "1. Annotate 40-60% of the Host lines \u2014 those with distinct emotional "
        "tone, delivery shifts, or notable reactions.\n"
        "2. SKIP routine/neutral connective lines (e.g. 'Right.', 'Exactly.', "
        "'Go on.', short acknowledgements).\n"
        "3. Cues are ALWAYS in English regardless of the script language.\n"
        "4. Keep cues concise \u2014 2-4 words per cue, 1-2 cues per line.\n"
        "5. Output ONLY the numbered annotations, nothing else \u2014 no preamble, "
        "no commentary, no script text."
    )

    try:
        # Extract Host lines with their original line indices
        all_lines = script_text.split('\n')
        host_entries = []  # (original_line_index, line_text)
        for i, line in enumerate(all_lines):
            if re.match(r'^\*{0,2}(?:Host\s+|ホスト\s*)\d\s*\*{0,2}\s*[:：]', line):
                host_entries.append((i, line))

        if not host_entries:
            logger.warning("  No Host lines found -- skipping reaction guidance")
            return script_text

        total_host_lines = len(host_entries)

        # Build numbered list for the LLM
        numbered_lines = []
        for idx, (_, line_text) in enumerate(host_entries, 1):
            # Truncate very long lines to keep prompt manageable
            display = line_text[:200] + "..." if len(line_text) > 200 else line_text
            numbered_lines.append(f"{idx}: {display}")

        user_prompt = (
            f"Generate emotion/delivery annotations for these {total_host_lines} "
            f"Host dialogue lines. Annotate approximately {total_host_lines * 2 // 5}-"
            f"{total_host_lines * 3 // 5} of them (40-60%).\n\n"
            + "\n".join(numbered_lines)
        )

        result = _call_smart_model(
            system=system,
            user=user_prompt,
            max_tokens=2000,
            temperature=0.4,
        )

        # Parse LLM output: extract "NUMBER: [cue]" lines
        annotations = {}  # host_entry_index (1-based) -> cue text
        for match in re.finditer(r'^(\d+)\s*:\s*(\[.+?\])\s*$', result, re.MULTILINE):
            line_num = int(match.group(1))
            cue = match.group(2)
            if 1 <= line_num <= total_host_lines:
                annotations[line_num] = cue

        if not annotations:
            logger.warning("  LLM returned no parseable annotations -- keeping original")
            return script_text

        # Build the annotated script by inserting ## [cue] before target Host lines
        # Map host_entry_index (1-based) -> original line index
        insert_before = {}  # original_line_index -> cue
        for entry_idx, cue in annotations.items():
            orig_line_idx = host_entries[entry_idx - 1][0]
            insert_before[orig_line_idx] = cue

        result_lines = []
        for i, line in enumerate(all_lines):
            if i in insert_before:
                result_lines.append(f"## {insert_before[i]}")
            result_lines.append(line)

        result = '\n'.join(result_lines)
        annotation_rate = len(annotations) / total_host_lines

        logger.info("  Reaction guidance complete -- %d annotations added (%d%% of %d Host lines)",
              len(annotations), int(annotation_rate * 100), total_host_lines)
        return result

    except Exception as e:
        logger.warning("  Reaction guidance failed: %s -- keeping original", e)
        return script_text


def _quick_content_audit(script_text: str, sot_content: str,
                         *, _call_smart_model, _truncate_at_boundary) -> str:
    """Check script for top-3 drift patterns against SOT. Returns issue string or None."""
    sot_excerpt = _truncate_at_boundary(sot_content, 4000)
    script_excerpt = _truncate_at_boundary(script_text, 4000)
    system = (
        "You are a scientific accuracy auditor. Compare the podcast script excerpt against "
        "the source-of-truth research. Check for these 3 drift patterns ONLY:\n"
        "1. CAUSATION INFLATION: Script states causation where source only shows correlation\n"
        "2. HEDGE REMOVAL: Script presents uncertain findings as settled fact\n"
        "3. CHERRY-PICKING: Script omits contradicting evidence that the source includes\n\n"
        "If the script is faithful to the source, respond with exactly: CLEAN\n"
        "If you find drift, respond with a 1-2 sentence description of the issue."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=(
                f"SOURCE OF TRUTH (excerpt):\n{sot_excerpt}\n\n"
                f"PODCAST SCRIPT (excerpt):\n{script_excerpt}"
            ),
            max_tokens=200,
            temperature=0.1,
        )
        if result.strip().upper() == "CLEAN":
            return None
        return result.strip()
    except Exception as e:
        logger.warning("  Content audit call failed: %s -- skipping", e)
        return None


def _validate_script(script_text: str, target_length: int, tolerance: float,
                     language_config: dict, sot_content: str, stage: str,
                     *, _call_smart_model=None, _truncate_at_boundary=None) -> dict:
    """
    Validate script for length, structure, repetition, and content accuracy.
    Returns: {'pass': bool, 'feedback': str, 'word_count': int, 'issues': list}
    """
    issues = []

    # Strip <think> blocks before measuring (Qwen3 safety net)
    script_text = strip_think_blocks(script_text)

    length_unit = language_config['length_unit']

    # 1. Measure word/char count
    if length_unit == 'chars':
        count = len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014*#]', '', script_text))
    else:
        content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', script_text, flags=re.MULTILINE)
        count = len(content_only.split())

    low = int(target_length * (1 - tolerance))
    high = int(target_length * (1 + tolerance))

    if count < low:
        issues.append(f"TOO SHORT: {count} {length_unit} (need \u2265{low})")
    elif count > high:
        issues.append(f"TOO LONG: {count} {length_unit} (need \u2264{high})")

    # 2. Degenerate repetition detection
    words = script_text.lower().split()
    consecutive = 1
    for i in range(1, len(words)):
        if words[i] == words[i-1] and len(words[i]) > 2:
            consecutive += 1
            if consecutive >= 4:
                issues.append(f"DEGENERATE REPETITION: '{words[i]}' repeated {consecutive}+ times consecutively")
                break
        else:
            consecutive = 1

    # 3. Structure check (for polish stage)
    if stage == 'polish':
        transition_count = script_text.count('[TRANSITION]')
        if transition_count < 3:
            issues.append(f"MISSING TRANSITIONS: only {transition_count} [TRANSITION] markers (need \u22653)")

    # 4. LLM content audit (only if Python checks pass -- saves tokens)
    if not issues and sot_content and _call_smart_model and _truncate_at_boundary:
        content_audit = _quick_content_audit(
            script_text, sot_content,
            _call_smart_model=_call_smart_model,
            _truncate_at_boundary=_truncate_at_boundary)
        if content_audit:
            issues.append(f"CONTENT: {content_audit}")

    return {
        'pass': len(issues) == 0,
        'feedback': '\n'.join(f"- {i}" for i in issues) if issues else 'PASS',
        'word_count': count,
        'issues': issues
    }


# --- SCRIPT HELPERS ---

def _count_words(text: str, language_config: dict) -> int:
    """Count words (English) or content characters (Japanese) in text."""
    if language_config['length_unit'] == 'chars':
        return len(re.sub(r'[\s\n\r\t\u3000\uff1a:\u300c\u300d\u3001\u3002\u30fb\uff08\uff09\-\u2014*#]', '', text))
    else:
        content_only = re.sub(r'^[A-Za-z0-9_ ]+:\s*', '', text, flags=re.MULTILINE)
        return len(content_only.split())


def _deduplicate_script(script_text: str, language_config: dict) -> str:
    """Remove repeated dialogue blocks from a script.

    Uses a sliding window to detect blocks of 3+ consecutive non-empty lines
    that appear verbatim more than once. Keeps the first occurrence, removes
    duplicates. Preserves [TRANSITION] markers and ## annotations.
    """
    lines = script_text.split('\n')
    WINDOW_SIZE = 3  # minimum consecutive lines to detect as a block

    # Build set of line-block fingerprints (tuples of WINDOW_SIZE consecutive non-empty lines)
    # Track which lines are part of duplicate blocks
    duplicate_line_indices = set()

    # Collect all non-empty line indices
    non_empty = [(i, lines[i]) for i in range(len(lines)) if lines[i].strip()]

    # Sliding window over non-empty lines to find repeated blocks
    seen_blocks = {}  # fingerprint -> first occurrence index in non_empty
    for start in range(len(non_empty) - WINDOW_SIZE + 1):
        block = tuple(non_empty[start + j][1] for j in range(WINDOW_SIZE))
        # Skip blocks that are only markers/annotations
        if all(l.startswith('[TRANSITION]') or l.startswith('## [') for l in block):
            continue
        if block in seen_blocks:
            first_pos = seen_blocks[block]
            # This is a duplicate -- mark for removal
            # But extend: keep scanning forward to find the full repeated span
            span = WINDOW_SIZE
            while (start + span < len(non_empty)
                   and first_pos + span < len(non_empty)
                   and non_empty[start + span][1] == non_empty[first_pos + span][1]):
                span += 1
            for j in range(span):
                duplicate_line_indices.add(non_empty[start + j][0])
        else:
            seen_blocks[block] = start

    if not duplicate_line_indices:
        return script_text

    # Remove duplicate lines, but keep empty lines that are not adjacent to removed lines
    cleaned = []
    for i, line in enumerate(lines):
        if i not in duplicate_line_indices:
            cleaned.append(line)

    # Clean up excessive blank lines left by removal
    result_lines = []
    blank_count = 0
    for line in cleaned:
        if not line.strip():
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)

    result = '\n'.join(result_lines)
    removed_count = len(duplicate_line_indices)
    removed_pct = removed_count / len(lines) * 100 if lines else 0
    length_unit = language_config.get('length_unit', 'chars')
    logger.info("  Deduplication: removed %d duplicate lines (%.0f%% of script)", removed_count, removed_pct)
    if removed_pct > 15:
        logger.warning("  Expansion produced %.0f%% duplicate content -- removed", removed_pct)

    return result


def _parse_blueprint_inventory(blueprint_text: str) -> dict:
    """
    Parse inline Discussion Points from Section 5 of the blueprint.
    Returns {act_label: [{question, answer}]} or {} if Section 5 absent.
    Logs a WARNING if parsing fails so callers can degrade gracefully.

    Also supports legacy Section 8 format for backward compatibility with
    existing blueprint outputs.
    """
    # Try Section 5 first (new inline format)
    section_match = re.search(
        r'(?:^|\n)(?:##\s*5\.?)\s*[^\n]+\n(.*?)(?=\n(?:##\s*\d+)|\Z)',
        blueprint_text, re.DOTALL | re.IGNORECASE
    )
    if not section_match:
        # Fall back to legacy Section 8
        section_match = re.search(
            r'(?:^|\n)(?:##\s*8\.?|#8\.?|\*\*8\.\*\*)\s*Discussion Inventory[^\n]*\n(.*?)(?=\n(?:##\s*\d+|#\d+|\*\*\d+\.\*\*)|\Z)',
            blueprint_text, re.DOTALL | re.IGNORECASE
        )
    if not section_match:
        logger.warning("  blueprint Section 5/8 absent; coverage checklist skipped")
        return {}

    section_text = section_match.group(1)

    # Split by ### Act N headers
    act_blocks = re.split(r'\n(?=###\s*Act\s+\d+)', section_text)

    inventory = {}
    for block in act_blocks:
        # Extract act label from header
        act_header = re.match(r'###\s*(Act\s+\d+[^\n]*)', block)
        if not act_header:
            continue
        act_label = act_header.group(1).strip()

        items = []

        # Try new format first: - Q: ... \n  A: ... (no tier prefix)
        # Handles both plain (- Q:) and bold markdown (- **Q:**) variants
        item_matches = list(re.finditer(
            r'-\s*\*{0,2}Q:\*{0,2}\s*([^\n]+)\n\s*\*{0,2}A:\*{0,2}\s*([^\n]+(?:\n(?!\s*-|\n)[^\n]+)*)',
            block
        ))

        if item_matches:
            for m in item_matches:
                question = m.group(1).strip()
                answer = m.group(2).strip()
                items.append({'question': question, 'answer': answer})
        else:
            # Fall back to legacy format: - [Tier] Q: ... \n  A: ...
            # Also handles bold markdown: - [Tier] **Q:** ...
            legacy_matches = re.finditer(
                r'-\s*\[(Basic|Context|Deep-dive|Unknown)\]\s*\*{0,2}Q:\*{0,2}\s*([^\n]+)\n\s*\*{0,2}A:\*{0,2}\s*([^\n]+(?:\n(?!\s*-|\n)[^\n]+)*)',
                block, re.IGNORECASE
            )
            for m in legacy_matches:
                question = m.group(2).strip()
                answer = m.group(3).strip()
                items.append({'question': question, 'answer': answer})

        if items:
            inventory[act_label] = items

    if not inventory:
        logger.warning("  blueprint discussion points found but no items parsed; coverage checklist skipped")
        return {}

    return inventory


# --- SECTION BUDGET ALLOCATION ---

# Percentages only — absolute budgets derived at runtime from target_length_int.
_SECTION_BUDGET_PCT = {
    'opening':   0.18,   # Channel Intro + Hook + Act 1
    'evidence':  0.50,   # Act 2
    'synthesis': 0.14,   # Act 3
    'closing':   0.18,   # Act 4 + Wrap-up + One Action
}

# Map inventory act labels to section IDs.
# Blueprint acts may have varied suffixes ("Act 1 — The Claim", "Act 1", etc.)
# so we match on the act number.
_ACT_NUM_TO_SECTION = {1: 'opening', 2: 'evidence', 3: 'synthesis', 4: 'closing'}

_SECTION_PACING = {
    'opening':   'High energy opening that hooks the listener, then shift to genuine curiosity as Act 1 establishes emotional stakes.',
    'evidence':  'Alternate between surprise/excitement for new findings and reflective pauses for nuance and limitations. Each study gets its own mini-arc.',
    'synthesis': 'Measured and thoughtful — connect the dots across all studies. Build toward a clear, grounded takeaway.',
    'closing':   'Practical urgency — translate science into action. Build momentum toward the One Action ending, then resolve with confidence.',
}

_SECTION_ACTS = {
    'opening':   ['intro', 'hook', 'act1'],
    'evidence':  ['act2'],
    'synthesis': ['act3'],
    'closing':   ['act4', 'wrapup', 'one_action'],
}


def _allocate_section_budgets(target_length: int, language_config: dict,
                              inventory: dict) -> list[dict]:
    """Divide total word/char budget across 4 sections and assign checklist items.

    All budgets are derived from percentages × target_length — no hard-coded counts.
    Returns a list of 4 section config dicts.
    """
    from dr2_podcast.prompt_strings import get_prompt

    length_unit = language_config['length_unit']
    language = 'ja' if length_unit == 'chars' else 'en'

    # Build section configs with budgets
    sections = []
    for section_id in ('opening', 'evidence', 'synthesis', 'closing'):
        budget = int(_SECTION_BUDGET_PCT[section_id] * target_length)

        # Gather checklist items for this section from the inventory
        checklist_items = []
        for act_label, items in inventory.items():
            # Extract act number from label like "Act 1 — The Claim" or "Act 2"
            act_num_match = re.search(r'Act\s+(\d+)', act_label, re.IGNORECASE)
            if act_num_match:
                act_num = int(act_num_match.group(1))
                if _ACT_NUM_TO_SECTION.get(act_num) == section_id:
                    checklist_items.extend(items)

        # Build act instructions from existing SCRIPT_PROMPTS
        act_instructions_parts = []
        for act in _SECTION_ACTS[section_id]:
            if act in ('intro', 'hook', 'wrapup', 'one_action'):
                continue  # These are handled in the user prompt directly
            try:
                act_key = act.replace('act', 'act')  # act1→act1, act2→act2, etc.
                instr = get_prompt("script", act_key, language,
                                   act2_min=f"{int(target_length * 0.45):,}",
                                   target_unit_plural=length_unit,
                                   core_target_or_default="the listener")
                act_instructions_parts.append(instr)
            except (KeyError, TypeError):
                pass

        sections.append({
            'section_id': section_id,
            'acts': _SECTION_ACTS[section_id],
            'word_budget': budget,
            'length_unit': length_unit,
            'checklist_items': checklist_items,
            'pacing': _SECTION_PACING[section_id],
            'act_instructions': '\n'.join(act_instructions_parts),
        })

    return sections


# --- SECTION GENERATION ---

# Map section_id to user prompt key in SECTION_GEN_PROMPTS
_SECTION_USER_PROMPT_KEY = {
    'opening':   'user_opening',
    'evidence':  'user_evidence',
    'synthesis': 'user_synthesis',
    'closing':   'user_closing',
}


def _generate_section(section_config: dict, previous_lines: list,
                      *, _call_smart_model, language_config: dict,
                      session_roles: dict, topic_name: str,
                      channel_intro: str = '',
                      target_min: int = 30) -> tuple:
    """Generate one section of the podcast script via a single LLM call.

    Args:
        section_config: dict from _allocate_section_budgets()
        previous_lines: last lines of the previous section (for continuity)
        _call_smart_model: callable for the Smart Model
        language_config: language settings dict
        session_roles: presenter/questioner role defs
        channel_intro: channel intro text (only used for opening section)
        target_min: total episode target in minutes

    Returns:
        (section_text, word_count, deficit) where deficit is the shortfall
        from the budget (0 if at or over budget).
    """
    from dr2_podcast.prompt_strings import get_prompt

    section_id = section_config['section_id']
    budget = section_config['word_budget']
    length_unit = section_config['length_unit']
    language = 'ja' if length_unit == 'chars' else 'en'

    presenter = session_roles['presenter']['label']
    questioner = session_roles['questioner']['label']

    # Build speakability rule for this language
    speakability_rule = get_prompt('section_gen', 'speakability_rule', language)

    # Build system prompt
    system = get_prompt('section_gen', 'system', language,
                        topic=topic_name,
                        presenter=presenter,
                        questioner=questioner,
                        presenter_personality=session_roles['presenter']['personality'],
                        questioner_personality=session_roles['questioner']['personality'],
                        speakability_rule=speakability_rule)

    # Build checklist block
    checklist_lines = []
    for item in section_config.get('checklist_items', []):
        checklist_lines.append(f"  Q: {item['question']}")
        checklist_lines.append(f"    -> {item['answer'][:120]}...")
    checklist_block = '\n'.join(checklist_lines) if checklist_lines else '(No checklist items for this section)'

    # Build lead-in from previous section
    lead_in = '\n'.join(previous_lines[-5:]) if previous_lines else '(This is the first section — no prior context)'

    # Channel intro directive for opening section
    if section_id == 'opening' and channel_intro:
        channel_intro_directive = f"Start with this EXACT text: \"{channel_intro}\""
    else:
        channel_intro_directive = f"{presenter}: [Brief show intro — who you are and what the show is about]"

    # Budget percentage
    budget_pct = str(round(budget / (target_min * language_config['speech_rate']) * 100))

    # Build user prompt
    user_key = _SECTION_USER_PROMPT_KEY[section_id]
    user_kwargs = dict(
        word_budget=str(budget),
        length_unit=length_unit,
        budget_pct=budget_pct,
        target_min=str(target_min),
        presenter=presenter,
        questioner=questioner,
        pacing=section_config['pacing'],
        checklist_block=checklist_block,
        lead_in=lead_in,
    )
    if section_id == 'opening':
        user_kwargs['channel_intro_directive'] = channel_intro_directive
    user = get_prompt('section_gen', user_key, language, **user_kwargs)

    # Calculate max_tokens: ~2 tokens/word for EN, ~1.5 tokens/char for JA, with buffer
    if length_unit == 'chars':
        max_tokens = int(budget * 1.5) + 500
    else:
        max_tokens = int(budget * 2) + 500

    # Attempt generation (up to 2 tries)
    floor = int(budget * 0.75)  # 25% under budget = retry threshold
    section_text = ''
    word_count = 0

    for attempt in range(1, 3):
        if attempt == 2:
            # Append retry feedback
            retry_feedback = get_prompt('section_gen', 'retry_feedback', language,
                                        actual_count=str(word_count),
                                        length_unit=length_unit,
                                        floor_count=str(floor))
            user = user + '\n\n' + retry_feedback

        result = _call_smart_model(
            system=system,
            user=user,
            max_tokens=max_tokens,
            temperature=0.5,
            frequency_penalty=0.15,
        )
        section_text = strip_think_blocks(result)
        word_count = _count_words(section_text, language_config)

        if word_count >= floor:
            break
        logger.warning("  Section %s attempt %d: %d %s (need >=%d) — retrying",
                        section_id, attempt, word_count, length_unit, floor)

    # Calculate deficit for redistribution
    deficit = max(0, budget - word_count)
    return section_text, word_count, deficit


def _run_condense_pass(script_text: str, inventory: dict, target_length: int,
                       language_config: dict, session_roles: dict,
                       topic_name: str, target_instruction: str,
                       *, _call_smart_model) -> str:
    """
    Condense an over-target script by rewriting verbose passages more concisely.
    Merges overlapping points, tightens language, reduces filler.
    Does NOT delete entire discussion topics — condenses them.
    Returns condensed text (or original if condensing didn't help).
    """
    from dr2_podcast.prompt_strings import get_prompt

    current = _count_words(script_text, language_config)
    target_with_buffer = int(target_length * 1.05)  # condense to 105%, leave room for polish
    length_unit = language_config['length_unit']

    if current <= target_with_buffer:
        return script_text  # already at or under target+buffer

    presenter = session_roles['presenter']['label']
    questioner = session_roles['questioner']['label']
    floor_count = int(target_length * 0.90)

    system_prompt = get_prompt("condense", "system", "en",
                               topic_name=topic_name,
                               current_count=str(current),
                               length_unit=length_unit,
                               target_count=str(target_with_buffer),
                               presenter=presenter,
                               questioner=questioner,
                               floor_count=str(floor_count),
                               target_instruction=target_instruction)

    user_prompt = get_prompt("condense", "user", "en",
                             current_count=str(current),
                             length_unit=length_unit,
                             script_text=script_text,
                             target_count=str(target_with_buffer))

    try:
        result = _call_smart_model(
            system=system_prompt,
            user=user_prompt,
            max_tokens=16000,
            temperature=0.3,
        )
        result = strip_think_blocks(result)
        result_count = _count_words(result, language_config)
        if result_count < current:
            logger.info("  Condense pass: %d -> %d %s", current, result_count, length_unit)
            return _deduplicate_script(result, language_config)
        else:
            logger.warning("  Condense pass did not reduce length (%d vs %d) -- using original", result_count, current)
            return script_text
    except Exception as e:
        logger.warning("  Condense pass failed (%s) -- using original", e)
        return script_text


# Keep old name as alias for backward compatibility with tests and call sites
_run_trim_pass = _run_condense_pass
