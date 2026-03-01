"""
Script validation, trimming, and deduplication utilities.

Extracted from pipeline.py (T4.1).
Contains: _validate_script, _count_words, _deduplicate_script,
_parse_blueprint_inventory, _run_trim_pass, _add_reaction_guidance,
_quick_content_audit.
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
            if re.match(r'^Host\s+\d\s*[:\uff1a]', line):
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
    Parse Section 8 Discussion Inventory from blueprint output.
    Returns {act_label: [{tier, question, answer}]} or {} if Section 8 absent.
    Logs a WARNING if parsing fails so callers can degrade gracefully.
    """
    # Find Section 8 (handle variations: ## 8., #8, **8.**)
    section_match = re.search(
        r'(?:^|\n)(?:##\s*8\.?|#8\.?|\*\*8\.\*\*)\s*Discussion Inventory[^\n]*\n(.*?)(?=\n(?:##\s*\d+|#\d+|\*\*\d+\.\*\*)|\Z)',
        blueprint_text, re.DOTALL | re.IGNORECASE
    )
    if not section_match:
        logger.warning("  blueprint Section 8 absent; coverage checklist skipped")
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

        # Find all items: - [Tier] Q: ... \n   A: ...
        items = []
        item_matches = re.finditer(
            r'-\s*\[(Basic|Context|Deep-dive|Unknown)\]\s*Q:\s*([^\n]+)\n\s*A:\s*([^\n]+(?:\n(?!\s*-|\n)[^\n]+)*)',
            block, re.IGNORECASE
        )
        for m in item_matches:
            tier_raw = m.group(1).strip()
            # Normalize tier label
            tier_map = {'basic': 'Basic', 'context': 'Context', 'deep-dive': 'Deep-dive', 'unknown': 'Unknown'}
            tier = tier_map.get(tier_raw.lower(), 'Unknown')
            if tier == 'Unknown' and tier_raw.lower() not in tier_map:
                logger.warning("  unrecognized tier '%s' in blueprint inventory; using 'Unknown'", tier_raw)
            question = m.group(2).strip()
            answer = m.group(3).strip()
            items.append({'tier': tier, 'question': question, 'answer': answer})

        if items:
            inventory[act_label] = items

    if not inventory:
        logger.warning("  blueprint Section 8 found but no items parsed; coverage checklist skipped")
        return {}

    return inventory


def _run_trim_pass(script_text: str, inventory: dict, target_length: int,
                   language_config: dict, session_roles: dict,
                   topic_name: str, target_instruction: str,
                   *, _call_smart_model) -> str:
    """
    Trim an over-target script by removing Deep-dive items first, then Context items.
    Within the same tier, prefer removing examples/analogies before mechanism explanations.
    Returns trimmed text (or original if trim didn't help).
    """
    current = _count_words(script_text, language_config)
    target_with_buffer = int(target_length * 1.05)  # trim to 105%, leave room for polish
    length_unit = language_config['length_unit']

    if current <= target_with_buffer:
        return script_text  # already at or under target+buffer

    # Collect removable items: Deep-dive first, then Context
    removable = []
    for act_label, items in inventory.items():
        for it in items:
            if it['tier'] == 'Deep-dive':
                removable.append((0, act_label, it))  # priority 0 = first to remove
    for act_label, items in inventory.items():
        for it in items:
            if it['tier'] == 'Context':
                removable.append((1, act_label, it))  # priority 1 = second

    if not removable:
        logger.info("  Trim pass: no removable items (no Deep-dive or Context inventory items)")
        return script_text

    presenter = session_roles['presenter']['label']
    questioner = session_roles['questioner']['label']

    # Build list of items to remove for the prompt
    remove_list_lines = ["Items to remove in priority order (Deep-dive first, then Context):"]
    for priority, act_label, it in removable:
        remove_list_lines.append(f"  [{it['tier']}] ({act_label}) {it['question'][:80]}")

    system_prompt = (
        f"You are trimming a two-host science podcast script about \"{topic_name}\" "
        f"from {current} {length_unit} down to approximately {target_with_buffer} {length_unit}.\n"
        f"Hosts: {presenter} (presenter) and {questioner} (questioner).\n\n"
        f"TRIMMING RULES:\n"
        f"- Remove the discussion items listed below in priority order (Deep-dive first, then Context)\n"
        f"- Within the same tier: remove examples and analogies BEFORE mechanism explanations\n"
        f"- Do NOT remove Basic items\n"
        f"- Preserve ALL [TRANSITION] markers exactly as-is\n"
        f"- Preserve the One Action ending\n"
        f"- Preserve speaker labels: {presenter}: and {questioner}:\n"
        f"- Do NOT trim below {int(target_length * 0.90):,} {length_unit}\n"
        f"- Return ONLY the trimmed script dialogue, no commentary\n"
        f"{target_instruction}\n\n"
        + '\n'.join(remove_list_lines)
    )

    user_prompt = (
        f"SCRIPT TO TRIM ({current} {length_unit}):\n\n{script_text}\n\n"
        f"Return the trimmed script at approximately {target_with_buffer} {length_unit}."
    )

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
            logger.info("  Trim pass: %d -> %d %s", current, result_count, length_unit)
            return _deduplicate_script(result, language_config)
        else:
            logger.warning("  Trim pass did not reduce length (%d vs %d) -- using original", result_count, current)
            return script_text
    except Exception as e:
        logger.warning("  Trim pass failed (%s) -- using original", e)
        return script_text
