"""
Translation pipeline for Source-of-Truth documents.

Extracted from pipeline.py (T4.1).
Contains: IMRaD-aware SOT splitting, pipelined translation + audit,
prompt translation, and script language audit.
"""

import logging
import os
import re
import asyncio
from dr2_podcast.utils import strip_think_blocks

logger = logging.getLogger(__name__)


# --- IMRaD-aware SOT splitting for translation prefix caching ---

_IMRAD_HEADERS = {
    "## Abstract",
    "## 1. Introduction",
    "## 2. Methods",
    "## 3. Results",
    "## 4. Discussion",
    "## 5. References",
}


def _split_sot_imrad(sot_content: str) -> list:
    """Split SOT at top-level IMRaD boundaries only.
    Returns [(header, body), ...]. First entry may have header="" (preamble).
    Embedded ## headers (case reports, framing doc) stay inside their parent section.
    """
    sections = []
    current_header = ""
    current_lines = []

    for line in sot_content.splitlines(keepends=True):
        stripped = line.rstrip()
        if stripped in _IMRAD_HEADERS:
            # Flush previous section
            sections.append((current_header, "".join(current_lines)))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    # Flush last section
    sections.append((current_header, "".join(current_lines)))
    return sections


def _estimate_translation_tokens(char_count: int) -> int:
    """Dynamic max_tokens based on input size. Japanese expands ~1.5x vs English in tokens."""
    if char_count < 5000:
        return 4000
    elif char_count <= 15000:
        return 6000
    else:
        return 10000


def _split_at_subheaders(body: str) -> list:
    """Split Discussion body at numbered ### 4.N boundaries only.
    Returns [(sub_header, sub_body), ...]. First entry may have sub_header="" (preamble).
    Embedded ### headers (e.g. ### **Study Designs**) stay inside their parent subsection.
    """
    sections = []
    current_header = ""
    current_lines = []

    for line in body.splitlines(keepends=True):
        stripped = line.rstrip()
        # Only split at numbered Discussion subsections: ### 4.1, ### 4.2, etc.
        if re.match(r"^### 4\.\d", stripped):
            sections.append((current_header, "".join(current_lines)))
            current_header = stripped
            current_lines = []
        else:
            current_lines.append(line)

    sections.append((current_header, "".join(current_lines)))
    return sections


def _translate_sot_pipelined(sot_content: str, language: str, language_config: dict,
                              *, _call_smart_model, _call_mid_model,
                              SMART_BASE_URL, SMART_MODEL,
                              MID_BASE_URL, MID_MODEL) -> str:
    """Pipelined: translate on mid-tier model, audit on smart model.
    Translation and audit overlap via asyncio producer-consumer pattern.
    Falls back to sequential smart-model-only if mid-tier is unavailable.
    """
    # Strip leftover <think> blocks from Qwen3 thinking mode that may be embedded in the SOT
    sot_content = strip_think_blocks(sot_content)

    imrad_sections = _split_sot_imrad(sot_content)

    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "ABSOLUTE RULE: Output MUST be in Japanese (\u65e5\u672c\u8a9e) ONLY. NEVER use Chinese (\u4e2d\u6587).\n"
            "WRONG: \u6267\u884c\u529f\u80fd \u2192 CORRECT: \u5b9f\u884c\u6a5f\u80fd; WRONG: \u8865\u5145 \u2192 CORRECT: \u88dc\u5145\n"
            "If unsure of the Japanese term, keep the English term \u2014 NEVER use Chinese.\n"
            "Common errors to avoid:\n"
            "\u6267\u884c\u2192\u5b9f\u884c, \u8865\u5145\u2192\u88dc\u5145, \u8ba4\u77e5\u2192\u8a8d\u77e5, \u6548\u679c\u2192\u52b9\u679c, \u8425\u517b\u2192\u6804\u990a,\n"
            "\u7ef4\u751f\u7d20\u2192\u30d3\u30bf\u30df\u30f3, \u5242\u91cf\u2192\u7528\u91cf, \u8bc1\u636e\u2192\u30a8\u30d3\u30c7\u30f3\u30b9, \u7ed3\u8bba\u2192\u7d50\u8ad6,\n"
            "\u663e\u8457\u2192\u986f\u8457, \u5206\u6790\u2192\u5206\u6790, \u4e34\u5e8a\u2192\u81e8\u5e8a, \u6444\u5165\u2192\u6442\u53d6, \u5065\u5eb7\u2192\u5065\u5eb7\n"
            "Double-check: NO Simplified Chinese characters in your output.\n\n"
        )

    translate_system = (
        "{chinese_ban}"
        "You are a medical translation specialist. Translate the following section into {lang_name}.\n"
        "RULES:\n"
        "- Preserve ALL markdown formatting (headers, tables, bullet points, bold, italic)\n"
        "- Keep study names, journal names, and URLs in English\n"
        "- Keep clinical abbreviations in English: ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR\n"
        "- Preserve ALL numerical values exactly (percentages, CI ranges, p-values, sample sizes)\n"
        "- Keep confidence labels (HIGH/MEDIUM/LOW/CONTESTED) in English\n"
        "- Translate meaning, not word-for-word\n"
        "- Output ONLY the translated text, no commentary"
    ).format(chinese_ban=chinese_ban, lang_name=lang_name)

    chinese_rules = ""
    if language == 'ja':
        chinese_rules = (
            "CRITICAL FOCUS \u2014 Chinese contamination:\n"
            "- Replace ANY Simplified Chinese characters with their Japanese equivalents\n"
            "- Common errors: \u6267\u884c\u2192\u5b9f\u884c, \u8865\u5145\u2192\u88dc\u5145, \u8ba4\u77e5\u2192\u8a8d\u77e5, \u7814\u7a76\u2192\u7814\u7a76, \u6548\u679c\u2192\u52b9\u679c, "
            "\u8425\u517b\u2192\u6804\u990a, \u7ef4\u751f\u7d20\u2192\u30d3\u30bf\u30df\u30f3, \u5242\u91cf\u2192\u7528\u91cf, \u8bc1\u636e\u2192\u30a8\u30d3\u30c7\u30f3\u30b9, \u7ed3\u8bba\u2192\u7d50\u8ad6, "
            "\u663e\u8457\u2192\u986f\u8457, \u5206\u6790\u2192\u5206\u6790, \u4e34\u5e8a\u2192\u81e8\u5e8a, \u6444\u5165\u2192\u6442\u53d6, \u5065\u5eb7\u2192\u5065\u5eb7\n"
            "- If you find Chinese sentences, translate them to Japanese\n\n"
        )

    audit_system = (
        "You are a {lang_name} language quality auditor for medical documents.\n\n"
        "{chinese_rules}"
        "Your task:\n"
        "1. Find and fix any non-{lang_name} text (Chinese or untranslated English sentences)\n"
        "2. Fix garbled or unnatural transliterations\n"
        "3. Ensure medical terminology is correct in {lang_name}\n"
        "4. KEEP in English: study names, journal names, URLs, clinical abbreviations "
        "(ARR, NNT, GRADE, CER, EER, RCT, RRR, CI, OR, HR), confidence labels\n"
        "5. Preserve ALL markdown formatting and numerical values exactly\n\n"
        "Return the COMPLETE corrected section. If no issues found, return the section unchanged.\n"
        "IMPORTANT: Output ONLY the corrected text. Do NOT include any commentary, explanation, or preamble."
    ).format(lang_name=lang_name, chinese_rules=chinese_rules)

    # --- Flatten sections into ordered chunk list ---
    # Each chunk: (index, header, body, passthrough_flag)
    # passthrough_flag: True = skip translate/audit, False = needs translate+audit,
    #                   "discussion_marker" = reassembly marker for Discussion section
    chunks = []
    for sec_idx, (header, body) in enumerate(imrad_sections):
        if header == "## 5. References":
            chunks.append((len(chunks), header, body, True))
            continue
        if (not body.strip() and not header) or (len(body.strip()) < 10 and not header):
            chunks.append((len(chunks), header, body, True))
            continue
        if header == "## 4. Discussion":
            sub_chunks = _split_at_subheaders(body)
            disc_parts = []
            for sub_idx, (sub_hdr, sub_body) in enumerate(sub_chunks):
                if (not sub_body.strip() and not sub_hdr) or (len(sub_body.strip()) < 10 and not sub_hdr):
                    disc_parts.append((len(chunks), sub_hdr, sub_body, True))
                else:
                    disc_parts.append((len(chunks), sub_hdr, sub_body, False))
                chunks.append(disc_parts[-1])
            # Store Discussion metadata for reassembly
            chunks.append((len(chunks), header, disc_parts, "discussion_marker"))
            continue
        chunks.append((len(chunks), header, body, False))

    # --- Build reassembly plan ---
    output_plan = []
    i = 0
    while i < len(chunks):
        idx, header, body_or_parts, flag = chunks[i]
        if flag == "discussion_marker":
            disc_indices = [p[0] for p in body_or_parts]
            output_plan.append((header, "discussion", disc_indices))
        else:
            output_plan.append((header, "single", [idx]))
        i += 1

    # --- Identify translatable chunks ---
    translatable = [(c[0], c[1], c[2]) for c in chunks if c[3] is False]
    total_translatable = len(translatable)
    if total_translatable == 0:
        logger.info("  No translatable sections found")
        return sot_content

    # --- Results storage (indexed by chunk index) ---
    results = {}

    # --- Probe mid-tier model with first chunk ---
    mid_tier_available = True
    first_chunk_idx, first_header, first_body = translatable[0]
    first_label = first_header if first_header else "preamble"
    max_tok = _estimate_translation_tokens(len(first_body))
    use_smart_first = len(first_body) > 8000
    try:
        if use_smart_first:
            translated = _call_smart_model(
                system=translate_system,
                user="Translate this section:\n\n" + first_body,
                max_tokens=max_tok, temperature=0.1,
            )
        else:
            from openai import OpenAI as _MidOpenAI
            _mid_client = _MidOpenAI(base_url=MID_BASE_URL, api_key=os.getenv("LLM_API_KEY", "NA"))
            _mid_resp = _mid_client.chat.completions.create(
                model=MID_MODEL,
                messages=[
                    {"role": "system", "content": translate_system},
                    {"role": "user", "content": "Translate this section:\n\n" + first_body},
                ],
                max_tokens=max_tok, temperature=0.1,
                timeout=max(180, int(max_tok / 40) + 60),
            )
            translated = _mid_resp.choices[0].message.content.strip()
        results[first_chunk_idx] = translated
        model_tag = "smart" if use_smart_first else "mid"
        logger.info("  Translated %s (%d -> %d chars) [%s, 1/%d]",
            first_label, len(first_body), len(translated), model_tag, total_translatable)
    except Exception as e:
        if not use_smart_first:
            logger.warning("  Mid-tier model (%s) unavailable: %s -- falling back to smart-model-only", MID_MODEL, e)
            mid_tier_available = False
            try:
                translated = _call_smart_model(
                    system=translate_system,
                    user="Translate this section:\n\n" + first_body,
                    max_tokens=max_tok, temperature=0.1,
                )
                results[first_chunk_idx] = translated
                logger.info("  Translated %s (%d -> %d chars) [smart fallback, 1/%d]",
                    first_label, len(first_body), len(translated), total_translatable)
            except Exception as e2:
                logger.warning("  Translation failed for %s: %s -- keeping original", first_label, e2)
                results[first_chunk_idx] = first_body
        else:
            logger.warning("  Translation failed for %s: %s -- keeping original", first_label, e)
            results[first_chunk_idx] = first_body

    remaining = translatable[1:]

    if not mid_tier_available:
        # --- FALLBACK: Sequential smart-model-only (old behavior) ---
        logger.info("  Running sequential translate+audit on Smart Model (no pipeline)")
        translate_count = 1
        for chunk_idx, header, body in remaining:
            label = header if header else "preamble"
            max_tok = _estimate_translation_tokens(len(body))
            try:
                translated = _call_smart_model(
                    system=translate_system,
                    user="Translate this section:\n\n" + body,
                    max_tokens=max_tok, temperature=0.1,
                )
                translate_count += 1
                results[chunk_idx] = translated
                logger.info("  Translated %s (%d -> %d chars) [smart, %d/%d]",
                    label, len(body), len(translated), translate_count, total_translatable)
            except Exception as e:
                logger.warning("  Translation failed for %s: %s -- keeping original", label, e)
                results[chunk_idx] = body

        # Sequential audit pass
        audit_count = 0
        for chunk_idx, header, body in translatable:
            label = header if header else "preamble"
            translated_body = results.get(chunk_idx, body)
            if not translated_body.strip():
                continue
            max_tok = _estimate_translation_tokens(len(translated_body))
            try:
                audited = _call_smart_model(
                    system=audit_system,
                    user="Audit and correct this {} medical document section:\n\n".format(lang_name) + translated_body,
                    max_tokens=max_tok, temperature=0.1,
                )
                audit_count += 1
                if len(audited) < len(translated_body) * 0.5:
                    logger.warning("  Audit output too short for %s (%d vs %d) -- keeping translation",
                        label, len(audited), len(translated_body))
                else:
                    results[chunk_idx] = audited
                    logger.info("  Audited %s (%d -> %d chars) [smart, %d/%d]",
                        label, len(translated_body), len(audited), audit_count, total_translatable)
            except Exception as e:
                logger.warning("  Audit failed for %s: %s -- keeping translation", label, e)

        logger.info("  Translation+audit complete: %d translate + %d audit calls (sequential fallback)",
            translate_count, audit_count)
    else:
        # --- PIPELINED: Mid-tier translates, smart model audits concurrently ---
        logger.info("  Running pipelined translate (mid-tier) + audit (smart) [%s]", MID_MODEL)

        async def _run_pipeline():
            queue = asyncio.Queue()
            translate_count = 1  # first chunk already done
            audit_count = 0

            async def producer():
                nonlocal translate_count
                for chunk_idx, header, body in remaining:
                    label = header if header else "preamble"
                    max_tok = _estimate_translation_tokens(len(body))
                    use_smart_for_chunk = len(body) > 8000
                    try:
                        if use_smart_for_chunk:
                            translated = await asyncio.to_thread(
                                _call_smart_model,
                                translate_system,
                                "Translate this section:\n\n" + body,
                                max_tok, 0.1,
                            )
                        else:
                            translated = await asyncio.to_thread(
                                _call_mid_model,
                                translate_system,
                                "Translate this section:\n\n" + body,
                                max_tok, 0.1,
                            )
                        translate_count += 1
                        results[chunk_idx] = translated
                        model_tag = "smart" if use_smart_for_chunk else "mid"
                        logger.info("  Translated %s (%d -> %d chars) [%s, %d/%d]",
                            label, len(body), len(translated), model_tag, translate_count, total_translatable)
                    except Exception as e:
                        logger.warning("  Translation failed for %s: %s -- keeping original", label, e)
                        results[chunk_idx] = body
                    await queue.put(chunk_idx)
                # Signal done
                await queue.put(None)

            async def consumer():
                nonlocal audit_count
                # First: audit the already-translated first chunk
                first_translated = results.get(first_chunk_idx, first_body)
                if first_translated.strip():
                    max_tok = _estimate_translation_tokens(len(first_translated))
                    try:
                        audited = await asyncio.to_thread(
                            _call_smart_model,
                            audit_system,
                            "Audit and correct this {} medical document section:\n\n".format(lang_name) + first_translated,
                            max_tok, 0.1,
                        )
                        audit_count += 1
                        if len(audited) >= len(first_translated) * 0.5:
                            results[first_chunk_idx] = audited
                            flabel = first_header if first_header else "preamble"
                            logger.info("  Audited %s (%d -> %d chars) [smart, %d/%d]",
                                flabel, len(first_translated), len(audited), audit_count, total_translatable)
                        else:
                            flabel = first_header if first_header else "preamble"
                            logger.warning("  Audit output too short for %s -- keeping translation", flabel)
                    except Exception as e:
                        flabel = first_header if first_header else "preamble"
                        logger.warning("  Audit failed for %s: %s -- keeping translation", flabel, e)

                # Then: audit chunks as they arrive from producer
                while True:
                    chunk_idx = await queue.get()
                    if chunk_idx is None:
                        break
                    translated_body = results.get(chunk_idx, "")
                    if not translated_body.strip():
                        continue
                    # Find label for this chunk
                    label = "section"
                    for c in translatable:
                        if c[0] == chunk_idx:
                            label = c[1] if c[1] else "preamble"
                            break
                    max_tok = _estimate_translation_tokens(len(translated_body))
                    try:
                        audited = await asyncio.to_thread(
                            _call_smart_model,
                            audit_system,
                            "Audit and correct this {} medical document section:\n\n".format(lang_name) + translated_body,
                            max_tok, 0.1,
                        )
                        audit_count += 1
                        if len(audited) >= len(translated_body) * 0.5:
                            results[chunk_idx] = audited
                            logger.info("  Audited %s (%d -> %d chars) [smart, %d/%d]",
                                label, len(translated_body), len(audited), audit_count, total_translatable)
                        else:
                            logger.warning("  Audit output too short for %s -- keeping translation", label)
                    except Exception as e:
                        logger.warning("  Audit failed for %s: %s -- keeping translation", label, e)

            await asyncio.gather(producer(), consumer())
            return translate_count, audit_count

        # Run the async pipeline
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop and loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                t_count, a_count = pool.submit(lambda: asyncio.run(_run_pipeline())).result()
        else:
            t_count, a_count = asyncio.run(_run_pipeline())
        logger.info("  Pipelined translation+audit complete: %d translate + %d audit calls", t_count, a_count)

    # --- Reassemble output in original section order ---
    assembled_parts = []
    for plan_entry in output_plan:
        header = plan_entry[0]
        plan_type = plan_entry[1]

        if plan_type == "discussion":
            disc_indices = plan_entry[2]
            disc_sub_parts = []
            for didx in disc_indices:
                for c in chunks:
                    if c[0] == didx:
                        sub_hdr = c[1]
                        if c[3] is True:
                            piece = (sub_hdr + "\n" + c[2]) if sub_hdr else c[2]
                        else:
                            body = results.get(didx, c[2])
                            piece = (sub_hdr + "\n" + body) if sub_hdr else body
                        disc_sub_parts.append(piece)
                        break
            discussion_body = "\n".join(disc_sub_parts)
            assembled_parts.append(header + "\n" + discussion_body)
        elif plan_type == "single":
            cidx = plan_entry[2][0]
            for c in chunks:
                if c[0] == cidx:
                    if c[3] is True:
                        piece = (c[1] + "\n" + c[2]) if c[1] else c[2]
                    elif c[3] == "discussion_marker":
                        continue
                    else:
                        body = results.get(cidx, c[2])
                        piece = (c[1] + "\n" + body) if c[1] else body
                    assembled_parts.append(piece)
                    break

    result_text = "\n".join(assembled_parts)

    # Completeness check: compare ## header counts
    source_headers = sot_content.count("\n## ")
    result_headers = result_text.count("\n## ")
    if result_headers < source_headers:
        missing = source_headers - result_headers
        logger.warning("  Translation missing %d section(s): source has %d ## headers, result has %d",
            missing, source_headers, result_headers)
    # Length sanity check
    if len(result_text) < len(sot_content) * 0.5:
        logger.warning("  Translated SOT suspiciously short: %d chars vs %d chars original",
            len(result_text), len(sot_content))

    return result_text


def _translate_prompt(prompt_text: str, language: str, language_config: dict,
                      *, _call_smart_model) -> str:
    """Translate a task prompt/instruction to the target language. Preserves structure."""
    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "ABSOLUTE RULE: Translate to Japanese (\u65e5\u672c\u8a9e) ONLY. NEVER use Chinese.\n"
            "WRONG: \u6267\u884c\u529f\u80fd \u2192 CORRECT: \u5b9f\u884c\u6a5f\u80fd\n\n"
        )
    system = (
        f"{chinese_ban}"
        f"Translate these podcast production instructions to {lang_name}.\n"
        f"KEEP intact: all markdown formatting (##, ###, numbered lists, bold), "
        f"variable placeholders, technical abbreviations (ARR, NNT, GRADE, RCT, CI, HR, OR), "
        f"speaker labels (Host 1:, Host 2:), [TRANSITION] markers.\n"
        f"This is an instruction template, not content \u2014 translate the instructional language only."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=f"Translate:\n\n{prompt_text}",
            max_tokens=6000,
            temperature=0.1,
        )
        if len(result) < len(prompt_text) * 0.3:
            logger.warning("  Prompt translation too short -- keeping original")
            return prompt_text
        return result
    except Exception as e:
        logger.warning("  Prompt translation failed: %s -- keeping original", e)
        return prompt_text


def _audit_script_language(script_text: str, language: str, language_config: dict,
                           *, _call_smart_model) -> str:
    """Post-Crew 3 audit: ensure script is consistently in the target language."""
    if language == 'en':
        return script_text
    lang_name = language_config['name']
    chinese_ban = ""
    if language == 'ja':
        chinese_ban = (
            "Also fix any Chinese characters \u2014 replace with Japanese equivalents.\n"
        )
    system = (
        f"You are a {lang_name} language consistency auditor for a podcast script.\n"
        f"Find any non-{lang_name} sentences (English or Chinese) and translate them to natural {lang_name}.\n"
        f"{chinese_ban}"
        f"KEEP in English: scientific terms (ARR, NNT, GRADE, RCT, CI, HR, OR), "
        f"study abbreviations, URLs, speaker labels (Host 1:, Host 2:).\n"
        f"Return the COMPLETE corrected script preserving ALL [TRANSITION] markers "
        f"and Host 1/Host 2 labels.\n"
        f"CRITICAL: Do NOT add any preamble, notes, or commentary before or after the script. "
        f"Output ONLY the corrected script lines. "
        f"If you must include any notes, prefix each note line with '## '."
    )
    try:
        result = _call_smart_model(
            system=system,
            user=f"Audit this podcast script for language consistency:\n\n{script_text}",
            max_tokens=12000,
            temperature=0.1,
        )
        # Sanity checks
        if len(result) < len(script_text) * 0.5:
            logger.warning("  Script audit output too short -- keeping original")
            return script_text
        orig_transitions = script_text.count('[TRANSITION]')
        result_transitions = result.count('[TRANSITION]')
        if orig_transitions > 0 and result_transitions < orig_transitions:
            logger.warning("  Script audit lost [TRANSITION] markers (%d->%d) -- keeping original",
                orig_transitions, result_transitions)
            return script_text
        logger.info("  Script language audit complete (%d -> %d chars)", len(script_text), len(result))
        return result
    except Exception as e:
        logger.warning("  Script language audit failed: %s -- keeping original", e)
        return script_text
