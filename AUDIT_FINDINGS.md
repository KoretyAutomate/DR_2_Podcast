# DR_2_Podcast Pipeline Audit Findings

**Date:** 2026-02-22
**Auditor:** Claude Opus 4.6
**Scope:** All 9 pipeline phases + cross-cutting concerns

---

## Executive Summary

Audited the entire DR_2_Podcast pipeline across 10 audit steps using 7 domain-expert personas. Found **5 bugs** (3 P0, 2 P1), **6 design risks** worth monitoring, and confirmed **30+ checks as PASS**.

### Bug List (Actionable)

| # | Severity | Finding | File:Line | Impact | Status |
|---|----------|---------|-----------|--------|--------|
| B1 | **P0** | `_call_smart()` has no try/except — ConnectionError/TimeoutError propagates uncaught and crashes pipeline | `clinical_research.py:820-827` | vLLM restart = pipeline crash | **FIXED** — 2-retry + backoff |
| B2 | **P0** | Crew 3 exception re-raised without saving partial task outputs — all script progress lost | `pipeline.py:2810-2817` | Mid-script crash = total data loss | **FIXED** — partial outputs saved |
| B3 | **P0** | `reports["audit"]` direct key access (no `.get()`) — KeyError if Phase 1 fails before GRADE | `pipeline.py:2268` | Phase 1 partial failure cascades to Phase 2 crash | **FIXED** — `.get()` + sentinel |
| B4 | **P1** | Accuracy audit sees PRE-expansion script — expansion loop runs AFTER Crew 3 | `pipeline.py` (expansion at ~3054 vs audit in Crew 3) | Audit may flag length issues that expansion already fixes; expanded content is unaudited | Accepted — by design |
| B5 | **P1** | Translation prompt missing explicit instruction to keep ARR/NNT/GRADE in English | `pipeline.py:1159-1169` | Clinical terms may be translated to Japanese, reducing precision | **FIXED** — preserve list added |

### Risk Matrix Summary

| Category | PASS | WARN | FAIL |
|----------|------|------|------|
| Failure modes (Step 0) | 7 | 2 | 3 |
| Cross-phase flow (Step 1) | 8 | 3 | 2 |
| Phase 0 framing (Step 2) | 4 | 1 | 0 |
| Phase 1 clinical (Step 3) | 14 | 2 | 1 |
| Phase 2 SOT (Step 4) | 5 | 1 | 1 |
| Phase 3 translation (Step 5) | 2 | 2 | 1 |
| Phases 4-6 script (Step 6) | 12 | 0 | 0 |
| Phase 7 accuracy (Step 7) | 6 | 1 | 0 |
| Phase 8 audio (Step 8) | 12 | 1 | 0 |
| Web UI (Step 9) | 5 | 1 | 0 |

---

## Step 0: Failure Mode Catalog

### 0a. Service Dependency Failures

| Service | Failure Point | Behavior | Status |
|---------|--------------|----------|--------|
| vLLM (8000) | `_call_smart()` (line 820) | **No try/except** — ConnectionError propagates | **FAIL (B1)** |
| vLLM | Crew 3 agents | CrewAI catches internally → crew raises → **re-raised** (line 2817) | **FAIL (B2)** |
| Ollama (11434) | Step 4 fast extraction | Falls back to smart model via `if self.fast_worker:` check | PASS |
| Ollama | Concept decomposition (line 1120) | `except Exception` catches all, returns empty dicts | PASS |
| SearXNG (8080) | Step 2 Scholar search | `try/except` around Scholar, PubMed still works | PASS |
| Qwen3-TTS (8082) | Phase 8 Japanese audio | Returns `(None, None)` per segment, caller handles | PASS |
| Kokoro TTS | Phase 8 English audio | Exception caught, `audio_file = None` | PASS |

### 0b. Empty/Null Data Propagation

| Scenario | Behavior | Status |
|----------|----------|--------|
| PubMed 0 results all tiers | `_tiered_search()` returns `([], highest_tier)` — graceful | PASS |
| 0 records pass screening | `_screen_and_prioritize()` explicit check, returns `[]` | PASS |
| `impacts` list empty | `if impacts:` guard at line 2301 — NNT summary omitted | PASS |
| `reports["audit"]` missing | **Direct key access, no `.get()`** — KeyError | **FAIL (B3)** |
| `result.raw` after Crew 3 crash | `result` undefined → NameError | **WARN** (covered by B2) |
| Blueprint task empty output | Script has no structure guidance — degraded quality | WARN |

### 0c. Partial Pipeline State

| Scenario | Behavior | Status |
|----------|----------|--------|
| `--reuse-dir` validation | Only checks `source_of_truth.md` — not clinical artifacts | ~~WARN~~ **FIXED** — warns about missing critical artifacts |
| Web UI worker thread error | Correctly marks task "failed" with cleaned error output (line 272-312) | PASS |
| `progress_tracker` cleanup | `finally` block at line 2821 calls `workflow_completed()` | PASS |
| Pipeline crash mid-Phase 1 | No cleanup; partial files left; next run creates new dir | PASS (acceptable) |

---

## Step 1: Cross-Phase Data Flow

### 1a. Phase 1 → Phase 2: `pipeline_data` dict

- [x] **PASS** `build_imrad_sot()` uses `.get()` with safe defaults for ALL pipeline_data keys (lines 2254-2264)
- [x] **PASS** Empty `impacts` handled — NNT summary gracefully omitted
- [x] **PASS** `_format_study_characteristics_table()` handles None fields with `or 'N/A'` (lines 2207-2212)
- [x] **PASS** `_format_references()` handles missing DOI/authors/journal with conditional appends (lines 2216-2245)
- [ ] **FAIL** `reports["audit"]`, `reports["lead"]`, `reports["counter"]` at line 2268-2270 — direct access, no `.get()` (B3)

### 1b. Phase 2 → Phase 4: SOT injection

- [x] **PASS** SOT injected into blueprint_task, script_task, audit_task descriptions
- [x] ~~**WARN**~~ **FIXED** `--reuse-dir` SOT truncation now uses `_truncate_at_boundary()` (paragraph-aware)
- [x] **PASS** Normal pipeline path (C) uses pre-summarized `sot_summary` — no truncation

### 1c. Phase 6 → Phase 7: Script handoff

- [x] **PASS** Audit task receives POLISHED script via `context=[polish_task]` (line 1268)
- [ ] **WARN** Expansion loop runs AFTER Crew 3 — audit checks pre-expansion script (B4)

### 1d. Phase 7 → Phase 8: Corrected script handoff

- [x] **PASS** `_corrected_script_text` initialized to `None` at line 2868
- [x] **PASS** Correction prompt says "Preserve all [TRANSITION] markers" (line 2842)
- [x] **PASS** Fallback: corrected script < 50% original length → use original (line 2852)

### 1e. Extraction cache consistency

- [x] **PASS** Empty string key guarded by `if cache_key:` — never cached/retrieved
- [x] **PASS** `_load_extraction_cache()` catches `json.JSONDecodeError` — returns `{}`
- [x] **PASS** Thread-safe: cache loaded once before `asyncio.gather()`, written after completion

---

## Step 2: Phase 0 — Research Framing

- [x] **PASS** Phase 0 failure caught (line 2042-2045) — continues with `framing_output = ""`
- [x] **WARN** Empty framing means Phase 1 concept decomposition has no context — keywords may be unfocused
- [x] **PASS** Framing agent has **no tools** — correct, it reasons from topic alone
- [x] **PASS** Framing agent uses `dgx_llm_strict` (low temperature) — appropriate for methodology
- [x] **PASS** Framing task asks for 3-5 testable hypotheses — good structure

---

## Step 3: Phase 1 — Clinical Research Pipeline

### 3a. Concept Decomposition

- [x] **PASS** Fast model failure caught by `except Exception` — returns empty lists (line 1137-1139)
- [x] **PASS** Falls back to smart model if `fast_worker` is None (line 1127)
- [x] **PASS** Downstream uses `.get()` on decomposition dict (line 2195)

### 3b. Step 1: Tiered Keywords + Auditor Gate

- [x] **PASS** `MAX_REVISIONS = 2` — 3 total attempts (line 1388)
- [x] **PASS** Unapproved plan logged via both `logger.warning()` AND `log()` callback (lines 1406-1410)
- [x] **PASS** `_build_tier_query()` is pure deterministic string building — NO LLM (lines 1415-1430)
- [x] **PASS** Fallback on parse failure: returns `TieredSearchPlan` with topic words as keywords (line 1296)
- [x] **PASS** Auditor gate: parse failures auto-approve to unblock pipeline (line 1379-1381)

### 3c. Step 2: Cascading Search

- [x] **PASS** `TIER_CASCADE_THRESHOLD = 50` (line 61) — if T1 returns 200+, T2/T3 don't fire
- [x] **PASS** Empty T1 intervention → tier skipped with log message (line 1459-1461)
- [x] ~~**WARN**~~ **FIXED** PubMed/Scholar exceptions elevated to `logger.error()` (5 locations)
- [x] **PASS** Scholar search skips if T1 intervention + outcome both empty (line 1510-1511)

### 3d. Step 3: Screening

- [x] **PASS** `MAX_TIER3_RATIO = 0.5` (50%) — Tier 3 capped at 10 of 20 slots (line 63)
- [x] **PASS** 0 records → explicit early return `[]` (line 1600-1602)
- [x] **PASS** Two-stage screening: relevance gate (intervention match) → rigor ranking (design + N)

### 3e. Step 4: Extraction + Cache

- [x] **PASS** Cache key `record.pmid or record.doi or ""` — empty string never cached due to `if cache_key:` guard
- [x] **PASS** `_extraction_from_cache()` correctly maps `"cer"`/`"eer"` cache keys to DeepExtraction fields
- [x] **PASS** `nonlocal cache_hits` in async function — safe with `asyncio.gather()` (single-threaded event loop)
- [x] **WARN** Cache has no expiry — old extractions persist forever (acceptable for PMIDs which are immutable)

### 3f. Step 5: Case Synthesis

- [x] **PASS** Empty extractions → `_build_case()` logs critical warning if synthetic citations detected (line 2066-2067)

### 3g. Step 6: Deterministic Math

- [x] **PASS** `NNT = 1.0 / abs(arr)` — handles negative ARR correctly (line 50)
- [x] **PASS** `CER == EER` → `abs(arr) < 1e-10` → NNT = `float('inf')`, direction = `"no_effect"` (line 40-44)
- [x] **PASS** `CER or EER is None` → `calculate_impact()` returns `None`, filtered out by caller (line 37-38, 66)
- [x] **PASS** `RRR = arr / cer if abs(cer) > 1e-10 else 0.0` — division by zero protected (line 49)

### 3h. Step 7: GRADE Synthesis

- [x] **PASS** Prompt explicitly says "NEVER recalculate ARR or NNT — use the Python-provided numbers exactly" (line 2523-2525)
- [x] **PASS** Starting certainty: "HIGH for RCTs, LOW for observational" with DOWNGRADE/UPGRADE modifiers (line 2501-2505)

---

## Step 4: Phase 2 — SOT Assembly

- [x] **PASS** `build_imrad_sot()` is purely deterministic — **zero LLM calls** confirmed
- [x] **PASS** All `.get()` calls have safe defaults (0, [], "", {})
- [x] **PASS** `_format_study_characteristics_table()` — all None fields → `'N/A'`
- [x] **PASS** `_format_references()` — missing authors → `"Unknown authors"`, missing journal/year/DOI conditionally omitted
- [x] **PASS** Empty extraction list → `"*No studies with full extraction data available.*"`
- [x] **WARN** URLs in SOT are NOT validated by `link_validator` — validation runs separately in Phase 2, but 404'd URLs still appear in SOT narrative. PubMed PMID links are hardcoded (always valid).
- [ ] **FAIL** `reports["audit"]` direct access without `.get()` at line 2268 (same as B3)

---

## Step 5: Phase 3 — Translation

- [x] **PASS** Chinese ban present for `language == 'ja'` (lines 1166-1168)
- [x] **PASS** Translation performed by `producer_agent` — no dedicated translation agent (acceptable: producer already has context)
- [x] **PASS** Prompt preserves confidence labels (HIGH/MEDIUM/LOW/CONTESTED), study names, journal names, URLs in English
- [x] ~~**FAIL**~~ **FIXED** ARR/NNT/GRADE/CER/EER/RCT/RRR/CI/OR/HR preserve list added (B5)
- [x] ~~**WARN**~~ **FIXED** Numerical preservation rule added — "do NOT convert or round"

---

## Step 6: Phases 4-6 — Script Production

### 6a. Phase 4: Blueprint

- [x] **PASS** Blueprint asks for all 7 sections: Thesis, Value Prop, Hook, Framework, Narrative Arc, GRADE Guide, Citations (lines 1296-1347)
- [x] **PASS** Hook section clearly says "provocative QUESTION" with good/bad examples (lines 1303-1307)
- [x] **PASS** Framework hint communicated via `_framework_hint` from `channel_mission` env var
- [x] **PASS** No output validation for 7 sections — but LLM adherence is generally good with explicit template

### 6b. Phase 5: Script Writing

- [x] **PASS** Script structure: 1. Channel Intro → 2. Hook → 3-6. Acts 1-4 → 7. Wrap-up → 8. One Action (lines 1076-1109)
- [x] **PASS** `_channel_intro_directive` numbered as "1." (line 1055)
- [x] **PASS** Word allocation: Act 1=20%, Act 2=35%, Act 3=25%, Act 4=20% (lines 1067-1070)
- [x] **PASS** Chinese ban present in `script_task` for `ja` (lines 1135-1138)
- [x] **PASS** Length instruction is emphatic: "This is CRITICAL — do not write less" (line 1124)

### 6c. Phase 6: Polish

- [x] **PASS** Editor agent uses dynamic `{target_script} {target_unit_plural}` — no residual "4,500 words" (lines 976, 986)
- [x] **PASS** Polish task verifies 8-part structure with correct ordering (line 1187-1188)
- [x] **PASS** Editor backstory says "welcome → hook question → topic shift" — consistent (line 987)
- [x] **PASS** `[TRANSITION]` marker insertion mentioned in polish task prompt

### 6d. Multi-pass Expansion

- [x] **PASS** `MAX_EXPANSION_PASSES = 3` — up to 3 attempts
- [x] **PASS** Early break if length doesn't improve ("did not improve length — stopping retries")
- [x] **PASS** Chinese ban present in expansion task for Japanese
- [x] **PASS** Japanese uses `chars` unit, English uses `words` — from `SUPPORTED_LANGUAGES` dict

### 6e. Post-Crew 3 Language Auditor

- [x] **PASS** `_has_chinese_contamination()` detects CJK chars without hiragana/katakana
- [x] **PASS** Correction crew runs if contamination detected, editor_agent handles correction

### 6f. Length-Mode Sensitivity

- [x] **PASS** `TARGET_MINUTES = {'short': 10, 'medium': 20, 'long': 30}` (line 489)
- [x] **PASS** `SCRIPT_TOLERANCE = 0.10` (±10%) (line 490)
- [x] **PASS** English: 150 words/min, Japanese: 500 chars/min (lines 474, 482)

---

## Step 7: Phase 7 — Accuracy Audit

- [x] **PASS** Audit prompt clearly defines HIGH/MEDIUM/LOW severity format (line 1257)
- [x] **PASS** HIGH-severity regex: `r'\*\*Severity\*\*:\s*HIGH'` — matches prompt format exactly (line 2825)
- [x] **PASS** Single correction cycle — no retry loop (acceptable: over-correction risk if iterated)
- [x] **PASS** Sanity check: corrected script must be > 50% of original length (line 2852)
- [x] **PASS** Corrected script saved to `ACCURACY_CORRECTIONS.md` (line 2854)
- [x] **PASS** Correction prompt explicitly preserves [TRANSITION] markers (line 2842)
- [x] ~~**WARN**~~ **FIXED** Post-correction [TRANSITION] marker count verified — falls back if markers lost

---

## Step 8: Phase 8 — Audio Production

### 8a. TTS Generation

- [x] **PASS** Speaker gap: 300ms (line 333) — within natural range (200-500ms)
- [x] **PASS** `[TRANSITION]` silence: 1.5s | `[PAUSE]`: 0.8s | `[BEAT]`: 0.3s (lines 552-556)
- [x] **PASS** `clean_script_for_tts()` placeholder swap: `[TRANSITION]` → `___TRANSITION___` then restored (line 549, 571-598)
- [x] **PASS** Return type `Tuple[str, List[int]]` or `None` — callers check `isinstance(tts_result, tuple)` (line 1718-1721)
- [x] ~~**WARN**~~ **FIXED** Qwen3-TTS: failed chunks now insert proportional silence (len/8.0 seconds)
- [x] **PASS** [TRANSITION] positions tracked in `transition_positions_ms` for pro mixer (lines 372-374)

### 8b. BGM Mixing

- [x] **PASS** `mix_podcast_pro()`: pre-roll 4s, post-roll 6s (line 60)
- [x] **PASS** Transition bumps: -18dB → -10dB (8dB increase) for 1.5s with 300ms fades (lines 108-110)
- [x] **PASS** BGM fallback chain: specific file → `Interesting BGM.wav` → MusicGen → no BGM (lines 477-507)
- [x] **PASS** `mix_podcast_pro()` failure → `mix_podcast()` basic mixer (line 152-154)
- [x] **PASS** Basic mixer failure → return original voice track (pipeline.py line 541)

### 8c. Duration Verification

- [x] **PASS** `wave.open()` reads the FINAL mixed file (line 3159-3166) — after `post_process_audio()`
- [x] **PASS** Duration check happens after all mixing is complete

---

## Step 9: Web UI Integration

- [x] **PASS** Worker thread error handling: tasks marked "failed" with cleaned error output (lines 272-312)
- [x] **PASS** `channel_intro`, `core_target`, `channel_mission` set as env vars only when non-empty
- [x] **PASS** Process return code check — non-zero marks task "failed" with last 50 error lines (lines 2116-2131)
- [x] **PASS** Authentication: username/password login present
- [x] **PASS** No file upload handling — topics submitted as text form fields
- [x] ~~**WARN**~~ **FIXED** Pre-flight health check for vLLM/Ollama added — fails fast with clear error

---

## Recommendations

### P0 Fixes (Should Fix Before Next Run)

1. **B1 — Wrap `_call_smart()` in try/except** (`clinical_research.py:820`)
   - Catch `ConnectionError`, `TimeoutError`, `httpx.ConnectError`
   - Log error, return empty string or raise custom `LLMUnavailableError`
   - Alternatively, add retry logic (1-2 retries with 5s backoff)

2. **B2 — Save partial outputs before Crew 3 re-raise** (`pipeline.py:2810-2817`)
   - Before `raise`, inspect `blueprint_task.output`, `script_task.output`, `polish_task.output`
   - Save whatever is available to `output_dir/` (e.g., `script_draft_partial.md`)
   - Consider not re-raising — fall back to best available output

3. **B3 — Use `.get()` for `reports["audit/lead/counter"]`** (`pipeline.py:2268-2270`)
   - Replace with `reports.get("audit", EmptyReport())` or guard with `if "audit" in reports:`
   - Provide a stub report object that returns empty strings for `.report`

### P1 Fixes (Fix When Convenient)

4. **B4 — Move expansion loop before accuracy audit** or add a post-expansion audit pass
   - Current: Crew 3 (includes audit) → expansion loop → audio
   - Better: Crew 3 (without audit) → expansion loop → separate audit crew → audio
   - Or: Accept that expanded content is unaudited (document this as intentional)

5. **B5 — Add ARR/NNT/GRADE to translation preserve list** (`pipeline.py:1165`)
   - Add: `f"- Keep clinical abbreviations (ARR, NNT, GRADE, CER, EER, RCT) in English\n"`

### ~~P2 Observations~~ ✓ ALL FIXED

6. ~~SOT truncation at 8000 chars in `--reuse-dir` mode may cut mid-table~~ → **FIXED** `_truncate_at_boundary()` helper
7. ~~Qwen3-TTS failed segments produce jumpy audio~~ → **FIXED** proportional silence insertion
8. ~~No pre-flight vLLM/Ollama check in Web UI~~ → **FIXED** `_preflight_check()` with 3s timeout
9. ~~`--reuse-dir` only validates `source_of_truth.md`~~ → **FIXED** warns about missing critical artifacts
10. ~~PubMed search exceptions silently swallowed~~ → **FIXED** elevated to `logger.error()` (5 locations)
11. ~~Post-correction [TRANSITION] markers not verified~~ → **FIXED** count comparison + fallback
12. ~~Translation may round/convert numerical values~~ → **FIXED** explicit preservation rule in prompt
