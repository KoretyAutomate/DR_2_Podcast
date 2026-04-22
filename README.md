# Deep-Research Podcast Crew

An AI-powered pipeline that deeply researches any scientific topic using a clinical systematic-review methodology (PICO/GRADE), synthesizes evidence from both affirmative and adversarial perspectives, and produces a broadcast-ready podcast with local TTS audio (Kokoro for English, VOICEVOX for Japanese) — all running on local models. Includes a FastAPI web UI for one-click production with live progress tracking.

## System Overview

```
                       ┌─────────────────────────────┐
                       │     Topic + Language Input  │
                       │   (Web UI or CLI)           │
                       └──────────────┬──────────────┘
                                      ▼
                  ┌──────────────────────────────────────────┐
                  │  Phase 0 — Research Framing (Crew 1)     │
                  │  · Domain classification (deterministic) │
                  │  · Domain-aware framing document         │
                  └───────────────────┬──────────────────────┘
                                      ▼
       ┌───────────────────────────────────────────────────────────────────┐
       │  Phase 1 — Research Pipeline (clinical or social science)         │
       │                                                                   │
       │  Pre-step: Concept Decomposition (Fast Model)                     │
       │                              ▼                                    │
       │  ┌─ AFFIRMATIVE (a) ────────────┐ ┌─ FALSIFICATION (b) ─────────┐ │
       │  │ 1a: Tiered keywords (Smart)  │ │ 1b: Tiered keywords         │ │
       │  │     + Auditor gate → loop    │ │     + Auditor gate → loop   │ │
       │  │ 2a: Cascade search           │ │ 2b: Cascade search          │ │
       │  │     PubMed or OpenAlex+ERIC  │ │     PubMed or OpenAlex+ERIC │ │
       │  │ 3a: Tier-aware screen → 20   │ │ 3b: Tier-aware screen → 20  │ │
       │  │ 4a: Full-text extraction     │ │ 4b: Full-text extraction    │ │
       │  │     (PMC/Unpaywall by Fast)  │ │     (PMC/Unpaywall by Fast) │ │
       │  │ 5a: Affirmative case (Smart) │ │ 5b: Falsification case      │ │
       │  └──────────────────────────────┘ └─────────────────────────────┘ │
       │          (both tracks run in parallel via asyncio.gather)         │
       │                              ▼                                    │
       │  ┌─ CLINICAL ───────────────────┐ ┌─ SOCIAL SCIENCE ────────────┐ │
       │  │ Step 6: ARR/NNT math         │ │ Step 6: Cohen's d /         │ │
       │  │         (Python, no LLM)     │ │         Hedges' g (no LLM)  │ │
       │  │ Step 7: GRADE synthesis      │ │ Step 7: Evidence quality    │ │
       │  │         Auditor (Smart)      │ │         synthesis (Smart)   │ │
       │  └──────────────────────────────┘ └─────────────────────────────┘ │
       └───────────────────────────────┬───────────────────────────────────┘
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │  Phase 2 — Source Validation (batch HEAD requests)            │
       │  Source-of-Truth synthesis — IMRaD format                     │
       └───────────────────────────────┬───────────────────────────────┘
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │  Phase 3 — Report Translation (Crew 2, conditional)           │
       └───────────────────────────────┬───────────────────────────────┘
                                       ▼
       ┌───────────────────────────────────────────────────────────────┐
       │ Crew 3 — Podcast Production                                   │
       │                                                               │
       │  Phase 4: Episode Blueprint (Producer)                        │
       │  Phase 5: Script Writing (Producer)                           │
       │  Phase 6: Script Polish (Editor)                              │
       │  Phase 7: Accuracy Audit (Auditor) [conditionally blocking]   │
       └───────────────────────────────┬───────────────────────────────┘
                                       ▼
                ┌─────────────────────────────────────────────┐
                │  Phase 8 — Audio Production                 │
                │  Kokoro TTS (EN), VOICEVOX (JA),             │
                │  two voices, 24kHz WAV + BGM                │
                └─────────────────────────────────────────────┘
```

## Web UI

A FastAPI-based web interface (`web_ui.py`) for managing podcast production:

- One-click topic submission with language, accessibility level, host selection, and channel branding (intro text, target audience, mission)
- **Live progress tracking**: phase name, progress bar, ETA, artifact count, studies scanned
- **Task queue**: submit multiple requests — confirmation dialog shows queue position, running task progress stays visible
- **Production History**: collapsible list of past runs with download links
- **Upload integration**: optional Buzzsprout (draft) and YouTube (private) publishing
- **Research reuse**: reuse previous research artifacts with optional LLM-assessed supplemental research
- **Stop button**: cancel a running task mid-pipeline
- **System status**: checks vLLM and Ollama availability before submission
- Basic authentication support (auto-generated credentials or via env vars)

Launch:
```bash
./start_podcast_web_ui.sh
# or directly:
python -m dr2_podcast.web.web_ui --port 8501
```

## Agents

The pipeline uses **4 CrewAI agents** (down from 7 — the clinical pipeline replaced the Lead Researcher, Adversarial Researcher, and Source Verifier):

| Agent | Variable | Role | Tools |
|-------|----------|------|-------|
| **Research Framing Specialist** | `framing_agent` | Defines scope, core questions, evidence criteria before any searching begins | — |
| **Scientific Auditor** | `auditor_agent` | Checks polished script for scientific drift against the Source-of-Truth | LinkValidator, ListResearchSources, ReadResearchSource, ReadFullReport, ReadValidationResults |
| **Podcast Producer** | `producer_agent` | Transforms research into a debate script targeting Masters/PhD-level depth | ReadFullReport |
| **Podcast Editor** | `editor_agent` | Polishes script for natural verbal delivery, enforces word count and depth | — |

## Two-Model Architecture

The system uses two local LLMs working in tandem:

| Role | Default Model | Hosted On | Purpose |
|------|---------------|-----------|---------|
| **Smart model** | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` | vLLM (port 8000) | PICO strategy, screening, case synthesis, GRADE audit, script writing, SOT translation |
| **Fast model** | `qwen3:8b` | Ollama (port 11434) | Parallel abstract screening, full-text clinical extraction, report condensation |

Model selection can be overridden via environment variables (`MODEL_NAME`, `LLM_BASE_URL`, `FAST_MODEL_NAME`, `FAST_LLM_BASE_URL`).

SOT translation is handled directly by the Smart Model (multilingual, including Japanese — set via `MODEL_NAME`). If the fast model is unavailable, the smart model handles all summarization and extraction (slower but functional).

## Evidence-Based Research Pipeline

The deep research pre-scan implements a 7-step systematic review methodology modelled on clinical trial standards. Steps 1–5 run as parallel affirmative (a) and falsification (b) tracks.

### Steps 1–5 — Parallel Tracks (a = Affirmative, b = Falsification)

Steps 1–5 run identically for both tracks via `asyncio.gather()`. The only differences are the search terms (b targets adverse-effects, null-results, harms, and bias terms) and the final case mandate (a argues FOR, b argues AGAINST).

**Pre-step — Concept Decomposition (Fast Model, ~5s)**
Before Step 1, the fast model extracts canonical scientific terms from the folk-language topic (e.g., "coffee" → canonical terms: caffeine, coffea; related concepts: adenosine, methylxanthine). These terms are fed into Step 1 to help the scientist generate accurate tier keywords.

**Step 1 — Tiered Keyword Generation + Auditor Gate (Smart Model, ~15s)**
A Scientist agent (Smart) generates a **3-tier plain keyword plan** — no Boolean operators, no MeSH notation — just simple English phrases organized into three escalating scope tiers:

| Tier | Label | Intervention | Outcome | Population |
|------|-------|-------------|---------|-----------|
| **1** | Established evidence | Exact folk/common names (e.g., "coffee") | Direct primary outcome labels | Specific population |
| **2** | Supporting evidence | *Inherited from Tier 1* | Superset of Tier 1 + broader proxies | Broader than Tier 1 |
| **3** | Speculative extrapolation | Compound class / mechanism (e.g., "caffeine") | *Inherited from Tier 2* | *Inherited from Tier 2* |

An **Auditor agent** (Smart) then reviews the plan against 5 criteria (intervention anchor, outcome broadening, population broadening, no Boolean syntax, coverage). If rejected, the scientist revises — up to **2 revision rounds** before proceeding with a warning. `_build_tier_query()` then deterministically builds PubMed Boolean strings from the approved plain keywords — no LLM is involved in query construction.

**Step 2 — Cascading PubMed Search + Scholar (PubMed + Fast Model, ~90s)**
Searches PubMed using a **cascading tier strategy** — the pipeline stops adding tiers once a sufficient candidate pool is reached:

1. **Tier 1** query (with `Humans[MeSH] AND English[la]` filters) → if pool ≥ 50, stop
2. **Tier 2** query (with `Humans[MeSH]` filter) → if pool ≥ 50, stop
3. **Tier 3** query (no filters) → ultrawide net

**Google Scholar** always runs using Tier 1 plain-text keywords (via SearXNG). **`PublicationType` in the PubMed XML is used directly** to classify study type (RCT, meta-analysis, systematic review, etc.) without LLM calls. The fast model only processes records where type cannot be determined from XML, extracting `study_type`, `sample_size`, and `primary_objective` from the abstract. Each record is tagged with its `research_tier` (1, 2, or 3) for downstream priority handling.

**Step 3 — Tier-Aware Screening (Smart Model, ~20s)**
Each tier is screened **independently** with a **two-stage process**:
1. **Relevance gate** — does the study directly investigate the tier-appropriate intervention?
2. **Rigor ranking** — among relevant studies, rank by meta-analyses first, then RCTs (by sample size), then large cohort studies.

Inclusion: RCTs, meta-analyses, systematic reviews, large cohort studies (n ≥ 30, prefer n ≥ 100). Exclusion: animal models, in vitro, case reports, conference abstracts, retractions.

The final top 20 are assembled via **priority fill**: Tier 1 first → Tier 2 fills remaining → Tier 3 capped at 50% of slots (minimum 3 if available). This ensures the evidence base is anchored in directly relevant studies while allowing speculative compound-class evidence to contribute.

**Step 4 — Full-Text Deep Extraction (Fast Model, ~120s)**
For each of the top 20 studies, the full text is retrieved via a 4-tier fallback:
1. **PubMed Central OA API** (`oai:pubmedcentral.nih.gov`)
2. **Europe PMC REST API** (free full-text XML for OA articles)
3. **Unpaywall API** (OA PDF location via DOI)
4. **Publisher page scrape** (existing `ContentFetcher` logic)

The fast model then extracts 20 clinical variables per article: `control_event_rate` (CER), `experimental_event_rate` (EER), effect size with CI, attrition, blinding, randomization, ITT analysis, funding source, conflicts of interest, risk of bias, demographics, follow-up period, and biological mechanism. Extractions are **cached by PMID** in `research_outputs/extraction_cache.json` — subsequent runs reuse cached CER/EER values, ensuring identical NNT calculations for the same paper across runs.

**Step 5 — Case Synthesis (Smart Model, ~30s each)**
Each track writes a structured case report from the extracted data:
- **5a (Affirmative):** argues FOR the hypothesis — clinical significance, biological plausibility, consistency, dose-response
- **5b (Falsification):** argues AGAINST — adverse effects, null results, methodological concerns, funding bias, publication bias

### Step 6 — Deterministic Math (Python, <1ms)

**No LLM involved.** A pure-Python calculator computes from the extracted CER/EER values:
- **ARR** = CER − EER (Absolute Risk Reduction)
- **RRR** = ARR / CER (Relative Risk Reduction)
- **NNT** = 1 / |ARR| (Number Needed to Treat)

This eliminates LLM math hallucinations. Only studies where both CER and EER were extracted contribute to the NNT table.

### Step 7 — GRADE Synthesis (Smart Model, ~45s)

The Auditor reads both cases and the Python-calculated NNT table, then issues a **GRADE-framework synthesis** (High / Moderate / Low / Very Low evidence quality) with evidence profile, downgrade/upgrade justifications, clinical impact interpretation, PRISMA flow diagram, and consolidated evidence table. The Auditor is instructed never to recalculate ARR/NNT — it uses the Python-provided numbers exactly.

**Total wall-clock time: ~5–6 minutes** (both tracks in parallel).

## Research Library

After the deep research pre-scan, all source-level data is saved to `research_sources.json`. Three CrewAI tools let agents browse and drill into this library during their tasks:

- **ListResearchSources** — Browse a numbered index of all sources (title, URL, research goal)
- **ReadResearchSource** — Read the full extracted summary for any specific source by index
- **ReadFullReport** — Read an entire research report from disk

Before injection into agent task descriptions, full reports are **condensed by the fast model** (~2000 words) instead of being hard-truncated.

## GRADE Audit Report Structure

The Step 7 GRADE synthesis follows this structure:

1. **Executive Summary** (3–4 sentences)
2. **Evidence Profile** (study designs, total N, risk of bias, consistency, directness, precision, publication bias)
3. **GRADE Assessment** (start at HIGH for RCTs / LOW for observational; apply upgrades/downgrades; final grade: ⊕⊕⊕⊕ → ⊕○○○)
4. **Clinical Impact** (NNT table from deterministic math, interpreted in context)
5. **Balanced Verdict** (weight of evidence, caveats, what would change the conclusion)
6. **Recommendations for Further Research**
7. **PRISMA Flow Diagram** (text-based: identified → screened → eligible → included)
8. **Consolidated Evidence Table** (Study, Design, N, Effect, CER, EER, ARR, NNT, Bias Risk, GRADE Impact)
9. **Full Reference List**

## Pipeline Phases

### Phase 0 — Research Framing (Crew 1)
Phase 0 has two internal steps that run back-to-back:
1. **Domain classification** — the topic is classified as `clinical` or `social_science` using deterministic keyword rules (fast, no LLM for most topics). The result is saved to `domain_classification.json`.
2. **Framing** — the Research Framing Specialist defines scope boundaries, core research questions, evidence criteria, suggested search directions, and hypotheses. The framing prompt is tailored to the domain (PICO/GRADE for clinical; PECO/effect sizes for social science).

### Phase 1 — Research Pipeline (domain-routed)
Dispatches to the appropriate 7-step pipeline based on the Phase 0 domain result:
- **Clinical topics** → PubMed cascade search, PICO framework, GRADE synthesis, ARR/NNT math. Produces `affirmative_case.md`, `falsification_case.md`, `grade_synthesis.md`, `clinical_math.md`, `research_sources.json`.
- **Social science topics** → OpenAlex + ERIC search, PECO framework, evidence quality hierarchy, Cohen's d / Hedges' g effect sizes. Produces equivalent artifacts with social science terminology.

### Phase 2 — Source Validation
Batch HEAD requests validate all cited URLs. The Source-of-Truth is then assembled in **IMRaD format** (Introduction, Methods, Results, and Discussion) — a structured scientific paper format derived deterministically from the pipeline's raw outputs. See [Source of Truth (IMRaD Format)](#source-of-truth-imrad-format) below.

### Phase 3 — Report Translation (conditional)
For non-English output, the Source-of-Truth is translated by the Smart Model (set via `MODEL_NAME`), which is multilingual and handles all IMRaD sections directly. Skipped entirely for English runs.

### Phase 4 — Episode Blueprint (Crew 3)
The Producer generates a 7-section Episode Blueprint: episode thesis, listener value proposition, hook question, content framework (PPP or QEI), 4-act narrative arc, GRADE-informed evidence framing, and citation plan. The Source-of-Truth summary is injected directly into task descriptions.

### Phase 5 — Script Writing (Crew 3)
The Producer generates the debate script following a Channel Intro → Hook Question → 4-Act structure (Claim, Evidence, Nuance, Protocol) → Wrap-up → One Action ending. The script uses `Host 1:` / `Host 2:` dialogue format with `[TRANSITION]` markers between acts.

### Phase 6 — Script Polish (Crew 3)
The Editor refines for natural verbal delivery and ensures balanced coverage.

### Phase 7 — Accuracy Audit (Crew 3, conditionally blocking)
The Auditor scans the polished script for drift patterns: correlation-to-causation, hedge removal, confidence inflation, cherry-picking, and contested-as-settled. If **HIGH-severity** drift is detected, a correction pass automatically rewrites the affected sections before audio generation. Corrections are logged to `ACCURACY_CORRECTIONS.md`.

### Phase 8 — Audio Production
Audio is rendered with two voices at 24kHz WAV, followed by BGM mixing. TTS engine is selected by language:

| Language | TTS Engine | Host 1 (Male) | Host 2 (Female) |
|----------|------------|---------------|-----------------|
| English  | Kokoro TTS (local, CPU) | `am_fenrir` (American male) | `af_heart` (American female) |
| Japanese | VOICEVOX (Docker, port 50021) | †聖騎士 紅桜† ノーマル (id=51) | 四国めたん ノーマル (id=2) |

## Multi-Language Support

The pipeline supports English and Japanese output:
- **English**: Default. All research, scripts, and audio in English.
- **Japanese**: A translation task runs in Crew 2 before Crew 3. Audio is rendered by **VOICEVOX** (Docker container at port 50021) — Kokoro is not used for Japanese. A post-Crew 3 language auditor detects Chinese contamination (CJK text without hiragana/katakana) and automatically runs a correction pass to translate any Chinese passages into natural Japanese.

### Script Length Enforcement

After Crew 3 completes, the pipeline measures script length against the target (language-aware: words for English, characters for Japanese) with a ±10% tolerance. If the script is too short, **up to 3 expansion passes** run automatically — each pass adds deeper scientific explanation, real-world examples, and host dialogue. Expansion stops early if a pass fails to increase the length.

## Podcast Hosts

| Host | Voice | Role |
|------|-------|------|
| **Host 1** | Male (Kokoro: `am_fenrir` / VOICEVOX: id=51) | Randomly assigned as presenter or questioner each session |
| **Host 2** | Female (Kokoro: `af_heart` / VOICEVOX: id=2) | Randomly assigned as presenter or questioner each session |

Personality is determined by role, not host: the **presenter** is an enthusiastic science communicator; the **questioner** is a curious, skeptical interviewer. Override with `PODCAST_HOSTS` env var (`host1_leads`, `host2_leads`, or `random`).

## Prerequisites

- **Python 3.11** (Kokoro TTS requires < 3.13; recommended conda env: `podcast_flow`)
- **NVIDIA GPU** (for vLLM; Kokoro falls back to CPU if GPU kernel is incompatible)
- **UniDic dictionary** (required by Kokoro TTS via fugashi/MeCab):
  ```bash
  python -m unidic download
  ```

### Required Services

**vLLM** — Default backend for the smart model:
```bash
./start_vllm_docker.sh
```
The launcher script mounts `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` from the local HuggingFace cache and serves it with `--reasoning-parser qwen3` and `--enable-auto-tool-choice --tool-call-parser hermes`. GPU memory utilization defaults to 82%. See `start_vllm_docker.sh` for all tunables.

**Ollama** — Required for the fast model:
```bash
ollama serve
ollama pull qwen3:8b                # Fast model (default)
```

**SearXNG** — Self-hosted search (optional, improves source diversity):
```bash
docker run -d -p 8080:8080 searxng/searxng:latest
```

**VOICEVOX** — Japanese TTS engine (Docker):
```bash
docker run -d --name voicevox -p 50021:50021 voicevox/voicevox_engine:cpu-latest
```

## Installation

```bash
conda activate podcast_flow
pip install -r requirements.txt
python -m unidic download               # Required for Kokoro TTS
```

## Environment Variables

```bash
# Required
export BRAVE_API_KEY="your_brave_search_api_key"

# Optional — PubMed
export PUBMED_API_KEY="your_ncbi_api_key"       # Increases rate limit from 3 to 10 req/sec
export UNPAYWALL_EMAIL="your@email.com"         # Required for Unpaywall OA PDF lookup (free)

# Optional — topic/language
export PODCAST_TOPIC="effects of intermittent fasting on cognitive performance"
export PODCAST_LANGUAGE="en"          # en or ja
export ACCESSIBILITY_LEVEL="simple"   # simple | moderate | technical
export PODCAST_LENGTH="long"          # short | medium | long
export PODCAST_HOSTS="random"         # random | host1_leads | host2_leads

# Optional — channel branding
export PODCAST_CHANNEL_INTRO="Welcome to Deep Research Podcast. Today we're..."
export PODCAST_CORE_TARGET="busy professionals aged 30-50"
export PODCAST_CHANNEL_MISSION="turning complex science into actionable protocols"

# Model config (defaults shown)
export MODEL_NAME="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4"
export LLM_BASE_URL="http://localhost:8000/v1"
export FAST_MODEL_NAME="qwen3:8b"
export FAST_LLM_BASE_URL="http://localhost:11434/v1"

# Service endpoints (defaults shown)
export SEARXNG_URL="http://localhost:8080"

# TTS engine configuration (defaults shown)
export TTS_ENGINE_JA="voicevox"         # Japanese engine (registry: voicevox)
export TTS_ENGINE_EN="kokoro"           # English engine (Kokoro in-process)
export TTS_API_URL="http://localhost:50021"   # HTTP endpoint for HTTP-based engines (VOICEVOX etc.)
export TTS_HOST1_ID="51"                # VOICEVOX: †聖騎士 紅桜† ノーマル
export TTS_HOST2_ID="2"                 # VOICEVOX: 四国めたん ノーマル

# Audio
export VOICE_DUCKING_DB="-20"         # BGM ducking in dB during speech

# Web UI authentication (auto-generated if not set)
export PODCAST_WEB_USER="admin"
export PODCAST_WEB_PASSWORD="your_password"

# Upload integration (optional)
export BUZZSPROUT_API_KEY="your_buzzsprout_api_key"
export BUZZSPROUT_ACCOUNT_ID="your_account_id"
export YOUTUBE_CLIENT_SECRET_PATH="/path/to/client_secret.json"
```

## Usage

```bash
# Via Web UI (recommended)
./start_podcast_web_ui.sh

# Via CLI
python -m dr2_podcast.pipeline --topic "neuroplasticity and exercise" --language en

# Reuse previous research (skip research phases, regenerate podcast only)
python -m dr2_podcast.pipeline --reuse-dir research_outputs/2025-01-15_10-30-00 --crew3-only

# Reuse with LLM-assessed supplemental research if needed
python -m dr2_podcast.pipeline --reuse-dir research_outputs/2025-01-15_10-30-00 --check-supplemental

# Resume an interrupted run from the last completed phase
python -m dr2_podcast.pipeline --resume research_outputs/2025-01-15_10-30-00
```

## Accessibility Levels

Control how aggressively scientific terminology is simplified:

| Level | Audience | Behavior |
|-------|----------|----------|
| `simple` | General educated (college-level) | Define every scientific term inline with plain-English analogies |
| `moderate` | Science enthusiasts | Define key terms once, assume basic cause-and-effect literacy |
| `technical` | Professionals in related fields | Standard terminology, no simplification, focus on depth |

## Prefect Orchestration

The pipeline is orchestrated by [Prefect](https://www.prefect.io/) (`pipeline_flow.py`). Each of the 9 phases (0–8) is a `@task` with `persist_result=True` — completed phases are cached and skipped on resume, so an interrupted run can pick up where it left off via `--resume`.

Key behaviors:
- **Cache key**: per-run directory basename + phase name (collision-safe hash)
- **Sequential execution**: phases run one at a time (single GPU constraint)
- **Retry policy**: configurable retries and delay per phase (e.g., Phase 1 research allows retries with 60s delay)
- **Timeouts**: per-phase timeouts (e.g., Phase 1 = 2 hours, Phase 8 audio = 1 hour)

The Prefect layer is transparent — CLI and Web UI both call `run_pipeline_flow()`, which delegates to the same underlying phase functions.

## Evaluation System

An optional post-run evaluation system (`dr2_podcast/evaluation/`) tracks pipeline quality over time:

- **Scorecard** (`scorecard.py`): Deterministic metric collection (no LLM) — parses logs and artifacts to compute screening tier counts, extraction timeouts, section budget adherence, script length, and audio duration. Outputs `run_scorecard.json`. Flags regressions when metrics drift >10% from a rolling 5-run average.
- **Lesson Generator** (`lesson_generator.py`): One Smart Model call per run to extract 1–3 actionable observations from the scorecard, tagged by phase group (`research`, `content`, `audio`). Appended to `lessons_pending.json`.
- **Lesson Reviewer** (`lesson_reviewer.py`): Threshold-triggered promotion — when pending lessons reach 10 entries, deduplicates via Smart Model, expires entries >90 days old, and promotes survivors to `lessons_promoted.json`.
- **Telegram Report** (`telegram_report.py`): Optional pipeline completion summary delivered via Telegram.

## Output Files

All outputs are saved to a timestamped directory under `research_outputs/`:

```
research_outputs/YYYY-MM-DD_HH-MM-SS/
├── research/
│   ├── research_framing.md           Phase 0 — domain-aware scope and hypotheses
│   ├── domain_classification.json    Phase 0 — domain routing decision (clinical vs. social science)
│   ├── affirmative_case.md           Step 5a — affirmative case
│   ├── falsification_case.md         Step 5b — falsification case
│   ├── grade_synthesis.md            Step 7 — GRADE synthesis
│   ├── research_sources.json         Research library (structured source data)
│   ├── clinical_math.md              Step 6 — deterministic ARR/NNT table
│   ├── search_strategy_aff.json      Step 1a — affirmative PICO/MeSH/Boolean
│   ├── search_strategy_neg.json      Step 1b — falsification PICO/MeSH/Boolean
│   ├── screening_results_aff.json    Step 3a — affirmative screening decisions
│   ├── screening_results_neg.json    Step 3b — falsification screening decisions
│   ├── source_of_truth.md            IMRaD scientific paper (Abstract, Intro, Methods, Results, Discussion, References)
│   ├── source_of_truth_ja.md         Phase 3 — translated SOT (Japanese only)
│   ├── url_validation_results.json   Phase 2 — batch URL validation results
│   ├── EPISODE_BLUEPRINT.md          Phase 4 — episode blueprint (thesis, hook, narrative arc, citations)
│   └── accuracy_audit.md             Phase 7 — drift detection
├── scripts/
│   ├── script_draft.md               Phase 5 — draft script
│   ├── script_polished.md            Phase 6 — polished script
│   ├── script_final.md               Phase 7/8 — final script (post-audit corrections if needed)
│   ├── script.txt                    Final script for TTS
│   └── ACCURACY_CORRECTIONS.md       Phase 7 — correction log (if HIGH-severity drift found)
├── audio/
│   ├── audio.wav                     Phase 8 — raw TTS audio (24kHz WAV, no BGM)
│   └── audio_mixed.wav               Phase 8 — BGM-mixed final audio (24kHz WAV)
└── meta/
    ├── session_metadata.txt          Topic, language, character assignments
    ├── podcast_generation.log        Execution log
    ├── checkpoint.json               Resume checkpoint for interrupted runs
    ├── extraction_cache.json         Step 4 — PMID-keyed extraction cache (per-run snapshot)
    ├── research_framing.pdf          PDF export of research framing
    ├── source_of_truth.pdf           PDF export of IMRaD SOT
    ├── source_of_truth_ja.pdf        PDF export of translated SOT (Japanese only)
    └── accuracy_audit.pdf            PDF export of accuracy audit

research_outputs/
├── run_scorecard.json                Run quality scorecard (evaluation module)
├── topic_index.json                  Cross-run topic index (used by reuse workflow)
└── extraction_cache.json             Step 4 — persistent PMID-keyed extraction cache (shared across all runs)
```

## Source of Truth (IMRaD Format)

The `source_of_truth.md` produced after Phase 1 follows the **IMRaD** scientific paper structure. Each section is assembled deterministically from the pipeline's raw outputs — no additional LLM calls are made for the SOT itself.

| IMRaD Section | Pipeline Source | Content |
|---|---|---|
| **Abstract** | Steps 1a, 6, 7 | PICO question, methods summary, key NNT finding, GRADE verdict |
| **1. Introduction** | Phase 0 + Steps 1a/1b | Research framing, clinical gap, dual affirmative/falsification hypotheses |
| **2. Methods** | Steps 1a/1b–4a/4b, 6–7 | PICO, MeSH terms, Boolean strings, databases, 500-result cap, screening criteria, 4-tier full-text retrieval, ARR/NNT formulas, GRADE methodology |
| **3. Results** | Steps 2–6 | PRISMA flow table (per-track), study characteristics table (design, N, demographics, follow-up, funding, bias), NNT/ARR table |
| **4. Discussion** | Steps 5a, 5b, 7 | §4.1 Affirmative case · §4.2 Falsification case · §4.3 GRADE assessment · §4.4 Balanced verdict · §4.5 Limitations · §4.6 Recommendations |
| **5. References** | Steps 4a/4b | Numbered bibliography from PMID/DOI/authors/journal/year — no LLM hallucination |

The evidence quality banner (`⚠ Evidence Quality Notice`) is prepended when the affirmative track retrieved fewer than 30 candidate studies.

## Project Structure

```
dr2_podcast/                          # Main package
├── pipeline.py                       # Main orchestrator (~3,200 lines)
├── pipeline_flow.py                  # Prefect @flow/@task orchestration (phases 0–8)
├── pipeline_crew.py                  # CrewAI agent/task definitions
├── pipeline_script.py                # Script validation and trimming
├── pipeline_sot.py                   # Source-of-Truth (IMRaD) builder
├── pipeline_translation.py           # Translation pipeline (translate only, no audit)
├── pipeline_types.py                 # TypedDicts for data contracts
├── config.py                         # Centralized configuration and model settings
├── utils.py                          # Shared utility functions
├── prompt_strings.py                 # Externalized LLM prompt strings (EN/JA)
├── sot_i18n.py                       # Internationalized IMRaD templates (EN/JA)
├── research/                         # Research sub-package
│   ├── clinical.py                   # 7-step clinical pipeline (PICO/GRADE)
│   ├── clinical_math.py              # Deterministic ARR/NNT calculator (no LLM)
│   ├── effect_size_math.py           # Cohen's d, Hedges' g, OR-to-d, r-to-d (no LLM)
│   ├── domain_classifier.py          # Clinical vs. social science classifier
│   ├── search_service.py             # SearXNG client, page scraping, content extraction
│   ├── fulltext_fetcher.py           # 4-tier full-text fetcher (PMC/Unpaywall/scrape)
│   ├── metadata_clients.py           # OpenAlex, Semantic Scholar, Crossref, ERIC clients
│   ├── social_science.py             # DEPRECATED — moved into clinical.py Orchestrator
│   └── wwc_database.py               # What Works Clearinghouse SQLite database
├── audio/
│   └── engine.py                     # Kokoro TTS (EN) + VOICEVOX (JA) + BGM mixing (dual-voice, 24kHz WAV)
├── evaluation/
│   ├── scorecard.py                  # Run quality scorecard generation
│   ├── lesson_generator.py           # LLM-assisted observation extraction from scorecards
│   ├── lesson_reviewer.py            # Lesson review and validation
│   └── telegram_report.py           # Telegram delivery of evaluation reports
├── web/
│   └── web_ui.py                     # FastAPI web UI (progress tracking, queue, uploads)
└── tools/
    ├── link_validator.py              # URL validation via HEAD requests
    └── upload_utils.py                # Buzzsprout and YouTube upload utilities
tests/                                # Test suite (22 files, ~195 tests)
```

| Support Files | Purpose |
|---------------|---------|
| `start_podcast_web_ui.sh` | Web UI launcher script |
| `start_vllm_docker.sh` | vLLM Docker container launcher |
| `docker/qwen3-tts/` | Legacy Qwen3-TTS server (unused — replaced by VOICEVOX) |
| `requirements.txt` | Core Python dependencies |
| `podcast_tasks.json` | Persistent task queue for the web UI |
| `Podcast BGM/` | Pre-built WAV background music library for BGM mixing |
| `asset/` | Kokoro TTS model weights (safetensors, voice files, tokenizer) |
| `voicevox_samples/` | Reference WAV samples for VOICEVOX speaker IDs |
| `educational_series/` | Hand-authored educational episode source-of-truth + scripts (bypasses CrewAI pipeline; see below) |
| `Intro Script ja.txt` | Reusable Japanese channel intro text |
| `regen_script_and_audio.py` | One-shot helper: regenerate polish + audio from an existing run's `script_draft.md` |
| `regen_edu_ep01.py`, `regen_edu_ep01_voicevox.py` | Educational Episode 1 regen (VOICEVOX variant uses the migrated TTS backend) |
| `regen_edu_ep02_voicevox.py`, `regen_edu_ep02_audio_only.py` | Educational Episode 2 regen (audio-only variant skips `_finalize_script()` when the draft already has `## [emotion]` cues) |
| `presentation/` | Project presentation assets (`build_slides.py`, `DR_2_Podcast_presentation.pptx`) |

## Educational Episode Workflow (Manual Script Path)

For the scientific-literacy series on the Japanese channel **仕組み化パパの効率化ラボ**, the CrewAI pipeline (tuned for evidence-review 4-Act structure) produces the wrong content shape. The working pattern for educational episodes bypasses CrewAI:

1. Author the Source-of-Truth by hand in `educational_series/epNN_source_of_truth.md`
2. Author the Host 1 / Host 2 script by hand in `educational_series/epNN_script_draft.md`
3. Run the matching `regen_edu_epNN_voicevox.py` helper, which loads the draft and invokes `_finalize_script()` + `_run_audio_pipeline()` directly for VOICEVOX + BGM rendering
4. For drafts that already include `## [emotion]` cues, use the `_audio_only.py` variant to skip `_finalize_script()` and avoid a redundant Smart-Model pass

See `educational_series/GRAND_PLAN.md` for the 14-day series outline.

## License

MIT
