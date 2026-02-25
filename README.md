# Deep-Research Podcast Crew

An AI-powered pipeline that deeply researches any scientific topic using a clinical systematic-review methodology (PICO/GRADE), synthesizes evidence from both affirmative and adversarial perspectives, and produces a broadcast-ready podcast with local TTS audio (Kokoro for English, Qwen3-TTS for Japanese) — all running on local models. Includes a FastAPI web UI for one-click production with live progress tracking.

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
                  └──────────────┬───────────────────────────┘
                                 ▼
             ┌───────────────────┴────────────────────┐
             │                                        │
             ▼                                        ▼
  ┌──────────────────────────┐           ┌────────────────────────────────┐
  │  Phase 1 — Clinical      │           │  Phase 1 — Social Science      │
  │  Research Pipeline       │           │  Research Pipeline             │
  │                          │           │                                │
  │  Pre-step: Concept       │           │  Pre-step: Concept             │
  │  Decomposition (Fast)    │           │  Decomposition (Fast)          │
  │           ▼              │           │           ▼                    │
  │  ┌─ AFF ──┐ ┌─ FAL ──┐  │           │  ┌─ AFF ──┐ ┌─ FAL ──┐       │
  │  │ 1–5a  │ │ 1–5b  │  │           │  │ 1–5a  │ │ 1–5b  │       │
  │  └───────┘ └───────┘  │           │  └───────┘ └───────┘       │
  │  (asyncio.gather)      │           │  (asyncio.gather)          │
  │           ▼              │           │           ▼                    │
  │  Step 6: ARR/NNT math   │           │  Step 6: Cohen's d / Hedges' g │
  │  Step 7: GRADE synthesis│           │  Step 7: Evidence quality      │
  └──────────┬───────────────┘           └──────────────┬─────────────────┘
             └───────────────────┬────────────────────────┘
                                 ▼
       ┌────────────────────────────────────────────────────────────────────┘
                      ▼
     ┌───────────────────────────────────────────────────────────────┐
     │  Phase 2 — Source Validation (batch HEAD requests)            │
     │  Source-of-Truth synthesis — IMRaD format                     │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
     ┌───────────────────────────────────────────────────────────────┐
     │  Phase 3 — Report Translation (Crew 2, conditional)           │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ Crew 3 — Podcast Production                                   │
     │                                                               │
     │  Phase 4: Episode Blueprint (Producer)                         │
     │  Phase 5: Script Writing (Producer)                           │
     │  Phase 6: Script Polish (Editor)                              │
     │  Phase 7: Accuracy Audit (Auditor) [conditionally blocking]   │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
              ┌─────────────────────────────────────────────┐
              │  Phase 8 — Audio Production                 │
              │  Kokoro TTS (EN), Qwen3-TTS (JA),           │
              │  two voices, 24kHz WAV + BGM                │
              └─────────────────────────────────────────────┘
```

## Web UI

A FastAPI-based web interface (`web_ui.py`) for managing podcast production:

- One-click topic submission with language, accessibility level, host selection, and channel branding (intro text, target audience, mission)
- **Live progress tracking**: phase name, progress bar, ETA, artifact count, source favicon grid
- **Task queue**: submit multiple requests — confirmation dialog shows queue position, running task progress stays visible
- **Production History**: collapsible list of past runs with download links
- **Upload integration**: optional Buzzsprout (draft) and YouTube (private) publishing
- **Research reuse**: reuse previous research artifacts with optional LLM-assessed supplemental research
- **System status**: checks vLLM and Ollama availability before submission
- Basic authentication support (auto-generated credentials or via env vars)

Launch:
```bash
./start_podcast_web_ui.sh
# or directly:
python web_ui.py --port 8501
```

## Agents

The pipeline uses **4 CrewAI agents** (down from 7 — the clinical pipeline replaced the Lead Researcher, Adversarial Researcher, and Source Verifier):

| Agent | Variable | Role | Tools |
|-------|----------|------|-------|
| **Research Framing Specialist** | `framing_agent` | Defines scope, core questions, evidence criteria before any searching begins | — |
| **Scientific Auditor** | `auditor_agent` | Checks polished script for scientific drift against the Source-of-Truth | LinkValidator, ListResearchSources, ReadResearchSource, ReadFullReport |
| **Podcast Producer** | `producer_agent` | Transforms research into a debate script targeting Masters/PhD-level depth | — |
| **Podcast Editor** | `editor_agent` | Polishes script for natural verbal delivery, enforces word count and depth | — |

## Tri-Model Architecture

The system uses three local LLMs working in tandem:

| Role | Default Model | Hosted On | Purpose |
|------|---------------|-----------|---------|
| **Smart model** | `Qwen/Qwen3-32B-AWQ` | vLLM (port 8000) | PICO strategy, screening, case synthesis, GRADE audit, script writing |
| **Mid-tier model** | `qwen2.5:7b` | Ollama (port 11434) | Pipelined report translation (Phase 3) |
| **Fast model** | `llama3.2:1b` | Ollama (port 11434) | Parallel abstract screening, full-text clinical extraction, report condensation |

Model selection can be overridden via environment variables (`MODEL_NAME`, `LLM_BASE_URL`, `MID_MODEL_NAME`, `MID_LLM_BASE_URL`, `FAST_MODEL_NAME`, `FAST_LLM_BASE_URL`).

The mid-tier model handles Phase 3 translation via a **pipelined architecture**: the 7B model translates each SOT section (~4x faster than the 32B smart model), while the smart model audits completed translations concurrently — so translation and auditing overlap instead of running sequentially. If the mid-tier model is unavailable, the pipeline falls back to smart-model-only mode (sequential translate then audit).

If the fast model is unavailable, the smart model handles all summarization (slower but functional).

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
For non-English output, the Source-of-Truth is translated using a **pipelined architecture**: the mid-tier model (`qwen2.5:7b`) translates each IMRaD section while the smart model (`Qwen3-32B`) audits completed translations concurrently — fixing Chinese contamination, garbled text, and terminology errors. Sections larger than 8K characters are translated directly by the smart model to avoid truncation. Skipped entirely for English runs.

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

| Language | TTS Engine | Host 1 (Kaz) | Host 2 (Erika) |
|----------|------------|--------------|----------------|
| English  | Kokoro TTS (local, CPU) | `am_fenrir` (American male) | `af_heart` (American female) |
| Japanese | Qwen3-TTS (GPU via FastAPI server) | Aiden (male) | Ono_Anna (native Japanese female) |

## Multi-Language Support

The pipeline supports English and Japanese output:
- **English**: Default. All research, scripts, and audio in English.
- **Japanese**: A translation task runs in Crew 2 before Crew 3. Audio is rendered by **Qwen3-TTS** (GPU, FastAPI server at port 8082) — Kokoro is not used for Japanese. Host names use katakana (カズ / エリカ). A post-Crew 3 language auditor detects Chinese contamination (CJK text without hiragana/katakana) and automatically runs a correction pass to translate any Chinese passages into natural Japanese.

### Script Length Enforcement

After Crew 3 completes, the pipeline measures script length against the target (language-aware: words for English, characters for Japanese) with a ±10% tolerance. If the script is too short, **up to 3 expansion passes** run automatically — each pass adds deeper scientific explanation, real-world examples, and host dialogue. Expansion stops early if a pass fails to increase the length.

## Podcast Characters

| Character | Personality | When Supporting | When Critical |
|-----------|-------------|-----------------|---------------|
| **Kaz** | Enthusiastic science advocate, optimistic, data-driven | Champions the evidence | Plays devil's advocate with energy |
| **Erika** | Skeptical analyst, cautious, evidence-focused | Presents findings carefully | Challenges methodology rigorously |

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
# or manually:
docker run --runtime nvidia --gpus all -p 8000:8000 \
  --name vllm-server \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --ipc=host \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  vllm/vllm-openai:v0.13.0 \
  --model Qwen/Qwen3-32B-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.65 \
  --dtype auto --trust-remote-code --enforce-eager --enable-prefix-caching
```

**Ollama** — Required for the fast model and mid-tier model:
```bash
ollama serve
ollama pull llama3.2:1b             # Fast model (default)
ollama pull qwen2.5:7b              # Mid-tier model (pipelined translation)
# Optional fast model upgrade:
ollama pull phi4-mini
# Then set: export FAST_MODEL_NAME="phi4-mini"
```

**SearXNG** — Self-hosted search (optional, improves source diversity):
```bash
docker run -d -p 8080:8080 searxng/searxng:latest
```

**Qwen3-TTS** — Optional high-quality Japanese TTS (conda env: `qwen3_tts`):
```bash
bash docker/qwen3-tts/run_server.sh
# First-time init:
bash docker/qwen3-tts/init_and_start.sh
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
export PODCAST_HOSTS="random"         # random | kaz_erika | erika_kaz

# Optional — channel branding
export PODCAST_CHANNEL_INTRO="Welcome to Deep Research Podcast. Today we're..."
export PODCAST_CORE_TARGET="busy professionals aged 30-50"
export PODCAST_CHANNEL_MISSION="turning complex science into actionable protocols"

# Model config (defaults shown)
export MODEL_NAME="Qwen/Qwen3-32B-AWQ"
export LLM_BASE_URL="http://localhost:8000/v1"
export FAST_MODEL_NAME="llama3.2:1b"
export FAST_LLM_BASE_URL="http://localhost:11434/v1"
export MID_MODEL_NAME="qwen2.5:7b"
export MID_LLM_BASE_URL="http://localhost:11434/v1"

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
python pipeline.py --topic "neuroplasticity and exercise" --language en

# Reuse previous research (skip research phases, regenerate podcast only)
python pipeline.py --reuse-dir research_outputs/2025-01-15_10-30-00 --crew3-only

# Reuse with LLM-assessed supplemental research if needed
python pipeline.py --reuse-dir research_outputs/2025-01-15_10-30-00 --check-supplemental
```

## Accessibility Levels

Control how aggressively scientific terminology is simplified:

| Level | Audience | Behavior |
|-------|----------|----------|
| `simple` | General educated (college-level) | Define every scientific term inline with plain-English analogies |
| `moderate` | Science enthusiasts | Define key terms once, assume basic cause-and-effect literacy |
| `technical` | Professionals in related fields | Standard terminology, no simplification, focus on depth |

## Output Files

All outputs are saved to a timestamped directory under `research_outputs/`:

```
research_outputs/YYYY-MM-DD_HH-MM-SS/
├── research_framing.md               Phase 0 — domain-aware scope and hypotheses
├── research_framing.pdf
├── domain_classification.json        Phase 0 — domain routing decision (clinical vs. social science)
├── affirmative_case.md               Step 5a — affirmative case
├── falsification_case.md             Step 5b — falsification case
├── grade_synthesis.md                Step 7 — GRADE synthesis
├── research_sources.json             Research library (structured source data)
├── clinical_math.md                  Step 6 — deterministic ARR/NNT table
├── search_strategy_aff.json          Step 1a — affirmative PICO/MeSH/Boolean
├── search_strategy_neg.json          Step 1b — falsification PICO/MeSH/Boolean
├── screening_results_aff.json        Step 3a — affirmative screening decisions
├── screening_results_neg.json        Step 3b — falsification screening decisions
├── source_of_truth.md                IMRaD scientific paper (Abstract, Intro, Methods, Results, Discussion, References)
├── source_of_truth.pdf
├── source_of_truth_ja.md             Phase 3 — translated SOT (Japanese only)
├── source_of_truth_ja.pdf            Phase 3 — translated SOT PDF (Japanese only)
├── url_validation_results.json       Phase 2 — batch URL validation results
├── EPISODE_BLUEPRINT.md              Phase 4 — episode blueprint (thesis, hook, narrative arc, citations)
├── script_draft.md                   Phase 5 — draft script
├── script_final.md                   Phase 6 — polished script
├── script.txt                        Final script for TTS
├── accuracy_audit.md                 Phase 7 — drift detection
├── accuracy_audit.pdf
├── ACCURACY_CORRECTIONS.md           Phase 7 — corrections applied (only if HIGH-severity drift)
├── audio.wav                         Phase 8 — raw TTS podcast audio (24kHz WAV)
├── audio_mixed.wav                   Phase 8 — final podcast with BGM mixing
├── session_metadata.txt              Topic, language, character assignments
└── podcast_generation.log            Execution log

research_outputs/
└── extraction_cache.json             Step 4 — PMID-keyed extraction cache (persistent across all runs)
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

| File / Directory | Purpose |
|------------------|---------|
| `pipeline.py` | Main pipeline — agents, tasks, orchestration, research library tools |
| `web_ui.py` | FastAPI web UI with live progress tracking, task queue, and upload integration |
| `clinical_research.py` | 7-step clinical research pipeline (concept decomposition → tiered keywords + auditor gate → cascade search → tier-aware screen → extract → cases → math → GRADE) |
| `clinical_math.py` | Deterministic ARR/NNT calculator — pure Python, zero LLM involvement |
| `effect_size_math.py` | Deterministic effect size calculator — Cohen's d, Hedges' g, OR-to-d, r-to-d (no LLM) |
| `domain_classifier.py` | Topic domain classifier (clinical vs. social science) — deterministic keyword rules + LLM fallback |
| `social_science_research.py` | 7-step social science research pipeline (PECO framework, OpenAlex + ERIC, effect sizes, evidence quality levels) |
| `metadata_clients.py` | Async API clients for OpenAlex, Semantic Scholar, Crossref, ERIC with SQLite rate-limit caching |
| `wwc_database.py` | Local SQLite database for What Works Clearinghouse education intervention quality ratings |
| `fulltext_fetcher.py` | 4-tier full-text fetcher: PMC OA → Europe PMC → Unpaywall → publisher scrape |
| `search_service.py` | SearXNG client, page scraping, content extraction |
| `audio_engine.py` | Kokoro TTS rendering with dual-voice stitching and BGM post-processing |
| `audio_mixer.py` | BGM mixing with pre-roll, post-roll, and transition bumps |
| `link_validator.py` | URL validation via HEAD requests |
| `upload_utils.py` | Buzzsprout and YouTube upload utilities |
| `test_clinical_math.py` | Unit tests — clinical_math (17), effect_size_math (34), domain_classifier (16), metadata_clients (31), wwc_database (19) — 117 tests total |
| `start_podcast_web_ui.sh` | Web UI launcher script |
| `start_vllm_docker.sh` | vLLM Docker container launcher |
| `docker/qwen3-tts/` | Qwen3-TTS FastAPI server for high-quality Japanese TTS (conda env: `qwen3_tts`) |
| `requirements.txt` | Core Python dependencies |
| `podcast_tasks.json` | Persistent task queue for the web UI |
| `Podcast BGM/` | Pre-built WAV background music library for BGM mixing |
| `asset/` | Kokoro TTS model weights (safetensors, voice files, tokenizer) |

## License

MIT
