# Deep-Research Podcast Crew

An AI-powered pipeline that deeply researches any scientific topic using a clinical systematic-review methodology, conducts adversarial peer review across 10 structured phases, and produces a broadcast-ready podcast with Kokoro TTS audio — all running on local models. Includes a FastAPI web UI for one-click production with live progress tracking.

## System Overview

```
                       ┌─────────────────────────────┐
                       │     Topic + Language Input   │
                       │   (Web UI or CLI)            │
                       └──────────────┬──────────────┘
                                      ▼
                  ┌──────────────────────────────────────┐
                  │  Phase 0 — Research Framing           │
                  │  (Research Framing Specialist)        │
                  └──────────────┬───────────────────────┘
                                 ▼
       ┌──────────────────────────────────────────────────────────┐
       │  Deep Research Pre-Scan (8-Step Clinical Pipeline)       │
       │                                                          │
       │  ┌─ AFFIRMATIVE TRACK ──────────────────────────────┐   │
       │  │ Step 1: PICO/MeSH/Boolean strategy (Smart)       │   │
       │  │ Step 2: Wide net — up to 500 results (PubMed +   │   │
       │  │         Google Scholar + Fast model screening)    │   │
       │  │ Step 3: Screen → top 20 (Smart)                  │   │
       │  │ Step 4: Full-text extraction + clinical vars      │   │
       │  │         (PMC/EuropePMC/Unpaywall + Fast model)    │   │
       │  │ Step 5: Affirmative case (Smart)                  │   │
       │  └───────────────────────────────────────────────────┘   │
       │  ┌─ FALSIFICATION TRACK ───────────────────────────┐    │
       │  │ Steps 1'–4': Same pipeline, adversarial framing  │    │
       │  │ Step 6: Falsification case (Smart)               │    │
       │  └───────────────────────────────────────────────────┘   │
       │  (Both tracks run in parallel)                           │
       │                          ▼                               │
       │  Step 7: Deterministic math — ARR/NNT (Python, no LLM)  │
       │  Step 8: GRADE synthesis — Auditor (Smart)               │
       │                                                          │
       │  Outputs: lead.md, counter.md, audit.md,                 │
       │           deep_research_sources.json,                    │
       │           deep_research_math.md,                         │
       │           deep_research_strategy_aff/neg.json,           │
       │           deep_research_screening.json                   │
       └──────────────┬───────────────────────────────────────────┘
                      ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ CREW 1 — Evidence Gathering                                   │
     │                                                               │
     │  Phase 1: Systematic Evidence Gathering (Lead Researcher)     │
     │  Phase 2: Research Gate & Gap Analysis (Auditor) → PASS/FAIL  │
     │  Phase 2b: Gap-Fill Research (Lead Researcher) [if FAIL]      │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ CREW 2 — Evidence Validation                                  │
     │                                                               │
     │  Phase 3: Counter-Evidence Research (Adversarial Researcher)  │
     │  Phase 4a: Source Validation (Source Verifier)                │
     │  Phase 4b: Source-of-Truth Synthesis (Auditor)               │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
     ┌───────────────────────────────────────────────────────────────┐
     │ CREW 3 — Podcast Production                                   │
     │                                                               │
     │  Phase 5: Show Notes & Citations (Producer)                   │
     │  Phase 6: Podcast Script Generation (Producer)               │
     │  Phase 7: Script Polishing (Personality Editor)              │
     │  Phase 8: Accuracy Check (Auditor) [advisory]               │
     │  [Translation task inserted before 5/7 for non-English]      │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
              ┌─────────────────────────────────────────────┐
              │  Audio Generation — Kokoro TTS               │
              │  Two voices, 24kHz WAV + BGM mixing          │
              └─────────────────────────────────────────────┘
```

## Web UI

A FastAPI-based web interface (`podcast_web_ui.py`) for managing podcast production:

- One-click topic submission with language, accessibility level, and host selection
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
python podcast_web_ui.py --port 8501
```

## Agents

The pipeline orchestrates **7 specialized CrewAI agents**, each with a distinct role:

| Agent | Role | Tools |
|-------|------|-------|
| **Research Framing Specialist** | Defines scope, core questions, evidence criteria before any searching begins | — |
| **Principal Investigator (Lead Researcher)** | Gathers supporting evidence organized by mechanism of action and clinical evidence | RequestSearch, ListResearchSources, ReadResearchSource, ReadFullReport |
| **Adversarial Researcher (The Skeptic)** | Hunts for contradictory evidence, methodology flaws, and null results | RequestSearch, ListResearchSources, ReadResearchSource, ReadFullReport |
| **Scientific Auditor (The Grader)** | Grades research quality, runs PASS/FAIL gate, synthesizes source-of-truth, checks script for scientific drift | LinkValidator, ListResearchSources, ReadResearchSource, ReadFullReport |
| **Scientific Source Verifier** | Validates every cited URL via HEAD requests, verifies claim-to-source accuracy | LinkValidator, ReadValidationResults |
| **Podcast Producer (The Showrunner)** | Transforms research into a debate script targeting Masters/PhD-level depth | — |
| **Podcast Personality (The Editor)** | Polishes script for natural verbal delivery, enforces word count and depth | — |

## Dual-Model Architecture

The system uses two local LLMs working in tandem:

| Role | Default Model | Hosted On | Purpose |
|------|---------------|-----------|---------|
| **Smart model** | `Qwen/Qwen2.5-32B-Instruct-AWQ` | vLLM (port 8000) | PICO strategy, screening, case synthesis, GRADE audit, script writing |
| **Fast model** | `llama3.2:1b` | Ollama (port 11434) | Parallel abstract screening, full-text clinical extraction, report condensation |

Model selection can be overridden via environment variables (`MODEL_NAME`, `LLM_BASE_URL`, `FAST_MODEL_NAME`, `FAST_LLM_BASE_URL`).

If the fast model is unavailable, the smart model handles all summarization (slower but functional).

## Evidence-Based Research Pipeline

The deep research pre-scan implements an 8-step systematic review methodology modelled on clinical trial standards. Both the affirmative and falsification tracks run in parallel.

### Step 1 — Search Strategy Formulation (Smart Model, ~15s)

Translates the topic into a structured **PICO framework** (Population, Intervention, Comparison, Outcome), generates **MeSH terms**, and writes **Boolean search strings** for PubMed, Cochrane CENTRAL, and Google Scholar. The adversarial track adds adverse-effects, null-result, toxicity, and funding-bias terms.

### Step 2 — Wide Net Search (PubMed + Fast Model, ~90s)

Queries PubMed with three Boolean variants (primary, broad, Cochrane subset `cochrane[sb]`) and Google Scholar, collecting up to 500 results. **`PublicationType` in the PubMed XML is used directly** to classify study type (RCT, meta-analysis, systematic review, etc.) without LLM calls. The fast model only processes the ~50% of records where type cannot be determined from XML, extracting `sample_size` and `primary_objective` from the abstract — reducing fast-model calls by ~50%.

### Step 3 — Screening (Smart Model, ~20s)

The smart model scans all records and selects the **top 20 most rigorous human clinical studies**. Inclusion: RCTs, meta-analyses, systematic reviews, large cohort studies (n ≥ 30, prefer n ≥ 100). Exclusion: animal models, in vitro, case reports, conference abstracts, retractions. Context-window overflow (>28K tokens) is handled by chunked screening with a merge step.

### Step 4 — Full-Text Deep Extraction (Fast Model, ~120s)

For each of the top 20 studies, the full text is retrieved via a 4-tier fallback:
1. **PubMed Central OA API** (`oai:pubmedcentral.nih.gov`)
2. **Europe PMC REST API** (free full-text XML for OA articles)
3. **Unpaywall API** (OA PDF location via DOI)
4. **Publisher page scrape** (existing `ContentFetcher` logic)

The fast model then extracts 20 clinical variables per article: `control_event_rate` (CER), `experimental_event_rate` (EER), effect size with CI, attrition, blinding, randomization, ITT analysis, funding source, conflicts of interest, risk of bias, demographics, follow-up period, and biological mechanism.

### Steps 5 & 6 — Affirmative & Falsification Cases (Smart Model, ~30s each)

Each track's smart model writes a structured case report from the extracted data — the affirmative case argues FOR the hypothesis (clinical significance, biological plausibility, consistency, dose-response), the falsification case argues AGAINST it (adverse effects, null results, methodological concerns, funding bias, publication bias).

### Step 7 — Deterministic Math (Python, <1ms)

**No LLM involved.** A pure-Python calculator computes from the extracted CER/EER values:
- **ARR** = CER − EER (Absolute Risk Reduction)
- **RRR** = ARR / CER (Relative Risk Reduction)
- **NNT** = 1 / |ARR| (Number Needed to Treat)

This eliminates LLM math hallucinations. Only studies where both CER and EER were extracted contribute to the NNT table.

### Step 8 — GRADE Synthesis (Smart Model, ~45s)

The Auditor reads both cases and the Python-calculated NNT table, then issues a **GRADE-framework synthesis** (High / Moderate / Low / Very Low evidence quality) with evidence profile, downgrade/upgrade justifications, clinical impact interpretation, PRISMA flow diagram, and consolidated evidence table. The Auditor is instructed never to recalculate ARR/NNT — it uses the Python-provided numbers exactly.

**Total wall-clock time: ~5–6 minutes** (both tracks in parallel).

## Research Library

After the deep research pre-scan, all source-level data is saved to `deep_research_sources.json`. Three CrewAI tools let agents browse and drill into this library during their tasks:

- **ListResearchSources** — Browse a numbered index of all sources (title, URL, research goal)
- **ReadResearchSource** — Read the full extracted summary for any specific source by index
- **ReadFullReport** — Read an entire research report from disk

Before injection into agent task descriptions, full reports are **condensed by the fast model** (~2000 words) instead of being hard-truncated.

## GRADE Audit Report Structure

The Step 8 GRADE synthesis follows this structure:

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

### Phase 0 — Research Framing & Hypothesis
The Research Framing Specialist defines scope boundaries, core research questions, evidence criteria, suggested search directions, and hypotheses to test.

### Deep Research Pre-Scan
See [Evidence-Based Research Pipeline](#evidence-based-research-pipeline) above.

### Phase 1 — Systematic Evidence Gathering
The Lead Researcher conducts a deep dive guided by the framing document and pre-scan evidence. Findings are grouped by evidence tier with "Author et al. (Year)" citations.

### Phase 2 — Research Gate & Gap Analysis
The Scientific Auditor evaluates coverage completeness and issues a **PASS/FAIL verdict**. If FAIL, Phase 2b triggers targeted gap-fill research.

### Phase 3 — Counter-Evidence Research
The Adversarial Researcher challenges lead findings by searching for contradictory RCTs, null results, confounders, and methodology flaws.

### Phase 4a — Source Validation
The Source Verifier validates every cited URL (HEAD requests) and checks that claims match sources.

### Phase 4b — Source-of-Truth Synthesis
The Auditor combines all evidence into an authoritative reference document with confidence levels (HIGH / MEDIUM / LOW / CONTESTED), a Reliability Scorecard, and a Caveat Box.

### Phase 5/6 — Show Notes & Script
The Producer generates show notes with citations and a debate script. The script uses `Kaz:` / `Erika:` dialogue format.

### Phase 7 — Script Polishing
The Personality Editor refines for natural verbal delivery and ensures balanced coverage.

### Phase 8 — Accuracy Check (advisory)
The Auditor scans the polished script for drift patterns: correlation-to-causation, hedge removal, confidence inflation, cherry-picking, and contested-as-settled. Non-blocking.

### Audio Generation
Kokoro TTS renders the polished script with two voices at 24kHz WAV, followed by BGM mixing:

| Language | Host 1 (Kaz) | Host 2 (Erika) |
|----------|--------------|----------------|
| English  | `am_fenrir` (American male) | `af_heart` (American female) |
| Japanese | `jm_kumo` (Japanese male) | `jf_alpha` (Japanese female) |

An alternative Japanese TTS engine (Qwen3-TTS) is available via a local FastAPI server (`docker/qwen3-tts/`).

## Multi-Language Support

The pipeline supports English and Japanese output:
- **English**: Default. All research, scripts, and audio in English.
- **Japanese**: A translation task is inserted into Crew 3 before script polishing and show notes. Kokoro uses Japanese voice models. Host names use katakana (カズ / エリカ).

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
  vllm/vllm-openai:latest \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --max-model-len 32768 \
  --gpu-memory-utilization 0.8 \
  --dtype auto --trust-remote-code --enforce-eager
```

**Ollama** — Required for the fast model:
```bash
ollama serve
ollama pull llama3.2:1b             # Fast model (default)
# Optional upgrade:
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
pip install -r web_ui_requirements.txt  # For the web UI
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

# Model config (defaults shown)
export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
export LLM_BASE_URL="http://localhost:8000/v1"
export FAST_MODEL_NAME="llama3.2:1b"
export FAST_LLM_BASE_URL="http://localhost:11434/v1"

# Web UI authentication (auto-generated if not set)
export PODCAST_WEB_USER="admin"
export PODCAST_WEB_PASSWORD="your_password"

# Upload integration (optional)
export BUZZSPROUT_API_KEY="your_buzzsprout_api_key"
export BUZZSPROUT_PODCAST_ID="your_podcast_id"
export YOUTUBE_CLIENT_SECRET_PATH="/path/to/client_secret.json"
```

## Usage

```bash
# Via Web UI (recommended)
./start_podcast_web_ui.sh

# Via CLI
python podcast_crew.py --topic "neuroplasticity and exercise" --language en

# Reuse previous research (skip research phases, regenerate podcast only)
python podcast_crew.py --reuse-dir research_outputs/2025-01-15_10-30-00 --crew3-only

# Reuse with LLM-assessed supplemental research if needed
python podcast_crew.py --reuse-dir research_outputs/2025-01-15_10-30-00 --check-supplemental
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
├── research_framing.md               Phase 0 — scope and hypotheses
├── research_framing.pdf
├── deep_research_lead.md             Pre-scan — affirmative case (Step 5)
├── deep_research_counter.md          Pre-scan — falsification case (Step 6)
├── deep_research_audit.md            Pre-scan — GRADE synthesis (Step 8)
├── deep_research_sources.json        Research library (structured source data)
├── deep_research_math.md             Step 7 — deterministic ARR/NNT table
├── deep_research_strategy_aff.json   Step 1 — affirmative PICO/MeSH/Boolean
├── deep_research_strategy_neg.json   Step 1' — adversarial PICO/MeSH/Boolean
├── deep_research_screening.json      Step 3 — screening decisions (500 → 20)
├── gap_analysis.md                   Phase 2 — gap analysis
├── supporting_research.md            Phase 1 — lead researcher paper
├── supporting_paper.pdf
├── adversarial_research.md           Phase 3 — counter-evidence paper
├── adversarial_paper.pdf
├── source_verification.md            Phase 4a — URL validation results
├── verified_sources_bibliography.pdf
├── source_of_truth.md                Phase 4b — authoritative reference
├── source_of_truth.pdf
├── show_notes.md                     Phase 5 — show notes and citations
├── podcast_script_raw.md             Phase 6 — raw script
├── podcast_script_polished.md        Phase 7 — polished script
├── podcast_script.txt                Final script for TTS
├── accuracy_check.md                 Phase 8 — drift detection
├── accuracy_check.pdf
├── podcast_final_audio.wav           Final podcast audio (24kHz WAV + BGM)
├── session_metadata.txt              Topic, language, character assignments
├── podcast_generation.log            Execution log
└── url_validation_results.json       Source URL validation data
```

## Project Structure

| File / Directory | Purpose |
|------------------|---------|
| `podcast_crew.py` | Main pipeline — agents, tasks, 10-phase orchestration, research library tools |
| `podcast_web_ui.py` | FastAPI web UI with live progress tracking, task queue, and upload integration |
| `deep_research_agent.py` | 8-step clinical research pipeline (PICO → wide net → screen → extract → cases → math → GRADE) |
| `clinical_math.py` | Deterministic ARR/NNT calculator — pure Python, zero LLM involvement |
| `fulltext_fetcher.py` | 4-tier full-text fetcher: PMC OA → Europe PMC → Unpaywall → publisher scrape |
| `search_agent.py` | SearXNG client, page scraping, content extraction |
| `research_planner.py` | Structured research planning with iterative gap-filling |
| `audio_engine.py` | Kokoro TTS rendering with dual-voice stitching and BGM post-processing |
| `link_validator_tool.py` | URL validation via HEAD requests |
| `upload_utils.py` | Buzzsprout and YouTube upload utilities |
| `test_clinical_math.py` | Unit tests for clinical_math.py (17 tests) |
| `start_podcast_web_ui.sh` | Web UI launcher script |
| `start_vllm_docker.sh` | vLLM Docker container launcher |
| `docker/qwen3-tts/` | Qwen3-TTS FastAPI server for high-quality Japanese TTS (conda env: `qwen3_tts`) |
| `requirements.txt` | Core Python dependencies |
| `web_ui_requirements.txt` | Additional dependencies for the web UI (FastAPI, uvicorn, Google API) |
| `environment.yml` | Conda environment specification (Python 3.11, `podcast_flow`) |
| `podcast_tasks.json` | Persistent task queue for the web UI |
| `Podcast BGM/` | Pre-built WAV background music library for BGM mixing |
| `asset/` | Kokoro TTS model weights (safetensors, voice files, tokenizer) |
| `archived_scripts/` | Deprecated utilities |

## License

MIT
