# Deep-Research Podcast Crew

An AI-powered pipeline that deeply researches any scientific topic, conducts adversarial peer review across 10 structured phases, and produces a broadcast-ready podcast with Kokoro TTS audio — all running on local models. Includes a FastAPI web UI for one-click production with live progress tracking.

## System Overview

```
                       ┌─────────────────────────────┐
                       │     Topic + Language Input   │
                       │   (Web UI or CLI)            │
                       └──────────────┬──────────────┘
                                      ▼
                  ┌──────────────────────────────────────┐
                  │  Phase 0 — Research Framing           │
                  │  (Research Framing Specialist)         │
                  └──────────────┬───────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────────┐
              │  Deep Research Pre-Scan (Dual-Model)         │
              │  Tiered search: PubMed + Google Scholar      │
              │  → general web (if academic insufficient)    │
              │  Smart model plans → workers fetch & extract │
              │  → Fast model summarizes → Smart synthesizes │
              │  PRISMA-style methodology tracking           │
              │  Outputs: lead.md, counter.md, audit.md,     │
              │           deep_research_sources.json          │
              └──────────────┬──────────────────────────────┘
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
- Basic authentication support

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
| **Principal Investigator (Lead Researcher)** | Gathers supporting evidence organized by mechanism of action and clinical evidence | BraveSearch, DeepSearch, Research Library |
| **Adversarial Researcher (The Skeptic)** | Hunts for contradictory evidence, methodology flaws, and null results | BraveSearch, DeepSearch, Research Library |
| **Scientific Auditor (The Grader)** | Grades research quality, runs PASS/FAIL gate, synthesizes source-of-truth, checks script for scientific drift | BraveSearch, DeepSearch, LinkValidator, Research Library |
| **Scientific Source Verifier** | Validates every cited URL via HEAD requests, verifies claim-to-source accuracy | LinkValidator |
| **Podcast Producer (The Showrunner)** | Transforms research into a debate script targeting Masters/PhD-level depth | — |
| **Podcast Personality (The Editor)** | Polishes script for natural verbal delivery, enforces word count and depth | — |

## Dual-Model Architecture

The system uses two local LLMs working in tandem:

| Model | Hosted On | Role |
|-------|-----------|------|
| **Qwen2.5-32B-Instruct-AWQ** | vLLM (port 8000) | Planning, research synthesis, script writing, auditing (32k context) |
| **phi4-mini** | Ollama (port 11434) | Parallel page summarization during deep research, report condensation before injection |

If phi4-mini is unavailable, the smart model handles all summarization (slower but functional).

## Tiered Academic Search

The deep research engine uses a 3-tier search strategy prioritizing academic sources:

1. **Tier 1 (Academic)**: PubMed (via NCBI E-utilities) + Google Scholar (via SearXNG)
2. **Tier 2 (Sufficiency check)**: If ≥5 academic results found, skip general web
3. **Tier 3 (General web)**: Google, Bing, Brave (only if academic sources insufficient)

Each source gets structured metadata extraction: study type, sample size, effect size, journal, authors, publication year, funding source, demographics, and limitations.

**PRISMA-style tracking**: The audit report includes an auto-generated methodology section with search date, databases searched, articles identified → screened → included, and tier breakdown.

## Research Library

After the deep research pre-scan, all source-level data is saved to `deep_research_sources.json`. Three CrewAI tools let agents browse and drill into this library during their tasks:

- **ListResearchSources** — Browse a numbered index of all sources (title, URL, research goal)
- **ReadResearchSource** — Read the full extracted summary for any specific source by index
- **ReadFullReport** — Read an entire research report from disk

Before injection into agent task descriptions, full reports are **condensed by phi4-mini** (~2000 words) instead of being hard-truncated.

## Scientific Report Structure

The audit report follows a unified scientific review format:

1. **Abstract** (200-300 words)
2. **Introduction** (background, scope, research questions)
3. **Methodology** (PRISMA flow: databases, search date, source selection, inclusion/exclusion criteria)
4. **Results** (grouped by evidence tier: meta-analyses → RCTs → cohort → mechanistic → expert opinion)
5. **Discussion** (synthesis, contradictions, evidence quality, recency analysis, conflicts of interest, cross-study comparison)
6. **Conclusions**
7. **Evidence Summary Table** (Author, Year, Study Type, N, Key Finding, Effect Size, Funding, Journal)
8. **References** (standardized: Author et al. (Year). Title. Journal. DOI/URL)

Citations use "Author et al. (Year)" format in body text. Industry-funded studies are flagged in the discussion.

## Pipeline Phases

### Phase 0 — Research Framing & Hypothesis
The Research Framing Specialist defines scope boundaries, core research questions, evidence criteria, suggested search directions, and hypotheses to test.

### Deep Research Pre-Scan (Dual-Model Map-Reduce)
An autonomous `Orchestrator` runs parallel research for lead and counter roles:
1. Smart model generates targeted search queries based on framing context
2. Tiered search (PubMed → Google Scholar → general web) fetches results
3. Pages are scraped and cleaned; fast model extracts structured metadata + key facts
4. Smart model reviews coverage, identifies gaps, generates follow-up queries (up to 3 iterations)
5. Smart model writes final synthesis reports

Outputs: `deep_research_lead.md`, `deep_research_counter.md`, `deep_research_audit.md`, `deep_research_sources.json`

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

**vLLM** — Smart model (Qwen2.5-32B-Instruct-AWQ):
```bash
./start_vllm_docker.sh
# or manually:
docker run --gpus all -p 8000:8000 \
  --name vllm-final \
  vllm/vllm-openai:latest \
  python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen2.5-32B-Instruct-AWQ \
  --max-model-len 32768
```

**Ollama** — Fast model (phi4-mini, optional but recommended):
```bash
ollama serve
ollama pull phi4-mini
```

**SearXNG** — Self-hosted search (optional, improves source diversity):
```bash
docker run -d -p 8080:8080 searxng/searxng:latest
```

## Installation

```bash
conda activate podcast_flow
pip install -r requirements.txt
python -m unidic download  # Required for Kokoro TTS
```

## Environment Variables

```bash
# Required
export BRAVE_API_KEY="your_brave_search_api_key"

# Optional
export PUBMED_API_KEY="your_ncbi_api_key"  # Higher rate limits for PubMed
export PODCAST_TOPIC="effects of intermittent fasting on cognitive performance"
export PODCAST_LANGUAGE="en"          # en or ja
export ACCESSIBILITY_LEVEL="simple"   # simple | moderate | technical
export PODCAST_LENGTH="medium"        # short | medium | long
export PODCAST_HOSTS="random"         # random | kaz_erika | erika_kaz

# Model config (defaults shown)
export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
export LLM_BASE_URL="http://localhost:8000/v1"
export FAST_MODEL_NAME="phi4-mini"
export FAST_LLM_BASE_URL="http://localhost:11434/v1"
```

## Usage

```bash
# Via Web UI (recommended)
./start_podcast_web_ui.sh

# Via CLI
python podcast_crew.py --topic "neuroplasticity and exercise" --language en
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
├── deep_research_lead.md             Pre-scan — supporting evidence
├── deep_research_counter.md          Pre-scan — opposing evidence
├── deep_research_audit.md            Pre-scan — unified scientific review
├── deep_research_sources.json        Research library (structured source data)
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

| File | Purpose |
|------|---------|
| `podcast_crew.py` | Main pipeline — agents, tasks, 10-phase orchestration, research library tools |
| `podcast_web_ui.py` | FastAPI web UI with live progress tracking, task queue, and upload integration |
| `deep_research_agent.py` | Dual-model map-reduce research engine with tiered academic search and PRISMA tracking |
| `search_agent.py` | SearXNG client, page scraping, content extraction |
| `research_planner.py` | Structured research planning with iterative gap-filling |
| `audio_engine.py` | Kokoro TTS rendering with dual-voice stitching and BGM post-processing |
| `link_validator_tool.py` | URL validation via HEAD requests |
| `upload_utils.py` | Buzzsprout and YouTube upload utilities |
| `start_podcast_web_ui.sh` | Web UI launcher script |
| `start_vllm_docker.sh` | vLLM Docker container launcher |

## License

MIT
