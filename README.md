# Deep-Research Podcast Crew

An AI-powered pipeline that deeply researches any scientific topic, conducts adversarial peer review across 10 structured phases, and produces a broadcast-ready podcast with Kokoro TTS audio — all running on local models.

## System Overview

```
                       ┌─────────────────────────────┐
                       │     Topic + Language Input   │
                       └──────────────┬──────────────┘
                                      ▼
                  ┌──────────────────────────────────────┐
                  │  Phase 0 — Research Framing           │
                  │  (Research Framing Specialist)         │
                  └──────────────┬───────────────────────┘
                                 ▼
              ┌─────────────────────────────────────────────┐
              │  Deep Research Pre-Scan (Map-Reduce)         │
              │  Smart model plans → SearXNG/Brave fetch →   │
              │  Fast model summarizes → Smart synthesizes   │
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
     │ CREW 2 — Synthesis & Production                               │
     │                                                               │
     │  Phase 3: Counter-Evidence Research (Adversarial Researcher)  │
     │  Phase 4: Source Validation (Source Verifier)                  │
     │  Phase 5: Source-of-Truth Synthesis (Auditor)                 │
     │  Phase 6a: Show Notes & Citations (Producer)                  │
     │  Phase 6b: Podcast Script Generation (Producer)               │
     │  Phase 7: Script Polishing (Personality Editor)               │
     │  Phase 8: Accuracy Check (Auditor) [advisory]                │
     └───────────────────────┬───────────────────────────────────────┘
                             ▼
              ┌─────────────────────────────────────────────┐
              │  Audio Generation — Kokoro TTS               │
              │  Two voices, 24kHz WAV, ~10 min episode      │
              └─────────────────────────────────────────────┘
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
| **Podcast Producer (The Showrunner)** | Transforms research into a 1,500-word Masters-level debate script | — |
| **Podcast Personality (The Editor)** | Polishes script for natural verbal delivery, enforces word count and depth | — |

## Dual-Model Architecture

The system uses two local LLMs working in tandem:

| Model | Hosted On | Role |
|-------|-----------|------|
| **Qwen2.5-32B-Instruct-AWQ** | vLLM (port 8000) | Planning, research synthesis, script writing, auditing (32k context) |
| **phi4-mini** | Ollama (port 11434) | Parallel page summarization during deep research, report condensation before injection |

During the **deep research pre-scan**, the smart model plans search queries (5-10 per iteration, up to 3 iterations), pages are fetched in parallel via SearXNG/BraveSearch, the fast model summarizes each page, and the smart model synthesizes findings into a final report. This map-reduce approach lets the system process 20-30 sources in minutes.

If phi4-mini is unavailable, the smart model handles all summarization (slower but functional).

## Research Library

After the deep research pre-scan, all source-level data is saved to `deep_research_sources.json`. Three CrewAI tools let agents browse and drill into this library during their tasks:

- **ListResearchSources** — Browse a numbered index of all sources (title, URL, research goal)
- **ReadResearchSource** — Read the full extracted summary for any specific source by index
- **ReadFullReport** — Read an entire research report from disk

Before injection into agent task descriptions, full reports are **condensed by phi4-mini** (~2000 words) instead of being hard-truncated at 6000 characters. This preserves ALL findings across the entire report rather than silently dropping everything after the cutoff.

## Pipeline Phases

### Phase 0 — Research Framing & Hypothesis
The Research Framing Specialist defines scope boundaries, core research questions, evidence criteria, suggested search directions, and hypotheses to test. This framing document guides all downstream phases.

### Deep Research Pre-Scan (Dual-Model Map-Reduce)
An autonomous `Orchestrator` runs iterative research for three roles (lead, counter, audit):
1. Smart model generates targeted search queries based on framing context
2. SearXNG + BraveSearch fetch results; pages are scraped and cleaned
3. Fast model extracts key facts from each page in parallel
4. Smart model reviews coverage, identifies gaps, generates follow-up queries
5. After up to 3 iterations, smart model writes a final synthesis report

Outputs: `deep_research_lead.md`, `deep_research_counter.md`, `deep_research_audit.md`, `deep_research_sources.json`

### Phase 1 — Systematic Evidence Gathering
The Lead Researcher conducts a deep dive guided by the framing document and pre-scan evidence. Findings are grouped by mechanism of action and clinical evidence level, with every source labeled (RCT, Observational, Animal Model).

### Phase 2 — Research Gate & Gap Analysis
The Scientific Auditor evaluates coverage completeness and issues a **PASS/FAIL verdict**. If FAIL, Phase 2b triggers targeted gap-fill research on weak areas.

### Phase 3 — Counter-Evidence Research
The Adversarial Researcher challenges lead findings by searching for contradictory RCTs, null results, confounders, and methodology flaws.

### Phase 4 — Source Validation
The Source Verifier validates every cited URL (HEAD requests) and checks that claims actually match what sources say.

### Phase 5 — Source-of-Truth Synthesis
The Auditor combines all evidence into a single authoritative reference document with confidence levels (HIGH / MEDIUM / LOW / CONTESTED), a Reliability Scorecard, and a Caveat Box.

### Phase 6a/6b — Show Notes & Script (parallel)
The Producer generates show notes with citations and a 1,500-word (~10 min) debate script. The script uses `Host 1:` / `Host 2:` dialogue format targeting Masters/PhD-level depth — no basic term definitions.

### Phase 7 — Script Polishing
The Personality Editor refines for natural verbal delivery, ensures Host 2 challenges weak evidence, and enforces the 1,500-word target.

### Phase 8 — Accuracy Check (advisory)
The Auditor scans the polished script for 5 drift patterns: correlation-to-causation, hedge removal, confidence inflation, cherry-picking, and contested-as-settled. Non-blocking — the pipeline continues regardless.

### Audio Generation
Kokoro TTS renders the polished script with two distinct voices at 24kHz WAV:
- **Host 1**: `bm_george` (British male)
- **Host 2**: `af_nicole` (American female)

## Podcast Characters

Each session randomly assigns two characters to the pro/con roles:

| Character | Personality | When Supporting | When Critical |
|-----------|-------------|-----------------|---------------|
| **Kaz** | Enthusiastic science advocate, optimistic, data-driven | Champions the evidence | Plays devil's advocate with energy |
| **Erika** | Skeptical analyst, cautious, evidence-focused | Presents findings carefully | Challenges methodology rigorously |

## Prerequisites

- **Python 3.11** (Kokoro TTS requires < 3.13; recommended conda env: `podcast_flow`)
- **NVIDIA GPU** (for vLLM; Kokoro falls back to CPU if GPU kernel is incompatible)

### Required Services

**vLLM** — Smart model (Qwen2.5-32B-Instruct-AWQ):
```bash
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
```

## Environment Variables

```bash
# Required
export BRAVE_API_KEY="your_brave_search_api_key"

# Optional (can also be passed as CLI args)
export PODCAST_TOPIC="effects of intermittent fasting on cognitive performance"
export PODCAST_LANGUAGE="en"          # en or ja
export ACCESSIBILITY_LEVEL="simple"   # simple | moderate | technical

# Model config (defaults shown)
export MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"
export LLM_BASE_URL="http://localhost:8000/v1"
```

## Usage

```bash
# Basic — uses PODCAST_TOPIC env var or default topic
python podcast_crew.py

# Specify topic and language
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
├── RESEARCH_FRAMING.md              Phase 0 — scope and hypotheses
├── deep_research_lead.md            Pre-scan — supporting evidence report
├── deep_research_counter.md         Pre-scan — opposing evidence report
├── deep_research_audit.md           Pre-scan — combined synthesis
├── deep_research_sources.json       Research library (structured source data)
├── SOURCE_OF_TRUTH.md               Phase 5 — authoritative reference
├── SHOW_NOTES.md                    Phase 6a — show notes and citations
├── ACCURACY_CHECK.md                Phase 8 — drift detection results
├── podcast_final_audio.wav          Final podcast audio (24kHz WAV)
├── session_metadata.txt             Topic, language, character assignments
├── podcast_generation.log           Execution log
├── research_framing.pdf             PDF exports of key documents
├── supporting_paper.pdf
├── adversarial_paper.pdf
├── verified_sources_bibliography.pdf
└── source_of_truth.pdf
```

## Project Structure

| File | Purpose |
|------|---------|
| `podcast_crew.py` | Main pipeline — agents, tasks, 10-phase orchestration, research library tools |
| `deep_research_agent.py` | Dual-model map-reduce deep research engine |
| `search_agent.py` | SearXNG client, page scraping, content extraction |
| `research_planner.py` | Structured research planning with iterative gap-filling |
| `audio_engine.py` | Kokoro TTS rendering with dual-voice stitching |
| `link_validator_tool.py` | URL validation via HEAD requests |
| `podcast_web_ui.py` | Web UI frontend (Gradio) |

## Search Backends

| Backend | Type | What it provides |
|---------|------|------------------|
| **BraveSearch** | Paid API | Title, URL, snippet (fast, broad coverage) |
| **SearXNG** | Self-hosted Docker | Full page content via scraping (deeper extraction) |

Both are used during the deep research pre-scan. Agents can also invoke them directly via CrewAI tools during their phases.

## License

MIT
