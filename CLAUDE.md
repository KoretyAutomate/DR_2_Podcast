# CLAUDE.md — DR_2_Podcast Project

## Project Context

**Primary project:** AI-powered deep-research podcast pipeline with adversarial peer review
**Language:** Python 3.11
**Project root:** `/home/korety/Project/DR_2_Podcast/`
**Conda env:** `podcast_flow`

---

## Architecture

10-phase pipeline orchestrated across 3 CrewAI crews + a dual-model deep research pre-scan:

```
Topic Input (Web UI or CLI)
  └── Phase 0 — Research Framing (Research Framing Specialist)
  └── Deep Research Pre-Scan (Dual-Model Map-Reduce)
       Smart model plans queries → workers fetch → Fast model summarizes → Smart synthesizes
       Outputs: lead.md, counter.md, audit.md, deep_research_sources.json

CREW 1 — Evidence Gathering
  Phase 1: Systematic Evidence Gathering (Lead Researcher)
  Phase 2: Research Gate & Gap Analysis (Auditor) → PASS/FAIL
  Phase 2b: Gap-Fill Research [if FAIL]

CREW 2 — Evidence Validation
  Phase 3: Counter-Evidence Research (Adversarial Researcher)
  Phase 4a: Source Validation (Source Verifier)
  Phase 4b: Source-of-Truth Synthesis (Auditor)

CREW 3 — Podcast Production
  Phase 5: Show Notes & Citations (Producer)
  Phase 6: Script Generation (Producer)
  Phase 7: Script Polishing (Personality Editor)
  Phase 8: Accuracy Check (Auditor) [advisory, non-blocking]
  [Translation task inserted before 5/7 for non-English]

Audio Generation — Kokoro TTS, two voices, 24kHz WAV + BGM mixing
```

---

## Agents

| Agent | Role |
|-------|------|
| Research Framing Specialist | Defines scope, questions, evidence criteria |
| Principal Investigator (Lead Researcher) | Gathers supporting evidence by mechanism and clinical tier |
| Adversarial Researcher (The Skeptic) | Hunts contradictory evidence, null results, methodology flaws |
| Scientific Auditor (The Grader) | PASS/FAIL gate, source-of-truth synthesis, script accuracy check |
| Scientific Source Verifier | Validates cited URLs, checks claim-to-source accuracy |
| Podcast Producer (The Showrunner) | Transforms research into debate script |
| Podcast Personality (The Editor) | Polishes for natural verbal delivery |

---

## Model Endpoints

| Role | Model | URL |
|------|-------|-----|
| Smart (plan/synthesize) | `Qwen/Qwen2.5-14B-Instruct-AWQ` | `http://localhost:8000/v1` (vLLM) |
| Fast (summarize) | `llama3.2:1b` | `http://localhost:11434/v1` (Ollama) |
| TTS (English/Japanese) | Kokoro | CPU fallback, in-process |
| TTS (Japanese, alt) | Qwen3-TTS | `http://localhost:8082/tts` |
| Search | SearXNG | `http://localhost:8080` |

Configured via env vars: `MODEL_NAME`, `LLM_BASE_URL`, `FAST_MODEL_NAME`, `FAST_LLM_BASE_URL`. Fast model upgrade: `phi4-mini`.

---

## Pre-flight Checks

```bash
# vLLM running? (user setup)
curl -s http://localhost:8000/v1/models | python3 -m json.tool

# Ollama running? (default or fast model)
curl -s http://localhost:11434/api/tags

# SearXNG running?
curl -s http://localhost:8080/healthz

# Qwen3-TTS running? (optional alt TTS)
curl -s http://localhost:8082/tts -X POST -H "Content-Type: application/json" \
  -d '{"text":"test","speaker":"Kaz"}'
```

---

## Starting Services

```bash
# Ollama (default backend — pull models first if needed)
ollama serve
ollama pull deepseek-r1:32b   # smart model default
ollama pull llama3.2:1b        # fast model default

# vLLM Docker container (optional, higher quality)
bash start_vllm_docker.sh

# Qwen3-TTS server (conda env: qwen3_tts)
bash docker/qwen3-tts/run_server.sh
# First-time init:
bash docker/qwen3-tts/init_and_start.sh

# Web UI
bash start_podcast_web_ui.sh
# or: python podcast_web_ui.py --port 8501
```

---

## Key Files

| File / Directory | Purpose |
|------------------|---------|
| `podcast_crew.py` | Main pipeline — agents, tasks, 10-phase orchestration, research library tools |
| `podcast_web_ui.py` | FastAPI web UI — live progress, task queue, upload integration |
| `deep_research_agent.py` | Dual-model map-reduce engine with tiered academic search + PRISMA tracking |
| `search_agent.py` | SearXNG client, page scraping, content extraction |
| `research_planner.py` | Structured research planning with iterative gap-filling |
| `audio_engine.py` | Kokoro TTS with dual-voice stitching and BGM post-processing |
| `link_validator_tool.py` | URL validation via HEAD requests |
| `upload_utils.py` | Buzzsprout and YouTube upload utilities |
| `start_podcast_web_ui.sh` | Web UI launcher |
| `start_vllm_docker.sh` | vLLM Docker container launcher |
| `podcast_tasks.json` | Persistent task queue for the web UI |
| `docker/qwen3-tts/` | Qwen3-TTS FastAPI server (conda env: `qwen3_tts`) |
| `asset/` | Kokoro TTS model weights, voice files, tokenizer |
| `Podcast BGM/` | WAV background music library for post-processing |
| `research_outputs/` | Timestamped output directories per run |
| `shared/checkpoints/` | Pipeline state checkpoints |
| `reflections/` | Agent post-task lessons |

---

## Known Footguns

- Kokoro TTS requires Python <3.13 — always use `podcast_flow` env, not `base`
- Kokoro must use CPU fallback — GPU sm_121 is incompatible with PyTorch 2.5.1
- phi4-mini runs 100% CPU when vLLM holds the GPU — set concurrency=2, timeout=180
- Two vLLM containers → CUDA OOM — check `docker ps` before starting a new container
- Stop tokens `"Observation:"` / `"Thought:"` cause empty LLM responses with AWQ models — do not add them
- SearxngClient needs `async with` context manager; param is `num_results` not `max_results`
- `ResearchAspect.source_count` is read-only — use `add_source()` method
- Junk URL filtering required — block dictionary.com, cambridge.org, crossword sites
- Qwen3-TTS: class is `Qwen3TTSModel.from_pretrained()` (NOT `QwenTTS`)
- Qwen3-TTS: `generate_custom_voice()` returns `(List[np.ndarray], sample_rate)` — must `np.concatenate` the list

---

## Access Rules

- **`.env` file**: Always ask the user before reading or modifying `.env`
- All other files and commands: no restrictions
