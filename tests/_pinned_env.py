"""Pin the env vars that dr2_podcast.config reads AT IMPORT TIME.

Must be imported before anything under dr2_podcast. `config.py` calls load_dotenv()
and binds SMART_MODEL/SMART_BASE_URL at module level, and load_dotenv() does not
override variables already present — so setting them here makes the suite read the
same values whether or not a developer `.env` exists.

Why it matters (2026-08-12): without this the golden fixture embeds whatever model
the developer's .env names. Locally 827 tests passed; CI, which has no .env, failed
23 — 13 in test_sot_golden and 10 in test_clinical_orchestrator_run — and had been
failing that way since before this branch. The autouse `mock_env_vars` fixture cannot
fix it: fixtures run after test modules are imported, and by then config has bound.

Importers use TEST_MODEL_NAME so the pin is a real dependency rather than an
import-for-side-effect that needs a suppression to survive lint.
"""

import os

TEST_MODEL_NAME = "test-model"
TEST_BASE_URL = "http://localhost:9999/v1"

# Assigned, NOT setdefault. setdefault preserves a value already exported by the
# shell or by CI, which leaves the suite environment-dependent in exactly the way this
# module exists to prevent — a developer with `export MODEL_NAME=...` would still get a
# golden mismatch. The tests must see the same values on every machine, so the pin wins
# over the ambient environment. load_dotenv() then declines to override these.
os.environ["MODEL_NAME"] = TEST_MODEL_NAME
os.environ["LLM_BASE_URL"] = TEST_BASE_URL
os.environ["LLM_API_KEY"] = "NA"
