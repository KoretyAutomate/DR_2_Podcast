"""Shared fixtures for the stage runner and CLI tests.

Split out so each module stays under the repo's file-size ceiling.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import stage as stage_mod
from dr2_podcast.stage import write_run_config

@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    write_run_config(tmp_path, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    return tmp_path


@pytest.fixture(autouse=True)
def _clean_adapters():
    """Adapters are module-global; no test may leak one into another.

    The real ones are imported FIRST. `load_adapters()` runs inside `_resolve`, so without this the
    first `run_stage` of a test module would import the adapter module and register the real
    adapters over any stub set before it — silently swapping a stub for a live LLM call.
    """
    stage_mod.load_adapters()
    original = dict(stage_mod.ADAPTERS)
    yield
    stage_mod.ADAPTERS.clear()
    stage_mod.ADAPTERS.update(original)


def _stub(name: str, writes: dict[str, str]) -> list[str]:
    """Register an adapter that writes fixed contents, and record when it ran."""
    calls: list[str] = []

    def _adapter(run_dir: Path, run_config: dict[str, Any]) -> None:
        calls.append(run_config["topic"])
        for artifact, text in writes.items():
            path = run_dir / artifact
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")

    stage_mod.ADAPTERS[name] = _adapter
    return calls


FRAMING_OUTPUTS = {
    "research/research_framing.md": "# framing\n",
    "research/domain_classification.json": '{"domain": "clinical"}',
    # Declared as a framing output: it fixes the presenter and questioner for every later Crew, so
    # editing or deleting it has to make framing non-current.
    "meta/session_roles.json": '{"presenter": "Host 1", "questioner": "Host 2"}',
}


