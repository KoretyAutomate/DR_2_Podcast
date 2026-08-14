"""Shared fixtures for the stage runner and CLI tests.

Split out so each module stays under the repo's file-size ceiling.
"""

from __future__ import annotations

import json
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
    "meta/session_roles.json": '{"hosts_setting": "", "roles": {"presenter": "Host 1"}}',
}




#: A minimal strategy, in the shape _restore_tiered_search_plan rebuilds.
STRATEGY = {
    "pico": {"population": "adults", "intervention": "vitamin D", "comparator": "placebo", "outcome": "fracture"},
    "tier1": {"intervention": ["vitamin D"], "outcome": ["fracture"], "population": ["adults"], "rationale": "r"},
    "tier2": {"intervention": ["cholecalciferol"], "outcome": ["hip fracture"], "population": [], "rationale": "r"},
    "tier3": {"intervention": ["secosteroid"], "outcome": [], "population": [], "rationale": "r"},
    "role": "affirmative",
    "auditor_approved": True,
}


def run_plan_search(run_dir: Path) -> None:
    """Run the plan_search STAGE (stubbed) and approve what it wrote.

    Writing the strategy files directly is not enough for anything that goes through the runner:
    the artifacts would exist while their producer had no manifest record, so `research` would
    refuse them as not current — correctly, since nothing recorded who made them.
    """
    from dr2_podcast.stage import run_stage
    from dr2_podcast.stages import get_stage

    _stub("plan_search", {a: json.dumps(STRATEGY) for a in get_stage("plan_search").produces})
    run_stage(run_dir, "plan_search")
    approve(run_dir)


def approve(run_dir: Path) -> None:
    """Record an approval over the artifacts exactly as they stand."""
    from dr2_podcast.approval import write_approval

    write_approval(run_dir, approver="test", approved_at="2026-08-13T00:00:00+09:00")


def plan_and_approve(run_dir: Path) -> None:
    """What Step 10 puts between framing and search: two strategies, and an approval over them.

    Every test that runs or stubs the `research` stage needs this now, because the stage consumes
    both strategy files and the approval — which is the point of the split, and the reason a fixture
    that only wrote a framing no longer describes a runnable run.
    """
    import json

    from dr2_podcast.approval import write_approval

    (run_dir / "research").mkdir(exist_ok=True)
    (run_dir / "meta").mkdir(exist_ok=True)
    (run_dir / "research/search_strategy_aff.json").write_text(json.dumps(STRATEGY))
    (run_dir / "research/search_strategy_neg.json").write_text(
        json.dumps({**STRATEGY, "role": "adversarial"})
    )
    write_approval(run_dir, approver="test", approved_at="2026-08-13T00:00:00+09:00")
