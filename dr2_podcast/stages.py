"""The stage graph: what each stage of a run consumes and produces.

PLAN.md Step 1: Claude cannot orchestrate a single 87-minute ``python -m dr2_podcast.pipeline``
subprocess, so the run is split into resumable stages whose contract is files on disk. This module
is that contract, and nothing else — no execution, no I/O. It is what makes staleness derivable:
a stage is current only while every artifact it recorded as an input still hashes to what it
recorded, and the graph is what says which stages that answer propagates to.

**Not every stage in the plan's list is separable yet, and this module says which.** The plan names
fourteen: framing, keywords, search, screen, extract, synthesize, grade, sot, translate, blueprint,
draft, polish, audit, audio. Six of them — keywords through grade — live inside a single call today:
``_run_research_track`` does plan → search → screen → extract in one pass, both tracks fire together
under ``asyncio.gather`` (``clinical.py:2952``), and the strategy JSON is not written until
``_save_artifacts`` after both tracks AND GRADE have finished (``clinical.py:3366``). Splitting them
is Step 10's work, sequenced at item 5. Until then those six are declared with ``available=False``
and the transitional ``research`` stage covers them as one unit. Declaring them now, unavailable,
rather than omitting them keeps the target shape visible and makes the CLI's refusal specific.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Where the legacy monolithic runner and the staged runner each record themselves. Separate files
#: so the two modes cannot collide on filenames while both are live (PLAN.md sequencing item 1).
MANIFEST_FILENAMES = {"staged": "meta/manifest.json", "legacy": "meta/manifest_legacy.json"}


#: stage name -> callable(run_dir, run_config) -> None. A stage writes its own artifacts; the runner
#: hashes and records them afterwards from the graph's declaration.
#:
#: It lives HERE rather than in the runner for a reason that is not stylistic: `python -m
#: dr2_podcast.stage` executes that file as `__main__`, and an adapter module importing
#: `dr2_podcast.stage` gets a SECOND module object with its own registry — so registrations landed
#: somewhere the running process could not see. A registry in this module is one dict either way.
ADAPTERS: dict[str, Callable[[Path, dict[str, Any]], None]] = {}


def register(name: str) -> Callable[[Callable[[Path, dict[str, Any]], None]], Callable[..., None]]:
    """Decorator registering a stage adapter against a declared stage."""

    def _wrap(func: Callable[[Path, dict[str, Any]], None]) -> Callable[..., None]:
        get_stage(name)
        ADAPTERS[name] = func
        return func

    return _wrap


@dataclass(frozen=True)
class Stage:
    """One re-runnable unit of a run. Paths are relative to the run directory."""

    name: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    engine: str
    available: bool = True
    unavailable_reason: str = ""
    #: Artifacts this stage may or may not write, depending on the run (e.g. a translated SOT only
    #: exists for a non-English episode). Absent optional outputs are not a failure.
    optional_outputs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.available and not self.unavailable_reason:
            raise ValueError(f"stage {self.name!r} is unavailable but says nothing about why")


_NOT_YET_SPLIT = (
    "inside phase 1 today: _run_research_track does plan/search/screen/extract in one call and the "
    "strategy JSON is not written until after both tracks and GRADE finish (clinical.py:3366). "
    "Separating it is PLAN.md Step 10, sequencing item 5. Use the transitional 'research' stage."
)

STAGES: tuple[Stage, ...] = (
    Stage(
        name="framing",
        consumes=(),
        produces=("research/research_framing.md", "research/domain_classification.json"),
        engine="smart",
    ),
    Stage("keywords", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("search", (), (), "python", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("screen", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("extract", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("synthesize", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("grade", (), (), "claude", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage(
        name="research",
        # domain_classification.json is a real input, not metadata: the phase passes
        # p0_result["domain"] into phase_1_research, where it selects the research domain and
        # framework. Unhashed, research could run against a changed classification and still be
        # recorded as current.
        consumes=("research/research_framing.md", "research/domain_classification.json"),
        produces=(
            "research/affirmative_case.md",
            "research/falsification_case.md",
            "research/grade_synthesis.md",
            "research/clinical_math.md",
            "research/research_sources.json",
            "research/search_strategy_aff.json",
            "research/search_strategy_neg.json",
            "research/screening_results_aff.json",
            "research/screening_results_neg.json",
            # The structured reports build_imrad_sot consumes. In the monolithic flow this crosses
            # from phase 1 to the SOT builder as a live dict; a staged run needs it on disk, and the
            # SOT cannot be rebuilt from the rendered Markdown.
            "meta/deep_reports.json",
        ),
        engine="smart",
    ),
    Stage(
        name="sot",
        consumes=("meta/deep_reports.json", "research/domain_classification.json"),
        produces=("research/source_of_truth.md",),
        engine="python",
    ),
    Stage(
        name="url_validation",
        consumes=("research/research_sources.json",),
        produces=("research/url_validation_results.json",),
        engine="python",
    ),
    Stage(
        name="translate",
        consumes=("research/source_of_truth.md",),
        produces=(),
        optional_outputs=("research/source_of_truth_ja.md",),
        engine="smart",
    ),
    Stage(
        name="blueprint",
        consumes=("research/source_of_truth.md",),
        produces=("research/EPISODE_BLUEPRINT.md",),
        engine="claude",
    ),
    Stage(
        name="draft",
        consumes=("research/EPISODE_BLUEPRINT.md", "research/source_of_truth.md"),
        produces=("scripts/script_draft.md",),
        engine="smart",
    ),
    Stage(
        name="polish",
        consumes=("scripts/script_draft.md",),
        produces=("scripts/script_polished.md",),
        engine="smart",
    ),
    Stage(
        name="audit",
        consumes=("scripts/script_polished.md", "research/source_of_truth.md"),
        produces=("research/accuracy_audit.md", "scripts/script_final.md"),
        engine="codex",
    ),
    Stage(
        name="audio",
        consumes=("scripts/script_final.md",),
        produces=("scripts/script.txt", "audio/audio.wav"),
        optional_outputs=("audio/audio_mixed.wav",),
        engine="python",
    ),
)

STAGES_BY_NAME: dict[str, Stage] = {stage.name: stage for stage in STAGES}
STAGE_NAMES: tuple[str, ...] = tuple(stage.name for stage in STAGES)
AVAILABLE_STAGE_NAMES: tuple[str, ...] = tuple(s.name for s in STAGES if s.available)


def get_stage(name: str) -> Stage:
    """Look a stage up by name, with a message that lists the alternatives."""
    try:
        return STAGES_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; known: {', '.join(STAGE_NAMES)}") from None


def producer_of(artifact: str) -> str | None:
    """Which stage writes this artifact, or None if nothing declares it."""
    for stage in STAGES:
        if artifact in stage.produces or artifact in stage.optional_outputs:
            return stage.name
    return None


def direct_producers(name: str) -> tuple[str, ...]:
    """Stages that write anything the named stage consumes."""
    consumed = set(get_stage(name).consumes)
    return tuple(
        other.name for other in STAGES if consumed & (set(other.produces) | set(other.optional_outputs))
    )


def direct_consumers(name: str) -> tuple[str, ...]:
    """Stages that consume anything the named stage produces."""
    stage = get_stage(name)
    written = set(stage.produces) | set(stage.optional_outputs)
    return tuple(other.name for other in STAGES if written & set(other.consumes))


def downstream_of(name: str) -> tuple[str, ...]:
    """Every stage reachable from this one, in declaration order.

    Declaration order is the run order, so a caller marking downstream stages stale walks them in
    the order they would have run.
    """
    reached: set[str] = set()
    frontier = [name]
    while frontier:
        for consumer in direct_consumers(frontier.pop()):
            if consumer not in reached:
                reached.add(consumer)
                frontier.append(consumer)
    return tuple(stage.name for stage in STAGES if stage.name in reached)
