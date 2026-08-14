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
    #: Entries may contain ``{language}``, resolved against the run config. The graph is static but
    #: some artifact names are not: the translated SOT is ``source_of_truth_ja.md`` for a Japanese
    #: run and does not exist at all for an English one. Without the placeholder, an English run that
    #: had previously been Japanese would still see the stale ``_ja`` file, treat ``translate`` as a
    #: required producer, and refuse to build a blueprint that never reads that file.
    #:
    #: Artifacts this stage reads WHEN THEY EXIST. They are hashed like any other input when
    #: present, so regenerating one makes this stage stale — but their absence is not a failure.
    #: Without this category a stage would either demand a file that may not exist, or read one the
    #: manifest never hashes, and the second is how a blueprint stays "current" after the
    #: translation it quoted was regenerated.
    optional_consumes: tuple[str, ...] = field(default_factory=tuple)

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
        # session_roles.json fixes the presenter and questioner for every later Crew. It is an OUTPUT of
        # framing, not loose metadata: unhashed, editing or deleting it left framing "current" while
        # the roles the rest of the run builds prompts from had changed underneath.
        produces=(
            "research/research_framing.md",
            "research/domain_classification.json",
            "meta/session_roles.json",
        ),
        engine="smart",
    ),
    Stage("keywords", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("search", (), (), "python", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("screen", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("extract", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("synthesize", (), (), "smart", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage("grade", (), (), "claude", available=False, unavailable_reason=_NOT_YET_SPLIT),
    Stage(
        name="plan_search",
        # The same inputs framing hands the search: what question, and in which domain.
        consumes=("research/research_framing.md", "research/domain_classification.json"),
        # It writes the strategies and STOPS. Nothing downstream runs, which is the whole point:
        # the post-search yield gate catches a strategy that is wrong in QUANTITY, and cannot catch
        # one that searches the wrong population or whose falsification track is not adversarial.
        produces=("research/search_strategy_aff.json", "research/search_strategy_neg.json"),
        engine="smart",
    ),
    Stage(
        name="research",
        # domain_classification.json is a real input, not metadata: the phase passes
        # p0_result["domain"] into phase_1_research, where it selects the research domain and
        # framework. Unhashed, research could run against a changed classification and still be
        # recorded as current.
        consumes=(
            "research/research_framing.md",
            "research/domain_classification.json",
            # Step 10: this stage IS the `search` half of the split. It consumes the strategies
            # rather than making them, so it cannot search against a plan nobody read, and it
            # consumes the approval so a strategy — or a framing — edited after approval fails
            # closed. dr2_podcast/approval.py holds the bundle rule.
            "research/search_strategy_aff.json",
            "research/search_strategy_neg.json",
            "meta/strategy_approval.json",
        ),
        produces=(
            "research/affirmative_case.md",
            "research/falsification_case.md",
            "research/grade_synthesis.md",
            "research/clinical_math.md",
            "research/research_sources.json",
            "research/screening_results_aff.json",
            "research/screening_results_neg.json",
            # The SOT is built HERE because phase 1 builds it here, on the live reports dict. See
            # the `sot` stage below for why it cannot be a stage of its own.
            "research/source_of_truth.md",
        ),
        # Structured GRADE (sequencing item 3). Optional because the social-science domain has an
        # evidence-quality ladder rather than GRADE's modifier arithmetic and produces no record —
        # declaring it required would fail every social-science run on an artifact that correctly
        # does not exist. drop_unproduced_optional_outputs keeps a clinical run's record from
        # surviving into a later social-science one.
        # step_pack.json is optional for the same reason grade_synthesis.json is: it projects the
        # GRADE verdict, which the social-science domain does not produce in that shape.
        optional_outputs=("research/grade_synthesis.json", "research/step_pack.json"),
        engine="smart",
    ),
    Stage(
        "sot",
        (),
        (),
        "python",
        available=False,
        unavailable_reason=(
            "not separable: build_imrad_sot runs inside phase 1 on the live deep_reports dict, and "
            "that dict cannot cross a process boundary — _serialize_dataclass REPR-STRINGIFIES the "
            "report objects, so 'audit' round-trips as the literal text \"namespace(report='…')\" "
            "and no rehydration can recover it. An adapter was written against a reconstructed "
            "artifact and withdrawn when a test with the real builder proved the artifact cannot "
            "exist. The 'research' stage produces research/source_of_truth.md."
        ),
    ),
    Stage(
        name="url_validation",
        consumes=("research/research_sources.json",),
        # The filtered library is a NEW artifact, not an edit of research's output. A stage that
        # rewrites another stage's output would make the producer stale on every run.
        produces=(
            "research/url_validation_results.json",
            "research/research_sources_validated.json",
            # The hash of the library the filtered copy was derived from, so consumers can CHECK
            # that rather than trusting file timestamps.
            "research/research_sources_validated.sha256",
        ),
        engine="python",
    ),
    Stage(
        name="translate",
        consumes=("research/source_of_truth.md",),
        produces=(),
        optional_outputs=("research/source_of_truth_{language}.md",),
        engine="smart",
    ),
    Stage(
        name="blueprint",
        consumes=(
            "research/source_of_truth.md",
            "research/domain_classification.json",
            "research/grade_synthesis.md",
            "meta/session_roles.json",
            # An ORDERING GATE, not a read: producer_agent has no tools, so the blueprint never
            # opens the library itself. Declaring it here is what stops an episode being designed
            # before its citations were checked at all — the monolithic flow runs phase 2 before
            # phase 4 for the same reason — and the cost is that re-validating re-runs the
            # blueprint, which is the safe direction.
            "research/research_sources_validated.json",
        ),
        optional_consumes=("research/source_of_truth_{language}.md",),
        # blueprint_inventory.json is what phases 5 and 6 take as the bp_inventory argument. In the
        # monolithic flow it is a return value; across a process boundary it has to be a file.
        produces=("research/EPISODE_BLUEPRINT.md", "meta/blueprint_inventory.json"),
        engine="claude",
    ),
    Stage(
        name="draft",
        # The blueprint TEXT is not read — the sectional draft is built from the parsed inventory
        # and the SOT. Declaring the text as well would make a blueprint reword that leaves the
        # inventory identical rerun the draft for nothing.
        consumes=(
            "meta/blueprint_inventory.json",
            "research/source_of_truth.md",
            "meta/session_roles.json",
        ),
        produces=("scripts/script_draft.md",),
        engine="smart",
    ),
    Stage(
        name="polish",
        consumes=(
            "scripts/script_draft.md",
            "meta/blueprint_inventory.json",
            "research/source_of_truth.md",
            "meta/session_roles.json",
        ),
        # Loaded into translation_task and passed into the polish loop, so a regenerated or edited
        # translation has to make the polish stale — it was polished against that evidence.
        optional_consumes=("research/source_of_truth_{language}.md",),
        produces=("scripts/script_polished.md",),
        engine="smart",
    ),
    Stage(
        name="audit",
        consumes=(
            "scripts/script_polished.md",
            "research/source_of_truth.md",
            "meta/session_roles.json",
            # The auditor agent — and ONLY the auditor; producer_agent carries no tools
            # (pipeline_crew.py:328 vs :356) — reads the source library through
            # pipeline.research_sources_file(), which consults all three of these to decide WHICH
            # library it serves. Hashing only the validated copy would let a change to the raw
            # library or the stamp leave this stage current while it would now read something else.
            "research/research_sources.json",
            "research/research_sources_validated.json",
            "research/research_sources_validated.sha256",
        ),
        # The translated SOT joins the audit task's context for a non-English episode, so a
        # regenerated translation has to make the audit stale.
        optional_consumes=("research/source_of_truth_{language}.md",),
        produces=("research/accuracy_audit.md", "scripts/script_final.md"),
        # Written only when the accuracy gate fires, which is most runs' quiet path.
        optional_outputs=("research/ACCURACY_CORRECTIONS.md",),
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


def resolve(artifacts: tuple[str, ...], substitutions: dict[str, str] | None = None) -> tuple[str, ...]:
    """Fill ``{language}``-style placeholders from the run's own configuration."""
    if not substitutions:
        return tuple(a for a in artifacts if "{" not in a)
    return tuple(a.format(**substitutions) for a in artifacts)


#: Inputs a PERSON writes, not a stage. The strategy approval is the only one: Step 10's whole point
#: is that a reviewer stands between the plan and the search, so an approval a stage could produce
#: would be an approval the pipeline grants itself.
REVIEWER_WRITTEN: frozenset[str] = frozenset({"meta/strategy_approval.json"})


def _pattern_matches(artifact: str, pattern: str) -> bool:
    """Whether an artifact name satisfies a possibly-placeholdered pattern.

    Compared on the literal prefix and suffix around the placeholder, so
    ``research/source_of_truth_{language}.md`` matches ``…_ja.md`` and ``…_en.md`` without the graph
    having to know which language a given run is.
    """
    if "{" not in pattern:
        return artifact == pattern
    head, _, tail = pattern.partition("{")
    return artifact.startswith(head) and artifact.endswith(tail.partition("}")[2])


def get_stage(name: str) -> Stage:
    """Look a stage up by name, with a message that lists the alternatives."""
    try:
        return STAGES_BY_NAME[name]
    except KeyError:
        raise KeyError(f"unknown stage {name!r}; known: {', '.join(STAGE_NAMES)}") from None


def producer_of(artifact: str) -> str | None:
    """Which stage writes this artifact, or None if nothing declares it.

    Pattern-aware, because an optional output may be declared as
    ``research/source_of_truth_{language}.md``. Exact matching here would silently answer None for
    the very artifact the graph exists to attribute.
    """
    for stage in STAGES:
        if artifact in stage.produces:
            return stage.name
        if any(_pattern_matches(artifact, pattern) for pattern in stage.optional_outputs):
            return stage.name
    return None


def _reads(stage: Stage) -> tuple[tuple[str, ...], tuple[str, ...]]:
    return stage.consumes, stage.optional_consumes


def direct_producers(name: str) -> tuple[str, ...]:
    """Stages that write anything the named stage consumes, optional inputs included."""
    required, optional = _reads(get_stage(name))
    return tuple(
        other.name
        for other in STAGES
        if (set(other.produces) | set(other.optional_outputs))
        & set(required)
        or any(
            _pattern_matches(written, pattern)
            for written in set(other.produces) | set(other.optional_outputs)
            for pattern in optional
        )
    )


def direct_consumers(name: str) -> tuple[str, ...]:
    """Stages that consume anything the named stage produces."""
    stage = get_stage(name)
    written = set(stage.produces) | set(stage.optional_outputs)
    return tuple(
        other.name
        for other in STAGES
        if (written & set(other.consumes))
        or any(_pattern_matches(w, pattern) for w in written for pattern in other.optional_consumes)
    )


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
