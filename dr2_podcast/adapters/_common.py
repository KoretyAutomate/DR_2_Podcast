"""Shared state reconstruction for the stage adapters.

Split out of ``__init__`` so the package can import its submodules at the top of the file: they
import these helpers, so keeping the helpers in ``__init__`` made the re-export order load-bearing
and cost an E402 suppression. A module nobody has to think about is worth more than the comment
explaining why the order mattered.

Originally:

The shared state reconstruction lives here; the adapters themselves are in ``research_stages``
(phases 0-3) and ``script_stages`` (phases 4-8), split only to keep each file under the repo's
size ceiling. Import everything from ``dr2_podcast.adapters``.

PLAN.md Step 1's remaining half. The phases were not callable across a process boundary — every one
took live Python objects from the phase before it, and the sixty lines of run initialisation that
build them lived inside ``pipeline.py``'s ``if __name__ == "__main__"`` block, reachable only by
being that script. An adapter's whole job is to reconstruct that state from the run directory and
then do what the phase does.

The state is reconstructed by calling ``pipeline.initialise_run_globals``, which was extracted from
``__main__`` for exactly this purpose rather than reimplemented here. Two runners producing
different episodes from the same inputs is the failure that would make a staged pipeline
untrustworthy, and duplicated initialisation is how that happens.

**Adapters fail closed.** The phases they replace do not, in places: ``phase_0_framing`` catches
every exception from the framing crew, logs "continuing", and returns an empty string, so a run
whose framing never happened proceeds to search for nothing in particular. A stage that produced
nothing is a failed stage.
"""

from __future__ import annotations

import logging
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import write_json_atomic

logger = logging.getLogger(__name__)



SESSION_ROLES_ARTIFACT = "meta/session_roles.json"


def _session_roles(run_dir: Path, *, reassign: bool = False) -> dict[str, Any]:
    """The run's host roles, assigned once and then read back.

    ``assign_roles()`` is RANDOM under the default ``PODCAST_HOSTS=random``, and every stage is a
    fresh process. Calling it per stage would reshuffle presenter and questioner between framing,
    blueprint and the script phases — an episode whose own roles change between its parts, with no
    manifest identity change to show for it, because the randomness is not in any input. The
    monolithic runner calls it exactly once per run; this makes "once per run" survive the process
    boundary.

    ``reassign`` is for the stage that DECLARES this artifact as an output — framing. Without it, a
    changed ``PODCAST_HOSTS`` makes framing stale, framing re-runs, and the old assignment is read
    straight back while the manifest records the stage as current under the new setting.
    """
    from dr2_podcast.artifacts import read_json_strict

    from dr2_podcast import pipeline

    path = run_dir / SESSION_ROLES_ARTIFACT
    if path.exists() and not reassign:
        return read_json_strict(path)
    roles = pipeline.assign_roles()
    write_json_atomic(path, roles)
    return roles


def _prepare_run(run_dir: Path, run_config: dict[str, Any], *, reassign_roles: bool = False) -> Any:
    """Rebuild the module state a Crew needs, from the run directory alone.

    Returns the :mod:`dr2_podcast.pipeline` module, whose globals the Crew builders read.
    """
    from dr2_podcast import pipeline

    pipeline.output_dir = run_dir
    pipeline.topic_name = run_config["topic"]
    pipeline.SESSION_ROLES = _session_roles(run_dir, reassign=reassign_roles)
    pipeline.initialise_run_globals(
        language_code=run_config["language"],
        target_minutes=run_config["target_length_minutes"],
    )
    pipeline._create_agents_and_tasks()
    return pipeline


def _classify_domain(topic: str) -> Any:
    """Domain classification, as phase 0 does it, but raising rather than degrading."""
    import asyncio

    from openai import AsyncOpenAI

    from dr2_podcast import config
    from dr2_podcast.research.domain_classifier import classify_topic

    client = AsyncOpenAI(base_url=config.SMART_BASE_URL, api_key="not-needed")
    return asyncio.run(classify_topic(topic=topic, smart_client=client, smart_model=config.SMART_MODEL))


def _domain_note(classification: Any) -> str:
    """The framing directive phase 0 appends, verbatim in effect."""
    from dr2_podcast.research.domain_classifier import ResearchDomain

    databases = ", ".join(classification.primary_databases)
    if classification.domain == ResearchDomain.SOCIAL_SCIENCE:
        return (
            "\n\nDOMAIN CONTEXT: This is a SOCIAL SCIENCE topic. "
            "Use PECO framework (Population, Exposure, Comparison, Outcome). "
            "Prioritise effect sizes (Cohen's d, Hedges' g), quasi-experimental designs, "
            f"and databases such as {databases}. "
            "Do NOT use clinical terminology (NNT, ARR, GRADE, MeSH terms)."
        )
    return (
        "\n\nDOMAIN CONTEXT: This is a CLINICAL/HEALTH topic. "
        "Use PICO framework (Population, Intervention, Comparison, Outcome). "
        "Prioritise RCTs, systematic reviews, GRADE evidence levels, NNT/ARR statistics, "
        f"and databases such as {databases}."
    )


def _prime_translation_task(pipeline: Any, run_dir: Path, language: str) -> None:
    """Give the freshly built translation task the output the monolithic run would have left on it.

    In the flow the translation task has already run by the time the script phases use it as
    context. A fresh process rebuilds it empty, so a Japanese episode would be polished against no
    translated evidence at all — and CrewAI context resolution can fail on an output-less task.

    What goes on the task is a COMPACT MARKER, not the translated text, exactly as the monolithic
    path does it (``pipeline.py:2422``). The comment there records why: the full SOT in a context
    task overflows the model window and sends CrewAI into an infinite summariser loop — "observed:
    36 cycles, 9.6h wasted". Both blueprint and polish put this task in their context, and the
    blueprint's own degradation guard cannot help, because degrading the injected summary never
    shrinks a task's output.
    """
    task = getattr(pipeline, "translation_task", None)
    if task is None or language == "en":
        return
    translated = run_dir / f"research/source_of_truth_{language}.md"
    if not translated.exists():
        return
    size = len(translated.read_text(encoding="utf-8"))
    task.output = _DraftOutput(
        f"[Translation complete — {size:,} chars]\n"
        f"Translated SOT saved: {translated}\n"
        f"Read that file for the translated evidence."
    )


def _script_context(pipeline: Any, run_config: dict[str, Any], sot: str) -> Any:
    """The ScriptRunContext both script phases build, from module state plus the run config.

    One helper rather than two copies: phases 5 and 6 assemble the same nine values, and the two
    drifting apart is how a draft and its polish end up written to different targets.
    """
    return pipeline.ScriptRunContext(
        language_config=pipeline.language_config,
        session_roles=pipeline.SESSION_ROLES,
        topic_name=run_config["topic"],
        target_instruction=pipeline.target_instruction,
        target_length_int=pipeline.target_length_int,
        sot_content=sot,
        channel_intro=pipeline.channel_intro,
        target_min=pipeline._target_min,
    )


@contextmanager
def staging_dir(run_dir: Path) -> Iterator[Path]:
    """A scratch tree inside the run for a helper that writes to final paths.

    Some of the pipeline's write helpers take an output directory and write straight into it —
    ``_run_audio_pipeline`` and ``_finalize_script`` both do, with a bare ``open(..., "w")``. An
    interruption partway leaves a truncated file where a complete one used to be, which the next
    stage reads as finished. Giving them a staging tree and promoting by rename restores the
    all-or-nothing the artifact contract promises.

    The subdirectories are created up front because ``output_path`` falls back to a FLAT path when
    its target subdirectory is absent (``pipeline.py:322``, for legacy runs) — an empty staging tree
    would scatter everything into its root and be promoted to the wrong places.
    """
    staging = run_dir / "meta/.stage_staging"
    shutil.rmtree(staging, ignore_errors=True)
    for subdir in ("scripts", "audio", "research", "meta"):
        (staging / subdir).mkdir(parents=True)
    try:
        yield staging
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def promote(staging: Path, run_dir: Path) -> list[str]:
    """Move everything a staged helper produced into the run, and return what was promoted."""
    promoted: list[str] = []
    for produced in sorted(path for path in staging.rglob("*") if path.is_file()):
        relative = produced.relative_to(staging)
        target = run_dir / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        os.replace(produced, target)
        promoted.append(str(relative))
    return promoted


def drop_unproduced_optional_outputs(
    run_dir: Path, stage_name: str, produced: list[str], substitutions: dict[str, str] | None = None
) -> list[str]:
    """Delete declared optional outputs this execution did not produce.

    Otherwise a previous run's artifact survives and ``Manifest.complete()`` records it as THIS
    execution's output: a rerender whose BGM pass fails leaves the old ``audio_mixed.wav`` beside
    the new raw audio, and both look current. An optional output means "this run may not produce
    one", not "keep whatever was there".
    """
    from dr2_podcast.stages import get_stage, resolve

    removed: list[str] = []
    for artifact in resolve(get_stage(stage_name).optional_outputs, substitutions):
        if artifact in produced:
            continue
        path = run_dir / artifact
        if path.exists():
            path.unlink()
            removed.append(artifact)
    return removed


class _DraftOutput:
    """What CrewAI leaves on a task once it has run, and what the loops read back off one."""

    def __init__(self, raw: str) -> None:
        self.raw = raw


# NOT HERE: an adapter for the `sot` stage. It was written, and then removed, because writing it
# proved its input artifact cannot exist in the form assumed.
#
# `build_imrad_sot` reads `reports["audit"].report` (pipeline_sot.py:809) and then walks the
# extractions and impacts as objects. The obvious way to persist that across a process boundary is
# `_serialize_deep_reports`, which delegates to `_serialize_dataclass` — and that function
# **repr-stringifies** the report objects: `audit` comes back as the literal text
# "namespace(report='### Overall Certainty…')". The structure is not merely flattened, it is gone,
# so no rehydration can recover it and the round trip cannot be made to work from that artifact.
#
# The `research` adapter therefore has to define and write a purpose-built artifact — the report
# bodies as explicit fields — rather than reuse the existing serialiser. Guessing that artifact's
# shape before its producer exists is what a test with the REAL builder caught here, and shipping
# an adapter that fails on its own intended input would have been worse than shipping none.
