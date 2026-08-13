"""Stage adapters: each phase as a function of ``(run_dir, run_config)``.

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

from pathlib import Path
from typing import Any

from dr2_podcast.artifacts import ArtifactError, write_atomic, write_json_atomic
from dr2_podcast.stages import register


def _prepare_run(run_dir: Path, run_config: dict[str, Any]) -> Any:
    """Rebuild the module state a Crew needs, from the run directory alone.

    Returns the :mod:`dr2_podcast.pipeline` module, whose globals the Crew builders read.
    """
    from dr2_podcast import pipeline

    pipeline.output_dir = run_dir
    pipeline.topic_name = run_config["topic"]
    pipeline.SESSION_ROLES = pipeline.assign_roles()
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


@register("framing")
def framing(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 0 — domain classification + the research framing crew.

    Differences from ``phase_0_framing``, both deliberate:

    * It writes both artifacts itself, atomically. The framing task carries an ``output_file`` so
      CrewAI writes ``research_framing.md`` directly, unatomically, mid-run; that is cleared here so
      the only write is the validated one.
    * An empty framing output raises. The phase logs "continuing" and returns ``""``, which sends a
      run into the search stage with no framework at all.
    """
    from crewai import Crew

    from dr2_podcast.pipeline_flow import _append_to_description_once

    pipeline = _prepare_run(run_dir, run_config)
    classification = _classify_domain(run_config["topic"])

    write_json_atomic(
        run_dir / "research/domain_classification.json",
        {
            "domain": classification.domain.value,
            "confidence": classification.confidence,
            "reasoning": classification.reasoning,
            "framework": classification.suggested_framework,
            "databases": classification.primary_databases,
        },
    )

    task = pipeline.framing_task
    _append_to_description_once(task, _domain_note(classification))
    task.output_file = None  # this module owns the write; see the docstring
    Crew(agents=[pipeline.framing_agent], tasks=[task], verbose=True, process="sequential").kickoff()

    output = task.output.raw if getattr(task, "output", None) else ""
    if not output.strip():
        raise ArtifactError(
            "the framing crew returned nothing. The monolithic phase logs this and continues, which "
            "sends the run into search with no framework; a stage that produced nothing has failed."
        )
    write_atomic(run_dir / "research/research_framing.md", output)


@register("url_validation")
def url_validation(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 2 — batch HEAD validation of every cited URL. Python only, no LLM.

    Reads ``research_sources.json`` from disk rather than taking the previous phase's return value,
    which is the whole point of the stage contract.
    """
    from dr2_podcast.artifacts import read_json_strict
    from dr2_podcast.tools.link_validator import validate_multiple_urls_parallel

    sources = read_json_strict(run_dir / "research/research_sources.json")
    urls = sorted({url for url in _iter_urls(sources) if url})
    results = validate_multiple_urls_parallel(urls, max_workers=15) if urls else {}
    write_json_atomic(run_dir / "research/url_validation_results.json", results)


def _iter_urls(node: Any) -> list[str]:
    """Every ``url`` value anywhere in the sources document, whatever shape it has."""
    found: list[str] = []
    if isinstance(node, dict):
        for key, value in node.items():
            if key == "url" and isinstance(value, str):
                found.append(value)
            else:
                found.extend(_iter_urls(value))
    elif isinstance(node, list):
        for value in node:
            found.extend(_iter_urls(value))
    return found


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


def registered() -> tuple[str, ...]:
    """Stage names this module registers. Imported for its side effects; this makes that visible."""
    from dr2_podcast.stage import ADAPTERS

    return tuple(sorted(ADAPTERS))


__all__ = ["framing", "registered", "url_validation"]
