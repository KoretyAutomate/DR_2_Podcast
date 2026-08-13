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

    # framing is the stage that declares meta/session_roles.json as an output, so it is the one that
    # writes it — otherwise a changed PODCAST_HOSTS would rerun framing and keep the old assignment.
    pipeline = _prepare_run(run_dir, run_config, reassign_roles=True)
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

    It writes the filtered library to a SEPARATE artifact rather than editing ``research_sources.json``
    the way ``phase_2_url_validation`` does. Under a manifest that is not a style preference: a stage
    that rewrites another stage's output makes the producer permanently stale — ``research`` would
    record a hash that ``url_validation`` immediately invalidates, on every single run. Downstream
    stages consume ``research_sources_validated.json``; the raw library stays as ``research`` left it.
    """
    from dr2_podcast.artifacts import read_json_strict
    from dr2_podcast.tools.link_validator import validate_multiple_urls_parallel

    sources = read_json_strict(run_dir / "research/research_sources.json")
    urls = sorted({url for url in _iter_urls(sources) if url})
    results = validate_multiple_urls_parallel(urls, max_workers=15) if urls else {}
    write_json_atomic(run_dir / "research/url_validation_results.json", results)
    write_json_atomic(
        run_dir / "research/research_sources_validated.json", _without_broken(sources, results)
    )


def _without_broken(sources: Any, results: dict[str, str]) -> Any:
    """The sources library with every URL the validator rejected removed.

    Broken, Invalid, or an ERROR status. The phase (``pipeline_flow.py:450``) tests
    ``status.startswith("ERROR")``, which **misses the single-URL path**: ``LinkValidatorTool._run``
    returns ``"✗ ERROR: …"`` with a leading marker (``link_validator.py:66``), while only the batch
    path returns a bare ``"ERROR: …"`` (``link_validator.py:110``). A substring test covers both,
    and matches how the same predicate already treats Broken and Invalid.
    """
    rejected = ("Broken", "Invalid", "ERROR")
    broken = {url for url, status in results.items() if any(bad in str(status) for bad in rejected)}
    if not broken or not isinstance(sources, dict):
        return sources
    filtered = dict(sources)
    for role, entries in sources.items():
        if isinstance(entries, list):
            filtered[role] = [e for e in entries if not (isinstance(e, dict) and e.get("url") in broken)]
    return filtered


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


@register("translate")
def translate(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 3 — translate the source of truth, for a non-English episode.

    Calls ``_translate_sot_pipelined`` directly rather than ``_translate_and_inject_sot``: the
    latter also injects the summary into Crew 3 task descriptions, which is meaningless across a
    process boundary because every stage rebuilds its own tasks. What survives the boundary is the
    file.

    An English run writes nothing and completes; the output is optional for exactly that reason. A
    translation that comes back empty RAISES, though — the phase returns None and carries on, which
    leaves a Japanese episode built from an English source of truth.
    """
    from dr2_podcast.artifacts import read_text_strict

    language = run_config["language"]
    if language == "en":
        return

    pipeline = _prepare_run(run_dir, run_config)
    sot = read_text_strict(run_dir / "research/source_of_truth.md")
    translated = pipeline._translate_sot_pipelined(sot, language, pipeline.language_config)
    if not translated or not translated.strip():
        raise ArtifactError(
            f"translation to {language!r} produced nothing. The monolithic phase returns None and "
            f"continues, which builds the episode from a source of truth in the wrong language."
        )
    write_atomic(run_dir / f"research/source_of_truth_{language}.md", translated)


@register("blueprint")
def blueprint(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 4 — the episode blueprint, via the producer agent.

    Two things it persists that the phase only returned in memory:

    * ``meta/blueprint_inventory.json`` — ``_parse_blueprint_inventory``'s output, which phases 5
      and 6 take as the ``bp_inventory`` argument. Across a process boundary it has to be a file.
    * The SOT summaries are RECOMPUTED here rather than threaded through, because the phase receives
      them from phases 1 and 3. Recomputing costs one Smart call and keeps the stage self-contained;
      threading them would mean another artifact whose only consumer is this stage.
    """
    from crewai import Crew

    from dr2_podcast.artifacts import read_json_strict, read_text_strict
    from dr2_podcast.pipeline_crew import CrewBudget, SotInjection, _crew_kickoff_guarded
    from dr2_podcast.pipeline_script import _parse_blueprint_inventory
    from dr2_podcast.utils import strip_think_blocks

    pipeline = _prepare_run(run_dir, run_config)
    topic = run_config["topic"]
    sot_file = run_dir / "research/source_of_truth.md"
    sot_summary = pipeline.summarize_report(read_text_strict(sot_file), "sot", topic)

    translated_file = run_dir / f"research/source_of_truth_{run_config['language']}.md"
    translated_summary = ""
    if translated_file.exists():
        translated_summary = pipeline.summarize_report(read_text_strict(translated_file), "sot_translated", topic)
    else:
        translated_file = None

    domain = read_json_strict(run_dir / "research/domain_classification.json")["domain"]
    grade_injection = ""
    if (run_dir / "research/grade_synthesis.md").exists():
        # The third argument is only a truthiness guard meaning "there is research to quote"
        # (pipeline.py:2439); the numbers themselves are read from grade_synthesis.md on disk.
        grade_injection = pipeline._build_grade_injection(run_dir, domain, {"grade_synthesis": True})

    task = pipeline.blueprint_task
    task.output_file = None  # this module owns the write, atomically
    _crew_kickoff_guarded(
        lambda: Crew(agents=[pipeline.producer_agent], tasks=[task], verbose=True),
        task,
        pipeline.translation_task,
        run_config["language"],
        SotInjection(
            sot_file=sot_file,
            translated_sot_file=translated_file,
            sot_summary=sot_summary,
            translated_sot_summary=translated_summary,
            grade_numbers_text=grade_injection,
            language_config=pipeline.language_config,
        ),
        CrewBudget("blueprint"),
    )

    raw = strip_think_blocks(task.output.raw if getattr(task, "output", None) else "")
    if not raw.strip():
        raise ArtifactError("the blueprint crew returned nothing; a stage that produced nothing has failed")
    write_atomic(run_dir / "research/EPISODE_BLUEPRINT.md", raw)
    write_json_atomic(run_dir / "meta/blueprint_inventory.json", _parse_blueprint_inventory(raw))


@register("audio")
def audio(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 8 — TTS and the BGM mix. Python plus the TTS engines, no Crew.

    Reads the final script from disk, which the phase received as an argument. It does not call
    :func:`_prepare_run`: the audio path needs ``output_dir`` and the language config, not the LLM
    handles or any Crew, and building them would make audio unrenderable whenever vLLM is down.

    ``_run_audio_pipeline`` returns ``(None, None)`` when it fails and the phase only logs a
    warning, so a run could reach its terminal state with no audio and nothing saying the run had
    failed. This raises.
    """
    from dr2_podcast import pipeline
    from dr2_podcast.artifacts import read_text_strict

    script = read_text_strict(run_dir / "scripts/script_final.md")
    pipeline.output_dir = run_dir
    language_config = pipeline.SUPPORTED_LANGUAGES[run_config["language"]]
    audio_file, duration_minutes = pipeline._run_audio_pipeline(script, run_dir, language_config)
    if not audio_file or not Path(audio_file).exists():
        raise ArtifactError(
            "audio generation produced no file. The monolithic phase logs a warning and returns, "
            "so a run reaches its terminal state with no audio and nothing saying it failed."
        )
    if not duration_minutes:
        raise ArtifactError(f"{audio_file} was written but reports no duration; treat that as a failed render")


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


__all__ = ["audio", "blueprint", "framing", "registered", "translate", "url_validation"]
