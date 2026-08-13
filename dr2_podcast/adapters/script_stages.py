"""Adapters for phases 4-8: blueprint, draft, polish, audit, audio.

See dr2_podcast.adapters for the shared state reconstruction and why it exists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dr2_podcast.adapters._common import (
    _DraftOutput,
    _prepare_run,
    _prime_translation_task,
    _script_context,
    drop_unproduced_optional_outputs,
    promote,
    staging_dir,
)
from dr2_podcast.artifacts import ArtifactError, write_atomic, write_json_atomic
from dr2_podcast.stages import register

logger = logging.getLogger(__name__)


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

    _prime_translation_task(pipeline, run_dir, run_config["language"])
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


@register("draft")
def draft(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 5 — the sectional script draft.

    Everything it needs is on disk already: the blueprint inventory the blueprint adapter persisted,
    and the English source of truth. The flow hands phase 5 the ENGLISH SOT even for a Japanese
    episode — translation reaches the script through the Crew task descriptions, not through this
    context — so that is what is read here. Deviating would change what the episode says.
    """
    from dr2_podcast.artifacts import read_json_strict, read_text_strict

    pipeline = _prepare_run(run_dir, run_config)
    inventory = read_json_strict(run_dir / "meta/blueprint_inventory.json")
    sot = read_text_strict(run_dir / "research/source_of_truth.md")

    text, count = pipeline._run_sectional_draft(
        inventory,
        _script_context(pipeline, run_config, sot),
        _call_smart_model=pipeline._call_smart_model,
    )
    if not text or not text.strip():
        raise ArtifactError("the sectional draft produced no script; a stage that produced nothing has failed")
    write_atomic(run_dir / "scripts/script_draft.md", text)
    logger.info("draft: %d %s", count, pipeline.language_config["length_unit"])


@register("polish")
def polish(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 6 — the polish loop, with its shrinkage guard.

    ``draft_count`` is RECOMPUTED from the draft on disk with ``_count_words``, the same function
    phase 5 used to produce it. The alternative was persisting the number, and a number that can be
    derived from the artifact it describes is a second source of truth waiting to disagree with it.

    ``_run_polish_loop`` reads the draft off ``script_task.output.raw``, so the task is primed the
    way the phase primes it. The polish task's description and expected output are taken straight
    from the freshly built task, which in a fresh process IS the base — the monolithic flow has to
    pass them separately precisely because it mutates the live task between phases.
    """
    from dr2_podcast.artifacts import read_json_strict, read_text_strict

    pipeline = _prepare_run(run_dir, run_config)
    draft_text = read_text_strict(run_dir / "scripts/script_draft.md")
    inventory = read_json_strict(run_dir / "meta/blueprint_inventory.json")
    sot = read_text_strict(run_dir / "research/source_of_truth.md")
    draft_count = pipeline._count_words(draft_text, pipeline.language_config)

    pipeline.script_task.output = _DraftOutput(draft_text)
    _prime_translation_task(pipeline, run_dir, run_config["language"])
    polished, _final_task = pipeline._run_polish_loop(
        draft_text,
        draft_count,
        inventory,
        _script_context(pipeline, run_config, sot),
        pipeline.Crew3Refs(
            script_task=pipeline.script_task,
            polish_task=pipeline.polish_task,
            translation_task=pipeline.translation_task,
            editor_agent=pipeline.editor_agent,
            polish_base_desc=pipeline.polish_task.description,
            polish_expected=pipeline.polish_task.expected_output,
        ),
    )
    if not polished or not polished.strip():
        raise ArtifactError("the polish loop produced no script; a stage that produced nothing has failed")
    write_atomic(run_dir / "scripts/script_polished.md", polished)


@register("audit")
def audit(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 7 — the accuracy audit, its two deterministic gates, the correction pass, and finalize.

    The stage spans more than ``phase_7_audit`` because the phase alone produces no artifact the
    next stage can use: the flow runs the gates, conditionally corrects, and only then calls
    ``_finalize_script`` to write ``script_final.md``, which is what audio renders. Splitting those
    apart would leave a stage boundary in the middle of a decision.

    **It fails closed where the flow ships anyway.** When the gate fires and the correction pass
    returns nothing, the flow logs "finalizing the UNCORRECTED script; MANUAL REVIEW NEEDED" and
    proceeds to audio (``pipeline_flow.py:1329``). A script that the pipeline's own accuracy gate
    rejected, and could not repair, is not a script to render — and "manual review needed" means a
    human has to look before this run continues, which under a staged runner is a failed stage.
    That is a deliberate deviation from the monolithic behaviour, recorded in PLAN.md. It is not
    Step 5: this is the existing Smart auditor, not the Codex loop.
    """
    from crewai import Crew

    from dr2_podcast.artifacts import read_text_strict
    from dr2_podcast.pipeline_flow import (
        _augment_audit_for_corrector,
        _deterministic_gate_issues,
        _run_inline_correction,
        _write_accuracy_corrections_md,
        flow_or_module_logger,
    )

    pipeline = _prepare_run(run_dir, run_config)
    language = run_config["language"]
    # A corrections report from an EARLIER audit must not survive this one: if the gate does not
    # fire this time, that file describes a different script and a different verdict, and
    # Manifest.complete() would record it as this execution's optional output.
    (run_dir / "research/ACCURACY_CORRECTIONS.md").unlink(missing_ok=True)
    polished = read_text_strict(run_dir / "scripts/script_polished.md")
    sot = read_text_strict(run_dir / "research/source_of_truth.md")

    pipeline.polish_task.output = _DraftOutput(polished)
    _prime_translation_task(pipeline, run_dir, language)

    task = pipeline.audit_task
    task.output_file = None  # this module owns the write, atomically
    task.context = [pipeline.polish_task]
    if pipeline.translation_task is not None:
        task.context = [pipeline.polish_task, pipeline.translation_task]
    Crew(agents=[pipeline.auditor_agent], tasks=[task], verbose=True).kickoff()

    audit_output = task.output.raw if getattr(task, "output", None) else ""
    if not audit_output.strip():
        raise ArtifactError("the accuracy audit returned nothing; an unaudited script is not an audited one")
    write_atomic(run_dir / "research/accuracy_audit.md", audit_output)

    flow_logger = flow_or_module_logger()
    citation_issues, grade_issues = _deterministic_gate_issues(polished, sot, flow_logger)
    corrected = None
    if pipeline._audit_requires_correction(audit_output) or citation_issues or grade_issues:
        logger.info(
            "accuracy gate TRIGGERED (citations=%d, grade=%d) — correcting",
            len(citation_issues),
            len(grade_issues),
        )
        corrected = _run_inline_correction(
            audit_output=_augment_audit_for_corrector(audit_output, citation_issues, grade_issues),
            polished_text=polished,
            editor_agent_ref=pipeline.editor_agent,
            target_instruction=pipeline.target_instruction,
            output_dir=run_dir,
        )
        _write_accuracy_corrections_md(
            run_dir, audit_output, citation_issues, grade_issues, corrected, flow_logger
        )
        if corrected is None:
            raise ArtifactError(
                "the accuracy gate fired and the correction pass produced no valid script. The "
                "monolithic flow finalises the UNCORRECTED script and renders it; a script this "
                "pipeline's own gate rejected and could not repair needs a human, not audio. See "
                "research/accuracy_audit.md and ACCURACY_CORRECTIONS.md."
            )

    # _finalize_script writes script_final.md with a bare open(), so an interruption partway would
    # replace the previous valid final script with a truncated one. Staged and promoted instead.
    with staging_dir(run_dir) as staging:
        final = pipeline._finalize_script(
            polished, pipeline.polish_task, language, pipeline.language_config, staging, corrected_text=corrected
        )
        if not final or not final.strip():
            raise ArtifactError("finalisation produced no script; there is nothing for audio to render")
        promote(staging, run_dir)


@register("audio")
def audio(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 8 — TTS and the BGM mix. Python plus the TTS engines, no Crew.

    Reads the final script from disk, which the phase received as an argument. It does not call
    :func:`_prepare_run`: the audio path needs ``output_dir`` and the language config, not the LLM
    handles or any Crew, and building them would make audio unrenderable whenever vLLM is down.

    **Rendering happens in a staging directory and the results are promoted by rename**, and any
    declared optional output this render did NOT produce is removed — otherwise a rerender whose BGM
    pass fails leaves the previous ``audio_mixed.wav`` beside the new raw audio, both looking
    current, and someone publishes mixed audio of a script that no longer exists.
    ``_run_audio_pipeline`` writes ``script.txt`` and the WAVs straight to their final paths, so a
    render interrupted midway would destroy the previous good audio or leave a truncated WAV that
    looks finished — the exact failure the atomic-artifact rule exists to prevent, and worse here
    because a WAV's truncation is not visible until someone listens to it. Staging sits inside the
    run directory, so the promotion is a same-filesystem rename.

    ``_run_audio_pipeline`` returns ``(None, None)`` when it fails and the phase only logs a
    warning, so a run could reach its terminal state with no audio and nothing saying the run had
    failed. This raises.
    """
    from dr2_podcast import pipeline
    from dr2_podcast.artifacts import read_text_strict

    script = read_text_strict(run_dir / "scripts/script_final.md")
    language_config = pipeline.SUPPORTED_LANGUAGES[run_config["language"]]

    previous_output_dir = pipeline.output_dir
    with staging_dir(run_dir) as staging:
        try:
            pipeline.output_dir = staging
            audio_file, duration_minutes = pipeline._run_audio_pipeline(script, staging, language_config)
            if not audio_file or not Path(audio_file).exists():
                raise ArtifactError(
                    "audio generation produced no file. The monolithic phase logs a warning and returns, "
                    "so a run reaches its terminal state with no audio and nothing saying it failed."
                )
            if not duration_minutes:
                raise ArtifactError(f"{audio_file} was written but reports no duration; that is a failed render")
            promoted = promote(staging, run_dir)
        finally:
            pipeline.output_dir = previous_output_dir

    dropped = drop_unproduced_optional_outputs(run_dir, "audio", promoted)
    if dropped:
        logger.info("audio: removed stale %s left by a previous render", ", ".join(dropped))
    logger.info("audio: %.2f min", duration_minutes)
