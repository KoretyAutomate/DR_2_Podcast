"""Prefect orchestration layer for DR_2_Podcast pipeline.

Each of the 9 pipeline phases is wrapped in a Prefect @task with
persist_result=True so that completed phases are automatically skipped
on resume (replacing the manual checkpoint.json pattern).

The @flow entry point run_pipeline_flow() replaces the monolithic
if __name__ == '__main__' orchestration block in pipeline.py.

Sequential execution is enforced throughout — all LLM calls share a
single GPU and cannot run in parallel without risking OOM.

Usage (from pipeline.py __main__):
    from dr2_podcast.pipeline_flow import run_pipeline_flow
    run_pipeline_flow(...)

Install requirement (run once):
    pip install "prefect>=3.0,<4"
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from prefect import flow, task, get_run_logger

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cache key helper — phases are keyed by (output_dir, phase_number) so that
# resuming the same run directory skips already-completed phases.
# ---------------------------------------------------------------------------

def _phase_cache_key(context, parameters):
    """Cache key = output_dir basename + phase name (filesystem-safe).

    The original implementation used the full absolute path with ``::`` as a
    separator, which produced a cache key like ``/home/.../outputs/2026-...::phase_1_research``.
    Prefect's LocalFileSystem storage resolves that key as a path, and the
    leading ``/`` causes it to escape the storage root, triggering:
      ValueError: '...' is not in the subpath of '/home/.../.prefect/storage'

    Fix: use only the run-directory basename (timestamp + random suffix),
    which is unique per run and contains no path separators.
    """
    import hashlib
    od = parameters.get("output_dir") or parameters.get("output_dir_str", "")
    # Use basename for readability; add a short hash of the full path as a
    # collision guard in case two runs share the same basename somehow.
    basename = od.rstrip("/").rsplit("/", 1)[-1] if "/" in od else od
    od_hash = hashlib.md5(od.encode()).hexdigest()[:8]
    return f"{basename}-{od_hash}-{context.task.name}"


# ---------------------------------------------------------------------------
# Phase 0 — Research Framing + Domain Classification
# ---------------------------------------------------------------------------

@task(
    name="phase_0_framing",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
)
def phase_0_framing(
    output_dir: str,
    topic_name: str,
    language: str,
    smart_base_url: str,
    smart_model: str,
    framing_task_ref,
    framing_agent_ref,
    blueprint_task_ref,
    script_task_ref,
    audit_task_ref,
):
    """Phase 0: Domain classification + research framing crew."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.research.domain_classifier import classify_topic, ResearchDomain
    from crewai import Crew

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 70)
    run_logger.info("PHASE 0: RESEARCH FRAMING + DOMAIN CLASSIFICATION")
    run_logger.info("=" * 70)

    # Domain classification
    try:
        from openai import AsyncOpenAI as _AOAIClassify
        _classify_client = _AOAIClassify(base_url=smart_base_url, api_key="not-needed")
    except Exception:
        _classify_client = None

    domain_classification = asyncio.run(classify_topic(
        topic=topic_name,
        smart_client=_classify_client,
        smart_model=smart_model,
    ))
    run_logger.info(
        "Domain: %s (confidence=%.2f, framework=%s)",
        domain_classification.domain.value,
        domain_classification.confidence,
        domain_classification.suggested_framework,
    )

    # Persist domain classification
    _dc_path = _pipeline.output_path(output_dir_path, "domain_classification.json")
    _dc_path.write_text(json.dumps({
        "domain": domain_classification.domain.value,
        "confidence": domain_classification.confidence,
        "reasoning": domain_classification.reasoning,
        "framework": domain_classification.suggested_framework,
        "databases": domain_classification.primary_databases,
    }, indent=2))

    # Build domain note for framing task
    if domain_classification.domain == ResearchDomain.SOCIAL_SCIENCE:
        _domain_framing_note = (
            "\n\nDOMAIN CONTEXT: This is a SOCIAL SCIENCE topic. "
            "Use PECO framework (Population, Exposure, Comparison, Outcome). "
            f"Prioritise effect sizes (Cohen's d, Hedges' g), quasi-experimental designs, "
            f"and databases such as {', '.join(domain_classification.primary_databases)}. "
            "Do NOT use clinical terminology (NNT, ARR, GRADE, MeSH terms)."
        )
    else:
        _domain_framing_note = (
            "\n\nDOMAIN CONTEXT: This is a CLINICAL/HEALTH topic. "
            "Use PICO framework (Population, Intervention, Comparison, Outcome). "
            "Prioritise RCTs, systematic reviews, GRADE evidence levels, NNT/ARR statistics, "
            f"and databases such as {', '.join(domain_classification.primary_databases)}."
        )
    framing_task_ref.description += _domain_framing_note

    # Run framing crew
    framing_output = ""
    try:
        crew_1 = Crew(
            agents=[framing_agent_ref],
            tasks=[framing_task_ref],
            verbose=True,
            process="sequential",
        )
        crew_1.kickoff()
        framing_output = (
            framing_task_ref.output.raw
            if hasattr(framing_task_ref, "output") and framing_task_ref.output
            else ""
        )
        run_logger.info("Phase 0 complete: %d chars framing output", len(framing_output))
    except Exception as exc:
        run_logger.warning("Phase 0 (Research Framing) failed: %s — continuing", exc)

    return {
        "framing_output": framing_output,
        "domain": domain_classification.domain.value,
        "confidence": domain_classification.confidence,
        "reasoning": domain_classification.reasoning,
        "framework": domain_classification.suggested_framework,
        "databases": domain_classification.primary_databases,
    }


# ---------------------------------------------------------------------------
# Phase 1 — Deep Research Pipeline
# ---------------------------------------------------------------------------

@task(
    name="phase_1_research",
    retries=1,
    retry_delay_seconds=60,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=7200,  # 2 hours max
)
def phase_1_research(
    output_dir: str,
    topic_name: str,
    language: str,
    framing_output: str,
    research_domain: str,
    evidence_limited_threshold: int,
):
    """Phase 1: Deep research pipeline (clinical or social science)."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.research.clinical import run_deep_research
    from dr2_podcast.config import EVIDENCE_LIMITED_THRESHOLD

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 70)
    run_logger.info("PHASE 1: RESEARCH PIPELINE")
    run_logger.info("=" * 70)

    _effective_domain = research_domain if research_domain in ("clinical", "social_science") else "clinical"
    brave_key = os.getenv("BRAVE_API_KEY", "")

    # Fast model availability check
    _fast_model_name = os.environ.get("FAST_MODEL_NAME", "")
    _fast_base_url = os.environ.get("FAST_LLM_BASE_URL", "http://localhost:11434/v1")
    fast_model_available = False
    try:
        import httpx
        _resp = httpx.get(f"{_fast_base_url}/models", timeout=3)
        if _resp.status_code == 200:
            _models = [m.get("id", "") for m in _resp.json().get("data", [])]
            fast_model_available = _fast_model_name in _models
            if fast_model_available:
                run_logger.info("Fast model ready: %s", _fast_model_name)
            else:
                run_logger.warning("Fast model '%s' not found. Available: %s", _fast_model_name, _models)
    except Exception:
        run_logger.warning("Fast model not available (Ollama unreachable). Running smart-only mode.")

    aff_candidates = 0
    neg_candidates = 0
    evidence_quality = "sufficient"
    deep_reports = None
    sot_content = ""
    sot_summary = ""

    try:
        deep_reports = asyncio.run(run_deep_research(
            topic=topic_name,
            brave_api_key=brave_key,
            results_per_query=15,
            fast_model_available=fast_model_available,
            framing_context=framing_output,
            output_dir=str(output_dir_path),
            domain=_effective_domain,
        ))

        # Read candidate counts from screening results
        for fname, varname in [("screening_results_aff.json", "aff"), ("screening_results_neg.json", "neg")]:
            p = _pipeline.output_path(output_dir_path, fname)
            if p.exists():
                try:
                    val = json.loads(p.read_text()).get("total_candidates", 0)
                    if varname == "aff":
                        aff_candidates = val
                    else:
                        neg_candidates = val
                except Exception:
                    pass

        if aff_candidates == 0:
            _pipeline._write_insufficient_evidence_report(topic_name, 0, neg_candidates, output_dir_path)
            from dr2_podcast.pipeline import InsufficientEvidenceError
            raise InsufficientEvidenceError(
                f"Affirmative track: 0 candidates for '{topic_name}'. "
                f"Adversarial found {neg_candidates}. "
                "See insufficient_evidence_report.md for suggested rephrasing."
            )

        if 0 < aff_candidates < evidence_limited_threshold:
            evidence_quality = "limited"

        # Save research reports
        REPORT_FILENAMES = {
            "lead": "affirmative_case.md",
            "counter": "falsification_case.md",
            "audit": "grade_synthesis.md",
        }
        for role_name, filename in REPORT_FILENAMES.items():
            report = deep_reports.get(role_name)
            if not report:
                run_logger.warning("%s report missing — skipping save", role_name.capitalize())
                continue
            report_file = _pipeline.output_path(output_dir_path, filename)
            report_file.write_text(report.report)
            run_logger.info("%s report saved: %s (%d sources)", role_name.capitalize(), filename, report.total_summaries)

        # Save sources JSON
        sources_json = {}
        for role_name in ("lead", "counter"):
            report = deep_reports[role_name]
            role_sources = []
            for idx, src in enumerate(report.sources):
                if src.error or not src.summary or src.summary.strip().upper() == "NO RELEVANT DATA":
                    continue
                if not src.url:
                    continue
                role_sources.append({
                    "index": idx,
                    "url": src.url,
                    "title": src.title,
                    "query": src.query,
                    "goal": src.goal,
                    "summary": src.summary,
                    "metadata": src.metadata.to_dict() if src.metadata else None,
                })
            sources_json[role_name] = role_sources
        sources_file = _pipeline.output_path(output_dir_path, "research_sources.json")
        sources_file.write_text(json.dumps(sources_json, indent=2, ensure_ascii=False))
        run_logger.info("Research library saved: %d lead, %d counter sources",
                        len(sources_json.get("lead", [])), len(sources_json.get("counter", [])))

        # Build Source-of-Truth
        sot_content = _pipeline.build_imrad_sot(
            topic=topic_name,
            reports=deep_reports,
            ev_quality=evidence_quality,
            aff_cand=aff_candidates,
            domain=research_domain,
        )
        if evidence_quality == "limited":
            from dr2_podcast.config import EVIDENCE_LIMITED_THRESHOLD as _ELT
            sot_content = (
                "## Evidence Quality Notice\n\n"
                f"The affirmative research track retrieved only **{aff_candidates} candidate studies** "
                f"(threshold: {_ELT}). "
                "The following synthesis is based on limited direct evidence. "
                "Claims should be interpreted cautiously.\n\n"
            ) + sot_content

        sot_file = _pipeline.output_path(output_dir_path, "source_of_truth.md")
        sot_file.write_text(sot_content)
        run_logger.info("Source of Truth (IMRaD) generated: %d chars", len(sot_content))

        sot_summary = _pipeline.summarize_report_with_fast_model(sot_content, "sot", topic_name)

    except Exception as exc:
        # Re-raise InsufficientEvidenceError (non-retryable — tells caller to abort)
        from dr2_podcast.pipeline import InsufficientEvidenceError
        if isinstance(exc, InsufficientEvidenceError):
            raise
        run_logger.warning("Deep research failed: %s — continuing without deep research", exc)

    # Serialize deep_reports for storage (plain dict of serializable values)
    _dr_serialized = None
    if deep_reports is not None:
        try:
            from dr2_podcast.pipeline import _serialize_dataclass
            _dr_serialized = _serialize_dataclass(deep_reports)
        except Exception:
            _dr_serialized = None

    return {
        "sot_content": sot_content,
        "sot_summary": sot_summary,
        "evidence_quality": evidence_quality,
        "aff_candidates": aff_candidates,
        "neg_candidates": neg_candidates,
        "research_domain": research_domain,
        "deep_reports_serialized": _dr_serialized,
    }


# ---------------------------------------------------------------------------
# Phase 2 — URL Validation
# ---------------------------------------------------------------------------

@task(
    name="phase_2_url_validation",
    retries=1,
    retry_delay_seconds=10,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=600,
)
def phase_2_url_validation(output_dir: str):
    """Phase 2: Validate and filter broken URLs from research library."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.tools.link_validator import validate_multiple_urls_parallel

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 70)
    run_logger.info("PHASE 2: SOURCE VALIDATION")
    run_logger.info("=" * 70)

    all_urls: set = set()
    url_pattern = re.compile(r'https?://[^\s\)\]\"\'<>]+')

    sources_file = _pipeline.output_path(output_dir_path, "research_sources.json")
    if sources_file.exists():
        try:
            src_data = json.loads(sources_file.read_text())
            for role_sources in src_data.values():
                if isinstance(role_sources, list):
                    for src in role_sources:
                        if src.get("url"):
                            all_urls.add(src["url"])
        except Exception:
            pass

    run_logger.info("Found %d unique URLs to validate", len(all_urls))

    if all_urls:
        validation_results = validate_multiple_urls_parallel(list(all_urls), max_workers=15)
        valid_count = sum(1 for v in validation_results.values() if "Valid" in v)
        broken_count = sum(1 for v in validation_results.values() if "Broken" in v or "Invalid" in v)
        run_logger.info("Results: %d valid, %d broken, %d other",
                        valid_count, broken_count,
                        len(validation_results) - valid_count - broken_count)

        validation_file = _pipeline.output_path(output_dir_path, "url_validation_results.json")
        validation_file.write_text(json.dumps(validation_results, indent=2, ensure_ascii=False))

        broken_urls = {url for url, status in validation_results.items()
                       if "Broken" in status or "Invalid" in status or status.startswith("ERROR")}
        if broken_urls and sources_file.exists():
            try:
                src_data = json.loads(sources_file.read_text())
                for role in src_data:
                    if isinstance(src_data[role], list):
                        before = len(src_data[role])
                        src_data[role] = [s for s in src_data[role] if s.get("url") not in broken_urls]
                        removed = before - len(src_data[role])
                        if removed:
                            run_logger.info("Filtered %d broken source(s) from '%s'", removed, role)
                sources_file.write_text(json.dumps(src_data, indent=2, ensure_ascii=False))
            except Exception as exc:
                run_logger.warning("Failed to filter broken sources: %s", exc)
    else:
        run_logger.warning("No URLs found to validate")
        # Write empty result so artifact count is satisfied
        validation_file = _pipeline.output_path(output_dir_path, "url_validation_results.json")
        validation_file.write_text("{}")

    return {"validated": True}


# ---------------------------------------------------------------------------
# Phase 3 — SOT Translation (skip for English)
# ---------------------------------------------------------------------------

@task(
    name="phase_3_translation",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=3600,
)
def phase_3_translation(
    output_dir: str,
    topic_name: str,
    language: str,
    language_config: dict,
    sot_content: str,
    sot_summary: str,
    grade_injection: str,
    blueprint_task_ref,
    script_task_ref,
    audit_task_ref,
    translation_task_ref,
    sot_file_path: Optional[str],
):
    """Phase 3: Translate Source-of-Truth for non-English pipelines."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.pipeline_crew import _build_sot_injection_for_stage

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    if language == "en" or translation_task_ref is None:
        run_logger.info("Phase 3: English pipeline — translation skipped")
        return {
            "translated_sot": None,
            "translated_sot_summary": "",
            "sot_translated_file": None,
        }

    run_logger.info("=" * 70)
    run_logger.info("PHASE 3: REPORT TRANSLATION")
    run_logger.info("=" * 70)

    sot_file = Path(sot_file_path) if sot_file_path else None
    translated_sot, sot_translated_file, translated_sot_summary = _pipeline._translate_and_inject_sot(
        sot_content, language, language_config, topic_name,
        output_dir_path, sot_file,
        sot_summary, grade_injection,
        blueprint_task_ref, script_task_ref, audit_task_ref, translation_task_ref,
    )

    return {
        "translated_sot": translated_sot,
        "translated_sot_summary": translated_sot_summary,
        "sot_translated_file": str(sot_translated_file) if sot_translated_file else None,
    }


# ---------------------------------------------------------------------------
# Phase 4 — Episode Blueprint
# ---------------------------------------------------------------------------

@task(
    name="phase_4_blueprint",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=1800,
)
def phase_4_blueprint(
    output_dir: str,
    topic_name: str,
    language: str,
    language_config: dict,
    sot_file_path: Optional[str],
    sot_translated_file_path: Optional[str],
    sot_summary: str,
    translated_sot_summary: str,
    grade_injection: str,
    blueprint_task_ref,
    producer_agent_ref,
    translation_task_ref,
):
    """Phase 4: Episode blueprint via producer agent."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.pipeline_crew import _crew_kickoff_guarded
    from dr2_podcast.utils import strip_think_blocks
    from dr2_podcast.pipeline_script import _parse_blueprint_inventory

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 60)
    run_logger.info("PHASE 4: EPISODE BLUEPRINT")
    run_logger.info("=" * 60)

    from crewai import Crew

    sot_file = Path(sot_file_path) if sot_file_path else None
    sot_translated_file = Path(sot_translated_file_path) if sot_translated_file_path else None

    _crew_kickoff_guarded(
        lambda: Crew(agents=[producer_agent_ref], tasks=[blueprint_task_ref], verbose=True),
        blueprint_task_ref, translation_task_ref, language,
        sot_file, sot_translated_file,
        sot_summary, translated_sot_summary,
        grade_injection, language_config, "Phase 4 Blueprint",
    )
    run_logger.info("Blueprint complete")

    # Save blueprint to disk
    _bp_raw = strip_think_blocks(blueprint_task_ref.output.raw)
    bp_file = _pipeline.output_path(output_dir_path, "EPISODE_BLUEPRINT.md")
    bp_file.write_text(_bp_raw)
    run_logger.info("Blueprint saved: %d chars", len(_bp_raw))

    # Parse inventory for downstream phases
    _bp_inventory = _parse_blueprint_inventory(_bp_raw)

    return {
        "blueprint_text": _bp_raw,
        "bp_inventory": _bp_inventory,
    }


# ---------------------------------------------------------------------------
# Phase 5 — Sectional Script Draft
# ---------------------------------------------------------------------------

@task(
    name="phase_5_script_draft",
    retries=1,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=7200,
)
def phase_5_script_draft(
    output_dir: str,
    topic_name: str,
    language: str,
    language_config: dict,
    target_length_int: int,
    target_instruction: str,
    channel_intro: str,
    target_min: int,
    session_roles: dict,
    sot_content: str,
    bp_inventory: dict,
):
    """Phase 5: Generate script draft via sequential section calls."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.pipeline_script import _count_words

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 60)
    run_logger.info("PHASE 5: SCRIPT DRAFT (SECTIONAL)")
    run_logger.info("=" * 60)

    script_draft_text, draft_count = _pipeline._run_sectional_draft(
        bp_inventory, target_length_int, language_config, sot_content,
        session_roles, topic_name, target_instruction, channel_intro,
        _call_smart_model=_pipeline._call_smart_model,
        target_min=target_min,
    )

    # Save draft to disk
    sd_path = _pipeline.output_path(output_dir_path, "script_draft.md")
    sd_path.write_text(script_draft_text, encoding="utf-8")
    run_logger.info("Script draft saved: %d %s", draft_count, language_config["length_unit"])

    return {
        "script_draft_text": script_draft_text,
        "draft_count": draft_count,
    }


# ---------------------------------------------------------------------------
# Phase 6 — Script Polish
# ---------------------------------------------------------------------------

@task(
    name="phase_6_polish",
    retries=1,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=7200,
)
def phase_6_polish(
    output_dir: str,
    script_draft_text: str,
    draft_count: int,
    bp_inventory: dict,
    target_length_int: int,
    language_config: dict,
    sot_content: str,
    topic_name: str,
    target_instruction: str,
    session_roles: dict,
    script_task_ref,
    polish_task_ref,
    editor_agent_ref,
    translation_task_ref,
    polish_task_base_description: str,
    polish_task_expected_output: str,
    max_attempts: int = 3,
):
    """Phase 6: Polish loop with shrinkage guard."""
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.pipeline_script import SCRIPT_TOLERANCE

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 60)
    run_logger.info("PHASE 6: SCRIPT POLISH (max %d attempts)", max_attempts)
    run_logger.info("=" * 60)

    # Sync script_task output so polish loop can read it
    class _FakeDraftOutput:
        def __init__(self, raw):
            self.raw = raw
    script_task_ref.output = _FakeDraftOutput(script_draft_text)

    polished_text, final_polish_task = _pipeline._run_polish_loop(
        script_draft_text, draft_count, bp_inventory, target_length_int,
        language_config, sot_content, script_task_ref, polish_task_ref,
        editor_agent_ref, translation_task_ref,
        polish_task_base_description, polish_task_expected_output,
        max_attempts,
        session_roles=session_roles,
        topic_name=topic_name,
        target_instruction=target_instruction,
    )

    # Save polished script to disk
    pol_path = _pipeline.output_path(output_dir_path, "script_polished.md")
    pol_path.write_text(polished_text, encoding="utf-8")
    run_logger.info("Polished script saved: %d chars", len(polished_text))

    return {"polished_text": polished_text}


# ---------------------------------------------------------------------------
# Phase 7 — Accuracy Audit
# ---------------------------------------------------------------------------

@task(
    name="phase_7_audit",
    retries=2,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=1800,
)
def phase_7_audit(
    output_dir: str,
    audit_task_ref,
    polish_task_ref,
    auditor_agent_ref,
    translation_task_ref,
):
    """Phase 7: Accuracy audit (advisory)."""
    from dr2_podcast import pipeline as _pipeline

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 60)
    run_logger.info("PHASE 7: ACCURACY AUDIT")
    run_logger.info("=" * 60)

    _pipeline._run_accuracy_audit(
        audit_task_ref, polish_task_ref, auditor_agent_ref, translation_task_ref
    )

    audit_output = (
        audit_task_ref.output.raw
        if hasattr(audit_task_ref, "output") and audit_task_ref.output
        else ""
    )

    # Save audit to disk
    aud_path = _pipeline.output_path(output_dir_path, "accuracy_audit.md")
    aud_path.write_text(audit_output, encoding="utf-8")
    run_logger.info("Accuracy audit saved: %d chars", len(audit_output))

    return {"audit_output": audit_output}


# ---------------------------------------------------------------------------
# Phase 8 — Audio Generation
# ---------------------------------------------------------------------------

@task(
    name="phase_8_audio",
    retries=1,
    retry_delay_seconds=30,
    persist_result=True,
    cache_key_fn=_phase_cache_key,
    log_prints=True,
    timeout_seconds=14400,  # TTS can take a long time
)
def phase_8_audio(
    output_dir: str,
    script_text: str,
    language_config: dict,
):
    """Phase 8: TTS + BGM mix."""
    from dr2_podcast import pipeline as _pipeline

    run_logger = get_run_logger()
    output_dir_path = Path(output_dir)

    run_logger.info("=" * 60)
    run_logger.info("PHASE 8: AUDIO GENERATION")
    run_logger.info("=" * 60)

    # Update module-level output_dir so output_path() resolves correctly
    _pipeline.output_dir = output_dir_path

    audio_file, duration_minutes = _pipeline._run_audio_pipeline(
        script_text, output_dir_path, language_config
    )

    if audio_file:
        run_logger.info("Audio complete: %s (%.2f min)", audio_file, duration_minutes or 0)
    else:
        run_logger.warning("Audio generation did not produce a file")

    return {
        "audio_file": str(audio_file) if audio_file else None,
        "duration_minutes": duration_minutes,
    }


# ---------------------------------------------------------------------------
# Main flow
# ---------------------------------------------------------------------------

@flow(
    name="dr2_podcast_pipeline",
    description="DR_2_Podcast: research-driven debate podcast generation",
    log_prints=True,
)
def run_pipeline_flow(
    topic_name: str,
    language: str,
    language_config: dict,
    output_dir: Path,
    target_length_int: int,
    target_instruction: str,
    channel_intro: str,
    target_min: int,
    session_roles: dict,
    smart_model: str,
    smart_base_url: str,
    accessibility_level: str,
    framing_task_ref,
    blueprint_task_ref,
    script_task_ref,
    polish_task_ref,
    audit_task_ref,
    translation_task_ref,
    framing_agent_ref,
    producer_agent_ref,
    editor_agent_ref,
    auditor_agent_ref,
    polish_task_base_description: str,
    polish_task_expected_output: str,
    evidence_limited_threshold: int,
    max_script_attempts: int = 3,
):
    """Run the full 9-phase podcast generation pipeline under Prefect orchestration.

    Each phase is a cached task — resuming the same output_dir skips completed phases
    automatically. Phases run strictly sequentially (single GPU constraint).
    """
    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.config import EVIDENCE_LIMITED_THRESHOLD

    flow_logger = get_run_logger()
    output_dir_str = str(output_dir)

    flow_logger.info("=" * 70)
    flow_logger.info("DR_2_PODCAST PIPELINE — Prefect Flow")
    flow_logger.info("Topic: %s | Language: %s", topic_name, language)
    flow_logger.info("Output: %s", output_dir_str)
    flow_logger.info("=" * 70)

    # Sync module-level output_dir so @tool decorators resolve paths correctly
    _pipeline.output_dir = output_dir

    # -------------------------------------------------------------------
    # Phase 0: Research Framing
    # -------------------------------------------------------------------
    p0_result = phase_0_framing(
        output_dir=output_dir_str,
        topic_name=topic_name,
        language=language,
        smart_base_url=smart_base_url,
        smart_model=smart_model,
        framing_task_ref=framing_task_ref,
        framing_agent_ref=framing_agent_ref,
        blueprint_task_ref=blueprint_task_ref,
        script_task_ref=script_task_ref,
        audit_task_ref=audit_task_ref,
    )
    framing_output = p0_result["framing_output"]
    research_domain = p0_result["domain"]

    # -------------------------------------------------------------------
    # Phase 1: Deep Research
    # -------------------------------------------------------------------
    p1_result = phase_1_research(
        output_dir=output_dir_str,
        topic_name=topic_name,
        language=language,
        framing_output=framing_output,
        research_domain=research_domain,
        evidence_limited_threshold=evidence_limited_threshold,
    )
    sot_content = p1_result["sot_content"]
    sot_summary = p1_result["sot_summary"]
    evidence_quality = p1_result["evidence_quality"]
    aff_candidates = p1_result["aff_candidates"]

    # -------------------------------------------------------------------
    # Phase 2: URL Validation (I/O-bound; errors non-fatal)
    # -------------------------------------------------------------------
    phase_2_url_validation(output_dir=output_dir_str)

    # -------------------------------------------------------------------
    # Inject SOT into task descriptions (needed before phases 3-7)
    # -------------------------------------------------------------------
    sot_file_path = str(_pipeline.output_path(output_dir, "source_of_truth.md"))

    _grade_injection = _pipeline._build_grade_injection(
        output_dir, research_domain, p1_result.get("deep_reports_serialized")
    ) if hasattr(_pipeline, "_build_grade_injection") else ""

    if sot_summary:
        sot_injection = (
            "\n\nSOURCE OF TRUTH (from deep research pipeline):\n"
            "Use this as your authoritative reference for all evidence and claims.\n\n"
            f"{sot_summary}\n\n"
            "For detailed sources, use ListResearchSources('lead') and ListResearchSources('counter').\n"
            "--- END SOURCE OF TRUTH ---\n"
        )
        script_task_ref.description += sot_injection
        audit_task_ref.description += sot_injection
        blueprint_task_ref.description += (
            "\n\nSOURCE OF TRUTH: Use ReadFullReport('sot') to read the full "
            "research document in the target language. Follow the two-pass workflow "
            "described above.\n"
        )

    if evidence_quality == "limited":
        script_task_ref.description += (
            "\n\nEVIDENCE QUALITY NOTE — READ CAREFULLY:\n"
            "The systematic review found limited direct scientific evidence for this question.\n"
            "Your script MUST:\n"
            "1. Acknowledge this in the HOOK or Act 1.\n"
            "2. Distinguish: (a) what limited direct evidence shows, "
            "(b) what related evidence suggests, (c) what remains unknown.\n"
            "3. Frame recommendations as 'based on current evidence' — not 'proven'.\n"
            "4. Do NOT invent citations.\n"
        )

    # -------------------------------------------------------------------
    # Phase 3: SOT Translation (skipped for English)
    # -------------------------------------------------------------------
    p3_result = phase_3_translation(
        output_dir=output_dir_str,
        topic_name=topic_name,
        language=language,
        language_config=language_config,
        sot_content=sot_content,
        sot_summary=sot_summary,
        grade_injection=_grade_injection,
        blueprint_task_ref=blueprint_task_ref,
        script_task_ref=script_task_ref,
        audit_task_ref=audit_task_ref,
        translation_task_ref=translation_task_ref,
        sot_file_path=sot_file_path,
    )
    translated_sot = p3_result["translated_sot"]
    translated_sot_summary = p3_result["translated_sot_summary"]
    sot_translated_file_path = p3_result["sot_translated_file"]

    # -------------------------------------------------------------------
    # Save base descriptions for audit-loop feedback injection
    # -------------------------------------------------------------------
    script_task_base_description = script_task_ref.description
    script_task_expected_output = script_task_ref.expected_output
    polish_base_desc = polish_task_base_description
    polish_expected = polish_task_expected_output

    # -------------------------------------------------------------------
    # Phase 4: Episode Blueprint
    # -------------------------------------------------------------------
    p4_result = phase_4_blueprint(
        output_dir=output_dir_str,
        topic_name=topic_name,
        language=language,
        language_config=language_config,
        sot_file_path=sot_file_path,
        sot_translated_file_path=sot_translated_file_path,
        sot_summary=sot_summary,
        translated_sot_summary=translated_sot_summary,
        grade_injection=_grade_injection,
        blueprint_task_ref=blueprint_task_ref,
        producer_agent_ref=producer_agent_ref,
        translation_task_ref=translation_task_ref,
    )
    bp_inventory = p4_result["bp_inventory"]
    blueprint_text = p4_result["blueprint_text"]

    # Inject blueprint checklist into script task description
    _pipeline._inject_blueprint_checklist(blueprint_task_ref, script_task_ref, script_task_base_description)

    # -------------------------------------------------------------------
    # Phase 5: Script Draft
    # -------------------------------------------------------------------
    p5_result = phase_5_script_draft(
        output_dir=output_dir_str,
        topic_name=topic_name,
        language=language,
        language_config=language_config,
        target_length_int=target_length_int,
        target_instruction=target_instruction,
        channel_intro=channel_intro,
        target_min=target_min,
        session_roles=session_roles,
        sot_content=sot_content,
        bp_inventory=bp_inventory,
    )
    script_draft_text = p5_result["script_draft_text"]
    draft_count = p5_result["draft_count"]

    # -------------------------------------------------------------------
    # Phase 6: Polish
    # -------------------------------------------------------------------
    p6_result = phase_6_polish(
        output_dir=output_dir_str,
        script_draft_text=script_draft_text,
        draft_count=draft_count,
        bp_inventory=bp_inventory,
        target_length_int=target_length_int,
        language_config=language_config,
        sot_content=sot_content,
        topic_name=topic_name,
        target_instruction=target_instruction,
        session_roles=session_roles,
        script_task_ref=script_task_ref,
        polish_task_ref=polish_task_ref,
        editor_agent_ref=editor_agent_ref,
        translation_task_ref=translation_task_ref,
        polish_task_base_description=polish_base_desc,
        polish_task_expected_output=polish_expected,
        max_attempts=max_script_attempts,
    )
    polished_text = p6_result["polished_text"]

    # Sync polish_task.output for downstream audit
    class _FakePolishOutput:
        def __init__(self, raw):
            self.raw = raw
    polish_task_ref.output = _FakePolishOutput(polished_text)

    # -------------------------------------------------------------------
    # Phase 7: Accuracy Audit
    # -------------------------------------------------------------------
    p7_result = phase_7_audit(
        output_dir=output_dir_str,
        audit_task_ref=audit_task_ref,
        polish_task_ref=polish_task_ref,
        auditor_agent_ref=auditor_agent_ref,
        translation_task_ref=translation_task_ref,
    )
    audit_output = p7_result["audit_output"]

    # -------------------------------------------------------------------
    # Conditional Correction (HIGH-severity drift)
    # -------------------------------------------------------------------
    corrected_script_text = _pipeline._run_correction_loop(
        audit_output=audit_output,
        polished_text=polished_text,
        editor_agent=editor_agent_ref,
        target_instruction=target_instruction,
        output_dir=output_dir,
    ) if hasattr(_pipeline, "_run_correction_loop") else None

    # Fallback: use inline correction logic from pipeline.py
    if corrected_script_text is None and audit_output:
        high_severity_found = bool(re.search(r'\*\*Severity\*\*:\s*HIGH', audit_output, re.IGNORECASE))
        if high_severity_found:
            corrected_script_text = _run_inline_correction(
                audit_output=audit_output,
                polished_text=polished_text,
                editor_agent_ref=editor_agent_ref,
                target_instruction=target_instruction,
                output_dir=output_dir,
            )

    # -------------------------------------------------------------------
    # Finalize + save outputs
    # -------------------------------------------------------------------
    script_text = _pipeline._finalize_script(
        polished_text, polish_task_ref, language, language_config, output_dir,
        corrected_text=corrected_script_text,
    )

    # Save markdown outputs
    _pipeline._save_task_outputs(output_dir, [
        ("Research Framing", framing_output, "research_framing.md"),
        ("Accuracy Audit", audit_task_ref, "accuracy_audit.md"),
        ("Episode Blueprint", blueprint_task_ref, "EPISODE_BLUEPRINT.md"),
        ("Script Draft", script_task_ref, "script_draft.md"),
    ])

    # Generate PDFs
    pdf_items = [
        ("Research Framing", framing_output, "research_framing.pdf"),
        ("Source of Truth", sot_content, "source_of_truth.pdf"),
        ("Accuracy Audit", audit_task_ref, "accuracy_audit.pdf"),
    ]
    # Translated SOT PDF for non-English runs
    if translated_sot:
        lang_code = language_config.get("code", "ja")
        pdf_items.append(
            (f"Source of Truth ({lang_code})", translated_sot, f"source_of_truth_{lang_code}.pdf")
        )
    for title, source, filename in pdf_items:
        try:
            _pipeline.create_pdf(title, source, filename)
        except Exception as exc:
            flow_logger.warning("PDF generation failed for %s: %s", filename, exc)

    # Session metadata
    _pipeline._save_session_metadata(
        output_dir=output_dir,
        topic_name=topic_name,
        language=language,
        language_config=language_config,
        session_roles=session_roles,
    )

    # -------------------------------------------------------------------
    # Phase 8: Audio Generation
    # -------------------------------------------------------------------
    p8_result = phase_8_audio(
        output_dir=output_dir_str,
        script_text=script_text,
        language_config=language_config,
    )

    # Write checkpoint.json for artifact completeness (Prefect handles resume
    # via result persistence, but the audit skill expects this file).
    try:
        from datetime import datetime as _dt
        ckpt = {
            "topic": topic_name,
            "language": language,
            "completed_phases": list(range(9)),
            "timestamp": _dt.now().isoformat(),
            "orchestrator": "prefect",
        }
        ckpt_path = _pipeline.output_path(output_dir, "checkpoint.json")
        ckpt_path.write_text(json.dumps(ckpt, indent=2, ensure_ascii=False))
    except Exception as exc:
        flow_logger.warning("Failed to write checkpoint.json: %s", exc)

    flow_logger.info("=" * 70)
    flow_logger.info("PIPELINE COMPLETE")
    flow_logger.info("Audio: %s (%.2f min)", p8_result["audio_file"], p8_result["duration_minutes"] or 0)
    flow_logger.info("Output: %s", output_dir_str)
    flow_logger.info("=" * 70)

    return {
        "output_dir": output_dir_str,
        "audio_file": p8_result["audio_file"],
        "duration_minutes": p8_result["duration_minutes"],
    }


# ---------------------------------------------------------------------------
# Inline correction helper (extracted from pipeline.py __main__ correction block)
# ---------------------------------------------------------------------------

def _run_inline_correction(
    audit_output: str,
    polished_text: str,
    editor_agent_ref,
    target_instruction: str,
    output_dir: Path,
    max_attempts: int = 2,
) -> Optional[str]:
    """Run script correction for HIGH-severity drift. Returns corrected text or None."""
    from dr2_podcast import pipeline as _pipeline
    from crewai import Crew, Task

    flow_logger = get_run_logger()
    orig_transitions = polished_text.count("[TRANSITION]") + polished_text.count("[INTRO_END]")
    corrected_script_text = None
    last_rejection_reason = ""

    flow_logger.info("HIGH-SEVERITY DRIFT — running script correction (max %d attempts)", max_attempts)

    for attempt in range(1, max_attempts + 1):
        retry_feedback = ""
        if last_rejection_reason:
            retry_feedback = (
                f"\n\nIMPORTANT — YOUR PREVIOUS CORRECTION WAS REJECTED:\n"
                f"Reason: {last_rejection_reason}\n"
                "Fix ONLY the minimum changes needed to address HIGH-severity items. "
                "Output the COMPLETE script with only the drifted passages changed.\n\n"
            )

        correction_task = Task(
            description=(
                f"The accuracy audit found HIGH-severity scientific drift in the podcast script.\n\n"
                f"AUDIT REPORT:\n{audit_output}\n\n"
                f"POLISHED SCRIPT:\n{polished_text}\n\n"
                "Fix ONLY the specific lines cited in the audit's 'Drift Instances Found' section.\n"
                "For each HIGH-severity issue: find the exact quote from 'Script says' and replace it "
                "with language consistent with 'Source-of-truth says'.\n"
                "Do NOT rewrite the entire script. Preserve all audio markers and structure.\n"
                f"{retry_feedback}"
                f"{target_instruction}"
            ),
            expected_output="Corrected podcast script with HIGH-severity drift fixed.",
            agent=editor_agent_ref,
        )
        try:
            correction_crew = Crew(agents=[editor_agent_ref], tasks=[correction_task], verbose=False)
            correction_result = correction_crew.kickoff()
            corrected = correction_result.raw if hasattr(correction_result, "raw") else str(correction_result)
            corrected_transitions = corrected.count("[TRANSITION]") + corrected.count("[INTRO_END]")

            if len(corrected) < len(polished_text) * 0.5:
                last_rejection_reason = (
                    f"Output too short ({len(corrected)} chars vs {len(polished_text)} original). "
                    "Output the COMPLETE script."
                )
                flow_logger.warning("Attempt %d: correction too short", attempt)
                continue
            elif orig_transitions > 0 and corrected_transitions < orig_transitions:
                last_rejection_reason = (
                    f"Lost audio markers ({orig_transitions} -> {corrected_transitions}). "
                    "Preserve ALL [TRANSITION] and [INTRO_END] markers."
                )
                flow_logger.warning("Attempt %d: lost audio markers", attempt)
                continue
            else:
                corrections_file = _pipeline.output_path(output_dir, "ACCURACY_CORRECTIONS.md")
                corrections_file.write_text(
                    "# Script Corrections Applied\n\n"
                    "HIGH-severity drift instances were corrected before audio generation.\n\n"
                    f"Correction succeeded on attempt {attempt}.\n\n"
                    f"## Original Audit\n{audit_output}\n"
                )
                flow_logger.info("Script correction applied (attempt %d)", attempt)
                corrected_script_text = corrected
                break
        except Exception as exc:
            last_rejection_reason = f"LLM correction raised exception: {exc}"
            flow_logger.warning("Attempt %d: correction failed: %s", attempt, exc)

    # Surgical fallback
    if corrected_script_text is None:
        high_items = []
        drift_blocks = re.split(r'(?=- \*\*Script says\*\*)', audit_output)
        for block in drift_blocks:
            if not re.search(r'\*\*Severity\*\*:\s*HIGH', block, re.IGNORECASE):
                continue
            s_match = re.search(r'\*\*Script says\*\*:\s*(.+?)(?:\n|$)', block)
            sot_match = re.search(r'\*\*Source-of-truth says\*\*:\s*(.+?)(?:\n|$)', block)
            if s_match and sot_match:
                sq = s_match.group(1).strip().strip('"').strip("'")
                sotq = sot_match.group(1).strip().strip('"').strip("'")
                if sq and sotq:
                    high_items.append((sq, sotq))

        if high_items:
            surgical = polished_text
            fixes_applied = 0
            for drifted, correct in high_items:
                if drifted in surgical:
                    surgical = surgical.replace(drifted, correct, 1)
                    fixes_applied += 1
            if fixes_applied > 0:
                corrections_file = _pipeline.output_path(output_dir, "ACCURACY_CORRECTIONS.md")
                corrections_file.write_text(
                    "# Script Corrections Applied (Surgical Replacement)\n\n"
                    f"LLM correction failed after {max_attempts} attempts.\n"
                    f"Applied deterministic replacement for {fixes_applied} HIGH-severity items.\n"
                    f"\n## Original Audit\n{audit_output}\n"
                )
                flow_logger.warning("Surgical fix: %d HIGH-severity item(s) replaced", fixes_applied)
                corrected_script_text = surgical

    return corrected_script_text
