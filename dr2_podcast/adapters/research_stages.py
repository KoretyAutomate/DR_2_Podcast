"""Adapters for phases 0-3: framing, URL validation, translation.

See dr2_podcast.adapters for the shared state reconstruction and why it exists.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from dr2_podcast.adapters._common import (
    _classify_domain,
    drop_unproduced_optional_outputs,
    _domain_note,
    _prepare_run,
    require_outputs_rewritten,
    snapshot_outputs,
)
from dr2_podcast.approval import require_approval
from dr2_podcast.translation_audit import audit_translation
from dr2_podcast.artifacts import ArtifactError, write_atomic, write_json_atomic
from dr2_podcast.stages import register

logger = logging.getLogger(__name__)


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


@register("framing_prior")
def framing_prior(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Stage 1b — what we believed before searching, authored by Claude and frozen here.

    The one stage in this pipeline whose output is a JUDGEMENT rather than a derivation, which is
    why it is Claude's under the allocation table and why it runs before anything is searched. A
    prior computed after the evidence is in is hindsight in costume — and it would break the
    confirmation-bias argument the episodes themselves teach.

    It does NOT call _prepare_run: this stage needs no Crew and no vLLM handle, and building them
    would make the prior unwritable whenever the Smart backend happens to be down.
    """
    import tempfile

    from dr2_podcast.artifacts import read_json_strict, read_text_strict
    from dr2_podcast.claude_runner import author_artifacts
    from dr2_podcast.schemas import validate_framing_prior

    framing = read_text_strict(run_dir / "research/research_framing.md")
    artifact = "research/framing_prior.json"
    topic = run_config["topic"]

    # Authored in a directory that contains ONLY the framing, and that is a guarantee rather than an
    # instruction. The prompt says not to look at the evidence, but the turn holds Read, Glob and
    # Grep — and on a re-run (framing edited after research) the run directory is full of search
    # results, so a prohibition is all that stood between the prior and the evidence it is supposed
    # to precede (prepush codex 2026-08-20). A prior that COULD have read the findings is not
    # demonstrably a pre-search prior, whatever it says.
    with tempfile.TemporaryDirectory(prefix="dr2-prior-") as scratch:
        isolated = Path(scratch)
        (isolated / "research").mkdir()
        (isolated / "research/research_framing.md").write_text(framing, encoding="utf-8")
        author_artifacts(
            # The WHOLE framing. A prefix silently drops the questions and scope constraints that
            # come after it, and the prior would still be recorded as applying to the full topic —
            # the same silent-truncation class Step 0 made loud everywhere else in this pipeline.
            _PRIOR_PROMPT.format(topic=topic, artifact=artifact, framing=framing),
            run_dir=isolated,
            expected=(artifact,),
            # WRITE ONLY. A scratch cwd does not sandbox Read, Glob or Grep — they still take
            # absolute paths — so isolation by directory was isolation by hope (prepush codex
            # 2026-08-20). The framing is already in the prompt, so the turn needs to read nothing
            # at all, and a capability it does not hold is the only kind it cannot use.
            allowed_tools=PRIOR_ALLOWED_TOOLS,
        )
        authored = (isolated / artifact).read_text(encoding="utf-8")
    write_atomic(run_dir / artifact, authored)

    # Fail closed on the shape. A prior that will not validate is worse than none: step 9 would do
    # ordinal arithmetic over it and the episode would state the result.
    record = read_json_strict(run_dir / artifact)
    validate_framing_prior(record)
    if record["topic"].strip() != topic.strip():
        # A prior about a different question is worse than none: step 9 would update it against
        # this run's evidence and the episode would state the result (prepush codex 2026-08-17).
        raise ArtifactError(
            f"the prior says it judged {record['topic']!r}, but this run is about {topic!r}"
        )
    logger.info("frozen prior: %s (authored before any search ran)", record["prior_level"])


#: The prior turn writes and does nothing else. Every input it needs is in the prompt, so read
#: capability would only be an opportunity to see the evidence this artifact must precede.
PRIOR_ALLOWED_TOOLS: tuple[str, ...] = ("Write",)


#: Prose, not a slash command — `claude -p` passes the message as literal text, so a slash command
#: never reaches the skill resolver. Every component asks for its BASIS, because a prior without one
#: is a number somebody made up and step 9 would be arithmetic over an invention.
_PRIOR_PROMPT = """You are setting the FROZEN PRIOR for a research podcast episode, before any \
literature search has run. Nothing has been searched yet, and that is deliberate: a prior written \
after the evidence is hindsight in costume.

Topic: {topic}

The research framing for this episode:
{framing}

Write {artifact} as JSON with exactly these keys:
  schema_version: 1
  authored_by: "claude"
  prior_level: one of まだ分からない / 低い / 中程度 / 高い / ほぼ確実
  plausibility, known_mechanism, class_effect, base_rate: each an object
      {{"stated": "<what you believe, or null if nothing is known>",
        "basis": "<WHY you believe it, from background knowledge only>"}}
  topic: the topic above
  frozen_at: today's date in ISO 8601

Rules. Use background knowledge only — do NOT search, and do not read anything in this run \
directory other than the framing above. "Nothing is known" is a real answer: write null for \
`stated` and say so in `basis`. Every `basis` must be non-empty. Write the file and nothing else."""


@register("plan_search")
def plan_search(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Stage 1b — both tracks' search strategies, and NOT the search (PLAN.md Step 10).

    The stage exists so there is a point where a strategy is on disk and no search has run. Before
    the split there was none: planning, searching, screening and extracting happened in one call,
    both tracks fired together, and the strategy JSON was not written until after GRADE finished —
    so reading a strategy "before the search" was not something anyone could do.

    What happens next is not this stage's business: somebody reads the two files against the framing
    and writes ``meta/strategy_approval.json``. The `research` stage will not search without one.
    """
    import asyncio
    import dataclasses
    import os

    from dr2_podcast.artifacts import read_json_strict, read_text_strict
    from dr2_podcast.pipeline_flow import flow_or_module_logger
    from dr2_podcast.research.clinical import ResearchConfig
    from dr2_podcast.research.clinical import plan_search as plan_both

    pipeline = _prepare_run(run_dir, run_config)
    framing = read_text_strict(run_dir / "research/research_framing.md")
    declared = read_json_strict(run_dir / "research/domain_classification.json")["domain"]
    domain = declared if declared in ("clinical", "social_science") else "clinical"
    log = flow_or_module_logger()

    plans = asyncio.run(
        plan_both(
            topic=run_config["topic"],
            config=ResearchConfig(
                brave_api_key=os.getenv("BRAVE_API_KEY", ""), results_per_query=15, domain=domain
            ),
            framing_context=framing,
            log=log.info,
        )
    )
    for key, artifact in (("aff_plan", "search_strategy_aff.json"), ("fal_plan", "search_strategy_neg.json")):
        plan = plans.get(key)
        if plan is None:
            raise ArtifactError(f"the {key} track produced no search strategy; there is nothing to approve")
        write_json_atomic(run_dir / f"research/{artifact}", dataclasses.asdict(plan))
    # The decomposition travels in neither file, so the search stage rebuilds the plans from the
    # artifacts a reviewer actually read — not from anything held over in memory.
    logger.info("search strategies written; nothing has been searched. Approve them before running 'research'")
    del pipeline


@register("research")
def research(run_dir: Path, run_config: dict[str, Any]) -> None:
    """Phase 1 — the deep research pipeline, and the source of truth it builds.

    **It produces the SOT because phase 1 does.** ``build_imrad_sot`` is called inside the phase,
    on the live ``deep_reports`` dict, and that dict cannot cross a process boundary:
    ``_serialize_dataclass`` repr-stringifies the report objects, so ``audit`` round-trips as the
    literal text ``"namespace(report='…')"``. A separate ``sot`` stage was written against a
    reconstructed artifact and withdrawn when a test with the real builder proved the artifact
    cannot exist. Keeping the SOT here is not a workaround — it is where the pipeline actually
    computes it, and it removes the need for the artifact that could not be built.

    **It fails closed.** The phase catches every exception, logs "continuing without deep research",
    and returns empty strings — so a run whose research never happened goes on to write an episode
    from nothing. ``InsufficientEvidenceError`` still propagates unchanged: that one is a real
    terminal verdict about the topic, with a report written for the human who has to rephrase it.

    **It does NOT stage its writes**, unlike ``audit`` and ``audio``. ``run_deep_research`` writes
    incrementally into the run directory and reads ``extraction_cache.json`` back out of it; a
    staging tree would hide that cache and force full-text re-extraction of every paper, which is
    the expensive part of a ~28-minute stage. The manifest still catches a partial run: every
    declared output must exist for the stage to complete, a failure is recorded as one, and the
    fail-closed readers downstream reject anything truncated.
    """
    from dr2_podcast.artifacts import read_json_strict, read_text_strict
    from dr2_podcast.pipeline_flow import (
        _evidence_limited_prefix,
        _read_candidate_counts,
        _save_research_reports,
        _save_sources_json,
        flow_or_module_logger,
    )

    import asyncio
    import os

    from dr2_podcast import config as app_config
    from dr2_podcast.pipeline import InsufficientEvidenceError
    from dr2_podcast.research.clinical import ResearchConfig, run_deep_research

    pipeline = _prepare_run(run_dir, run_config)
    topic = run_config["topic"]
    before = snapshot_outputs(run_dir, "research")
    framing = read_text_strict(run_dir / "research/research_framing.md")
    declared = read_json_strict(run_dir / "research/domain_classification.json")["domain"]
    domain = declared if declared in ("clinical", "social_science") else "clinical"
    log = flow_or_module_logger()

    # BEFORE anything is searched. The approval covers the framing and the prior as well as the two
    # strategy files, so a strategy edited after approval, a framing edited after approval, or a
    # prior appearing after approval all stop the run here — with zero search calls issued, which is
    # what Step 10's exit criterion asserts.
    require_approval(run_dir)
    plans = _approved_plans(run_dir)

    # Held as Any, as the flow holds it: DeepResearchResult is a TypedDict whose values are typed
    # `object`, so narrowing it here only moves the complaint from the call to the attribute access.
    reports: Any = asyncio.run(
        run_deep_research(
            topic=topic,
            config=ResearchConfig(
                brave_api_key=os.getenv("BRAVE_API_KEY", ""),
                results_per_query=15,
                domain=domain,
            ),
            framing_context=framing,
            output_dir=str(run_dir),
            plans=plans,
        )
    )

    aff_candidates, neg_candidates = _read_candidate_counts(run_dir, log)
    if aff_candidates == 0:
        pipeline._write_insufficient_evidence_report(topic, 0, neg_candidates, run_dir)
        raise InsufficientEvidenceError(
            f"Affirmative track: 0 candidates for {topic!r}. Adversarial found {neg_candidates}. "
            f"See insufficient_evidence_report.md for suggested rephrasing."
        )

    _save_research_reports(reports, run_dir, log)
    _save_sources_json(reports, run_dir, log)

    sot = pipeline.build_imrad_sot(topic=topic, reports=reports, domain=domain)
    if not sot or not sot.strip():
        raise ArtifactError("the source of truth came back empty; there is nothing to write an episode from")
    if 0 < aff_candidates < app_config.EVIDENCE_LIMITED_THRESHOLD:
        log.warning("evidence limited: %d affirmative candidates", aff_candidates)
        sot = _evidence_limited_prefix(aff_candidates) + sot
    write_atomic(run_dir / "research/source_of_truth.md", sot)
    # The structured GRADE record beside the prose it was read from (sequencing item 3). Written
    # only when there is one: social science has no GRADE modifiers, and a stale record from a
    # previous clinical run must not be recorded as this run's output.
    produced = ["research/source_of_truth.md"]
    record = (reports.get("pipeline_data") or {}).get("grade_record")
    if domain != "social_science" and not record:
        # A clinical run without a record has not assessed its evidence, whatever prose it produced.
        # _grade_record raises when it cannot ground the assessment, but the synthesis CALL itself
        # has a degraded mode — a timeout returns "GRADE synthesis failed. Raw inputs below." and
        # never reaches the record pass (prepush codex 2026-08-13). The artifact is optional in the
        # graph because social science has no GRADE modifiers, so its absence cannot carry this;
        # the stage boundary is where "completed" is decided, so it is decided here.
        raise ArtifactError(
            "the clinical run produced no structured GRADE record, so nothing assessed the evidence "
            "this episode is about to speak with confidence. See research/grade_synthesis.md for "
            "what step 7 did manage to write."
        )
    if record:
        write_json_atomic(run_dir / "research/grade_synthesis.json", record, schema="grade")
        produced.append("research/grade_synthesis.json")
    if _write_step_pack(run_dir, reports, sot, domain):
        produced.append("research/step_pack.json")
    drop_unproduced_optional_outputs(run_dir, "research", produced)
    # Existence is not authorship for a stage that writes in place: a rerun producing fewer
    # artifacts would otherwise complete on a mix of this run's and the previous one's.
    require_outputs_rewritten(run_dir, "research", before)


def _approved_plans(run_dir: Path) -> dict[str, Any]:
    """The two strategies as the approver read them, rebuilt from the artifacts on disk.

    From the FILES, never from anything held in memory: the approval is a statement about those
    bytes, so the search has to run against those bytes or the approval covers something else.
    """
    from dr2_podcast.artifacts import read_json_strict
    from dr2_podcast.pipeline import _restore_tiered_search_plan

    plans: dict[str, Any] = {"decomposition": {}}
    for key, artifact in (("aff_plan", "search_strategy_aff.json"), ("fal_plan", "search_strategy_neg.json")):
        plan = _restore_tiered_search_plan(read_json_strict(run_dir / f"research/{artifact}"))
        if plan is None or not getattr(plan, "tier1", None):
            raise ArtifactError(f"research/{artifact} is not a usable search strategy")
        plans[key] = plan
    return plans


def _write_step_pack(run_dir: Path, reports: Any, sot: str, domain: str) -> bool:
    """The nine-step projection of the SOT this stage just wrote (PLAN.md Step 9b).

    Validated against the SOT it projects BEFORE it is written: every non-absent answer has to name
    a section that is actually in that document. The pack is derived with no LLM anywhere, so a
    validation failure here means the projection and the document disagree — which is the one thing
    a projection must never do, and writing it anyway would create the second source of truth the
    design exists to prevent.
    """
    from dr2_podcast.research.step_pack import build_step_pack
    from dr2_podcast.schemas import validate_step_pack

    pipeline_data = (reports.get("pipeline_data") or {}) if isinstance(reports, dict) else {}
    extractions = list(pipeline_data.get("aff_extractions") or []) + list(
        pipeline_data.get("fal_extractions") or []
    )
    if not extractions:
        logger.warning("no extractions, so there is nothing to project into a step pack")
        return False

    pack = build_step_pack(pipeline_data=pipeline_data, extractions=extractions, sot=sot, domain=domain)
    validate_step_pack(pack, {"research/source_of_truth.md": sot})
    write_json_atomic(run_dir / "research/step_pack.json", pack, schema="step_pack")
    answered = sum(1 for step in pack["steps"].values() if step["sufficiency"] != "absent")
    logger.info("step pack: %d of %d steps answered from this run's evidence", answered, len(pack["steps"]))
    return True


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

    import hashlib

    raw = run_dir / "research/research_sources.json"
    sources = read_json_strict(raw)
    urls = sorted({url for url in _iter_urls(sources) if url})
    results = validate_multiple_urls_parallel(urls, max_workers=15) if urls else {}
    write_json_atomic(run_dir / "research/url_validation_results.json", results)
    write_json_atomic(
        run_dir / "research/research_sources_validated.json", _without_broken(sources, results)
    )
    # The hash of the library this was filtered FROM. pipeline.research_sources_file() serves the
    # validated copy only while this still matches, so "derived from the current sources" is a fact
    # it can check rather than an ordering it has to trust.
    write_atomic(
        run_dir / "research/research_sources_validated.sha256",
        hashlib.sha256(raw.read_bytes()).hexdigest() + "\n",
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
            kept = [e for e in entries if not (isinstance(e, dict) and e.get("url") in broken)]
            # Renumbered, because the listing and the lookup are two different things that have to
            # mean the same number: pipeline.py:1440 prints each entry's stored `index`, while
            # read_research_source resolves the number it is given POSITIONALLY. Removing a
            # non-final entry leaves a gap, and an agent asking for the index it was shown then
            # gets a DIFFERENT source — the wrong evidence attached to a claim, silently, or an
            # out-of-range error (prepush codex 2026-08-13).
            filtered[role] = [
                {**e, "index": i} if isinstance(e, dict) and "index" in e else e
                for i, e in enumerate(kept)
            ]
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

    An English run writes nothing and completes — and removes any translated SOT left behind by an
    earlier implementation or a migration, because an optional output means "this run may not
    produce one", not "keep whatever was there". A
    translation that comes back empty RAISES, though — the phase returns None and carries on, which
    leaves a Japanese episode built from an English source of truth.
    """
    from dr2_podcast.artifacts import read_text_strict

    language = run_config["language"]
    substitutions = {"language": language}
    if language == "en":
        # Not just "produce nothing": REMOVE a translated SOT left by an earlier implementation, a
        # manual copy or an interrupted migration. Left there, Manifest.complete() would record it as
        # this execution's optional output and blueprint would treat it as this run's translation.
        drop_unproduced_optional_outputs(run_dir, "translate", [], substitutions)
        return

    pipeline = _prepare_run(run_dir, run_config)
    sot = read_text_strict(run_dir / "research/source_of_truth.md")
    translated = pipeline._translate_sot_pipelined(sot, language, pipeline.language_config)
    if not translated or not translated.strip():
        raise ArtifactError(
            f"translation to {language!r} produced nothing. The monolithic phase returns None and "
            f"continues, which builds the episode from a source of truth in the wrong language."
        )
    # PLAN.md Step 11. Nothing checked the translation, and the translated SOT is the basis for
    # every claim in a Japanese episode — a moved number or a reassigned PMID poisons everything
    # downstream, silently. What Python can prove it checks here, before the file is written.
    audit = audit_translation(sot, translated)
    if not audit["ok"]:
        raise ArtifactError(
            f"the {language!r} translation does not carry the same claims as the source of truth: "
            + "; ".join(audit["errors"][:3])
            + ". The episode would state numbers the research does not support."
        )
    logger.info(
        "translation audit: %d claim(s) checked, %d token(s) matched. NOT checked: %s",
        audit["checked_claims"],
        audit["checked_tokens"],
        audit["not_checked"],
    )
    produced = f"research/source_of_truth_{language}.md"
    write_atomic(run_dir / produced, translated)
    drop_unproduced_optional_outputs(run_dir, "translate", [produced], substitutions)
