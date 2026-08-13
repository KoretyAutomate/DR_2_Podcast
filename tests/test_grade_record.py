"""Structured GRADE: the assessment as a record, not as prose someone regex-scrapes.

PLAN.md sequencing item 3. `pipeline_sot.py`'s scraper defaults to "Not Determined" when its
pattern misses, the status map turns that into a mild "Under Evaluation", and the episode goes out
speaking a confidence nobody computed. A record either says what the level is or fails validation
upstream, where someone can see it.
"""

from __future__ import annotations

import asyncio
import json
import types
from typing import Any

import pytest

from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.research.clinical import Orchestrator

AFF_CASE = (
    "The pooled estimate favours treatment. Two of the four trials were open-label, which raises "
    "concerns about performance bias.\n"
)
FAL_CASE = (
    "The confidence interval spans both benefit and harm at the primary endpoint.\n"
    "No trial enrolled adults over 75, so the population is narrower than the question.\n"
)
ARTIFACTS = {"case:affirmative": AFF_CASE, "case:falsification": FAL_CASE}

#: The synthesis closes its GRADE Assessment with a block naming what it applied, so the
#: transcription can be checked for completeness rather than only for groundedness.
PROSE = (
    "### 3. GRADE Assessment\n"
    "Start at HIGH for RCTs. FINAL GRADE: MODERATE\n"
    "APPLIED MODIFIERS:\n"
    "- DOWNGRADE risk_of_bias 1 — two open-label trials\n"
    "\n"
    "### 4. Clinical Impact\n"
)


def _payload(**overrides: Any) -> dict:
    base = {
        "level": "moderate",
        "downgrades": [
            {
                "domain": "risk_of_bias",
                "steps": 1,
                "reason": "two open-label trials",
                "artifact_id": "case:affirmative",
                "quote": "Two of the four trials were open-label, which raises concerns about performance bias.",
            }
        ],
        "upgrades": [],
    }
    base.update(overrides)
    return base


def _orchestrator(domain: str = "clinical") -> Orchestrator:
    agent = Orchestrator.__new__(Orchestrator)
    agent.domain = domain
    agent.smart_client = object()
    agent.smart_model = "stub"
    return agent


def _answering(monkeypatch: pytest.MonkeyPatch, *payloads: Any) -> list[str]:
    """Stub the model with one response per attempt; records the system prompts it received."""
    from dr2_podcast.research import clinical

    prompts: list[str] = []
    queue = list(payloads)

    async def _create(client, **kwargs):
        prompts.append(kwargs["messages"][0]["content"])
        body = queue.pop(0) if queue else queue
        text = body if isinstance(body, str) else json.dumps(body)
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setattr(clinical, "gated_create", _create)
    return prompts


def _synthesise(orchestrator: Orchestrator, track: Any):
    """_grade_synthesis with a stand-in _TrackResult. Typed loosely on purpose: the real dataclass
    carries a search plan and an extraction list this test has no use for."""
    return orchestrator._grade_synthesis("topic", track, track, "math", "2026-08-13", log=lambda *a: None)


def _run_maybe(orchestrator: Orchestrator) -> dict | None:
    return asyncio.run(orchestrator._grade_record(PROSE, ARTIFACTS, log=lambda *a: None))


def _run(orchestrator: Orchestrator) -> dict:
    """The clinical path always returns a record or raises; social science uses _run_maybe."""
    record = _run_maybe(orchestrator)
    assert record is not None
    return record


# --------------------------------------------------------------------------- #
# The happy path, and what "grounded" means
# --------------------------------------------------------------------------- #
def test_a_stated_assessment_becomes_a_validated_record(monkeypatch: pytest.MonkeyPatch) -> None:
    _answering(monkeypatch, _payload())
    record = _run(_orchestrator())

    assert record["level"] == "moderate"
    assert [d["domain"] for d in record["downgrades"]] == ["risk_of_bias"]
    assert record["upgrades"] == []


def test_python_finds_the_offset_the_model_never_states(monkeypatch: pytest.MonkeyPatch) -> None:
    """Asking a model to count characters produces a number that satisfies the contract while
    pointing nowhere. The model supplies the span; the offset is Python's."""
    _answering(monkeypatch, _payload())
    locator = _run(_orchestrator())["downgrades"][0]["locator"]

    quoted = locator["quoted_span"]
    assert AFF_CASE[locator["char_offset"] : locator["char_offset"] + len(quoted)] == quoted


def test_the_record_travels_with_the_case_it_was_read_from(monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.schemas import grade_errors

    _answering(monkeypatch, _payload())
    assert grade_errors(_run(_orchestrator()), ARTIFACTS) == []


# --------------------------------------------------------------------------- #
# Fail closed — the whole reason the record exists
# --------------------------------------------------------------------------- #
def test_a_modifier_whose_quote_is_not_in_the_case_is_not_silently_dropped(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Dropping the modifier would change sum(downgrades[].steps) and therefore net_direction —
    the record would validate while describing a different assessment."""
    fabricated = _payload(
        downgrades=[
            {
                "domain": "risk_of_bias",
                "steps": 1,
                "reason": "invented",
                "artifact_id": "case:affirmative",
                "quote": "Every trial was triple-blinded and preregistered.",
            }
        ]
    )
    _answering(monkeypatch, fabricated, fabricated)
    with pytest.raises(ArtifactError, match="could not be stated as a record"):
        _run(_orchestrator())


def test_a_repeated_domain_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Two imprecision downgrades summed twice make net_direction wrong."""
    twice = _payload(
        downgrades=[
            {
                "domain": "imprecision",
                "steps": 1,
                "reason": "wide interval",
                "artifact_id": "case:falsification",
                "quote": "The confidence interval spans both benefit and harm at the primary endpoint.",
            },
            {
                "domain": "imprecision",
                "steps": 1,
                "reason": "again",
                "artifact_id": "case:falsification",
                "quote": "No trial enrolled adults over 75, so the population is narrower than the question.",
            },
        ]
    )
    _answering(monkeypatch, twice, twice)
    with pytest.raises(ArtifactError):
        _run(_orchestrator())


def test_an_unparseable_answer_never_becomes_not_determined(monkeypatch: pytest.MonkeyPatch) -> None:
    """The behaviour this replaces: the regex misses, the level reads "Not Determined", and the
    run continues to audio with a confidence nobody computed."""
    _answering(monkeypatch, "I could not assess this.", "still not JSON")
    with pytest.raises(ArtifactError):
        _run(_orchestrator())


def test_the_retry_loop_is_bounded_and_says_what_was_wrong(monkeypatch: pytest.MonkeyPatch) -> None:
    bad = _payload(level="excellent")
    prompts = _answering(monkeypatch, bad, bad, bad, bad)
    with pytest.raises(ArtifactError):
        _run(_orchestrator())

    assert len(prompts) == Orchestrator.GRADE_RECORD_ATTEMPTS, "an unbounded loop is a wedged stage"
    assert "was rejected" in prompts[1], "the second attempt must be told what failed"


def test_a_second_attempt_can_succeed(monkeypatch: pytest.MonkeyPatch) -> None:
    _answering(monkeypatch, _payload(level="excellent"), _payload())
    assert _run(_orchestrator())["level"] == "moderate"


# prepush codex 2026-08-13 [P1]: grade_errors checks the modifiers that ARE there — shape, one per
# domain, and whether each quotes its case. It cannot see one that is missing, and a dropped
# downgrade changes sum(downgrades[].steps), so net_direction says the evidence moved the
# confidence the other way. The synthesis therefore declares what it applied, in a fixed grammar,
# and the record has to be the whole of that declaration.
def test_a_dropped_downgrade_is_caught_even_though_the_record_is_grounded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prose = PROSE.replace(
        "- DOWNGRADE risk_of_bias 1 — two open-label trials\n",
        "- DOWNGRADE risk_of_bias 1 — two open-label trials\n"
        "- DOWNGRADE imprecision 2 — the interval spans benefit and harm\n",
    )
    # The payload is valid, quoted and schema-clean. It is simply not all of what the prose said.
    _answering(monkeypatch, _payload(), _payload())
    orchestrator = _orchestrator()
    with pytest.raises(ArtifactError, match="imprecision"):
        asyncio.run(orchestrator._grade_record(prose, ARTIFACTS, log=lambda *a: None))


def test_a_modifier_the_synthesis_never_applied_is_caught_too(monkeypatch: pytest.MonkeyPatch) -> None:
    """The other direction: a transcription that invents a downgrade moves the confidence as surely
    as one that drops it."""
    prose = PROSE.replace("- DOWNGRADE risk_of_bias 1 — two open-label trials\n", "NONE\n")
    _answering(monkeypatch, _payload(), _payload())
    with pytest.raises(ArtifactError, match="did not apply"):
        asyncio.run(_orchestrator()._grade_record(prose, ARTIFACTS, log=lambda *a: None))


def test_steps_have_to_match_not_just_the_domain(monkeypatch: pytest.MonkeyPatch) -> None:
    """One step versus two is the difference between MODERATE and LOW."""
    prose = PROSE.replace("DOWNGRADE risk_of_bias 1", "DOWNGRADE risk_of_bias 2")
    _answering(monkeypatch, _payload(), _payload())
    with pytest.raises(ArtifactError, match="risk_of_bias"):
        asyncio.run(_orchestrator()._grade_record(prose, ARTIFACTS, log=lambda *a: None))


def test_a_synthesis_that_declares_nothing_cannot_be_transcribed(monkeypatch: pytest.MonkeyPatch) -> None:
    """No block is not the same as declaring none: the first cannot be checked, and a record nobody
    can check is the state this whole pass exists to remove."""
    _answering(monkeypatch, _payload(level="moderate", downgrades=[], upgrades=[]),
               _payload(level="moderate", downgrades=[], upgrades=[]))
    with pytest.raises(ArtifactError, match="APPLIED MODIFIERS"):
        asyncio.run(
            _orchestrator()._grade_record("### GRADE\nFinal GRADE: MODERATE\n", ARTIFACTS, log=lambda *a: None)
        )


def test_a_synthesis_that_applied_nothing_transcribes_to_an_empty_record(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The control for the test above: NONE is a declaration, and it validates."""
    prose = PROSE.replace("- DOWNGRADE risk_of_bias 1 — two open-label trials\n", "NONE\n")
    _answering(monkeypatch, _payload(downgrades=[], upgrades=[]))
    record = asyncio.run(_orchestrator()._grade_record(prose, ARTIFACTS, log=lambda *a: None))
    assert record is not None
    assert record["downgrades"] == [] and record["upgrades"] == []


def test_the_prompt_asks_for_the_block_the_parser_reads() -> None:
    """The prompt/parser pin, as for the extraction prompt: a check nobody feeds is not a check."""
    import inspect

    from dr2_podcast.research.clinical import Orchestrator as O

    source = inspect.getsource(O._grade_synthesis)
    assert "APPLIED MODIFIERS" in source
    for domain in ("risk_of_bias", "imprecision", "large_effect", "dose_response"):
        assert domain in source, domain


# --------------------------------------------------------------------------- #
# Social science has no GRADE modifiers
# --------------------------------------------------------------------------- #
def test_social_science_produces_no_record_and_costs_no_call(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts = _answering(monkeypatch, _payload())
    assert _run_maybe(_orchestrator("social_science")) is None
    assert prompts == [], "an evidence-quality ladder has no modifier arithmetic to transcribe"


# --------------------------------------------------------------------------- #
# What the SOT does with it
# --------------------------------------------------------------------------- #
def test_the_sot_reads_the_record_rather_than_the_prose() -> None:
    from dr2_podcast.pipeline_sot import _extract_conclusion_status

    prose = "### GRADE Assessment\nFinal GRADE: Very Low\n"
    level, _status, _summary = _extract_conclusion_status(
        prose, grade_record={"schema_version": 1, "level": "high", "downgrades": [], "upgrades": []}
    )
    assert level == "High", "the record is the assessment; the prose is how it was explained"


def test_the_scrape_still_serves_a_run_that_has_no_record() -> None:
    from dr2_podcast.pipeline_sot import _extract_conclusion_status

    level, _status, _summary = _extract_conclusion_status("Final GRADE: Moderate\n")
    assert level == "Moderate"


def test_a_missed_scrape_is_exactly_the_failure_the_record_prevents() -> None:
    """Characterising the old behaviour, so the reason for the record stays visible."""
    from dr2_podcast.pipeline_sot import _extract_conclusion_status

    level, status, _summary = _extract_conclusion_status("The evidence is quite strong overall.\n")
    assert level == "Not Determined"
    assert status, "and it still produces a status, which is how it reached the episode unnoticed"


# --------------------------------------------------------------------------- #
# The contract has to survive the caller
# --------------------------------------------------------------------------- #
# prepush codex 2026-08-13 [P1]: _grade_synthesis wraps its model call in a broad handler that
# degrades to fallback prose. With the record call inside it, an ArtifactError became a warning, the
# adapter saw grade_record=None, and — because the artifact is optional for social science — a
# clinical stage completed with no grounded assessment. Fail-closed only counts where it is caught.
def test_a_record_that_cannot_be_grounded_stops_the_clinical_step(monkeypatch: pytest.MonkeyPatch) -> None:
    from dr2_podcast.research import clinical

    prose = "### GRADE Assessment\nFinal GRADE: Moderate\n"
    calls = {"n": 0}

    async def _create(client, **kwargs):
        calls["n"] += 1
        # First call is the prose synthesis; every later one is the record pass, answering with
        # something that cannot validate.
        text = prose if calls["n"] == 1 else "I am unable to produce that."
        message = types.SimpleNamespace(content=text)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setattr(clinical, "gated_create", _create)

    orchestrator = _orchestrator()
    monkeypatch.setattr(orchestrator, "_summarize_metadata_for_grade", lambda *a, **k: "")
    track = types.SimpleNamespace(
        case_report=AFF_CASE,
        plan=types.SimpleNamespace(pico={"population": "p"}),
        extractions=[],
        wide_net_total=0,
        screened_in=0,
        fulltext_ok=0,
        fulltext_err=0,
    )
    with pytest.raises(ArtifactError, match="could not be stated as a record"):
        asyncio.run(_synthesise(orchestrator, track))


def test_a_failed_prose_synthesis_still_degrades_rather_than_raising(monkeypatch: pytest.MonkeyPatch) -> None:
    """The control: the prose call keeps its degraded mode, which is what the handler is for."""
    from dr2_podcast.research import clinical

    async def _dies(client, **kwargs):
        raise RuntimeError("the backend is down")

    monkeypatch.setattr(clinical, "gated_create", _dies)

    orchestrator = _orchestrator()
    monkeypatch.setattr(orchestrator, "_summarize_metadata_for_grade", lambda *a, **k: "")
    track = types.SimpleNamespace(
        case_report=AFF_CASE,
        plan=types.SimpleNamespace(pico={}),
        extractions=[],
        wide_net_total=0,
        screened_in=0,
        fulltext_ok=0,
        fulltext_err=0,
    )
    out = asyncio.run(_synthesise(orchestrator, track))
    assert "synthesis failed" in out
    assert orchestrator.grade_record is None
