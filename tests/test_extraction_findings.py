"""Step 9a: the extraction produces findings[] that Python vouches for.

Three things the model does not get to decide, and each is a test here: the finding_key (computed
from the identity tuple, never authored), the char_offset (found by searching the source, because a
model asked for an offset is being asked to count), and whether a quote is real (a span that is not
in the source drops the finding).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from dr2_podcast.research.clinical import (
    DeepExtraction,
    Finding,
    FundingBlock,
    ResearchAgent,
    locate_span,
)
from dr2_podcast.schemas import compute_finding_key, finding_errors, verify_locator_span
from tests._extraction_fixtures import SOURCE, _raw_finding, _record

# --------------------------------------------------------------------------- #
# locate_span — Python owns the offset
# --------------------------------------------------------------------------- #
def test_a_span_is_found_at_its_real_offset() -> None:
    hit = locate_span(SOURCE, "Absolute risk reduction 5.0%")
    assert hit is not None
    offset, literal = hit
    assert literal == "Absolute risk reduction 5.0%"
    assert SOURCE[offset : offset + len(literal)] == literal


def test_a_span_survives_the_rewrapping_extraction_does() -> None:
    """Extracted full text re-wraps lines; the quote is still the same quote."""
    text = "results\nAbsolute risk\nreduction 5.0% here"
    hit = locate_span(text, "Absolute risk reduction 5.0%")
    assert hit is not None
    offset, literal = hit
    assert offset == len("results\n")
    # prepush codex 2026-08-13: the returned literal is the SOURCE's wrapping, not the model's.
    # Storing the model's spaces at this offset builds a locator that can never verify.
    assert literal == "Absolute risk\nreduction 5.0%"
    assert text[offset : offset + len(literal)] == literal


def test_a_span_that_is_not_there_is_not_found() -> None:
    assert locate_span(SOURCE, "Absolute risk reduction 50.0%") is None
    assert locate_span(SOURCE, "") is None
    assert locate_span("", "anything") is None


# --------------------------------------------------------------------------- #
# The prompt and the parser have to agree
# --------------------------------------------------------------------------- #
# prepush codex 2026-08-13, and this one would have shipped: the parser required `identity_quote`
# while the prompt never asked for it, so EVERY finding would have been dropped on every real run —
# silently, with an empty findings[] and no error. The unit tests could not see it because they feed
# _build_findings directly. This pins the two together.
@pytest.mark.parametrize(
    "key",
    [
        "findings",
        "identity_quote",
        "quote",
        "population",
        "intervention",
        "comparator",
        "endpoint",
        "timepoint",
        "direction",
        "is_primary",
        "control_event_rate",
        "experimental_event_rate",
        "outcome_is_adverse",
        "trial_registration",
        "author_group",
        "funding_raw",
        "funding_category",
        "funding_disclosure",
        "funding_quote",
    ],
)
def test_every_field_the_parser_reads_is_asked_for_in_the_prompt(key: str) -> None:
    import inspect

    from dr2_podcast.research.clinical import ResearchAgent

    # The extraction function itself, which is where both prompt branches live. Slicing the module
    # by the first prompt header picked up the social-science branch instead.
    prompt_source = inspect.getsource(ResearchAgent._deep_extract_batch)
    assert f'"{key}"' in prompt_source, f"{key} is read by the parser but never requested in the prompt"


# --------------------------------------------------------------------------- #
# _build_findings
# --------------------------------------------------------------------------- #
def test_a_finding_is_built_with_a_python_computed_key() -> None:
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, SOURCE, "pmid:12345678")
    assert len(built) == 1
    finding = built[0]
    assert finding.finding_key == compute_finding_key(
        {
            "population": "adults aged 50 or older",
            "intervention": "vitamin D 800 IU/day",
            "comparator": "placebo",
            "endpoint": "hip fracture",
            "timepoint": "12 months",
        }
    )
    assert len(finding.finding_key) == 40


def test_a_model_supplied_key_is_ignored() -> None:
    """The key is Python's to produce; accepting one would make replication grouping semantic again."""
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(finding_key="0" * 40)]}, SOURCE, "pmid:12345678"
    )
    assert built[0].finding_key != "0" * 40


def test_the_locator_offset_points_at_the_quote() -> None:
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, SOURCE, "pmid:12345678")
    locator = built[0].locators[0]
    span = locator["quoted_span"]
    assert SOURCE[locator["char_offset"] : locator["char_offset"] + len(span)] == span


def test_a_fabricated_quote_drops_the_finding() -> None:
    """An unverifiable quote is not evidence. The paper never says 50%."""
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(quote="Absolute risk reduction 50.0% (95% CI 40 to 60)")]},
        SOURCE,
        "pmid:12345678",
    )
    assert built == []


def test_a_finding_with_no_quote_is_dropped() -> None:
    built = ResearchAgent._build_findings({"findings": [_raw_finding(quote=None)]}, SOURCE, "pmid:12345678")
    assert built == []


def test_a_finding_with_no_endpoint_is_dropped() -> None:
    built = ResearchAgent._build_findings({"findings": [_raw_finding(endpoint=None)]}, SOURCE, "pmid:12345678")
    assert built == []


def test_two_endpoints_of_one_paper_stay_two_findings() -> None:
    """A paper is not a finding: benefit on one endpoint and a null result on another."""
    falls = _raw_finding(
        endpoint="falls",
        direction="null_result",
        value=None,
        unit=None,
        ci_low=None,
        ci_high=None,
        p_value=0.41,
        is_primary=False,
        control_event_rate=None,
        experimental_event_rate=None,
        outcome_is_adverse=None,
        quote="No significant difference in falls was observed between groups (p=0.41)",
    )
    built = ResearchAgent._build_findings({"findings": [_raw_finding(), falls]}, SOURCE, "pmid:12345678")
    assert len(built) == 2
    assert built[0].finding_key != built[1].finding_key
    assert {f.direction for f in built} == {"decrease", "null_result"}


def test_a_built_finding_satisfies_the_contract_unmodified() -> None:
    """The end-to-end point of 9a: what the extractor produces is what the schema accepts, with no
    adjustment. An earlier version of this test had to rewrite the locator's fields before
    validating — which was the tell that the extractor was not actually meeting its contract."""
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, SOURCE, "pmid:12345678")
    assert finding_errors(built[0].to_dict(), {"pmid:12345678": SOURCE}) == []


def test_the_two_quotes_cover_what_each_actually_says() -> None:
    """One span cannot honestly substantiate both who was studied and what happened to them."""
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, SOURCE, "pmid:12345678")
    identity_locator, result_locator = built[0].locators
    assert set(identity_locator["fields"]) == {"population", "intervention", "comparator", "timepoint"}
    assert "endpoint" in result_locator["fields"]
    assert {"value", "ci_low", "p_value"} <= set(result_locator["fields"])
    assert "population" not in result_locator["fields"]


def test_a_finding_with_no_identity_quote_is_dropped() -> None:
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(identity_quote=None)]}, SOURCE, "pmid:12345678"
    )
    assert built == []


def test_a_null_result_covers_only_the_fields_it_has() -> None:
    """Coverage follows what the record carries: no value, no locator naming one."""
    built = ResearchAgent._build_findings(
        {
            "findings": [
                _raw_finding(
                    endpoint="falls",
                    direction="null_result",
                    value=None,
                    unit=None,
                    ci_low=None,
                    ci_high=None,
                    control_event_rate=None,
                    experimental_event_rate=None,
                    outcome_is_adverse=None,
                    quote="No significant difference in falls was observed between groups (p=0.41)",
                )
            ]
        },
        SOURCE,
        "pmid:12345678",
    )
    assert finding_errors(built[0].to_dict(), {"pmid:12345678": SOURCE}) == []
    result_fields = set(built[0].locators[1]["fields"])
    assert "value" not in result_fields
    assert {"endpoint", "direction", "p_value"} <= result_fields


def test_no_findings_at_all_is_an_empty_list_not_a_crash() -> None:
    assert ResearchAgent._build_findings({}, SOURCE, "pmid:1") == []
    assert ResearchAgent._build_findings({"findings": None}, SOURCE, "pmid:1") == []
    assert ResearchAgent._build_findings({"findings": ["not a dict"]}, SOURCE, "pmid:1") == []


# prepush codex 2026-08-13. The default identity quote in _raw_finding() spans a line wrap in
# SOURCE, so this fires on exactly the defect it was written for: an offset into the wrapped source
# stored beside the model's unwrapped spelling of the quote. The locator schema accepts that pair —
# only reading the artifact refutes it, which is what verify_locator_span does.
def test_every_locator_a_finding_carries_verifies_against_the_source() -> None:
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, SOURCE, "pmid:12345678")
    assert built
    for locator in built[0].locators:
        assert verify_locator_span(locator, SOURCE), locator


def test_locators_verify_when_the_model_unwraps_a_quote_the_source_wrapped() -> None:
    wrapped = SOURCE.replace("Absolute risk reduction 5.0%", "Absolute risk\nreduction 5.0%")
    built = ResearchAgent._build_findings({"findings": [_raw_finding()]}, wrapped, "pmid:12345678")
    assert built, "a quote differing only in whitespace is the same quote"
    for locator in built[0].locators:
        assert verify_locator_span(locator, wrapped), locator


# prepush codex 2026-08-13: bool("false") is True. The primary finding is what supplies the
# paper-level CER/EER the deterministic ARR/NNT math runs on, so a string here moves the numbers.
@pytest.mark.parametrize("stated", ["false", "true", "no", 0, 1, "", None])
def test_only_a_real_json_boolean_makes_a_finding_primary(stated: object) -> None:
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(is_primary=stated)]}, SOURCE, "pmid:12345678"
    )
    assert built
    assert built[0].is_primary is False, f"{stated!r} is not the model stating a boolean"


def test_a_json_boolean_true_makes_a_finding_primary() -> None:
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(is_primary=True)]}, SOURCE, "pmid:12345678"
    )
    assert built[0].is_primary is True


# prepush codex 2026-08-13: schemas/ was written and then called only by its own tests. A contract
# nothing on the production path invokes is documentation, not a guarantee — these pin the call.
@pytest.mark.parametrize(
    ("override", "why"),
    [
        ({"direction": "sideways"}, "direction is a closed enum"),
        ({"population": ""}, "an identity field cannot be blank"),
        ({"experimental_event_rate": None}, "one of CER/EER without the other computes nothing"),
        ({"control_event_rate": 1.5}, "an event rate outside [0, 1] is not a rate"),
    ],
)
def test_a_finding_that_fails_the_contract_never_reaches_the_math(override: dict, why: str) -> None:
    built = ResearchAgent._build_findings(
        {"findings": [_raw_finding(**override)]}, SOURCE, "pmid:12345678"
    )
    assert built == [], why


def test_one_invalid_finding_does_not_cost_the_paper_its_valid_ones() -> None:
    good = _raw_finding()
    bad = _raw_finding(endpoint="falls", direction="sideways", quote="No significant difference in falls")
    built = ResearchAgent._build_findings({"findings": [bad, good]}, SOURCE, "pmid:12345678")
    assert [f.endpoint for f in built] == ["hip fracture"]


# --------------------------------------------------------------------------- #
# The record as a whole
# --------------------------------------------------------------------------- #
def test_the_new_fields_round_trip_through_to_dict() -> None:
    record = DeepExtraction(
        pmid="1",
        doi=None,
        title="t",
        url="u",
        findings=[Finding(population="p", intervention="i", comparator="c", endpoint="e")],
        funding=FundingBlock(funding_category="undisclosed", funding_disclosure="undisclosed"),
        trial_registration="NCT01234567",
        author_group="Tanaka H; Osaka University",
    )
    d = record.to_dict()
    assert d["trial_registration"] == "NCT01234567"
    assert d["author_group"] == "Tanaka H; Osaka University"
    assert d["funding"]["funding_disclosure"] == "undisclosed"
    assert d["findings"][0]["endpoint"] == "e"
    assert Finding.from_dict(d["findings"][0]).endpoint == "e"
    assert FundingBlock.from_dict(d["funding"]).funding_category == "undisclosed"


# --------------------------------------------------------------------------- #
# The batch, driven end to end with the model stubbed out
# --------------------------------------------------------------------------- #
# prepush codex 2026-08-13. Two defects lived here, both invisible to a unit test of the builders:
# the paper-level CER/EER fell back to the model's own top-level numbers whenever no finding
# survived — unquoted, unlocatable, and fed straight to the deterministic ARR/NNT math — and the
# resulting record was cached, so the next run inherited the failure without retrying.
def _run_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    payload: dict,
    *,
    social: bool = False,
    source_text: str = SOURCE,
    respond: Any = None,
) -> Any:
    import asyncio
    import json as _json
    import types

    from dr2_podcast.research import clinical

    agent = ResearchAgent.__new__(ResearchAgent)
    agent.smart_client = object()
    agent.smart_model = "stub"
    agent._domain = "social_science" if social else "clinical"

    async def _fake_create(client, **kwargs):
        message = types.SimpleNamespace(content=_json.dumps(payload))
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])

    monkeypatch.setattr(clinical, "gated_create", respond or _fake_create)
    article = types.SimpleNamespace(full_text=source_text)
    return asyncio.run(
        agent._deep_extract_batch(
            [article], [_record()], {}, log=lambda *a, **k: None, output_dir=str(tmp_path)
        )
    )[0]


def test_a_paper_with_no_verified_finding_contributes_no_event_rates(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "study_design": "RCT",
        "raw_facts": "something readable",
        # paper-level rates the model volunteered, with no quote behind either of them
        "control_event_rate": 0.15,
        "experimental_event_rate": 0.10,
        "outcome_is_adverse": True,
        "findings": [],
    }
    extraction = _run_batch(monkeypatch, tmp_path, payload)
    assert extraction.findings == []
    assert extraction.control_event_rate is None, "an unquoted rate must not reach the ARR/NNT math"
    assert extraction.experimental_event_rate is None
    assert extraction.outcome_is_adverse is None


def test_an_unverified_clinical_extraction_is_not_cached(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _run_batch(monkeypatch, tmp_path, {"study_design": "RCT", "raw_facts": "readable", "findings": []})
    cached = list(tmp_path.rglob("*extraction*cache*")) + list(tmp_path.rglob("*.json"))
    for path in cached:
        assert "12345678" not in path.read_text(), f"{path.name} remembers an unverified extraction"


def test_a_verified_clinical_extraction_is_cached(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """The control: the same path caches normally when a finding survives, so the test above is
    measuring the guard and not some unrelated reason nothing was written."""
    payload = {"study_design": "RCT", "raw_facts": "readable", "findings": [_raw_finding()]}
    extraction = _run_batch(monkeypatch, tmp_path, payload)
    assert extraction.findings, "the fixture finding must survive, or the control proves nothing"
    assert extraction.control_event_rate == 0.15
    written = [p for p in tmp_path.rglob("*.json") if "12345678" in p.read_text()]
    assert written, "a verified extraction is worth remembering"


# prepush codex 2026-08-13: the v2 cache is keyed by PMID/DOI alone. The same paper fetched from a
# different provider is a different string, so a restored locator's offsets can point at the wrong
# words while its CER/EER keep feeding the deterministic math.
def test_a_cache_entry_is_reused_when_it_still_verifies(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {"study_design": "RCT", "raw_facts": "readable", "findings": [_raw_finding()]}
    first = _run_batch(monkeypatch, tmp_path, payload)
    assert first.findings

    def _must_not_be_called(client, **kwargs):
        raise AssertionError("a verifying cache entry must be reused, not re-extracted")

    from dr2_podcast.research import clinical

    monkeypatch.setattr(clinical, "gated_create", _must_not_be_called)
    second = _run_batch(monkeypatch, tmp_path, payload)
    assert [f.finding_key for f in second.findings] == [f.finding_key for f in first.findings]


def test_a_cache_entry_that_no_longer_verifies_is_re_extracted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {"study_design": "RCT", "raw_facts": "readable", "findings": [_raw_finding()]}
    assert _run_batch(monkeypatch, tmp_path, payload).findings

    # The same PMID, fetched this run from a provider whose text says something else entirely.
    other_text = "Unrelated paper about tides. " * 40
    reextracted = _run_batch(monkeypatch, tmp_path, payload, source_text=other_text)
    assert reextracted.findings == [], (
        "the cached locators do not hold against this text, and the re-extraction's quotes are not "
        "in it either — so the paper contributes no verified finding"
    )
    assert reextracted.control_event_rate is None


# prepush codex 2026-08-13: the same argument as the findings above, for the other locator on the
# record. A funding block quoted from the paper points into that paper's text; a cache hit against
# text from a different provider must not carry the quote through unchecked.
def test_a_cache_entry_whose_funding_quote_no_longer_holds_is_re_extracted(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    payload = {
        "study_design": "RCT",
        "raw_facts": "readable",
        "findings": [],
        "funding_raw": "Supported by grant R01-AG000000 from the National Institute on Aging.",
        "funding_quote": "Supported by grant R01-AG000000 from the National Institute on Aging.",
        "funding_category": "government",
        "funding_disclosure": "disclosed",
    }
    # Social, so an empty findings list is legitimate and the entry reaches the cache on its funding.
    first = _run_batch(monkeypatch, tmp_path, payload, social=True)
    assert first.funding.funding_source_type == "extracted_text"

    kept_source = SOURCE.replace("National Institute on Aging", "Acme Pharma")
    second = _run_batch(monkeypatch, tmp_path, payload, social=True, source_text=kept_source)
    assert second.funding.funding_source_type != "extracted_text", (
        "the cached funding quote does not appear in the text this run fetched"
    )


# prepush codex 2026-08-13: the template is assembled from adjacent Python string literals, and a
# description wrapped across two source lines renders as `"...or null" " if the paper is silent"` —
# two quoted fragments where JSON allows one string. A model copying the shape it was shown returns
# something the parser rejects, on every paper.
def _prompt_json_template(prompt: str) -> str:
    start = prompt.index("{", prompt.index("Return ONLY valid JSON:"))
    depth = 0
    for i in range(start, len(prompt)):
        if prompt[i] == "{":
            depth += 1
        elif prompt[i] == "}":
            depth -= 1
            if depth == 0:
                return prompt[start : i + 1]
    raise AssertionError("the template's braces never close")


@pytest.mark.parametrize("social", [False, True])
def test_the_prompt_shows_the_model_a_template_that_is_valid_json(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, social: bool
) -> None:
    import json as _json

    seen: dict[str, str] = {}

    async def _capture(client, **kwargs):
        seen["prompt"] = kwargs["messages"][0]["content"]
        raise RuntimeError("stop here; the prompt is what is under test")

    _run_batch(monkeypatch, tmp_path, {}, social=social, respond=_capture)

    assert "prompt" in seen, "the extraction call was never made"
    _json.loads(_prompt_json_template(seen["prompt"]))
