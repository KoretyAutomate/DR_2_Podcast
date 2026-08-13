"""The funding block: two provenances, and only one of them can carry a locator."""

from __future__ import annotations

import pytest

from dr2_podcast.research.clinical import (
    DeepExtraction,
    Finding,
    FundingBlock,
    PaperMetadata,
    ResearchAgent,
)
from tests._extraction_fixtures import SOURCE, _record

# --------------------------------------------------------------------------- #
# _build_funding — the two provenances
# --------------------------------------------------------------------------- #
def test_funding_quoted_from_the_paper_carries_a_locator() -> None:
    block = ResearchAgent._build_funding(
        {
            "funding_raw": "Supported by grant R01-AG000000 from the National Institute on Aging.",
            "funding_quote": "Supported by grant R01-AG000000 from the National Institute on Aging.",
            "funding_category": "government",
            "funding_disclosure": "disclosed",
        },
        _record(),
        SOURCE,
        "pmid:12345678",
    )
    assert block.funding_source_type == "extracted_text"
    assert block.funding_locator is not None
    assert block.funding_locator["fields"] == ["funding_raw"]


def test_funding_from_api_metadata_has_no_locator() -> None:
    """It exists nowhere in the paper, so it cannot satisfy the locator contract — and saying so is
    the whole point of splitting provenance from value."""
    record = _record()
    record.paper_metadata = PaperMetadata(funding_sources=["National Institute on Aging"])
    block = ResearchAgent._build_funding({}, record, SOURCE, "pmid:12345678")
    assert block.funding_source_type == "api_metadata"
    assert block.funding_locator is None
    assert block.funding_raw == "National Institute on Aging"


# prepush codex 2026-08-13: a model-supplied funder whose quote could not be located was kept and
# labelled api_metadata — fabricating the very provenance the split exists to guarantee.
def test_an_unverifiable_funder_is_discarded_not_relabelled(caplog: pytest.LogCaptureFixture) -> None:
    block = ResearchAgent._build_funding(
        {
            "funding_raw": "Acme Pharma",
            "funding_quote": "Wholly funded by Acme Pharma.",  # nowhere in SOURCE
            "funding_category": "industry",
            "funding_disclosure": "disclosed",
        },
        _record(),
        SOURCE,
        "pmid:12345678",
    )
    assert block.funding_source_type != "api_metadata", "nothing from an API produced this"
    assert block.funding_raw is None
    assert block.funding_disclosure == "unknown"


def test_an_unverifiable_funder_still_falls_back_to_real_api_metadata() -> None:
    record = _record()
    record.paper_metadata = PaperMetadata(funding_sources=["National Institute on Aging"])
    block = ResearchAgent._build_funding(
        {"funding_raw": "Acme Pharma", "funding_quote": "Wholly funded by Acme Pharma."},
        record,
        SOURCE,
        "pmid:12345678",
    )
    assert block.funding_source_type == "api_metadata"
    assert block.funding_raw == "National Institute on Aging", "the API's value, not the model's"


# prepush codex 2026-08-13: the API fallback returned disclosed + api_metadata + category unknown,
# a combination the funding contract's legal-combination table did not admit — and nothing on the
# production path was validating it, so the illegal record was built and cached.
def test_the_api_branch_never_inherits_the_models_guess_at_a_category() -> None:
    """The model's category describes the funding statement it could not quote — the one discarded
    a few lines earlier. Carrying it onto an API-supplied name labels a guess as a classification."""
    record = _record()
    record.paper_metadata = PaperMetadata(funding_sources=["National Institute on Aging"])
    block = ResearchAgent._build_funding(
        {
            "funding_raw": "Acme Pharma",
            "funding_quote": "Wholly funded by Acme Pharma.",  # nowhere in SOURCE
            "funding_category": "industry",
            "funding_disclosure": "disclosed",
        },
        record,
        SOURCE,
        "pmid:12345678",
    )
    assert block.funding_source_type == "api_metadata"
    assert block.funding_category == "unknown", "an unquotable 'industry' label must not survive"


def test_an_unclassified_api_funder_is_still_a_legal_block() -> None:
    from dr2_podcast.schemas import funding_errors

    record = _record()
    record.paper_metadata = PaperMetadata(funding_sources=["Some Unclassifiable Body"])
    block = ResearchAgent._build_funding({}, record, SOURCE, "pmid:12345678")
    assert block.funding_category == "unknown", "the API proves a funder, not what kind it is"
    assert block.funding_disclosure == "disclosed"
    assert funding_errors(block.to_dict(), {"pmid:12345678": SOURCE}) == []


# prepush codex 2026-08-13: a cache hit dropped findings, funding and the new paper-level fields, so
# every run after the first silently lost what the first one paid to extract.
def test_a_cache_hit_restores_the_structured_fields() -> None:
    original = DeepExtraction(
        pmid="12345678",
        doi=None,
        title="t",
        url="u",
        findings=[Finding(population="p", intervention="i", comparator="c", endpoint="e", finding_key="k" * 40)],
        funding=FundingBlock(funding_category="government", funding_disclosure="disclosed", funding_raw="NIA"),
        trial_registration="NCT01234567",
        author_group="Tanaka H",
        paper_metadata=PaperMetadata(citation_count=7),
    )
    cached = ResearchAgent._cache_extraction(original)
    restored = ResearchAgent._extraction_from_cache(_record(), cached)

    assert [f.endpoint for f in restored.findings] == ["e"]
    assert restored.findings[0].finding_key == "k" * 40
    assert restored.funding is not None and restored.funding.funding_category == "government"
    assert restored.trial_registration == "NCT01234567"
    assert restored.author_group == "Tanaka H"
    assert restored.paper_metadata is not None and restored.paper_metadata.citation_count == 7


def test_a_silent_paper_is_undisclosed_not_unknown() -> None:
    """Ep09's thesis makes that distinction the finding."""
    block = ResearchAgent._build_funding({"funding_disclosure": "undisclosed"}, _record(), SOURCE, "p")
    assert block.funding_disclosure == "undisclosed"
    assert block.funding_category == "undisclosed"
    assert block.funding_raw is None


def test_a_failed_extraction_is_unknown() -> None:
    block = ResearchAgent._build_funding({}, _record(), SOURCE, "p")
    assert block.funding_disclosure == "unknown"
    assert block.funding_category == "unknown"


@pytest.mark.parametrize(
    "data",
    [
        {
            "funding_raw": "Acme Pharma",
            "funding_quote": "Supported by grant R01-AG000000 from the National Institute on Aging.",
            "funding_category": "industry",
            "funding_disclosure": "disclosed",
        },
        {"funding_disclosure": "undisclosed"},
        {},
    ],
)
def test_every_funding_block_the_extractor_builds_is_legal(data: dict) -> None:
    """The five fields have a legal-combination table; a block that violates it is a validation
    failure, not a warning."""
    from dr2_podcast.schemas import funding_errors

    block = ResearchAgent._build_funding(data, _record(), SOURCE, "pmid:12345678").to_dict()
    artifacts = {"pmid:12345678": SOURCE}
    assert funding_errors(block, artifacts) == [], block


# --------------------------------------------------------------------------- #
# What the block looks like where a reader meets it
# --------------------------------------------------------------------------- #
# Step 9a slice 2. `(ext.funding_source or 'N/A')[:30]` collapsed "the paper is silent" and "we
# failed to extract" into one cell, which is the distinction Ep09's thesis is built on.
def _ext(**overrides):
    from types import SimpleNamespace

    base = dict(title="t", pmid="1", findings=[], funding=None, funding_source=None)
    base.update(overrides)
    return SimpleNamespace(**base)


def test_a_silent_paper_and_a_failed_extraction_read_differently_in_the_sot() -> None:
    from dr2_podcast.pipeline_sot import _funding_cell

    silent = _ext(funding=FundingBlock(funding_category="undisclosed", funding_disclosure="undisclosed"))
    failed = _ext(funding=FundingBlock())
    assert _funding_cell(silent) != _funding_cell(failed)
    assert "undisclosed" in _funding_cell(silent)
    assert "unknown" in _funding_cell(failed)


def test_api_derived_funding_is_flagged_as_unverifiable_in_the_sot() -> None:
    from dr2_podcast.pipeline_sot import _funding_cell

    cell = _funding_cell(
        _ext(
            funding=FundingBlock(
                funding_raw="National Institute on Aging",
                funding_category="unknown",
                funding_disclosure="disclosed",
                funding_source_type="api_metadata",
            )
        )
    )
    assert "API" in cell and "unverified" in cell


def test_quoted_funding_is_not_flagged() -> None:
    from dr2_podcast.pipeline_sot import _funding_cell

    cell = _funding_cell(
        _ext(
            funding=FundingBlock(
                funding_raw="Supported by the NIA",
                funding_category="government",
                funding_disclosure="disclosed",
                funding_source_type="extracted_text",
                funding_locator={"fields": ["funding_raw"], "source_artifact_id": "a", "char_offset": 0,
                                 "quoted_span": "Supported by the NIA"},
            )
        )
    )
    assert "unverified" not in cell and "government" in cell


def test_the_case_prompt_states_every_finding_not_just_the_primary() -> None:
    """A study reporting benefit on one endpoint and no effect on another presented as unambiguous
    support, because only one CER/EER pair was serialised."""
    from dr2_podcast.research.clinical import Finding, _findings_block

    ext = _ext(
        findings=[
            Finding(population="p", intervention="i", comparator="c", endpoint="hip fracture",
                    direction="decrease", value=5.0, unit="%", is_primary=True, finding_key="k" * 40),
            Finding(population="p", intervention="i", comparator="c", endpoint="falls",
                    direction="null_result", p_value=0.41, finding_key="j" * 40),
        ]
    )
    block = _findings_block(ext)
    assert "hip fracture" in block and "falls" in block
    assert "null_result" in block, "the result a falsification case most needs must reach the model"


def test_a_legacy_record_still_states_its_rates() -> None:
    from dr2_podcast.research.clinical import _findings_block

    legacy = _ext(control_event_rate=0.2, experimental_event_rate=0.1)
    assert "CER: 0.2" in _findings_block(legacy)
