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
