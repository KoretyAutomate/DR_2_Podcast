"""Tests for synthetic citation detection (_detect_synthetic_citations & _CITATION_RE).

File under test: dr2_podcast/research/clinical.py
"""

from types import SimpleNamespace

import pytest

from dr2_podcast.research.clinical import _detect_synthetic_citations, _CITATION_RE


def _ext(title="", raw_facts=""):
    """Create a minimal extraction-like object with .title and .raw_facts."""
    return SimpleNamespace(title=title, raw_facts=raw_facts)


class TestDetectSyntheticCitations:

    def test_no_citations_in_report(self):
        """Report with no Author (YYYY) patterns returns (False, [])."""
        result = _detect_synthetic_citations("No citations here at all.", [_ext("Study")])
        assert result == (False, [])

    def test_real_citation_matches_extraction_title(self):
        """Citation whose author name appears in extraction title is NOT flagged."""
        extractions = [_ext(title="Smith randomized trial 2022")]
        has_syn, flagged = _detect_synthetic_citations(
            "According to Smith (2022), the effect was significant.", extractions
        )
        assert has_syn is False
        assert flagged == []

    def test_hallucinated_citation_not_in_extractions(self):
        """Citation whose author is NOT in any extraction is flagged."""
        extractions = [_ext(title="Jones meta-analysis")]
        has_syn, flagged = _detect_synthetic_citations(
            "Fakename (2021) reported improvements.", extractions
        )
        assert has_syn is True
        assert "Fakename (2021)" in flagged

    def test_empty_extractions_with_citations(self):
        """All citations are flagged when extractions list is empty."""
        has_syn, flagged = _detect_synthetic_citations(
            "Smith (2020) and Jones (2021) found results.", []
        )
        assert has_syn is True
        assert len(flagged) == 2

    def test_mix_of_real_and_fake_citations(self):
        """Only the unknown citation is flagged."""
        extractions = [_ext(title="Smith clinical trial")]
        has_syn, flagged = _detect_synthetic_citations(
            "Smith (2022) confirmed it. Ghostauthor (2019) disagreed.", extractions
        )
        assert has_syn is True
        assert len(flagged) == 1
        assert "Ghostauthor (2019)" in flagged

    def test_et_al_in_citation(self):
        """'et al.' variant is handled correctly and stripped during lookup."""
        extractions = [_ext(title="Johnson longitudinal study")]
        has_syn, flagged = _detect_synthetic_citations(
            "Johnson et al. (2023) showed benefits.", extractions
        )
        assert has_syn is False
        assert flagged == []

    def test_author_name_in_extraction_title_not_flagged(self):
        """Author name appearing anywhere in extraction titles clears it."""
        extractions = [
            _ext(title="unrelated study"),
            _ext(title="Chen meta-analysis of caffeine"),
        ]
        has_syn, flagged = _detect_synthetic_citations(
            "Chen (2021) demonstrated efficacy.", extractions
        )
        assert has_syn is False

    def test_author_name_in_raw_facts_not_flagged(self):
        """Author name found in raw_facts of an extraction is NOT flagged."""
        extractions = [_ext(title="some study", raw_facts="Lead investigator Lee reported")]
        has_syn, flagged = _detect_synthetic_citations(
            "Lee (2020) found significant results.", extractions
        )
        assert has_syn is False

    def test_citation_with_extreme_year_still_matched(self):
        """Regex matches any 4-digit year (e.g. 1800, 2099)."""
        matches = _CITATION_RE.findall("Oldman (1800) and Futurista (2099)")
        assert len(matches) == 2
        assert ("Oldman", "1800") in matches
        assert ("Futurista", "2099") in matches

    def test_no_uppercase_word_before_year_no_match(self):
        """Regex requires capitalized word; lowercase before year does not match."""
        matches = _CITATION_RE.findall("someone (2022) said something")
        assert matches == []

    def test_multiple_citations_all_real(self):
        """Multiple citations that all match extractions returns (False, [])."""
        extractions = [
            _ext(title="Alpha study"),
            _ext(title="Beta trial"),
            _ext(title="Gamma review"),
        ]
        has_syn, flagged = _detect_synthetic_citations(
            "Alpha (2020), Beta (2021), and Gamma (2019) all agreed.",
            extractions,
        )
        assert has_syn is False
        assert flagged == []
