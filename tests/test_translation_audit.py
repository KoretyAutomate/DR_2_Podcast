"""The translation audit's Python half — PLAN.md Step 11.

The exit criterion is a mutation matrix, and it names which side must catch each mutation. Python
catches what moves a number or a citation; negation and hedge strength move neither, and Python must
MISS those by design — pretending otherwise would be a keyword list wearing a proof's clothes.
"""

from __future__ import annotations

import pytest

from dr2_podcast.translation_audit import audit_translation, claim_tokens, normalise, translation_errors

SOURCE = (
    "## 3.3 Clinical Impact\n"
    "\n"
    "| Study | ARR | NNT |\n"
    "|-------|-----|-----|\n"
    "| PMID:12345678 | 5.0% | 20 |\n"
    "| PMID:87654321 | 3.0% | 33 |\n"
    "\n"
    "Absolute risk reduction was 5.0% (95% CI 2.0 to 8.0) for hip fracture at 12 months.\n"
    "No significant difference in falls was observed (p=0.41).\n"
)

FAITHFUL = (
    "## 3.3 臨床的インパクト\n"
    "\n"
    "| 研究 | ARR | NNT |\n"
    "|-------|-----|-----|\n"
    "| PMID:12345678 | 5.0% | 20 |\n"
    "| PMID:87654321 | 3.0% | 33 |\n"
    "\n"
    "絶対リスク減少は12か月時点の大腿骨骨折について5.0%（95% CI 2.0〜8.0）でした。\n"
    "転倒については有意差は認められませんでした（p=0.41）。\n"
)


def test_a_faithful_translation_passes() -> None:
    assert translation_errors(SOURCE, FAITHFUL) == []


def test_the_audit_reports_what_it_checked() -> None:
    result = audit_translation(SOURCE, FAITHFUL)
    assert result["ok"]
    assert result["checked_tokens"] > 0


# --------------------------------------------------------------------------- #
# Formatting must not false-fail — a check that cries wolf gets removed
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("source_form", "translated_form"),
    [
        ("5.0%", "５．０％"),      # full width
        ("1,000", "1000"),          # thousands separator
        ("-0.05", "−0.05"),         # minus sign vs U+2212
        ("20", "20.0"),             # trailing zero
    ],
)
def test_japanese_formatting_is_not_a_mistranslation(source_form: str, translated_form: str) -> None:
    assert translation_errors(f"ARR は {source_form} です。\n", f"ARR は {translated_form} です。\n") == []


def test_normalise_folds_full_width_without_touching_meaning() -> None:
    assert normalise("５．０％") == "5.0%"


def test_headings_do_not_drown_the_comparison() -> None:
    """Section numbers appear identically on both sides and carry no claim."""
    assert [t.value for t in claim_tokens("## 4.1 Study Characteristics\nARR 5.0%\n")] == ["5"]


# --------------------------------------------------------------------------- #
# The mutation matrix — what Python must catch
# --------------------------------------------------------------------------- #
def test_a_swapped_numeral_is_caught() -> None:
    broken = FAITHFUL.replace("5.0%（95% CI", "0.5%（95% CI")
    errors = translation_errors(SOURCE, broken)
    assert errors and "5" in errors[0]


def test_a_number_reassigned_between_claims_is_caught() -> None:
    """Every number still present, the multiset unchanged, and both claims now wrong. This is the
    case set equality cannot see, and it is why the comparison is ordered."""
    broken = FAITHFUL.replace("| PMID:12345678 | 5.0% | 20 |\n| PMID:87654321 | 3.0% | 33 |",
                              "| PMID:12345678 | 3.0% | 20 |\n| PMID:87654321 | 5.0% | 33 |")
    assert sorted(t.value for t in claim_tokens(FAITHFUL)) == sorted(t.value for t in claim_tokens(broken)), (
        "the fixture must leave the multiset identical, or this tests something easier"
    )
    assert translation_errors(SOURCE, broken)


def test_a_duplicated_value_going_missing_is_caught() -> None:
    """Multiplicity: a set cannot tell two 5.0s from one."""
    source = "ARR は 5.0% でした。\n別の研究でも 5.0% でした。\n"
    broken = "ARR は 5.0% でした。\n別の研究でも同様でした。\n"
    errors = translation_errors(source, broken)
    assert errors and "missing" in errors[-1]


def test_a_citation_reassigned_to_a_different_claim_is_caught() -> None:
    broken = FAITHFUL.replace("| PMID:12345678 | 5.0%", "| PMID:87654321 | 5.0%").replace(
        "| PMID:87654321 | 3.0%", "| PMID:12345678 | 3.0%"
    )
    assert translation_errors(SOURCE, broken)


def test_a_whole_claim_sentence_omitted_is_caught() -> None:
    broken = FAITHFUL.replace("転倒については有意差は認められませんでした（p=0.41）。\n", "")
    errors = translation_errors(SOURCE, broken)
    assert errors and "0.41" in errors[-1]


def test_a_doi_that_changes_is_caught() -> None:
    source = "See 10.1001/jama.2026.1234 for the trial.\n"
    broken = "試験は 10.1001/jama.2026.9999 を参照。\n"
    assert translation_errors(source, broken)


# prepush codex 2026-08-13: nothing makes a translator preserve line wrapping, and grouping by
# physical line read a reflowed paragraph as a moved claim. A check that cries wolf gets removed
# rather than satisfied, so the boundary is a block: table rows individually, prose by paragraph.
def test_a_reflowed_paragraph_is_not_a_moved_claim() -> None:
    source = "ARR was 5.0% for hip fracture at 12 months, and the NNT is 20.\n"
    reflowed = "12か月時点の大腿骨骨折におけるARRは5.0%であり、\nNNTは20です。\n"
    assert translation_errors(source, reflowed) == []


def test_a_merged_pair_of_sentences_is_not_a_moved_claim() -> None:
    source = "ARR was 5.0%.\nNNT is 20.\n"
    merged = "ARRは5.0%、NNTは20です。\n"
    assert translation_errors(source, merged) == []


def test_table_rows_stay_separate_claims_despite_the_reflow_tolerance() -> None:
    """A table row IS a claim, and per-row association is what catches a value swapped between two
    studies — so rows are their own blocks even though prose is not."""
    source = "| PMID:12345678 | 5.0% |\n| PMID:87654321 | 3.0% |\n"
    swapped = "| PMID:12345678 | 3.0% |\n| PMID:87654321 | 5.0% |\n"
    assert translation_errors(source, swapped)


def test_a_doi_keeps_its_identity_across_japanese_punctuation() -> None:
    """The DOI character class runs into whatever ends the sentence, and Japanese ends it with 。"""
    source = "See 10.1001/jama.2026.1234.\n"
    translated = "10.1001/jama.2026.1234を参照。\n"
    assert translation_errors(source, translated) == []


# --------------------------------------------------------------------------- #
# The mutation matrix — what Python must MISS, on purpose
# --------------------------------------------------------------------------- #
def test_a_dropped_negation_is_invisible_here_and_that_is_the_design() -> None:
    """No number moves when a negation disappears. Claiming to catch this with a keyword list would
    be a check that passes for the wrong reason — Claude's half of Step 11 covers it."""
    broken = FAITHFUL.replace("有意差は認められませんでした", "有意差が認められました")
    assert translation_errors(SOURCE, broken) == []


def test_an_inflated_hedge_is_invisible_here_too() -> None:
    source = "この結果は効果を示唆する。\n値は 5.0% であった。\n"
    broken = "この結果は効果を示した。\n値は 5.0% であった。\n"
    assert translation_errors(source, broken) == []


def test_two_claims_in_one_paragraph_can_exchange_numbers_unnoticed() -> None:
    """The price of tolerating reflow, asserted rather than left to be discovered. Tightening it
    would fail legitimate translations; Claude's half of Step 11 is what covers this."""
    source = "ARR was 5.0% for fracture and 3.0% for falls.\n"
    swapped = "骨折のARRは3.0%、転倒は5.0%でした。\n"
    assert translation_errors(source, swapped) == []


def test_the_audit_says_out_loud_what_it_did_not_check() -> None:
    """A limit written only in a docstring gets read as a guarantee nobody made."""
    not_checked = audit_translation(SOURCE, FAITHFUL)["not_checked"]
    assert "negation" in not_checked
    assert "paragraph" in not_checked, "the reflow trade-off is a limit too, so it is stated too"
