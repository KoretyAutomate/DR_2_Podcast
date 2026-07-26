"""Tests for validate_grade_consistency — the deterministic GRADE/NNT gate.

Regression origin: the 2026-05-05 sleep episode shipped a script claiming
GRADE "中程度から高い" against a basis of LOW, and projected "年間数万単位" of
prevented cases against a computed NNT = inf. The LLM auditor graded the run
FAIL but missed both.
"""
import pytest
from dr2_podcast.pipeline_validators import validate_grade_consistency

BASIS_LOW = """
### 3. GRADE 評価
*   **最終 GRADE: LOW**
| 29438540 | 0.000 | 0.000 | +0.0000 | +0.00% | inf | no_effect |
"""
BASIS_MODERATE = "**FINAL GRADE: MODERATE**\n"


def test_flags_grade_overstatement():
    s = "Host 1: 「GRADE」の基準でいうと「中程度」から「高い」レベルと言えるでしょう。"
    out = validate_grade_consistency(s, basis_text=BASIS_LOW)
    assert any("GRADE_CONTRADICTION" in i for i in out)


def test_accepts_correct_grade():
    s = "Host 2: GRADE の基準では「低い」と評価されています。"
    assert validate_grade_consistency(s, basis_text=BASIS_LOW) == []


def test_flags_nnt_projection_against_null():
    s = "Host 2: 厳密な NNT は出せませんが、試算では年間数万人を防げる可能性があります。"
    out = validate_grade_consistency(s, basis_text=BASIS_LOW)
    assert any("NNT_NULL_CONTRADICTION" in i for i in out)


def test_accepts_faithful_nnt_statement():
    s = "Host 2: 今回の計算では治療必要数は無限大、測定可能な差が出ませんでした。"
    assert validate_grade_consistency(s, basis_text=BASIS_LOW) == []


def test_grade_direction_is_basis_relative():
    """MODERATE is correct against a MODERATE basis, wrong against LOW."""
    s = "Host 1: GRADE では中程度と評価されています。"
    assert validate_grade_consistency(s, basis_text=BASIS_MODERATE) == []
    assert validate_grade_consistency(s, basis_text=BASIS_LOW) != []


def test_failsafe_no_basis():
    s = "Host 1: GRADE は高いです。"
    assert validate_grade_consistency(s) == []
    assert validate_grade_consistency(s, basis_text="no grade stated here") == []


def test_ignores_lines_not_about_grade():
    """'高い' is common Japanese; only GRADE-bearing lines are judged."""
    s = "Host 2: 睡眠不足のリスクは高いと言えます。効果量も大きいです。"
    assert validate_grade_consistency(s, basis_text=BASIS_LOW) == []


def test_missing_files_do_not_raise():
    assert validate_grade_consistency("x", "/nonexistent/a.md", "/nonexistent/b.md") == []
