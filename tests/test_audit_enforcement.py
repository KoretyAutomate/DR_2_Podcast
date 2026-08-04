"""Tests for the accuracy-audit enforcement trigger.

Regression anchor (2026-07-05): the flow's only correction trigger was
``re.search(r'\\*\\*Severity\\*\\*:\\s*HIGH', ...)``, which matched NEITHER the
Japanese ``**重大度**: **HIGH**`` nor ``HIGH Severity`` — so the sleep-week
fabrications (audit verdict FAIL) never triggered the (already-built) corrector.
"""

import pytest

from dr2_podcast.pipeline import _audit_requires_correction


@pytest.mark.parametrize(
    "text",
    [
        "## Overall Assessment\n**FAIL**\n",  # English FAIL
        "## 総合評価\n**FAIL (不合格)**\n",  # JA FAIL
        "評価: 不合格",  # JA verdict word
        "- **重大度**: **HIGH**",  # JA severity label (the bug case)
        "This is a HIGH severity drift.",  # inline EN
        "重要な問題 (HIGH Severity) が見つかりました。",  # parenthetical
        "- **Severity**: HIGH",  # original EN format
        "- Severity: HIGH",
    ],
)
def test_triggers_correction(text):
    assert _audit_requires_correction(text) is True


@pytest.mark.parametrize(
    "text",
    [
        "",
        "## Overall Assessment\nPASS\nNo drift found.",
        "Everything checks out. All claims supported.",
        "- **Severity**: MEDIUM",  # medium must NOT trigger
        "- **Severity**: LOW",
    ],
)
def test_does_not_trigger(text):
    assert _audit_requires_correction(text) is False


def test_real_sleep_audits_trigger():
    """The three real sleep audits that were ignored must now all trigger."""
    import pathlib

    base = pathlib.Path("research_outputs")
    for folder in (
        "Ep020_睡眠とホルモン_テストステロンと成長ホルモン",
        "Ep018_睡眠と脳_グリンパティック系と認知症",
        "Ep019_睡眠と代謝_血糖と食欲",
    ):
        audit = base / folder / "research" / "accuracy_audit.md"
        if audit.exists():  # skip gracefully if the fixture run dir is absent
            assert _audit_requires_correction(audit.read_text(encoding="utf-8")) is True
