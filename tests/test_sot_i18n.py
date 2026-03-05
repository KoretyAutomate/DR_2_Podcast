"""Tests for sot_i18n.py — EN/JA template parity and correctness.

File under test: dr2_podcast/sot_i18n.py
"""

import re
import pytest

from dr2_podcast.sot_i18n import SOT_TEMPLATES, get_templates, t


# ---------------------------------------------------------------------------
# Template parity — all keys present in both EN and JA
# ---------------------------------------------------------------------------

class TestTemplateParity:

    def _collect_keys(self, d, prefix=""):
        """Recursively collect all key paths from a nested dict."""
        keys = set()
        for k, v in d.items():
            path = f"{prefix}.{k}" if prefix else k
            if isinstance(v, dict):
                keys.update(self._collect_keys(v, path))
            else:
                keys.add(path)
        return keys

    def test_en_and_ja_have_same_top_level_keys(self):
        en_keys = set(SOT_TEMPLATES["en"].keys())
        ja_keys = set(SOT_TEMPLATES["ja"].keys())
        assert en_keys == ja_keys, f"Missing in JA: {en_keys - ja_keys}, extra in JA: {ja_keys - en_keys}"

    def test_en_and_ja_section_keys_match(self):
        """Every section in EN should have exactly the same keys in JA."""
        for section in SOT_TEMPLATES["en"]:
            en_val = SOT_TEMPLATES["en"][section]
            ja_val = SOT_TEMPLATES["ja"][section]
            if isinstance(en_val, dict) and isinstance(ja_val, dict):
                en_keys = self._collect_keys(en_val)
                ja_keys = self._collect_keys(ja_val)
                missing = en_keys - ja_keys
                extra = ja_keys - en_keys
                assert not missing, f"Section '{section}': missing in JA: {missing}"
                assert not extra, f"Section '{section}': extra in JA: {extra}"


# ---------------------------------------------------------------------------
# Placeholder name matching between EN and JA
# ---------------------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r'\{(\w+)(?:[^}]*)?\}')


class TestPlaceholderParity:

    def _extract_placeholders(self, text):
        """Extract {placeholder} names from a template string."""
        if not isinstance(text, str):
            return set()
        return set(_PLACEHOLDER_RE.findall(text))

    def _walk_templates(self, en_dict, ja_dict, path=""):
        """Walk both dicts in parallel, comparing placeholder names."""
        errors = []
        for key in en_dict:
            en_val = en_dict[key]
            ja_val = ja_dict.get(key)
            if ja_val is None:
                continue  # parity checked elsewhere
            full_path = f"{path}.{key}" if path else key
            if isinstance(en_val, str) and isinstance(ja_val, str):
                en_ph = self._extract_placeholders(en_val)
                ja_ph = self._extract_placeholders(ja_val)
                if en_ph != ja_ph:
                    errors.append(f"{full_path}: EN={en_ph}, JA={ja_ph}")
            elif isinstance(en_val, dict) and isinstance(ja_val, dict):
                errors.extend(self._walk_templates(en_val, ja_val, full_path))
            elif isinstance(en_val, list) and isinstance(ja_val, list):
                # Lists (e.g., tier_labels) — just check same length
                if len(en_val) != len(ja_val):
                    errors.append(f"{full_path}: EN list len={len(en_val)}, JA list len={len(ja_val)}")
        return errors

    def test_all_placeholders_match(self):
        errors = self._walk_templates(SOT_TEMPLATES["en"], SOT_TEMPLATES["ja"])
        assert not errors, "Placeholder mismatches:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# JA templates contain no Chinese characters
# ---------------------------------------------------------------------------

# Simplified Chinese characters commonly confused with Japanese
# These are characters that exist ONLY in Simplified Chinese (not shared with Japanese)
_CHINESE_ONLY_CHARS = set(
    "执补认效营维剂证结显临摄"  # from the known error list
    "随筛杂混"  # 随機化, 篩査, 混杂因子
)


class TestNoChinese:

    def _check_string(self, text, path):
        """Check a string for Chinese-only characters."""
        if not isinstance(text, str):
            return []
        found = []
        for ch in text:
            if ch in _CHINESE_ONLY_CHARS:
                found.append(f"{path}: found Chinese char '{ch}' (U+{ord(ch):04X})")
        return found

    def _walk(self, d, path=""):
        errors = []
        for k, v in d.items():
            full_path = f"{path}.{k}" if path else k
            if isinstance(v, str):
                errors.extend(self._check_string(v, full_path))
            elif isinstance(v, dict):
                errors.extend(self._walk(v, full_path))
            elif isinstance(v, list):
                for i, item in enumerate(v):
                    if isinstance(item, str):
                        errors.extend(self._check_string(item, f"{full_path}[{i}]"))
        return errors

    def test_ja_templates_no_chinese_characters(self):
        errors = self._walk(SOT_TEMPLATES["ja"])
        assert not errors, "Chinese characters found in JA templates:\n" + "\n".join(errors)


# ---------------------------------------------------------------------------
# get_templates() fallback
# ---------------------------------------------------------------------------

class TestGetTemplates:

    def test_returns_en_for_en(self):
        tmpl = get_templates("en")
        assert tmpl is SOT_TEMPLATES["en"]

    def test_returns_ja_for_ja(self):
        tmpl = get_templates("ja")
        assert tmpl is SOT_TEMPLATES["ja"]

    def test_falls_back_to_en_for_unknown(self):
        tmpl = get_templates("fr")
        assert tmpl is SOT_TEMPLATES["en"]

    def test_falls_back_to_en_for_empty(self):
        tmpl = get_templates("")
        assert tmpl is SOT_TEMPLATES["en"]


# ---------------------------------------------------------------------------
# t() function interpolation
# ---------------------------------------------------------------------------

class TestTFunction:

    def test_basic_lookup(self):
        tmpl = get_templates("en")
        result = t(tmpl, "abstract", "header")
        assert result == "## Abstract\n"

    def test_ja_basic_lookup(self):
        tmpl = get_templates("ja")
        result = t(tmpl, "abstract", "header")
        assert result == "## 要約\n"

    def test_interpolation(self):
        tmpl = get_templates("en")
        result = t(tmpl, "abstract", "methods",
                   total_wide=100, total_screened=20, total_ft_ok=10)
        assert "100" in result
        assert "20" in result
        assert "10" in result

    def test_ja_interpolation(self):
        tmpl = get_templates("ja")
        result = t(tmpl, "abstract", "methods",
                   total_wide=100, total_screened=20, total_ft_ok=10)
        assert "100" in result
        assert "20" in result
        assert "10" in result

    def test_missing_key_raises(self):
        tmpl = get_templates("en")
        with pytest.raises(KeyError):
            t(tmpl, "abstract", "nonexistent_key")


# ---------------------------------------------------------------------------
# Key JA translations (correctness of known-problematic terms)
# ---------------------------------------------------------------------------

class TestJACorrectTerms:

    @pytest.fixture
    def ja(self):
        return get_templates("ja")

    def test_falsification_not_perjury(self, ja):
        """反証 (falsification), NOT 偽証 (perjury)."""
        dual = t(ja, "introduction", "dual_hypothesis")
        assert "反証" in dual
        assert "偽証" not in dual

    def test_confounders_correct(self, ja):
        """交絡因子, NOT 混雑因子."""
        dual = t(ja, "introduction", "dual_hypothesis")
        assert "交絡因子" in dual

    def test_screening_katakana(self, ja):
        """スクリーニング, NOT 篩査."""
        screening = t(ja, "methods", "screening_header")
        assert "スクリーニング" in screening

    def test_track_katakana(self, ja):
        """トラック, NOT トレック."""
        dual = t(ja, "introduction", "dual_hypothesis")
        assert "トラック" in dual
        assert "トレック" not in dual

    def test_rct_correct(self, ja):
        """無作為化比較試験, NOT 随機化."""
        screening = t(ja, "methods", "screening_body",
                      aff_screened=0, fal_screened=0, total_screened=0)
        assert "無作為化比較試験" in screening

    def test_ja_section_headers(self, ja):
        """JA SoT should contain Japanese section headers."""
        assert "要約" in t(ja, "abstract", "header")
        assert "序論" in t(ja, "introduction", "header")
        assert "方法" in t(ja, "methods", "header")
        assert "結果" in t(ja, "results", "header")
        assert "考察" in t(ja, "discussion", "header")
        assert "参考文献" in t(ja, "references", "header")
