"""Tests for the deterministic post-generation validators (pipeline_validators).

Regression anchors from the 2026-07-05 sleep-week investigation:
  - Tue: lowercase-underscore `host_1：` labels leaked past clean_script_for_tts.
  - Sat: ~40 consecutive Host-1 turns (mislabeled block).
  - Fri: fabricated citations 'Vandewalle 2007' / 'Pillai 2014'.
  - Mon: identical closing line repeated 3-4 times.
"""

import json
import textwrap
from pathlib import Path

from dr2_podcast.pipeline_validators import (
    normalize_speaker_labels,
    check_speaker_alternation,
    validate_citations,
    detect_duplicate_blocks,
    deduplicate_lines,
    validate_script_structure,
)


# --------------------------------------------------------------------------- #
# Speaker-label normalization
# --------------------------------------------------------------------------- #
def test_normalize_lowercase_underscore_label():
    text = "host_1：こんにちは。\nhost_2：はい。"
    out, fixed = normalize_speaker_labels(text)
    assert out == "Host 1: こんにちは。\nHost 2: はい。"
    assert fixed == 2


def test_normalize_fullwidth_and_bold_and_katakana():
    text = "Host 1：A\n**Host 2**: B\nホスト1： C\nSpeaker 2: D"
    out, fixed = normalize_speaker_labels(text)
    lines = out.split("\n")
    assert lines[0] == "Host 1: A"
    assert lines[1] == "Host 2: B"
    assert lines[2] == "Host 1: C"
    assert lines[3] == "Host 2: D"
    assert fixed == 4


def test_normalize_leaves_canonical_untouched():
    text = "Host 1: already fine\nHost 2: also fine"
    out, fixed = normalize_speaker_labels(text)
    assert out == text
    assert fixed == 0


# --------------------------------------------------------------------------- #
# Alternation / single-voice
# --------------------------------------------------------------------------- #
def test_speaker_run_flagged():
    text = "\n".join(f"Host 1: line {i}" for i in range(40))
    issues = check_speaker_alternation(text)
    assert any(i.startswith("SPEAKER_RUN") for i in issues)


def test_single_voice_flagged():
    text = "\n".join(["Host 1: a"] * 9 + ["Host 2: b"])
    issues = check_speaker_alternation(text)
    assert any(i.startswith("SINGLE_VOICE") for i in issues)


def test_healthy_alternation_clean():
    text = "\n".join(f"Host {1 if i % 2 == 0 else 2}: line" for i in range(20))
    assert check_speaker_alternation(text) == []


def test_no_labels_flagged():
    assert check_speaker_alternation("just prose, no labels") == [
        "NO_SPEAKER_LABELS: no canonical 'Host N:' turns found"
    ]


# --------------------------------------------------------------------------- #
# Citations
# --------------------------------------------------------------------------- #
def _write_sot(tmp_path):
    sot = tmp_path / "sot.md"
    sot.write_text(
        textwrap.dedent(
            """
        ## References
        1. Leproult et al.. *Effect of 1 week of sleep restriction on testosterone*. JAMA. (2011). PMID: [21632481](https://pubmed.ncbi.nlm.nih.gov/21632481/).
        5. Van Cauter et al.. *Age-related changes in slow wave sleep*. JAMA. (2000). PMID: [10938176](https://pubmed.ncbi.nlm.nih.gov/10938176/).
        """
        ),
        encoding="utf-8",
    )
    return str(sot)


def test_fabricated_citation_flagged(tmp_path):
    sot = _write_sot(tmp_path)
    script = "Host 1: Vandewalle et al. (2007) の研究があります。\nHost 2: Pillai et al. (2014) もそうですね。"
    issues = validate_citations(script, sot_path=sot)
    joined = " ".join(issues)
    assert "Vandewalle" in joined and "FABRICATED" in joined
    assert "Pillai" in joined


def test_real_citation_passes(tmp_path):
    sot = _write_sot(tmp_path)
    script = "Host 1: Leproult & Van Cauter 2011 の研究では15%減少しました。"
    assert validate_citations(script, sot_path=sot) == []


def test_coauthored_real_citation_not_flagged(tmp_path):
    sot = _write_sot(tmp_path)
    # co-authored real citation must not false-positive even if one co-author's
    # other paper is a different year
    script = "Host 1: Leproult & Van Cauter 2011 の研究。"
    assert validate_citations(script, sot_path=sot) == []


def test_no_whitelist_never_flags():
    script = "Host 1: Vandewalle et al. (2007) の研究。"
    assert validate_citations(script, sot_path=None, sources_json_path=None) == []


def test_allcaps_and_nonauthor_not_flagged(tmp_path):
    sot = _write_sot(tmp_path)
    # trial name / acronym / journal should not be treated as authors
    script = "Host 1: PREDIMED 2013 と VITAL 2018 の試験。\nHost 2: Nature 2015 に掲載されました。"
    assert validate_citations(script, sot_path=sot) == []


def test_pmids_loaded_from_sources_json(tmp_path):
    sj = tmp_path / "src.json"
    sj.write_text(
        json.dumps({"lead": [{"url": "https://pubmed.ncbi.nlm.nih.gov/21632481/", "title": "x"}]}), encoding="utf-8"
    )
    # no SOT -> no author whitelist -> no flags even though json exists
    assert validate_citations("Host 1: Foo (2020).", sources_json_path=str(sj)) == []


# --------------------------------------------------------------------------- #
# Duplicate blocks
# --------------------------------------------------------------------------- #
def test_duplicate_ending_flagged():
    line = "Host 1: ありがとうございます！これからも一緒に科学の不思議を探求していきましょうね！"
    text = f"Host 2: 中身の話。\n{line}\nHost 2: 別の話。\n{line}"
    issues = detect_duplicate_blocks(text)
    assert any(i.startswith("DUPLICATE_LINE") for i in issues)


def test_short_lines_not_flagged():
    text = "Host 1: はい。\nHost 2: はい。\nHost 1: はい。"
    assert detect_duplicate_blocks(text) == []


def test_deduplicate_lines_removes_repeats():
    line = "Host 1: ありがとうございます！これからも一緒に科学の不思議を探求していきましょうね！"
    text = f"Host 2: 中身の話をしましょう、十分な長さの本文です。\n{line}\nHost 2: 別の十分に長い本文の話。\n{line}"
    out, removed = deduplicate_lines(text)
    assert removed == 1
    assert out.count("これからも一緒に科学") == 1


def test_deduplicate_keeps_short_backchannels():
    text = "Host 1: はい。\nHost 2: はい。\nHost 1: はい。"
    out, removed = deduplicate_lines(text)
    assert removed == 0 and out == text


# --------------------------------------------------------------------------- #
# Aggregate
# --------------------------------------------------------------------------- #
def test_validate_script_structure_normalizes_and_flags(tmp_path):
    sot = _write_sot(tmp_path)
    script = "host_1：Vandewalle et al. (2007) の話。\nhost_2：なるほど。"
    r = validate_script_structure(script, sot_path=sot)
    assert r["labels_fixed"] == 2
    assert r["normalized_text"].startswith("Host 1: ")
    assert not r["pass"]
    assert any("FABRICATED" in i for i in r["issues"])


def test_validate_structure_accepts_sot_text(tmp_path):
    sot = _write_sot(tmp_path)
    sot_text = Path(sot).read_text(encoding="utf-8")
    script = "Host 1: Vandewalle et al. (2007) の話。\nHost 2: なるほど。"
    r = validate_script_structure(script, sot_text=sot_text)
    assert not r["pass"]
    assert any("FABRICATED" in i for i in r["issues"])


def test_clean_script_passes():
    script = "\n".join(f"Host {1 if i % 2 == 0 else 2}: 十分な長さの通常の対話文です。" for i in range(10))
    r = validate_script_structure(script)
    assert r["pass"], r["issues"]
