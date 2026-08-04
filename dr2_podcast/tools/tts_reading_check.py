"""TTS misreading detector — PLAN.md Step 7, Layers 0/1.

Extracts what AivisSpeech will ACTUALLY say for each line, without synthesizing audio,
and flags lines where an independent G2P disagrees. Replaces "render 20 minutes and
listen to find one misreading".

Layer 0  engine reading   — POST /audio_query, read accent_phrases[].moras[].text.
                            This is the synthesis design, not an ASR guess: what the
                            engine returns here IS what it will pronounce.
Layer 1  pyopenjtalk      — independent G2P over the same line. Shared OpenJTalk lineage
                            means AGREEMENT proves nothing, but DISAGREEMENT is a strong
                            signal, so it is used only to rank candidates for Layer 2.

Runs on the post-glossary text (clean_script_for_tts output) because that is what is
actually sent to the engine.

Usage:
    python -m dr2_podcast.tools.tts_reading_check <script.txt> [...] [--json OUT]
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import requests

from dr2_podcast.audio.engine import clean_script_for_tts

AIVISSPEECH_URL = "http://localhost:10101"
DEFAULT_SPEAKER = 1937616896

# Confirmed misreading families (memory 64e977ae + listening rounds).
#
# A bad READING alone is not evidence — it is only a misreading if the SOURCE TEXT
# actually contains the hazardous surface form. Matching on the reading alone produced
# 13 false positives on 2026-07-28: コワサ matched the correct 壊さなかった/怖さ,
# ゼロゼロ matched the correct 0.001 -> レエテンゼロゼロイチ, and ガイソオ matched a
# substring spanning a word boundary in 違いそう -> チガイソオ. So each entry pairs a
# source pattern with the reading that would be WRONG for it.
HAZARD_READINGS = {
    "五つ目": ("ゴツメ", "五つ目 misread as ごつめ (user listening, ep09 2026-07-24)"),
    "5つ目": ("ゴツメ", "5つ目 misread as ごつめ (user confirmed 2026-07-28)"),
    "4つ目": ("ヨンツメ", "4つ目 misread as よんつめ (ep08 2026-07-24)"),
    "六つ目": ("ロクツメ", "六つめ misread as ろくつめ"),
    "一つ目": ("イチツメ", "一つめ misread as いちつめ"),
    "〇〇": ("ゼロゼロ", "〇〇 misread as ゼロゼロ"),
    "△△": ("", "△△ produces NO reading — silent in the audio"),
    "笑い者": ("ワライシャ", "笑い者 misread as わらいしゃ"),
    "建前": ("ケンマエ", "建前 misread as けんまえ"),
    "放って": ("ハナッテ", "放っておく misread as はなって"),
    "強さ": ("コワサ", "強さ misread as こわさ"),
    "母数": ("ハハスウ", "母数 misread as ははすう"),
    "一行": ("イッコオ", "一行 misread as いっこう"),
    "捕食者": ("プレデタア", "捕食者 misread as プレデター"),
    "外そう": ("ガイソオ", "外そう misread as がいそう"),
    "数行": ("スウコオ", "数行 misread as すうこう"),
    "NAD": ("ナッド", "NAD+ misread as ナッド"),
}

# Mirrors engine.py:325 — only group(3) is ever sent to the engine, and
# engine.py:386 treats [TRANSITION] as a BGM marker that is never spoken.
SPEAKER_RE = re.compile(r"^(Speaker\s*(\d+))\s*[:：]\s*(.*)", re.IGNORECASE)
MARKER_RE = re.compile(r"^\[[A-Z_]+\]$")

_KATA = re.compile(r"[^゠-ヿ]")

# vowel each katakana mora ends on, for expanding the ー long-vowel mark
_VOWEL = {}
for _row, _v in (("アカサタナハマヤラワガザダバパャァ", "ア"),
                 ("イキシチニヒミリギジヂビピィ", "イ"),
                 ("ウクスツヌフムユルグズヅブプュゥヴ", "ウ"),
                 ("エケセテネヘメレゲゼデベペェ", "エ"),
                 ("オコソトノホモヨロヲゴゾドボポョォ", "オ")):
    for _c in _row:
        _VOWEL[_c] = _v


def _norm(kana: str) -> str:
    """Normalise a katakana reading so the two G2P sources are comparable.

    They disagree on spelling, not pronunciation: pyopenjtalk writes ヨーコソ where
    the engine writes ヨオコソ, and uses ヲ where the engine uses オ. Without this,
    every single line reads as a disagreement and Layer 1 is useless as a ranker.
    """
    s = _KATA.sub("", kana or "")
    out: list[str] = []
    for ch in s:
        prev_vowel = _VOWEL.get(out[-1]) if out else None
        if ch == "ー" and out:
            # long-vowel mark -> repeat the preceding vowel
            out.append(prev_vowel or "")
        elif ch == "ウ" and prev_vowel == "オ":
            # o-row + ウ IS a long o in Japanese: ホウホウ ≡ ホオホオ, ヨウ ≡ ヨオ, コウ ≡ コオ.
            # Handling this as the literal pair "オウ" (as this function first did) misses
            # every case where the o-vowel comes from a consonant mora, which is most of them
            # and was the single largest source of false MISREADs on 2026-07-28.
            out.append("オ")
        elif ch == "イ" and prev_vowel == "エ":
            # likewise e-row + イ is a long e: エイヨウ ≡ エエヨオ, ケイケン ≡ ケエケン
            out.append("エ")
        else:
            out.append(ch)
    s = "".join(out)
    for a, b in (("ヲ", "オ"), ("ヅ", "ズ"), ("ヂ", "ジ"), ("ヱ", "エ"), ("ヰ", "イ"),
                 ("・", ""), ("゠", "")):
        s = s.replace(a, b)
    return s


def spoken_lines(cleaned: str) -> list[str]:
    """The text the engine actually receives, in render order."""
    out = []
    for ln in cleaned.splitlines():
        ln = ln.strip()
        if not ln or MARKER_RE.match(ln):
            continue
        m = SPEAKER_RE.match(ln)
        text = m.group(3).strip() if m else ln
        if text:
            out.append(text)
    return out


def engine_reading(text: str, speaker: int, session: requests.Session) -> str:
    r = session.post(
        f"{AIVISSPEECH_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return "".join(m["text"] for ap in data["accent_phrases"] for m in ap["moras"])


def openjtalk_reading(text: str) -> str | None:
    try:
        import pyopenjtalk

        return pyopenjtalk.g2p(text, kana=True)
    except Exception:
        return None


def check_line(text: str, speaker: int, session: requests.Session) -> dict | None:
    """Return a finding dict if the line looks suspicious, else None."""
    try:
        eng = engine_reading(text, speaker, session)
    except Exception as exc:  # engine down / bad line — surface, do not swallow
        return {"line": text, "reason": "engine_error", "detail": str(exc)[:120]}

    # a line with visible content but no reading is SILENT in the audio (the △△ bug)
    if text.strip() and not eng.strip():
        return {"line": text, "engine_reading": "", "reason": "empty_reading",
                "detail": "line produces NO audio — silent content loss"}

    hazards = [why for src, (bad, why) in HAZARD_READINGS.items()
               if src in text and (bad in eng if bad else not eng.strip())]

    # Layer 1 is only a RANKER, and it is blind on ASCII: the engine says エピソドワン
    # for "Episode 1" while pyopenjtalk spells the letters out. Comparing those produces
    # pure noise, and acronyms are already owned by the glossary (Layer 1 of the 3-layer
    # system), so skip the G2P vote on lines containing Latin letters.
    ojt = None if re.search(r"[A-Za-z]", text) else openjtalk_reading(text)
    disagree = bool(ojt) and _norm(ojt) != _norm(eng)

    if hazards or disagree:
        return {
            "line": text,
            "engine_reading": eng,
            "openjtalk_reading": ojt,
            "reason": "known_hazard" if hazards else "g2p_disagreement",
            "hazards": hazards,
            "priority": "HIGH" if hazards else "REVIEW",
        }
    return None


def check_script(path: Path, speaker: int = DEFAULT_SPEAKER) -> dict:
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_script_for_tts(raw)  # exactly what gets sent to the engine
    lines = spoken_lines(cleaned)
    session = requests.Session()
    findings = [f for ln in lines if (f := check_line(ln, speaker, session))]
    return {
        "script": str(path),
        "lines_checked": len(lines),
        "findings": findings,
        "high": sum(1 for f in findings if f.get("priority") == "HIGH"),
        "review": sum(1 for f in findings if f.get("priority") == "REVIEW"),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("scripts", nargs="+", type=Path)
    ap.add_argument("--speaker", type=int, default=DEFAULT_SPEAKER)
    ap.add_argument("--json", type=Path, help="write full report here")
    args = ap.parse_args(argv)

    reports = []
    for p in args.scripts:
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            continue
        rep = check_script(p, args.speaker)
        reports.append(rep)
        print(f"{p.parent.name:34} lines={rep['lines_checked']:<5} "
              f"HIGH={rep['high']:<4} review={rep['review']}")

    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.json}")
    return 1 if any(r["high"] for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
