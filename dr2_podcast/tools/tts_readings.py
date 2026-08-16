"""What each of the two G2P sources says a line is pronounced as.

Split out of `tts_reading_check` so that module can stay about DECIDING whether a
line is suspicious; everything here is about OBTAINING a reading and making two
readings comparable, and none of it knows what a finding is.

Layer 0  engine reading   — POST /audio_query, read accent_phrases[].moras[].text.
                            This is the synthesis design, not an ASR guess: what the
                            engine returns here IS what it will pronounce.
Layer 1  pyopenjtalk      — independent G2P over the same line. Shared OpenJTalk lineage
                            means AGREEMENT proves nothing, but DISAGREEMENT is a strong
                            signal, so it is used only to rank candidates for Layer 2.

The two sources disagree on SPELLING as well as on pronunciation, which is what
:func:`_norm` is for — without it every line reads as a disagreement and Layer 1
is useless as a ranker.
"""

from __future__ import annotations

import re

import requests

AIVISSPEECH_URL = "http://localhost:10101"

# A line whose reading proves the engine is answering with real content. Used by
# tts_reading_check.preflight() — a reachable engine that returns empty accent_phrases
# would otherwise make every line look clean.
PROBE_LINE = "読み上げの確認です。"

_KATA = re.compile(r"[^゠-ヿ]")

# vowel each katakana mora ends on, for expanding the ー long-vowel mark
_VOWEL = {}
for _row, _v in (
    ("アカサタナハマヤラワガザダバパャァ", "ア"),
    ("イキシチニヒミリギジヂビピィ", "イ"),
    ("ウクスツヌフムユルグズヅブプュゥヴ", "ウ"),
    ("エケセテネヘメレゲゼデベペェ", "エ"),
    ("オコソトノホモヨロヲゴゾドボポョォ", "オ"),
):
    for _c in _row:
        _VOWEL[_c] = _v


class EngineUnavailable(RuntimeError):
    """The engine could not be reached, or answered with nothing.

    Raised rather than returned so a check CANNOT report "clean" when it never read
    anything: an unreachable engine used to produce zero HIGH findings and exit 0.
    """


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
    for a, b in (("ヲ", "オ"), ("ヅ", "ズ"), ("ヂ", "ジ"), ("ヱ", "エ"), ("ヰ", "イ"), ("・", ""), ("゠", "")):
        s = s.replace(a, b)
    return s


def engine_phrases(text: str, speaker: int, session: requests.Session) -> list[str]:
    """The engine's reading, split at the accent-phrase boundaries it will speak.

    The boundaries are the point: 「その大きさ自体が」 comes back as オオキ / サジタイガ,
    which is inaudible in the concatenated string and obvious here.
    """
    r = session.post(
        f"{AIVISSPEECH_URL}/audio_query",
        params={"text": text, "speaker": speaker},
        timeout=30,
    )
    r.raise_for_status()
    data = r.json()
    return ["".join(m["text"] for m in ap["moras"]) for ap in data["accent_phrases"]]


def openjtalk_reading(text: str) -> str | None:
    try:
        import pyopenjtalk

        return pyopenjtalk.g2p(text, kana=True)
    except Exception:
        return None


def openjtalk_phrases(text: str) -> list[str] | None:
    """pyopenjtalk's own accent-phrase segmentation of the same line.

    NJD's chain_flag is what VOICEVOX-lineage engines build accent phrases from: 0 (or -1
    at the start) opens a phrase, 1 attaches the word to the one before. `pron` rather than
    `read` because pron carries the particle readings (は→ワ, を→オ) the engine also speaks.
    """
    try:
        import pyopenjtalk

        out: list[str] = []
        for w in pyopenjtalk.run_frontend(text):
            if not isinstance(w, dict) or not w.get("mora_size"):
                continue
            if w.get("chain_flag", 0) == 1 and out:
                out[-1] += w["pron"]
            else:
                out.append(w["pron"])
        return out or None
    except Exception:
        return None


def boundary_conflicts(eng: list[str], ojt: list[str] | None) -> list[str]:
    """Accent-phrase boundaries the ENGINE places that pyopenjtalk does not.

    One-directional on purpose. Measured over the 95 comparable lines of Ep014's script:
    a plain "do the segmentations differ" test fires on 83 of them, because pyopenjtalk
    habitually splits finer than the engine (ドンナ|ケンキューガ|コノ|ブンヤニワ vs the
    engine's ...|コノブンヤニワ) — that is granularity, not a misreading. The engine
    OPENING a phrase inside a word pyopenjtalk keeps whole is the rare case, 13 of 95, and
    it is exactly what 大きさ自体 -> オオキ / サジタイガ looks like.

    Comparable only when the two flat readings already agree; otherwise the mora offsets
    address different strings. Returns one "…before/after…" context string per conflict.
    """
    if not ojt:
        return []
    flat_e, flat_o = _norm("".join(eng)), _norm("".join(ojt))
    if flat_e != flat_o:
        return []

    def interior_offsets(phrases: list[str]) -> set[int]:
        # Normalise the PREFIX rather than summing per-phrase lengths: _norm is left-to-
        # right and context-dependent (a phrase opening on ー has no preceding mora of its
        # own), so per-phrase lengths need not add up to the length of the normalised whole.
        return {len(_norm("".join(phrases[: k + 1]))) for k in range(len(phrases) - 1)}

    extra = sorted(interior_offsets(eng) - interior_offsets(ojt))
    return [f"…{flat_e[max(0, x - 6):x]}/{flat_e[x:x + 6]}…" for x in extra]
