"""TTS misreading detector — PLAN.md Step 7, Layers 0/1.

Extracts what AivisSpeech will ACTUALLY say for each line, without synthesizing audio,
and flags lines where an independent G2P disagrees. Replaces "render 20 minutes and
listen to find one misreading".

Layer 0 (the engine's own reading) and Layer 1 (pyopenjtalk's independent G2P) are
obtained in `tts_readings`; this module decides what a disagreement between them means.

Runs on the post-glossary text (clean_script_for_tts output) because that is what is
actually sent to the engine, AT THE VOICE THAT WILL SPEAK EACH LINE (see resolve_voices).

TWO MODES
  full script    — every spoken line. Catches the hazards listed in HAZARD_READINGS.
  changed lines  — `--changed-vs PREVIOUS`. Reads back only the lines that differ from a
                   previous version of the same script, which is what actually catches the
                   misreadings HAZARD_READINGS cannot express (see that table's comment).

Usage:
    python -m dr2_podcast.tools.tts_reading_check <script.txt> [...] [--json OUT]
    python -m dr2_podcast.tools.tts_reading_check <script.txt> --changed-vs <script.txt.bak>
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import requests

from dr2_podcast.audio.engine import (
    MARKER_SILENCE,
    _chunk_japanese_text,
    _get_tts_speaker_ids_int,
    assign_seeded_voices,
    clean_script_for_tts,
)
from dr2_podcast.config import TTS_RANDOM_VOICE
from dr2_podcast.tools.tts_readings import (
    AIVISSPEECH_URL,
    PROBE_LINE,
    EngineUnavailable,
    _norm,
    boundary_conflicts,
    engine_phrases,
    openjtalk_phrases,
    openjtalk_reading,
)

# engine.py:_generate_audio_aivisspeech routes unlabelled leading prose to Speaker 2.
CHANNEL_INTRO_SPEAKER = 2

# Confirmed misreading families (memory 64e977ae + listening rounds).
#
# A bad READING alone is not evidence — it is only a misreading if the SOURCE TEXT
# actually contains the hazardous surface form. Matching on the reading alone produced
# 13 false positives on 2026-07-28: コワサ matched the correct 壊さなかった/怖さ,
# ゼロゼロ matched the correct 0.001 -> レエテンゼロゼロイチ, and ガイソオ matched a
# substring spanning a word boundary in 違いそう -> チガイソオ. So each entry pairs a
# source pattern with the reading that would be WRONG for it.
#
# LIMIT OF THIS TABLE (2026-08-13): it encodes hazard as a property of the WORD. The four
# misreadings caught by hand that day were properties of POSITION — 次の研究で read ジノ
# after 」+読点, 大きさ自体 re-segmented to オオキ/サジタイ, 点推定 split after という,
# またいで split sentence-initially — and every one of them reads correctly in isolation,
# so none is expressible here. One of them was CREATED by an edit that fixed something
# else. That is why `--changed-vs` exists: read back what just changed, at the real voice.
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

# Mirrors engine.py:_generate_audio_aivisspeech — only group(3) is ever sent to the engine.
SPEAKER_RE = re.compile(r"^(Speaker\s*(\d+))\s*[:：]\s*(.*)", re.IGNORECASE)
# Shape of an audio marker, for callers that want to recognise one. spoken_turns does NOT
# use it: the renderer keys off membership in MARKER_SILENCE, and an unknown [BRACKETED]
# line is not a marker there — it is appended to the open turn and spoken.
MARKER_RE = re.compile(r"^\[[A-Z_]+\]$")


# ---------------------------------------------------------------------------
# Which voice actually speaks a line
# ---------------------------------------------------------------------------


class VoiceResolutionError(RuntimeError):
    """The configured voice IDs are not usable for this engine."""


@dataclass(frozen=True)
class VoiceAssignment:
    """The voice each Speaker N will actually be rendered with."""

    speaker1: int
    speaker2: int
    swapped: bool
    random_voice: bool
    #: Set only by `--speaker`, which overrides both. Kept so the report can say the
    #: check ran at a forced voice rather than at the one the renderer would use.
    forced: int | None = None

    def for_speaker(self, speaker: int) -> int:
        """engine.py:_AivisTimeline.flush_turn — host1 for Speaker 1, host2 otherwise."""
        return self.speaker1 if speaker == 1 else self.speaker2

    @classmethod
    def forced_to(cls, voice: int) -> VoiceAssignment:
        """`--speaker N`: every line is checked at N, nothing is resolved.

        A forced voice is still an assignment, so it is one — the alternative,
        carrying `VoiceAssignment | None` and reading the forced voice out of a
        second variable, is two representations of one decision and made
        `voice_of` unable to state that it returns an int.
        """
        return cls(speaker1=voice, speaker2=voice, swapped=False, random_voice=False, forced=voice)

    def as_report(self) -> dict:
        if self.forced is not None:
            return {"forced": self.forced}
        return {
            "speaker1": self.speaker1,
            "speaker2": self.speaker2,
            "swapped": self.swapped,
            "random_voice": self.random_voice,
        }


def resolve_voices(cleaned_script: str) -> VoiceAssignment:
    """Resolve Speaker 1 / Speaker 2 to the voice IDs THIS script will be rendered with.

    Same two steps the renderer takes, in the same order and off the same text:
    engine._get_tts_speaker_ids_int() reads TTS_HOST1_ID/TTS_HOST2_ID, then
    engine.assign_seeded_voices() may swap them from a sha256 of the cleaned script when
    TTS_RANDOM_VOICE is on (pipeline.py hands the renderer clean_script_for_tts output, so
    the seed text is the cleaned script — and cleaning is idempotent, so re-cleaning an
    already-cleaned script.txt gives the same seed).

    Checking every line at one hardcoded voice — what this module did until 2026-08-13 —
    could report the reading of a voice that never speaks those lines.
    """
    host1, host2 = _get_tts_speaker_ids_int()
    if host1 is None or host2 is None:
        raise VoiceResolutionError(
            "TTS_HOST1_ID/TTS_HOST2_ID are not integer style IDs — cannot resolve the "
            "voices this script renders with. Pass --speaker to force one."
        )
    spk1, spk2, swapped = assign_seeded_voices(cleaned_script, host1, host2)
    return VoiceAssignment(speaker1=spk1, speaker2=spk2, swapped=swapped, random_voice=bool(TTS_RANDOM_VOICE))


# ---------------------------------------------------------------------------
# What the engine is asked to speak
# ---------------------------------------------------------------------------


class Turn(NamedTuple):
    """One turn as the renderer assembles it, with the Speaker N it is attributed to."""

    speaker: int
    text: str


def spoken_turns(cleaned: str) -> list[Turn]:
    """The turns the renderer will build from this script, in render order.

    Line-for-line mirror of engine.py:_generate_audio_aivisspeech's parser, because a
    positional misreading is only reproduced if the engine is asked about the SAME text:
    a `Speaker N:` prefix flushes the previous turn and opens a new one, an unlabelled
    line is appended to the open turn (joined with a space, as the renderer does), a
    MARKER_SILENCE line flushes without ending the speaker, `##` and `---` lines are
    skipped, and an unlabelled line before any prefix is the channel intro, which the
    renderer speaks with Speaker 2's voice.

    Checking each physical line separately — what this did until 2026-08-13 — asks the
    engine about a fragment that is never synthesized on its own.
    """
    out: list[Turn] = []
    buffer = ""
    current: int | None = None

    def flush() -> None:
        # engine.py:_AivisTimeline.flush_turn — `if not (text and speaker): return`, and
        # that truthiness test is why `Speaker 0:` is never spoken: 0 is falsy there. A
        # `current is not None` test here would report a line the render silently drops.
        nonlocal buffer
        if buffer and current:
            out.append(Turn(current, buffer))
        buffer = ""

    for raw in cleaned.splitlines():
        line = raw.strip()
        if not line or line.startswith("##") or re.match(r"^-{3,}$", line):
            continue
        if line in MARKER_SILENCE:
            flush()
            continue
        m = SPEAKER_RE.match(line)
        if m:
            flush()
            current = int(m.group(2))
            buffer = m.group(3).strip()
        elif current is None:
            if not line.startswith("["):
                out.append(Turn(CHANNEL_INTRO_SPEAKER, line))
        else:
            buffer = f"{buffer} {line}".strip()
    flush()
    return out


def spoken_lines(cleaned: str) -> list[str]:
    """The text the engine actually receives, in render order."""
    return [t.text for t in spoken_turns(cleaned)]


def engine_inputs(turn: Turn) -> list[str]:
    """The strings the renderer will actually POST for this turn.

    engine.py:_synthesize_turn splits a turn with _chunk_japanese_text before calling
    /audio_query, so a long turn is analysed by the engine in pieces. Reading back the
    whole turn in one query can produce a reading the render never makes.
    """
    return _chunk_japanese_text(turn.text) or [turn.text]


def changed_turns(previous: str, current: str) -> list[tuple[int, Turn]]:
    """Turns present in `current` that a previous version of the same script did not have.

    Diffs the spoken lines themselves, not line numbers: inserting one turn must not mark
    every turn after it as changed. difflib's autojunk heuristic is disabled because a
    script legitimately repeats short turns (そうですね。), and treating those as junk
    would move the alignment.

    Returns (index in current's spoken order, Turn).
    """
    prev_turns, cur_turns = spoken_turns(previous), spoken_turns(current)
    # Compared as (speaker, text), not text alone. Production has two configured voices, so moving a
    # line from speaker 1 to speaker 2 changes the voice that says it — and with it the reading the
    # engine produces. Diffing the text only, that edit looked like no change at all and the tool
    # read back zero lines (prepush codex 2026-08-13).
    sm = difflib.SequenceMatcher(a=list(prev_turns), b=list(cur_turns), autojunk=False)
    out: list[tuple[int, Turn]] = []
    for tag, _i1, _i2, j1, j2 in sm.get_opcodes():
        if tag in ("replace", "insert"):
            out.extend((j, cur_turns[j]) for j in range(j1, j2))
    return out



# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


def check_line(
    text: str,
    speaker: int,
    session: requests.Session,
    *,
    segmentation: bool = False,
    elevate: bool = False,
    always_report: bool = False,
) -> dict | None:
    """Return a finding dict if the line looks suspicious, else None.

    segmentation   also compare accent-phrase boundaries (see boundary_conflicts).
    elevate        rank a Layer 0/1 disagreement ELEVATED instead of REVIEW. Set in
                   changed-lines mode, where a disagreement is the primary signal because
                   there is no previous reading of that line to compare against.
    always_report  emit a READBACK entry for a clean line too, so changed-lines mode shows
                   the reading of everything that changed — that is how the 2026-08-13
                   misreadings were caught, none of which any rule in this module predicts.
    """
    # One /audio_query per line either way — the flat reading is the phrases joined,
    # so `segmentation` only decides whether the boundaries are USED, never whether
    # they are fetched.
    try:
        phrases = engine_phrases(text, speaker, session)
    except Exception as exc:  # engine down / bad line — surface, do not swallow
        return {"line": text, "reason": "engine_error", "priority": "ERROR", "detail": str(exc)[:120]}
    eng = "".join(phrases)

    # a line with visible content but no reading is SILENT in the audio (the △△ bug)
    if text.strip() and not eng.strip():
        return {
            "line": text,
            "engine_reading": "",
            "reason": "empty_reading",
            "priority": "HIGH",
            "detail": "line produces NO audio — silent content loss",
        }

    hazards = [
        why for src, (bad, why) in HAZARD_READINGS.items() if src in text and (bad in eng if bad else not eng.strip())
    ]

    # Layer 1 is only a RANKER, and it is blind on ASCII: the engine says エピソドワン
    # for "Episode 1" while pyopenjtalk spells the letters out. Comparing those produces
    # pure noise, and acronyms are already owned by the glossary (Layer 1 of the 3-layer
    # system), so skip the G2P vote on lines containing Latin letters. (Those lines are
    # still READ BACK in changed-lines mode — 45% of Ep014's lines contain "Episode N",
    # and a human reading the kana is not blinded by them.)
    ojt = None if re.search(r"[A-Za-z]", text) else openjtalk_reading(text)
    disagree = bool(ojt) and _norm(ojt) != _norm(eng)
    conflicts = boundary_conflicts(phrases, openjtalk_phrases(text)) if (segmentation and ojt is not None) else []

    if not (hazards or disagree or conflicts or always_report):
        return None

    if hazards:
        reason, priority = "known_hazard", "HIGH"
    elif disagree or conflicts:
        reason = "g2p_disagreement" if disagree else "segmentation_disagreement"
        priority = "ELEVATED" if elevate else "REVIEW"
    else:
        reason, priority = "readback", "READBACK"

    finding = {
        "line": text,
        "engine_reading": eng,
        "openjtalk_reading": ojt,
        "reason": reason,
        "hazards": hazards,
        "priority": priority,
    }
    if segmentation:
        finding["engine_reading_segmented"] = " | ".join(phrases)
    if conflicts:
        finding["boundary_conflicts"] = conflicts
    return finding


def preflight(session: requests.Session, speaker: int) -> None:
    """Prove the engine is up AND returning readings, or raise EngineUnavailable."""
    try:
        r = session.get(f"{AIVISSPEECH_URL}/version", timeout=10)
        r.raise_for_status()
        probe = "".join(engine_phrases(PROBE_LINE, speaker, session))
    except Exception as exc:
        raise EngineUnavailable(
            f"AivisSpeech unreachable at {AIVISSPEECH_URL} ({str(exc)[:120]}). "
            f"Start it: docker run -d --name aivisspeech -p 10101:10101 "
            f"ghcr.io/aivis-project/aivisspeech-engine:cpu-latest"
        ) from exc
    if not probe.strip():
        raise EngineUnavailable(
            f"AivisSpeech answered at {AIVISSPEECH_URL} but returned NO reading for the probe "
            f"line {PROBE_LINE!r} (speaker {speaker}) — every line would look clean."
        )


def check_script(
    path: Path,
    speaker: int | None = None,
    *,
    baseline: Path | None = None,
    segmentation: bool = True,
    elevate: bool | None = None,
    session: requests.Session | None = None,
) -> dict:
    """Check one script.

    speaker    force every line to one voice (escape hatch). None = resolve per line.
    baseline   a previous version of the same script; only the lines that differ from it
               are checked, and every one of them is reported with its reading.
    elevate    default: on in changed-lines mode, off for a full script.
    """
    raw = path.read_text(encoding="utf-8")
    cleaned = clean_script_for_tts(raw)  # exactly what gets sent to the engine
    turns = spoken_turns(cleaned)

    voices = VoiceAssignment.forced_to(speaker) if speaker is not None else resolve_voices(cleaned)

    def voice_of(turn: Turn) -> int:
        return voices.for_speaker(turn.speaker)

    if baseline is not None:
        selected = changed_turns(clean_script_for_tts(baseline.read_text(encoding="utf-8")), cleaned)
        always_report = True
    else:
        selected = list(enumerate(turns))
        always_report = False
    if elevate is None:
        elevate = baseline is not None

    session = session or requests.Session()
    preflight(session, voice_of(turns[0]) if turns else voices.for_speaker(1))

    findings = []
    for idx, turn in selected:
        for chunk_no, chunk in enumerate(engine_inputs(turn)):
            f = check_line(
                chunk,
                voice_of(turn),
                session,
                segmentation=segmentation,
                elevate=elevate,
                always_report=always_report,
            )
            if f:
                findings.append(
                    {**f, "index": idx, "chunk": chunk_no, "speaker": turn.speaker, "voice": voice_of(turn)}
                )

    def count(p: str) -> int:
        return sum(1 for f in findings if f.get("priority") == p)

    return {
        "script": str(path),
        "baseline": str(baseline) if baseline else None,
        "mode": "changed" if baseline else "full",
        "voices": voices.as_report(),
        "lines_total": len(turns),
        "lines_checked": len(selected),
        "findings": findings,
        "high": count("HIGH"),
        "elevated": count("ELEVATED"),
        "review": count("REVIEW"),
        "readback": count("READBACK"),
        "errors": count("ERROR"),
    }


def _print_report(rep: dict, verbose: bool) -> None:
    name = Path(rep["script"]).parent.name or Path(rep["script"]).name
    v = rep["voices"]
    voice_note = (
        f"voice forced={v['forced']}"
        if "forced" in v
        else f"S1={v['speaker1']} S2={v['speaker2']}{' SWAPPED' if v['swapped'] else ''}"
    )
    print(
        f"{name:34} {rep['mode']:7} lines={rep['lines_checked']}/{rep['lines_total']:<5} "
        f"HIGH={rep['high']:<3} elevated={rep['elevated']:<3} review={rep['review']:<4} "
        f"err={rep['errors']:<3} [{voice_note}]"
    )
    if not verbose:
        return
    for f in rep["findings"]:
        if f["reason"] == "engine_error":
            print(f"  [{f['priority']}] #{f['index']} ENGINE ERROR: {f['detail']}")
            continue
        print(f"  [{f['priority']}] #{f['index']} Speaker {f['speaker']} (voice {f['voice']}): {f['line'][:70]}")
        print(f"      READ: {f.get('engine_reading_segmented') or f['engine_reading']}")
        for c in f.get("boundary_conflicts", []):
            print(f"      BOUNDARY: {c}")
        for h in f.get("hazards", []):
            print(f"      HAZARD: {h}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scripts", nargs="+", type=Path)
    ap.add_argument(
        "--changed-vs",
        type=Path,
        metavar="PREVIOUS",
        help="previous version of the script; check only the lines that differ (one script only)",
    )
    ap.add_argument(
        "--speaker",
        type=int,
        default=None,
        help="force every line to this voice ID (default: resolve the voice that will actually speak it)",
    )
    ap.add_argument("--no-segmentation", action="store_true", help="skip the accent-phrase boundary comparison")
    ap.add_argument(
        "--no-elevate",
        action="store_true",
        help="keep changed-line disagreements at REVIEW so tts_reading_judge (Layer 2) consumes them",
    )
    ap.add_argument("-v", "--verbose", action="store_true", help="print every finding with its reading")
    ap.add_argument("--json", type=Path, help="write full report here")
    args = ap.parse_args(argv)

    if args.changed_vs is not None:
        if len(args.scripts) != 1:
            print("--changed-vs takes exactly one script", file=sys.stderr)
            return 2
        if not args.changed_vs.exists():
            print(f"MISSING baseline: {args.changed_vs}", file=sys.stderr)
            return 2

    reports, missing = [], 0
    for p in args.scripts:
        if not p.exists():
            print(f"MISSING: {p}", file=sys.stderr)
            missing += 1
            continue
        try:
            rep = check_script(
                p,
                args.speaker,
                baseline=args.changed_vs,
                segmentation=not args.no_segmentation,
                elevate=False if args.no_elevate else None,
            )
        except (EngineUnavailable, VoiceResolutionError) as exc:
            print(f"CANNOT CHECK {p}: {exc}", file=sys.stderr)
            return 2
        reports.append(rep)
        _print_report(rep, args.verbose or rep["mode"] == "changed")
        if rep["mode"] == "changed" and rep["lines_checked"] == 0:
            print("  no changed lines — the two versions speak the same text")
        if rep["elevated"]:
            # Say it out loud rather than let it be discovered: tts_reading_judge (Layer 2)
            # only judges priority == REVIEW, so elevating a finding takes it OUT of the
            # judge's queue and hands it to a human.
            print(
                f"  {rep['elevated']} ELEVATED finding(s) are for a HUMAN to read — "
                f"tts_reading_judge (Layer 2) consumes REVIEW only; re-run with --no-elevate to judge them"
            )

    if args.json:
        args.json.write_text(json.dumps(reports, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"\nreport -> {args.json}")

    if missing or not reports or any(r["errors"] for r in reports):
        return 2
    return 1 if any(r["high"] or r["elevated"] for r in reports) else 0


if __name__ == "__main__":
    raise SystemExit(main())
