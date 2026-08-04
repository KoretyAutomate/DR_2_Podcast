"""TTS misreading Layer 2 — LLM as JUDGE over the Layer 0/1 candidates.

PLAN.md Step 7. Consumes the REVIEW-priority findings from `tts_reading_check` and asks a
model whether the engine's reading is plausible for that sentence.

DESIGN — judge, do not generate. The obvious approach (ask the LLM for the correct reading,
diff against the engine) checks a fallible engine against a fallible oracle: Japanese
readings of proper nouns and technical terms are exactly where an LLM hallucinates. Judging
a supplied candidate is a far easier task and collapses the false-positive rate.

Role/model separation: the ROLE is "reading judge"; the MODEL is swappable via
TTS_JUDGE_MODEL / TTS_JUDGE_BASE_URL. Defaults to the Smart model because Scientific
Accuracy > TTS Quality only holds if the judge is actually competent at Japanese.

Usage:
    python -m dr2_podcast.tools.tts_reading_judge <report.json> [--out OUT] [--limit N]
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

from dr2_podcast.tools.tts_reading_check import _norm

JUDGE_MODEL = os.environ.get("TTS_JUDGE_MODEL", "Intel/Qwen3.5-122B-A10B-int4-AutoRound")
JUDGE_BASE_URL = os.environ.get("TTS_JUDGE_BASE_URL", "http://localhost:8000/v1")
CONCURRENCY = int(os.environ.get("TTS_JUDGE_CONCURRENCY", "4"))

SYSTEM = """あなたは日本語音声合成の読み上げ校正者です。
与えられた「文」と、TTSエンジンがその文を読み上げる際の「読み(カタカナ)」を照合し、
読みが正しいかどうかだけを判定してください。

重要な制約:
- あなたの仕事は「判定」であり、文全体の正しい読みを生成することではありません。
- 読みが自然で意味の通る読み方であれば OK と判定してください。
- 表記のゆれは誤りではありません。必ず OK と判定してください:
  長音の書き方(データ=デエタ、コーヒー=コオヒイ、フレームワーク=フレエムワアク、
  ニュース=ニュウス、レビュー=レビュウ、平均=ヘエキン、経験=ケエケン)、句読点や記号の扱い。
  カタカナ語が「ー」ではなく母音を重ねて書かれていても、発音は同じなので OK です。
- 助詞の「は」がワ、「を」がオ と読まれるのは正しい日本語です。OK です。
- 誤りと判定するのは、意味が変わる読み間違いだけです。
  例: 「強さ」をコワサ(怖さ)と読む、「建前」をケンマエと読む、「五つ目」をゴツメと読む、
      「一行」をイッコオと読む、動詞連用形+方(読み方など)をヨミホウと読む。

出力は次のJSONのみ。説明文やマークダウンは一切書かないこと:
{"verdict":"OK"|"MISREAD","span":"誤読された語(MISREADのときのみ、なければ空文字)","expected":"正しい読みカタカナ(なければ空文字)","confidence":"high"|"low"}"""

USER_TMPL = """文:
{line}

エンジンの読み:
{reading}

この読みは正しいですか。JSONのみで答えてください。"""

_JSON_RE = re.compile(r"\{.*\}", re.S)


def judge_one(finding: dict, session: requests.Session) -> dict:
    line = finding["line"]
    reading = eng_reading = finding.get("engine_reading", "")
    try:
        r = session.post(
            f"{JUDGE_BASE_URL}/chat/completions",
            json={
                "model": JUDGE_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": USER_TMPL.format(line=line, reading=reading)},
                ],
                "temperature": 0,
                "max_tokens": 300,
                # The Smart model is a reasoning model: with thinking ON it spends the whole
                # budget in `reasoning`, returns content=None, and takes ~22s per line. Off,
                # it answers this judge-a-candidate task correctly in ~0.9s. 25x cheaper for
                # the same verdict, which is what makes judging all 222 candidates practical.
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=180,
        )
        r.raise_for_status()
        msg = r.json()["choices"][0]["message"]
        # fall back to `reasoning` if a thinking-enabled backend still routes output there
        content = msg.get("content") or msg.get("reasoning") or ""
    except Exception as exc:
        return {**finding, "judge": {"verdict": "ERROR", "detail": str(exc)[:140]}}

    m = _JSON_RE.search(content)
    if not m:
        return {**finding, "judge": {"verdict": "ERROR", "detail": f"unparseable: {content[:100]}"}}
    try:
        verdict = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {**finding, "judge": {"verdict": "ERROR", "detail": f"bad json: {m.group(0)[:100]}"}}

    # Mechanically discard vacuous MISREADs rather than trusting the prompt to prevent them.
    # Measured on the 2026-07-28 smoke run: the judge flagged 27/42, and the overwhelming
    # majority were katakana long-vowel NOTATION (デエタ vs データ, コオヒイ vs コーヒー,
    # フレエムワアク vs フレームワーク) — identical pronunciation, not a misreading. Others
    # returned span == expected, which asserts nothing. Both collapse under _norm().
    if verdict.get("verdict") == "MISREAD":
        span, exp = verdict.get("span") or "", verdict.get("expected") or ""
        n_exp, n_eng = _norm(exp), _norm(eng_reading)
        reason = None
        if not exp:
            reason = "vacuous (no expected reading given)"
        elif _norm(span) == n_exp:
            reason = "notation-only (span ≡ expected after normalisation)"
        elif n_exp and n_exp in n_eng:
            # The judge often returns the SOURCE text as `span` (e.g. 一番した) rather than a
            # reading, so span/expected cannot be compared. But if the engine ALREADY produces
            # the reading the judge is asking for, there is nothing to fix either way.
            reason = "engine already produces the expected reading"
        if reason:
            verdict = {**verdict, "verdict": "OK", "downgraded_from": "MISREAD", "downgrade_reason": reason}
    return {**finding, "judge": verdict}


def judge_report(report: list[dict], limit: int | None = None) -> list[dict]:
    """Judge every REVIEW-priority finding. HIGH findings are already decided by Layer 0."""
    session = requests.Session()
    out = []
    for ep in report:
        cands = [f for f in ep["findings"] if f.get("priority") == "REVIEW"]
        if limit:
            cands = cands[:limit]
        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            judged = list(pool.map(lambda f: judge_one(f, session), cands))
        misread = [j for j in judged if j["judge"].get("verdict") == "MISREAD"]
        errors = [j for j in judged if j["judge"].get("verdict") == "ERROR"]
        out.append(
            {
                "script": ep["script"],
                "judged": len(judged),
                "misread": misread,
                "n_misread": len(misread),
                "n_ok": sum(1 for j in judged if j["judge"].get("verdict") == "OK"),
                "n_error": len(errors),
            }
        )
        name = Path(ep["script"]).parent.name
        print(f"{name:34} judged={len(judged):<4} MISREAD={len(misread):<4} ok={out[-1]['n_ok']:<4} err={len(errors)}")
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("report", type=Path, help="JSON from tts_reading_check --json")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--limit", type=int, help="max candidates per episode (smoke testing)")
    args = ap.parse_args(argv)

    if not args.report.exists():
        print(f"missing: {args.report}", file=sys.stderr)
        return 2
    report = json.loads(args.report.read_text(encoding="utf-8"))
    print(f"judge model: {JUDGE_MODEL} @ {JUDGE_BASE_URL}\n")
    results = judge_report(report, args.limit)

    total = sum(r["n_misread"] for r in results)
    print(f"\nTOTAL MISREAD: {total} | errors: {sum(r['n_error'] for r in results)}")
    if args.out:
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"report -> {args.out}")
    return 1 if total else 0


if __name__ == "__main__":
    raise SystemExit(main())
