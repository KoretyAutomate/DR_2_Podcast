"""Characterization-golden generator for pipeline_script's section generators.

Covers _generate_section, _generate_section_subsplit, _generate_section_single
and _run_condense_pass. The Smart Model is replaced by a deterministic fake, so
the golden pins two things:

  1. the (text, count, deficit) tuple each generator returns, and
  2. the EXACT prompts handed to the model — system, user, max_tokens,
     temperature, frequency_penalty — for every call, in order.

(2) is the part that matters. These functions exist to build prompts; a refactor
that quietly drops a checklist block or a retry-feedback append would still
return a plausible tuple.

    python -m tests.gen_section_golden          # write tests/golden_section.json
    python -m tests.gen_section_golden --check  # regenerate and diff
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dr2_podcast import pipeline_script as ps

GOLDEN_PATH = Path(__file__).parent / "golden_section.json"

EN_CFG = {
    "name": "English",
    "tts_code": "a",
    "instruction": "Write all content in English.",
    "speech_rate": 150,
    "length_unit": "words",
    "prompt_unit": "word",
}
JA_CFG = {
    "name": "日本語 (Japanese)",
    "tts_code": "j",
    "instruction": "すべてのコンテンツを日本語で書いてください。",
    "speech_rate": 350,
    "length_unit": "chars",
    "prompt_unit": "character",
}

ROLES = {
    "presenter": {"label": "Host 1", "stance": "expert", "personality": "Enthusiastic communicator."},
    "questioner": {"label": "Host 2", "stance": "skeptic", "personality": "Analytical generalist."},
}

CHECKLIST = [
    {"question": "What did the trial measure?", "answer": "A" * 200},
    {"question": "How large was the effect?", "answer": "Effect was modest."},
    {"question": "Who was studied?", "answer": "Adults aged 40-70."},
]


class FakeModel:
    """Deterministic stand-in for _call_smart_model that records every call.

    `yield_ratio` controls output length as a fraction of the requested budget,
    which is how the retry path (word_count < 75% of budget) gets exercised.
    """

    def __init__(self, language_config, budget, yield_ratio=1.0, escalate_on_retry=True):
        self.language_config = language_config
        self.budget = budget
        self.yield_ratio = yield_ratio
        self.escalate_on_retry = escalate_on_retry
        self.calls: list[dict] = []

    def __call__(self, **kwargs):
        self.calls.append(
            {
                "system": kwargs.get("system", ""),
                "user": kwargs.get("user", ""),
                "max_tokens": kwargs.get("max_tokens"),
                "temperature": kwargs.get("temperature"),
                "frequency_penalty": kwargs.get("frequency_penalty"),
            }
        )
        ratio = self.yield_ratio
        if self.escalate_on_retry and len(self.calls) > 1:
            ratio = 1.0
        n = max(1, int(self.budget * ratio))
        if self.language_config["length_unit"] == "chars":
            body = "あ" * n
            return f"Host 1: {body}\nHost 2: はい。"
        words = " ".join(["word"] * n)
        return f"Host 1: {words}\nHost 2: Right."


def _section_config(section_id, budget, length_unit, *, checklist=True):
    return {
        "section_id": section_id,
        "word_budget": budget,
        "length_unit": length_unit,
        "pacing": "Measured and thoughtful.",
        "checklist_items": list(CHECKLIST) if checklist else [],
    }


def _run(fn, cfg, previous_lines, model, language_config, *, channel_intro="", target_min=30):
    deps = ps.SectionGenDeps(
        call_smart_model=model,
        language_config=language_config,
        session_roles=ROLES,
        topic_name="Coffee and productivity",
        channel_intro=channel_intro,
        target_min=target_min,
    )
    text, count, deficit = fn(cfg, previous_lines, deps)
    return {
        "text": text,
        "count": count,
        "deficit": deficit,
        "calls": model.calls,
    }


def _cases() -> dict:
    out: dict = {}

    # --- English, one call per section id -----------------------------------
    for sid in ("opening", "evidence", "synthesis", "closing"):
        model = FakeModel(EN_CFG, 300)
        out[f"en_{sid}"] = _run(
            ps._generate_section, _section_config(sid, 300, "words"), ["Host 1: prior line."], model, EN_CFG
        )

    # opening with and without a channel intro (different directive text)
    model = FakeModel(EN_CFG, 300)
    out["en_opening_with_intro"] = _run(
        ps._generate_section,
        _section_config("opening", 300, "words"),
        [],
        model,
        EN_CFG,
        channel_intro="Welcome to the Deep Research Podcast.",
    )

    # no checklist items -> the "(No checklist items...)" placeholder
    model = FakeModel(EN_CFG, 300)
    out["en_no_checklist"] = _run(
        ps._generate_section,
        _section_config("evidence", 300, "words", checklist=False),
        ["Host 2: earlier."],
        model,
        EN_CFG,
    )

    # first section: no previous lines -> the "(This is the first section...)" lead-in
    model = FakeModel(EN_CFG, 300)
    out["en_no_previous_lines"] = _run(
        ps._generate_section, _section_config("evidence", 300, "words"), [], model, EN_CFG
    )

    # long previous_lines: only the last 5 are used as lead-in
    model = FakeModel(EN_CFG, 300)
    out["en_long_previous_lines"] = _run(
        ps._generate_section,
        _section_config("evidence", 300, "words"),
        [f"Host 1: line {i}." for i in range(12)],
        model,
        EN_CFG,
    )

    # under-floor first attempt -> retry with feedback appended (2 model calls)
    model = FakeModel(EN_CFG, 400, yield_ratio=0.10)
    out["en_retry_then_pass"] = _run(ps._generate_section, _section_config("evidence", 400, "words"), [], model, EN_CFG)

    # both attempts under floor -> loop exhausts, deficit is non-zero
    model = FakeModel(EN_CFG, 400, yield_ratio=0.10, escalate_on_retry=False)
    out["en_retry_still_short"] = _run(
        ps._generate_section, _section_config("evidence", 400, "words"), [], model, EN_CFG
    )

    # --- Japanese ------------------------------------------------------------
    # below the sub-section threshold -> single call
    model = FakeModel(JA_CFG, 2000)
    out["ja_below_threshold"] = _run(
        ps._generate_section, _section_config("evidence", 2000, "chars"), ["Host 1: 前の行。"], model, JA_CFG
    )

    # exactly at the threshold -> still a single call (boundary is strict >)
    model = FakeModel(JA_CFG, ps._JA_SUBSECTION_THRESHOLD)
    out["ja_at_threshold"] = _run(
        ps._generate_section,
        _section_config("evidence", ps._JA_SUBSECTION_THRESHOLD, "chars"),
        [],
        model,
        JA_CFG,
    )

    # one over the threshold -> sub-split path
    model = FakeModel(JA_CFG, ps._JA_SUBSECTION_THRESHOLD + 1)
    out["ja_just_over_threshold"] = _run(
        ps._generate_section,
        _section_config("evidence", ps._JA_SUBSECTION_THRESHOLD + 1, "chars"),
        [],
        model,
        JA_CFG,
    )

    # large budget -> several sub-parts, checklist distributed round-robin
    model = FakeModel(JA_CFG, 12000)
    out["ja_subsplit_many_parts"] = _run(
        ps._generate_section, _section_config("evidence", 12000, "chars"), ["Host 1: 前。"], model, JA_CFG
    )

    # sub-split opening: channel_intro must reach part 0 only
    model = FakeModel(JA_CFG, 9000)
    out["ja_subsplit_opening_intro"] = _run(
        ps._generate_section,
        _section_config("opening", 9000, "chars"),
        [],
        model,
        JA_CFG,
        channel_intro="ディープリサーチポッドキャストへようこそ。",
    )

    # target_min changes budget_pct in the prompt
    model = FakeModel(JA_CFG, 2000)
    out["ja_short_episode"] = _run(
        ps._generate_section,
        _section_config("evidence", 2000, "chars"),
        [],
        model,
        JA_CFG,
        target_min=10,
    )

    # --- condense pass -------------------------------------------------------
    def _long_script(cfg, units):
        """A script comfortably over target so the early-return does not fire."""
        if cfg["length_unit"] == "chars":
            return "\n".join(f"Host {1 + i % 2}: {'あ' * 60}。" for i in range(units // 60 + 2))
        return "\n".join(f"Host {1 + i % 2}: {' '.join(['word'] * 20)}" for i in range(units // 20 + 2))

    def _condense(name, cfg, model, *, script=None, inventory=None, target=500):
        out[f"condense_{name}"] = {
            "text": ps._run_condense_pass(
                script if script is not None else _long_script(cfg, 900),
                {"evidence": [{"question": "Q1", "answer": "A1"}]} if inventory is None else inventory,
                target,
                "Target 500 units.",
                ps.SectionGenDeps(
                    call_smart_model=model,
                    language_config=cfg,
                    session_roles=ROLES,
                    topic_name="Coffee and productivity",
                ),
            ),
            "calls": model.calls,
        }

    # model returns something shorter -> condensed text is used
    for name, cfg in (("en", EN_CFG), ("ja", JA_CFG)):
        _condense(name, cfg, FakeModel(cfg, 400, escalate_on_retry=False))

    # already under target+5% -> early return, model never called
    _condense("under_target", EN_CFG, FakeModel(EN_CFG, 400), script="Host 1: short.", target=500)

    # exactly at the 105% buffer boundary -> still an early return
    boundary_script = "\n".join(f"Host 1: {' '.join(['word'] * 21)}" for _ in range(25))
    _condense("at_buffer_boundary", EN_CFG, FakeModel(EN_CFG, 400), script=boundary_script, target=500)

    # model returns something LONGER -> original kept
    _condense("no_reduction", EN_CFG, FakeModel(EN_CFG, 5000, escalate_on_retry=False))

    class Boom(FakeModel):
        def __call__(self, **kwargs):
            super().__call__(**kwargs)
            raise RuntimeError("model exploded")

    # model raises -> original kept, failure swallowed by design
    _condense("model_raises", EN_CFG, Boom(EN_CFG, 400))

    # empty inventory, still over target
    _condense("empty_inventory", EN_CFG, FakeModel(EN_CFG, 400, escalate_on_retry=False), inventory={})

    return out


def generate() -> dict:
    return _cases()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    current = generate()
    if args.check:
        if not GOLDEN_PATH.exists():
            print(f"golden missing: {GOLDEN_PATH}")
            return 1
        stored = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
        if stored == current:
            print(f"golden matches ({len(current)} cases)")
            return 0
        for key in sorted(set(stored) | set(current)):
            if stored.get(key) != current.get(key):
                print(f"DRIFT: {key}")
        return 1

    GOLDEN_PATH.write_text(json.dumps(current, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"wrote {GOLDEN_PATH} ({len(current)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
