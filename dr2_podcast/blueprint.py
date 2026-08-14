"""The episode blueprint: derived where it can be, validated where it must be.

PLAN.md Step 2. The episode takes a fixed shape — speaker 1 states the conclusion at a calibrated
confidence, speaker 2 drives the nine-step interrogation, speaker 1 answers with the evidence.

**What is derived and what is authored.** Which steps run, what each asks, which artifact answers it,
what it contributes to the verdict, and the confidence the opening states are all computed here from
`step_pack.json` and the structured GRADE record. What a claim SAYS is authored — and under the
allocation table it is Claude's to author, not the Smart model's. So the scaffold is built with empty
claim lists and `authored: false`, and it says so rather than looking finished.

**What the validator can and cannot see.** The three-way hedge equality (opening == step 10 ==
GRADE-derived 確信度) is a METADATA check. It cannot see an opening whose label says 中程度 while its
prose says 「確実に効きます」 — that is a tier-2 read, on the blueprint and on the spoken script,
because the script can drop or inflate a conclusion the blueprint stated correctly. Everything here
is the cheap half, and the cheap half must not be mistaken for the sufficient one.
"""

from __future__ import annotations

from typing import Any

from dr2_podcast.schemas import CONFIDENCE_LADDER, schema_errors

#: Prior, numbers, update, verdict. An episode that skips one of these is not performing the method,
#: whatever else it does — so unlike the others they cannot be skipped with a reason.
MANDATORY_STEPS: tuple[int, ...] = (1, 4, 9, 10)

#: The name each step goes by in the episode.
STEP_NAMES: dict[int, str] = {
    1: "事前確率",
    2: "関連研究の探索",
    3: "研究デザインの評価",
    4: "数字の確認",
    5: "資金提供者の確認",
    6: "再現性のチェック",
    8: "バイアスと誤謬の特定",
    9: "ベイズ更新",
    10: "確率の主張",
}

#: Roughly how long each step gets. Derived from its role, not from how interesting it turned out:
#: the numbers and the update carry the argument, the rest support it.
STEP_TARGET_CHARS: dict[int, int] = {
    1: 1200, 2: 1600, 3: 2400, 4: 3000, 5: 1600, 6: 2000, 8: 2400, 9: 2400, 10: 1200,
}

OPENING_TARGET_CHARS = 900


class BlueprintUnavailable(RuntimeError):
    """The blueprint cannot be built from what this run produced, and why."""


def _verdict_allows(confidence: str, contributions: list[str]) -> bool:
    """Whether a confidence is reachable from what the steps actually contributed.

    Narrow on purpose: an episode whose every step contributes `downgrade` cannot land at 高い. It
    says nothing about the magnitude of the update — that is GRADE's job and it has already been
    done — only that the direction and the destination are not in open contradiction.
    """
    if not contributions:
        return True
    if CONFIDENCE_LADDER.index(confidence) < CONFIDENCE_LADDER.index("高い"):
        return True
    return any(c != "downgrade" for c in contributions)


def build_skeleton(step_pack: dict[str, Any], confidence_ja: str) -> dict[str, Any]:
    """The derived scaffold: every step the pack can answer, in interrogation order.

    The skip decision comes from the pack's ``sufficiency``, not from anyone's taste — and `absent`
    does not always mean skip. Where the absence IS the finding, undisclosed funding being Ep09's
    whole thesis, the step runs and says so; that is why only steps with nothing at all to project
    are skipped here.
    """
    if confidence_ja not in CONFIDENCE_LADDER:
        raise BlueprintUnavailable(f"{confidence_ja!r} is not a 確信度; the blueprint's opening has nothing to state")

    steps: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for key in sorted(step_pack.get("steps", {}), key=int):
        number = int(key)
        entry = step_pack["steps"][key]
        if entry["sufficiency"] == "absent":
            skipped.append(
                {
                    "step": number,
                    "skip_reason": _skip_reason(number, entry),
                }
            )
            continue
        steps.append(
            {
                "step": number,
                "name": STEP_NAMES[number],
                "driver": 2,
                "answerer": 1,
                "question_ja": entry["question_ja"],
                "source_artifact": f"step_pack.json#/steps/{key}",
                "verdict_contribution": entry["verdict_contribution"],
                "target_chars": STEP_TARGET_CHARS[number],
                "beats": ["問い", "証拠", "数字", "限界"],
                "claims": [],
            }
        )

    missing = [n for n in MANDATORY_STEPS if n not in {s["step"] for s in steps}]
    if missing:
        reasons = {s["step"]: s["skip_reason"] for s in skipped}
        raise BlueprintUnavailable(
            "the episode cannot be built without "
            + ", ".join(f"step {n} ({STEP_NAMES[n]}): {reasons.get(n, 'not in the step pack')}" for n in missing)
            + ". These four are the method — prior, numbers, update, verdict — and an episode that "
            "skips one is not performing it."
        )

    return {
        "schema_version": 1,
        "authored": False,
        "confidence_ja": confidence_ja,
        "opening": {
            "step": 0,
            "arc_role": "conclusion_first",
            "speaker": 1,
            "hedge_level": confidence_ja,
            "claim_ja": "",
            "target_chars": OPENING_TARGET_CHARS,
        },
        "steps": steps,
        "skipped": skipped,
    }


def _skip_reason(number: int, entry: dict[str, Any]) -> str:
    stated = entry.get("answer", {}).get("unavailable")
    if stated:
        return f"{STEP_NAMES[number]}: {stated}"
    return f"{STEP_NAMES[number]}: この回では扱える証拠がありませんでした"


def blueprint_errors(blueprint: dict[str, Any], artifacts: dict[str, str]) -> list[str]:
    """Every write-time rule, as reasons a person can act on.

    ``artifacts`` maps an id to its text — the SOT, and any extraction a claim quotes. A claim whose
    span is in none of them is a claim about a document nobody has.
    """
    errors = schema_errors("blueprint", blueprint)
    if errors:
        return errors

    confidence = blueprint["confidence_ja"]
    errors.extend(_hedge_errors(blueprint, confidence))
    errors.extend(_coverage_errors(blueprint))

    contributions = [s["verdict_contribution"] for s in blueprint["steps"]]
    if not _verdict_allows(confidence, contributions):
        errors.append(
            f"/confidence_ja: every step contributes 'downgrade' and the episode still lands at "
            f"{confidence!r} — the interrogation and the verdict cannot disagree about direction"
        )

    for index, step in enumerate(blueprint["steps"]):
        for claim in step["claims"]:
            span = claim["evidence"]["quoted_span"]
            if not any(span in text for text in artifacts.values()):
                errors.append(
                    f"/steps/{index}/claims: {claim['id']!r} quotes {span[:40]!r}, which is in none of "
                    f"the artifacts this episode is built from"
                )
            if CONFIDENCE_LADDER.index(claim["hedge_level"]) > CONFIDENCE_LADDER.index(confidence):
                errors.append(
                    f"/steps/{index}/claims: {claim['id']!r} is hedged at {claim['hedge_level']!r}, "
                    f"stronger than the episode's own {confidence!r}"
                )
    return errors


def _hedge_errors(blueprint: dict[str, Any], confidence: str) -> list[str]:
    """The three labels that must agree. A metadata check, and only that."""
    errors = []
    if blueprint["opening"]["hedge_level"] != confidence:
        errors.append(
            f"/opening/hedge_level: {blueprint['opening']['hedge_level']!r} but the GRADE-derived "
            f"確信度 is {confidence!r} — the opening states a confidence nobody computed"
        )
    for step in blueprint["steps"]:
        if step["step"] != 10:
            continue
        for claim in step["claims"]:
            if claim["hedge_level"] != confidence:
                errors.append(
                    f"/steps step 10: claim {claim['id']!r} is hedged at {claim['hedge_level']!r} while "
                    f"the opening states {confidence!r}; the episode contradicts itself end to end"
                )
    return errors


def _coverage_errors(blueprint: dict[str, Any]) -> list[str]:
    errors = []
    present = {s["step"] for s in blueprint["steps"]}
    skipped = {s["step"] for s in blueprint["skipped"]}
    for number in MANDATORY_STEPS:
        if number not in present:
            errors.append(
                f"/steps: step {number} ({STEP_NAMES[number]}) is mandatory — prior, numbers, update "
                f"and verdict are the method, and an episode without one is not performing it"
            )
    overlap = present & skipped
    if overlap:
        errors.append(f"/skipped: step(s) {sorted(overlap)} are both included and skipped")
    return errors
