"""The blueprint's write-time validator — PLAN.md Step 2.

The exit criterion is a mutation matrix, not a spot check, and it is explicit that the cheap check
must not be mistaken for a sufficient one: the three-way hedge equality is METADATA, and every
label-equal mutation below is one it cannot see. Those are asserted as invisible here, on purpose,
so their absence from this file is never read as coverage.
"""

from __future__ import annotations

import pytest

from dr2_podcast.blueprint import (
    MANDATORY_STEPS,
    BlueprintUnavailable,
    blueprint_errors,
    build_skeleton,
)

SOT = "# Source of Truth\n\nAbsolute risk reduction was 5.0% for hip fracture at 12 months.\n"
ARTIFACTS = {"research/source_of_truth.md": SOT}


def _pack_step(number: int, *, sufficiency: str = "complete", verdict: str = "neutral", unavailable=None):
    return {
        "step": number,
        "question_ja": f"質問{number}",
        "answer": {"unavailable": unavailable} if unavailable else {"count": 1},
        "sot_sections": [] if sufficiency == "absent" else ["4.1"],
        "locators": [],
        "verdict_contribution": verdict,
        "sufficiency": sufficiency,
    }


def _pack(overrides: dict | None = None):
    steps = {str(n): _pack_step(n) for n in (1, 2, 3, 4, 5, 6, 8, 9, 10)}
    steps.update({str(n): entry for n, entry in (overrides or {}).items()})
    return {"schema_version": 1, "sot_domain": "clinical", "steps": steps}


def _claim(claim_id="4a", hedge="高い", span="Absolute risk reduction was 5.0%"):
    return {
        "id": claim_id,
        "claim_ja": "骨折リスクは5.0%下がりました。",
        "allowed_numbers": ["5.0%"],
        "hedge_level": hedge,
        "must_not_say": ["治る", "確実に"],
        "evidence": {"pmid": "12345678", "quoted_span": span},
    }


def _blueprint(confidence="高い"):
    blueprint = build_skeleton(_pack(), confidence)
    blueprint["authored"] = True
    blueprint["opening"]["claim_ja"] = "効果はありますが、大きくはありません。"
    for step in blueprint["steps"]:
        step["claims"] = [_claim(f"{step['step']}a", hedge=confidence)]
    return blueprint


# --------------------------------------------------------------------------- #
# The shape
# --------------------------------------------------------------------------- #
def test_a_correct_blueprint_passes() -> None:
    assert blueprint_errors(_blueprint(), ARTIFACTS) == []


def test_the_scaffold_is_derived_and_says_it_is_not_yet_authored() -> None:
    """Which steps run and what they contribute is Python's; what a claim SAYS is Claude's."""
    skeleton = build_skeleton(_pack(), "中程度")
    assert skeleton["authored"] is False
    assert all(step["claims"] == [] for step in skeleton["steps"])
    assert skeleton["opening"]["hedge_level"] == "中程度"


def test_speaker_two_owns_every_question() -> None:
    """This is what resolves the 20-30% substantive balance by construction rather than by
    instruction: the questioner's share IS the nine questions."""
    for step in build_skeleton(_pack(), "高い")["steps"]:
        assert step["driver"] == 2
        assert step["answerer"] == 1


def test_step_seven_has_no_place_in_the_shape() -> None:
    assert 7 not in {step["step"] for step in build_skeleton(_pack(), "高い")["steps"]}


# --------------------------------------------------------------------------- #
# The mutation matrix
# --------------------------------------------------------------------------- #
def test_an_opening_confidence_that_disagrees_with_step_ten_is_rejected() -> None:
    broken = _blueprint()
    broken["opening"]["hedge_level"] = "ほぼ確実"
    errors = blueprint_errors(broken, ARTIFACTS)
    assert errors and "確信度" in errors[0]


def test_a_claim_hedged_stronger_than_the_grade_table_allows_is_rejected() -> None:
    broken = _blueprint(confidence="低い")
    broken["steps"][0]["claims"] = [_claim(hedge="ほぼ確実")]
    assert any("stronger than" in e for e in blueprint_errors(broken, ARTIFACTS))


def test_a_claim_quoting_a_span_absent_from_the_sot_is_rejected() -> None:
    broken = _blueprint()
    broken["steps"][0]["claims"] = [_claim(span="a sentence nobody wrote")]
    assert any("none of the artifacts" in e for e in blueprint_errors(broken, ARTIFACTS))


@pytest.mark.parametrize("number", MANDATORY_STEPS)
def test_a_missing_mandatory_step_is_rejected(number: int) -> None:
    broken = _blueprint()
    broken["steps"] = [s for s in broken["steps"] if s["step"] != number]
    assert any("mandatory" in e for e in blueprint_errors(broken, ARTIFACTS))


def test_an_omitted_step_with_no_reason_cannot_even_be_expressed() -> None:
    """A skip carries its reason in the schema, so a silent skip is not a blueprint. It is said
    aloud in the episode — a named skip keeps the framework visible."""
    broken = _blueprint()
    broken["skipped"] = [{"step": 5}]
    assert blueprint_errors(broken, ARTIFACTS)


def test_an_all_downgrade_interrogation_cannot_land_at_high_confidence() -> None:
    broken = _blueprint(confidence="高い")
    for step in broken["steps"]:
        step["verdict_contribution"] = "downgrade"
    assert any("cannot disagree about direction" in e for e in blueprint_errors(broken, ARTIFACTS))


def test_an_all_downgrade_interrogation_may_land_low() -> None:
    """The control: the rule is about contradiction, not about magnitude, which is GRADE's job."""
    fine = _blueprint(confidence="低い")
    for step in fine["steps"]:
        step["verdict_contribution"] = "downgrade"
    assert blueprint_errors(fine, ARTIFACTS) == []


def test_a_step_both_included_and_skipped_is_rejected() -> None:
    broken = _blueprint()
    broken["skipped"] = [{"step": broken["steps"][0]["step"], "skip_reason": "…"}]
    assert any("both included and skipped" in e for e in blueprint_errors(broken, ARTIFACTS))


# --------------------------------------------------------------------------- #
# The label-equal mutations — invisible here, and that is the whole point
# --------------------------------------------------------------------------- #
def test_labels_can_agree_while_the_opening_prose_oversells() -> None:
    """All three labels say 中程度 and the opening says 「確実に効きます」. The metadata check cannot
    see it, and pretending otherwise would make a tier-2 read look unnecessary."""
    broken = _blueprint(confidence="中程度")
    broken["opening"]["claim_ja"] = "確実に効きます。"
    assert blueprint_errors(broken, ARTIFACTS) == []


def test_labels_can_agree_while_a_claim_states_the_opposite_direction() -> None:
    broken = _blueprint()
    broken["steps"][0]["claims"] = [
        {**_claim(), "claim_ja": "骨折リスクはむしろ上がりました。"}
    ]
    assert blueprint_errors(broken, ARTIFACTS) == []


# --------------------------------------------------------------------------- #
# When the blueprint cannot be built at all
# --------------------------------------------------------------------------- #
def test_a_pack_that_cannot_answer_a_mandatory_step_declines_with_the_reason() -> None:
    """Today's real state: no stage authors the frozen prior, so the pack marks steps 1 and 9
    absent and the episode cannot be built. Declining loudly beats emitting a blueprint whose
    opening states a prior nobody set."""
    pack = _pack(
        {
            1: _pack_step(1, sufficiency="absent", unavailable="no frozen prior artifact"),
            9: _pack_step(9, sufficiency="absent", unavailable="the update needs a frozen prior"),
        }
    )
    with pytest.raises(BlueprintUnavailable, match="事前確率"):
        build_skeleton(pack, "高い")


def test_an_optional_step_with_nothing_to_say_is_skipped_aloud() -> None:
    pack = _pack({5: _pack_step(5, sufficiency="absent", unavailable="資金の記載がありませんでした")})
    skeleton = build_skeleton(pack, "高い")
    [skip] = skeleton["skipped"]
    assert skip["step"] == 5
    assert "資金" in skip["skip_reason"]
    assert 5 not in {s["step"] for s in skeleton["steps"]}


def test_a_confidence_off_the_ladder_is_refused() -> None:
    with pytest.raises(BlueprintUnavailable):
        build_skeleton(_pack(), "たぶん")
