"""The draft and polish gates — PLAN.md Steps 3, 4 and 6.

The exit criteria are mutation matrices. Both name cases Python must catch AND cases it must let
through: a gate that rejects a legitimate rewrite forbids the work the phase exists to do, and a
gate that fires on 「3つ」 gets switched off rather than fixed.
"""

from __future__ import annotations

import pytest

from dr2_podcast.script_gates import (
    MAX_REVISION_ROUNDS,
    Finding,
    LoopState,
    claim_fingerprints,
    invariance_errors,
    parse_draft,
    tier0_errors,
    tier1_errors,
    write_loop_events,
)

STEP = {
    "claims": [
        {
            "id": "4a",
            "claim_ja": "…",
            "allowed_numbers": ["5.0%", "12週", "1447"],
            "hedge_level": "中程度",
            "must_not_say": ["治る", "確実に"],
            "evidence": {"quoted_span": "…"},
        }
    ]
}

APPROVED = (
    "Host 1: 骨折リスクは5.0%下がりました。 [[4a]]\n"
    "Host 2: それはどのくらいの期間ですか。 [[none]]\n"
    "Host 1: 12週の追跡です。 [[4a]]\n"
)


# --------------------------------------------------------------------------- #
# Tier 0 — shape
# --------------------------------------------------------------------------- #
def test_a_well_formed_draft_clears_tier_zero() -> None:
    assert tier0_errors(APPROVED, {"4a"}) == []


def test_an_unannotated_sentence_is_malformed() -> None:
    """The annotation is what makes every later check possible."""
    findings = tier0_errors("Host 1: 骨折リスクは下がりました。\n", {"4a"})
    assert [f.rule_id for f in findings] == ["tier0.unannotated"]


def test_a_claim_id_the_blueprint_never_declared_is_malformed() -> None:
    findings = tier0_errors("Host 1: なにか。 [[9z]]\n", {"4a"})
    assert any(f.rule_id == "tier0.unknown_claim" for f in findings)


def test_connective_tissue_is_annotated_none_not_left_bare() -> None:
    assert tier0_errors("Host 2: なるほど。 [[none]]\n", {"4a"}) == []
    assert parse_draft("Host 2: なるほど。 [[none]]\n")[0].claim_id is None


# --------------------------------------------------------------------------- #
# Tier 1 — lexical, and the things it must NOT reject
# --------------------------------------------------------------------------- #
def test_an_allowed_number_passes() -> None:
    assert tier1_errors(APPROVED, STEP) == []


def test_a_number_outside_the_allowed_set_is_caught() -> None:
    findings = tier1_errors("Host 1: 骨折リスクは50%下がりました。 [[4a]]\n", STEP)
    assert any(f.rule_id == "tier1.numeral" for f in findings)


@pytest.mark.parametrize(
    "line",
    [
        "Host 1: ポイントは3つあります。 [[none]]\n",
        "Host 1: 1日2回の服用です。 [[none]]\n",
        "Host 1: 2番目の研究を見ましょう。 [[none]]\n",
        "Host 1: 第3章で扱います。 [[none]]\n",
    ],
)
def test_counting_expressions_are_not_claims(line: str) -> None:
    """A gate that rejects 「3つ」 fires on every well-written section, and one that fires on correct
    work gets switched off rather than fixed."""
    assert tier1_errors(line, STEP) == []


def test_a_forbidden_phrase_is_caught() -> None:
    findings = tier1_errors("Host 1: これで確実に治ると言えます。 [[4a]]\n", STEP)
    assert {f.rule_id for f in findings} == {"tier1.must_not_say"}


def test_a_speaker_taking_two_turns_running_is_caught() -> None:
    draft = "Host 1: ひとつめ。 [[none]]\nHost 1: ふたつめ。 [[none]]\n"
    assert any(f.rule_id == "tier1.alternation" for f in tier1_errors(draft, STEP))


def test_full_width_numerals_are_the_same_numbers() -> None:
    assert tier1_errors("Host 1: ５．０％下がりました。 [[4a]]\n", STEP) == []


# --------------------------------------------------------------------------- #
# What tier 1 cannot do — asserted, so it is never mistaken for a correctness gate
# --------------------------------------------------------------------------- #
def test_an_allowed_number_on_the_wrong_endpoint_passes_tier_one() -> None:
    """Every numeral resolves and the claim is false. This is why tier 2 is not optional."""
    assert tier1_errors("Host 1: 転倒リスクが5.0%下がりました。 [[4a]]\n", STEP) == []


def test_a_flipped_negation_passes_tier_one() -> None:
    """「効果が確認された」 and 「効果が確認されなかった」 carry the same tokens; the negation is the
    whole meaning, and no lexical rule separates them."""
    assert tier1_errors("Host 1: 効果は確認されませんでした。 [[4a]]\n", STEP) == []


# --------------------------------------------------------------------------- #
# Step 4 — the invariance gate preserves the claim, not the wording
# --------------------------------------------------------------------------- #
def test_a_naturalness_rewrite_of_a_claim_sentence_passes() -> None:
    """Rewriting sentences IS polishing. A literal-preservation rule would forbid the work."""
    polished = (
        "Host 1: 骨折のリスクは5.0%ほど下がったんです。 [[4a]]\n"
        "Host 2: 期間はどれくらいでしたか。 [[none]]\n"
        "Host 1: 追跡は12週でした。 [[4a]]\n"
    )
    assert invariance_errors(APPROVED, polished, STEP) == []


def test_a_reordering_that_preserves_every_claim_passes() -> None:
    reordered = (
        "Host 1: 12週の追跡です。 [[4a]]\n"
        "Host 2: それはどのくらいの期間ですか。 [[none]]\n"
        "Host 1: 骨折リスクは5.0%下がりました。 [[4a]]\n"
    )
    assert invariance_errors(APPROVED, reordered, STEP) == []


def test_a_dropped_claim_is_rejected() -> None:
    polished = "Host 2: それはどのくらいの期間ですか。 [[none]]\n"
    assert any(f.rule_id == "polish.claim_dropped" for f in invariance_errors(APPROVED, polished, STEP))


def test_a_swapped_number_is_rejected() -> None:
    polished = APPROVED.replace("5.0%", "0.5%")
    findings = invariance_errors(APPROVED, polished, STEP)
    assert any(f.rule_id in ("polish.numbers_moved", "tier1.numeral") for f in findings)


def test_a_claim_reassigned_to_another_id_is_rejected() -> None:
    step = {"claims": STEP["claims"] + [{**STEP["claims"][0], "id": "4b"}]}
    polished = APPROVED.replace("[[4a]]", "[[4b]]")
    findings = invariance_errors(APPROVED, polished, step)
    assert {f.rule_id for f in findings} >= {"polish.claim_dropped", "polish.claim_added"}


def test_a_new_number_that_no_claim_allows_is_rejected() -> None:
    polished = APPROVED.replace("12週の追跡です。", "12週の追跡で、対象は9999人でした。")
    assert any(f.rule_id == "tier1.numeral" for f in invariance_errors(APPROVED, polished, STEP))


def test_the_fingerprint_is_numbers_not_words() -> None:
    a = claim_fingerprints("Host 1: 5.0%下がった。 [[4a]]\n")
    b = claim_fingerprints("Host 1: 5.0%の低下が見られました。 [[4a]]\n")
    assert a == b


# --------------------------------------------------------------------------- #
# Step 6 — the bound, and what it records
# --------------------------------------------------------------------------- #
def _finding(rule="tier1.numeral", claim="4a", location="line 1"):
    return Finding(rule, claim, location, "message that changes between rounds")


def test_the_loop_stops_at_exactly_three_rounds() -> None:
    state = LoopState("step-4")
    for round_number in range(MAX_REVISION_ROUNDS):
        assert not state.exhausted, f"stopped early at round {round_number}"
        state.record([_finding(location=f"line {round_number}")])
    assert state.exhausted
    assert state.should_stop


def test_an_identical_finding_set_two_rounds_running_is_thrash() -> None:
    """Compared as a SET, not a count: equal counts can mean one defect fixed and another
    introduced, and a falling count can cycle among blockers."""
    state = LoopState("step-4")
    state.record([_finding()])
    state.record([_finding()])
    assert state.thrashing


def test_a_changing_finding_set_is_not_thrash() -> None:
    state = LoopState("step-4")
    state.record([_finding(location="line 1")])
    state.record([_finding(location="line 7")])
    assert not state.thrashing


def test_a_message_that_changes_does_not_hide_a_repeat() -> None:
    """Identity is (rule_id, claim_id, location) precisely so wording drift cannot defeat it."""
    state = LoopState("step-4")
    state.record([Finding("tier1.numeral", "4a", "line 1", "round one wording")])
    state.record([Finding("tier1.numeral", "4a", "line 1", "round two wording")])
    assert state.thrashing


def test_a_clean_round_ends_the_loop() -> None:
    state = LoopState("step-4")
    state.record([])
    assert state.should_stop
    assert state.event()["outcome"] == "converged"


def test_the_event_records_the_section_and_the_surviving_findings(tmp_path) -> None:
    import json

    (tmp_path / "meta").mkdir()
    state = LoopState("step-4")
    for _ in range(MAX_REVISION_ROUNDS):
        state.record([_finding()])
    write_loop_events(tmp_path, [state.event()])

    events = json.loads((tmp_path / "meta/loop_events.json").read_text())["events"]
    assert events[0]["section"] == "step-4"
    assert events[0]["rounds"] == MAX_REVISION_ROUNDS
    assert events[0]["surviving_findings"][0]["rule_id"] == "tier1.numeral"


def test_events_accumulate_rather_than_replace(tmp_path) -> None:
    """A section type that repeatedly escapes is a blueprint-template bug, and only the whole record
    makes that visible."""
    import json

    (tmp_path / "meta").mkdir()
    first, second = LoopState("step-4"), LoopState("step-8")
    first.record([_finding()])
    second.record([_finding()])
    write_loop_events(tmp_path, [first.event()])
    write_loop_events(tmp_path, [second.event()])

    events = json.loads((tmp_path / "meta/loop_events.json").read_text())["events"]
    assert [e["section"] for e in events] == ["step-4", "step-8"]
