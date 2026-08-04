"""Characterization tests pinning pipeline_script's section generators.

These exist so the _generate_section* / _run_condense_pass parameter-object
refactor is provable: the returned tuples AND every prompt sent to the model
must be unchanged.

Regenerate with `python -m tests.gen_section_golden`, and only when the change
is intended and reviewed.
"""

import json

import pytest

from tests.gen_section_golden import GOLDEN_PATH, generate

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) if GOLDEN_PATH.exists() else {}


def test_golden_file_exists():
    assert GOLDEN_PATH.exists(), "run `python -m tests.gen_section_golden` to create the golden"
    assert GOLDEN


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_section_output_matches_golden(case):
    current = generate()
    assert case in current, f"case {case} disappeared from the generator grid"
    assert current[case] == GOLDEN[case]


def test_no_cases_added_without_regenerating():
    assert sorted(generate()) == sorted(GOLDEN)


def test_golden_pins_the_prompts_not_just_the_return():
    """The prompts are the product here — a golden that only pinned the tuple
    would not notice a dropped checklist block."""
    for case, data in GOLDEN.items():
        assert "calls" in data, f"{case} records no model calls"
        for call in data["calls"]:
            assert set(call) == {"system", "user", "max_tokens", "temperature", "frequency_penalty"}


def test_golden_covers_the_branches_the_refactor_touches():
    keys = set(GOLDEN)
    # single-call vs sub-split dispatch, on both sides of the JA threshold
    assert {"ja_at_threshold", "ja_just_over_threshold", "ja_subsplit_many_parts"} <= keys
    # the retry loop, both outcomes
    assert GOLDEN["en_retry_then_pass"]["calls"].__len__() == 2
    assert GOLDEN["en_retry_still_short"]["deficit"] > 0
    # opening-only channel-intro directive
    assert "en_opening_with_intro" in keys and "ja_subsplit_opening_intro" in keys
    # condense: early return, success, no-reduction, and the exception path
    assert GOLDEN["condense_under_target"]["calls"] == []
    assert GOLDEN["condense_at_buffer_boundary"]["calls"] == []
    assert len(GOLDEN["condense_model_raises"]["calls"]) == 1


def test_channel_intro_only_reaches_the_first_subsplit_part():
    """Pins a real behaviour: part 0 gets the intro, later parts must not."""
    calls = GOLDEN["ja_subsplit_opening_intro"]["calls"]
    assert len(calls) > 1
    intro = "ディープリサーチポッドキャストへようこそ。"
    assert intro in calls[0]["user"]
    assert all(intro not in c["user"] for c in calls[1:])
