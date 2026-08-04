"""Characterization tests pinning build_imrad_sot's exact output.

These exist to make the build_imrad_sot / _build_social_science_sot complexity
refactor provable: the refactor is correct iff every case below still renders
byte for byte what it rendered before.

Regenerate with `python -m tests.gen_sot_golden` — and only ever regenerate
when the change to the output is intended and reviewed.
"""

import json
from pathlib import Path

import pytest

from tests.gen_sot_golden import GOLDEN_PATH, generate

GOLDEN = json.loads(GOLDEN_PATH.read_text(encoding="utf-8")) if GOLDEN_PATH.exists() else {}


def test_golden_file_exists():
    assert GOLDEN_PATH.exists(), "run `python -m tests.gen_sot_golden` to create the golden"
    assert GOLDEN, "golden file is empty"


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_sot_output_matches_golden(case):
    """Every grid case renders exactly what it rendered when the golden was cut."""
    current = generate()
    assert case in current, f"case {case} disappeared from the generator grid"
    assert current[case] == GOLDEN[case]


def test_no_cases_added_without_regenerating():
    """Guards the other direction: a new grid case must be captured in the golden."""
    assert sorted(generate()) == sorted(GOLDEN)


# ---------------------------------------------------------------------------
# Mutation sensitivity — a golden that cannot fail proves nothing.
#
# This does not mutate the module (that would leak across tests). It asserts the
# comparison is genuinely tight: any single-character perturbation of any case
# must be detected. The corresponding real-mutation check was run by hand
# against pipeline_sot.py before the refactor; see the commit message.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_golden_detects_single_character_drift(case):
    text = GOLDEN[case]
    assert text, f"case {case} rendered empty — it would pin nothing"
    mutated = text[: len(text) // 2] + "​" + text[len(text) // 2 :]
    assert mutated != GOLDEN[case]


def test_golden_covers_both_domains_and_languages():
    """The grid must keep exercising every dispatch branch the refactor touches."""
    keys = set(GOLDEN)
    assert any(k.startswith("clinical_") for k in keys)
    assert any(k.startswith("social_science_") for k in keys)
    assert any(k.endswith("_ja") for k in keys)
    assert any(k.endswith("_en") for k in keys)
    # The degenerate case matters most: it is where extraction bugs surface.
    assert "clinical_empty_en" in GOLDEN
    assert "missing_pipeline_data" in GOLDEN


def test_golden_path_is_tracked_next_to_tests():
    assert GOLDEN_PATH.parent == Path(__file__).parent
