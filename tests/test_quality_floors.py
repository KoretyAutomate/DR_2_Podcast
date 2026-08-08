"""Absolute quality floors.

The behaviour worth pinning is the reason this module exists: a rolling-average
check cannot see gradual decay, because the baseline descends with the metric.
`test_a_slow_decay_is_invisible_to_the_rolling_average_but_not_to_a_floor` is
the whole argument, executed.
"""

from __future__ import annotations

import json

import pytest

from dr2_podcast.evaluation import floors
from dr2_podcast.evaluation.scorecard import _detect_regressions


def card(**metrics) -> dict:
    """A scorecard carrying just the metrics a test cares about."""
    return {
        "metrics": {
            "research": {
                "extraction_timeout_rate": metrics.get("extraction_timeout_rate"),
                "url_validation_pass_rate": metrics.get("url_validation_pass_rate"),
            },
            "script": {
                "adherence_pct": metrics.get("script_adherence_pct"),
                "accuracy_audit_findings": metrics.get("accuracy_audit_findings"),
                "degenerate_repetition_pct": metrics.get("degenerate_repetition_pct"),
            },
            "audio": {"adherence_pct": metrics.get("audio_adherence_pct")},
        }
    }


def test_a_slow_decay_is_invisible_to_the_rolling_average_but_not_to_a_floor():
    """The point of the module, as a test.

    Each run is 5% below the last. Against a 5-run trailing mean that settles
    at roughly 14% below average — comfortably inside the 20% the rolling check
    needs to fire — so it never fires, while the metric loses more than half its
    value. An absolute floor catches it on the way past.

    The decay rate matters and is not arbitrary: for geometric decay at rate r,
    the steady-state ratio of current to trailing-5 mean is 5 / Σ(r⁻ᵏ, k=1..5).
    That stays above 0.8 while r is above about 0.93. Anything faster does trip
    the rolling check, which is the case it handles correctly.
    """
    value = 1.0
    history = [card(script_adherence_pct=value) for _ in range(5)]
    floor = {"script_adherence_pct": 0.8}

    saw_rolling_flag = False
    for _ in range(16):
        value *= 0.95
        current = card(script_adherence_pct=value)
        if _detect_regressions(current, history[-5:]):
            saw_rolling_flag = True
        history.append(current)

    assert value < 0.5, "the decay should have more than halved the metric"
    assert not saw_rolling_flag, "rolling average should never have fired — that is the flaw"
    assert floors.compare(floors.extract(history[-1]), floor), "the floor must catch it"


def test_extract_pulls_the_floorable_metrics_by_name():
    got = floors.extract(card(script_adherence_pct=1.5, accuracy_audit_findings=4))
    assert got["script_adherence_pct"] == 1.5
    assert got["accuracy_audit_findings"] == 4.0


def test_extract_omits_a_metric_the_run_never_reported():
    """A null audio_adherence_pct means the audio step did not report, which is
    not the same fact as the audio being bad."""
    assert "audio_adherence_pct" not in floors.extract(card(script_adherence_pct=1.0))


def test_compare_respects_direction_for_lower_is_better_metrics():
    breaches = floors.compare({"accuracy_audit_findings": 9.0}, {"accuracy_audit_findings": 6.0})
    assert len(breaches) == 1
    assert not breaches[0].higher_is_better
    # ...and the same number under the ceiling is fine
    assert floors.compare({"accuracy_audit_findings": 4.0}, {"accuracy_audit_findings": 6.0}) == []


def test_compare_flags_a_higher_is_better_metric_that_fell():
    assert floors.compare({"script_adherence_pct": 0.5}, {"script_adherence_pct": 0.99})
    assert floors.compare({"script_adherence_pct": 1.5}, {"script_adherence_pct": 0.99}) == []


def test_a_metric_missing_from_the_run_is_not_a_breach():
    """Otherwise every partial run reads as a quality collapse."""
    assert floors.compare({}, {"audio_adherence_pct": 0.9}) == []


def test_derive_uses_the_worst_run_not_the_mean():
    """A mean floor is under water half the time by construction, and a gate
    that fires on half of all healthy runs gets routed around."""
    cards = [card(script_adherence_pct=v) for v in (2.0, 1.5, 1.0)]
    assert floors.derive(cards, margin=1.0)["script_adherence_pct"] == 1.0


def test_derive_relaxes_a_ceiling_upward_not_downward():
    cards = [card(accuracy_audit_findings=v) for v in (3, 5)]
    assert floors.derive(cards, margin=0.5)["accuracy_audit_findings"] == 10.0


@pytest.mark.parametrize("bad", [0, -1, 2.0])
def test_derive_rejects_a_nonsense_margin(bad):
    with pytest.raises(ValueError):
        floors.derive([card(script_adherence_pct=1.0)], margin=bad)


def test_vacuous_names_floors_that_can_never_fire():
    """url_validation_pass_rate is the live case: one recorded run scored 0.000,
    so a floor derived from history is 0 and no run can breach it."""
    assert floors.vacuous({"url_validation_pass_rate": 0.0}) == ["url_validation_pass_rate"]
    assert floors.vacuous({"script_adherence_pct": 0.9}) == []
    # a ceiling of 0 is meaningful — it demands the metric stay at zero
    assert floors.vacuous({"degenerate_repetition_pct": 0.0}) == []


def test_tighten_never_relaxes_an_existing_limit():
    kept = floors.tighten({"script_adherence_pct": 1.2}, {"script_adherence_pct": 0.5})
    assert kept["script_adherence_pct"] == 1.2
    tighter = floors.tighten({"accuracy_audit_findings": 6.0}, {"accuracy_audit_findings": 4.0})
    assert tighter["accuracy_audit_findings"] == 4.0
    looser = floors.tighten({"accuracy_audit_findings": 6.0}, {"accuracy_audit_findings": 9.0})
    assert looser["accuracy_audit_findings"] == 6.0


def test_floors_round_trip_through_disk(tmp_path):
    floors.save(tmp_path, {"script_adherence_pct": 0.99}, note="test")
    assert floors.load(tmp_path) == {"script_adherence_pct": 0.99}
    assert json.loads((tmp_path / floors.FLOOR_FILE).read_text())["note"] == "test"


def test_absent_floor_file_loads_as_no_floors(tmp_path):
    assert floors.load(tmp_path) == {}


def test_the_committed_floor_file_is_loadable_and_not_all_theatre():
    """Guards the real artifact: a floor file whose every entry is vacuous
    would pass every other test in here while enforcing nothing."""
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parent.parent
    committed = floors.load(repo_root)
    assert committed, "quality_floor.json should exist and carry floors"
    assert floors.vacuous(committed) == []
