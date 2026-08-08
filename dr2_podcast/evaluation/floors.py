"""Absolute quality floors for a pipeline run.

`scorecard._detect_regressions` flags a metric that moved more than 20% against
the ROLLING AVERAGE OF THE LAST 5 RUNS. That is a useful early warning for a
sudden drop, and it is unfit as the only guard, for one reason: the baseline
moves with the thing it is measuring.

Five consecutive runs at -19% each pass on their own, and each one drags the
average down for the next comparison, so the sixth run is measured against a
standard two thirds of the way to the floor. The check reports "no regressions"
the whole way down. Quality that decays gradually is invisible to any test
whose reference point is recent history.

So this module adds the other half: floors that do not move. `quality_floor.json`
at the repo root records what a run has to reach, in absolute terms, and a run
under any of them is a breach regardless of what the last five runs looked like.

Both checks stay. The rolling average catches a cliff; the floor catches a
slope. Neither sees what the other does.

Pure functions over plain dicts — the only I/O is load/save — which is why the
comparison is unit-tested without a pipeline run.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

FLOOR_FILE = "quality_floor.json"

# (name, path into scorecard["metrics"], higher_is_better)
#
# Single definition, imported by scorecard.py so the rolling-average check and
# the floor check can never drift apart about what a metric means or which
# direction is good.
METRICS: tuple[tuple[str, tuple[str, ...], bool], ...] = (
    ("extraction_timeout_rate", ("research", "extraction_timeout_rate"), False),
    ("url_validation_pass_rate", ("research", "url_validation_pass_rate"), True),
    ("script_adherence_pct", ("script", "adherence_pct"), True),
    ("audio_adherence_pct", ("audio", "adherence_pct"), True),
    ("accuracy_audit_findings", ("script", "accuracy_audit_findings"), False),
    ("degenerate_repetition_pct", ("script", "degenerate_repetition_pct"), False),
)

_HIGHER_IS_BETTER = {name: hib for name, _, hib in METRICS}

# How far under a derived reference a run may legitimately land. Pipeline runs
# vary with the topic — a sparse literature genuinely lowers
# url_validation_pass_rate — and a floor set at the exact historical best fails
# on ordinary variance, which is how a gate gets bypassed instead of fixed.
DEFAULT_MARGIN = 0.85


@dataclass(frozen=True)
class Breach:
    metric: str
    limit: float
    actual: float
    higher_is_better: bool

    def __str__(self) -> str:
        direction = "below floor" if self.higher_is_better else "above ceiling"
        return f"{self.metric}: {self.actual:.3f} {direction} {self.limit:.3f}"


def extract(scorecard: dict) -> dict[str, float]:
    """The floorable metrics of one scorecard, flattened by name.

    A metric the run did not produce is OMITTED rather than defaulted. A null
    audio_adherence_pct means the audio step did not report, which is a
    different fact from the audio being bad, and inventing a 0 for it would
    fail every run whose audio is still rendering.
    """
    out: dict[str, float] = {}
    metrics = scorecard.get("metrics", {})
    for name, path, _ in METRICS:
        value = metrics
        for key in path:
            value = value.get(key) if isinstance(value, dict) else None
            if value is None:
                break
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            out[name] = float(value)
    return out


def unmeasured(current: dict[str, float], floors: dict[str, float]) -> list[str]:
    """Floored metrics this run did not report.

    `compare` skips them, because a null audio_adherence_pct means the audio
    step has not reported yet rather than that the audio is bad. But silence is
    then a way past a floor: a producer that degrades to emitting nothing for a
    metric breaches nothing. So the omission is surfaced separately instead of
    being folded into either the pass or the failure.
    """
    return sorted(name for name in floors if name not in current)


def compare(current: dict[str, float], floors: dict[str, float]) -> list[Breach]:
    """Every floored metric that came in the wrong side of its limit.

    A metric absent from the run is NOT a breach here — see `extract`. The
    missing-artifact case belongs to the scorecard's own completeness checks,
    not to a quality floor, and failing on it would make every partial run look
    like a quality collapse.
    """
    breaches = []
    for name, limit in floors.items():
        if name not in current:
            continue
        actual = current[name]
        higher_better = _HIGHER_IS_BETTER.get(name, True)
        if (higher_better and actual < limit) or (not higher_better and actual > limit):
            breaches.append(Breach(name, float(limit), actual, higher_better))
    return sorted(breaches, key=lambda b: b.metric)


def derive(scorecards: list[dict], margin: float = DEFAULT_MARGIN) -> dict[str, float]:
    """Propose floors from a body of past runs.

    Uses the WORST observed value, relaxed by the margin — not the mean. A mean
    floor would be under water half the time by construction, and a gate that
    fires on half of all healthy runs is one people route around. The worst
    accepted run is the honest statement of "this much, at least".
    """
    if not 0 < margin <= 1:
        raise ValueError(f"margin must be in (0, 1], got {margin}")
    observed: dict[str, list[float]] = {}
    for card in scorecards:
        for name, value in extract(card).items():
            observed.setdefault(name, []).append(value)

    floors: dict[str, float] = {}
    for name, values in observed.items():
        if _HIGHER_IS_BETTER.get(name, True):
            floors[name] = round(min(values) * margin, 4)
        else:
            # A ceiling relaxes upward, so divide rather than multiply.
            floors[name] = round(max(values) / margin, 4)
    return floors


def vacuous(floors: dict[str, float]) -> list[str]:
    """Floors that no possible run can breach, and are therefore theatre.

    `derive` takes the worst run in history, so a metric whose history contains
    a total failure inherits a limit of 0 — which for a rate can never fire.
    url_validation_pass_rate is the live example: one recorded run scored 0.000,
    so no defensible floor can be read off the past for it. Better to leave that
    metric unfloored and say so than to commit a number that looks like a
    guarantee and enforces nothing.
    """
    return sorted(
        name
        for name, limit in floors.items()
        if _HIGHER_IS_BETTER.get(name, True) and limit <= 0
    )


def tighten(existing: dict[str, float], proposed: dict[str, float]) -> dict[str, float]:
    """Limits only ever get STRICTER, and only when asked.

    A floor that could also relax would rebuild the drifting baseline this
    module exists to replace: each bad run would license the next.
    """
    out = dict(existing)
    for name, value in proposed.items():
        if name not in out:
            out[name] = value
        elif _HIGHER_IS_BETTER.get(name, True):
            out[name] = max(out[name], value)
        else:
            out[name] = min(out[name], value)
    return out


def load(repo_root: Path) -> dict[str, float]:
    path = Path(repo_root) / FLOOR_FILE
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {str(k): float(v) for k, v in data.get("floors", {}).items()}


def save(repo_root: Path, floors: dict[str, float], note: str) -> None:
    payload = {
        "version": 1,
        "note": note,
        "higher_is_better": {name: hib for name, _, hib in METRICS},
        "floors": dict(sorted(floors.items())),
    }
    (Path(repo_root) / FLOOR_FILE).write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
