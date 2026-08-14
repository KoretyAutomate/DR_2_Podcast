"""Replication, funding and bias, counted from the structured records rather than read off prose.

PLAN.md Step 9b items 2 and 3. All three were impossible before Step 9a: replication needs
``finding_key``/``author_group``/``trial_registration``, the funding rollup needs the five-field
block instead of one free-text line, and the bias rollup needs GRADE's downgrades as records rather
than as reasons written into a paragraph.

Every count here reports ``n of N``, and ``unknown`` is broken out rather than folded into a
denominator — Ep09's thesis is that a missing disclosure is a finding, and a rollup that quietly
shrinks its denominator turns that finding into silence.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

#: What a funding rollup counts. `unknown` is here as a category like any other, because the whole
#: point is that it is reported rather than dropped.
FUNDING_CATEGORIES: tuple[str, ...] = (
    "industry",
    "government",
    "foundation",
    "institutional",
    "mixed",
    "none_declared",
    "undisclosed",
    "unknown",
)


def unique_studies(extractions) -> list:
    """One entry per paper, by stable identity.

    The affirmative and falsification tracks search separately and routinely land on the SAME paper,
    so `all_extractions` carries it twice. §4.1's study table already deduplicates by PMID/DOI/title,
    so counting both copies here made the aggregates contradict the table printed directly above
    them and overstated every category the duplicate belonged to (prepush codex 2026-08-13).

    Identity is the first of PMID, DOI or title that the record actually has — the same ladder the
    table uses, so the two cannot disagree about what counts as one study.
    """
    seen: set[str] = set()
    unique = []
    for extraction in extractions or []:
        identity = (
            (getattr(extraction, "pmid", None) or "").strip()
            or (getattr(extraction, "doi", None) or "").strip()
            or (getattr(extraction, "title", None) or "").strip()
        )
        if identity and identity in seen:
            continue
        if identity:
            seen.add(identity)
        unique.append(extraction)
    return unique


@dataclass
class ReplicationGroup:
    """One finding, and who has reported it."""

    finding_key: str
    endpoint: str
    direction: str
    author_groups: list[str] = field(default_factory=list)
    registrations: list[str] = field(default_factory=list)
    #: Papers reporting this finding whose author group could not be read. They are counted as
    #: studies but cannot count as INDEPENDENT ones, which is a different thing and is said so.
    unattributed: int = 0
    #: How many papers reported this finding at all, registered or not. Needed to tell "everyone
    #: named the same trial" from "one paper named a trial and the others named nothing".
    reports: int = 0

    @property
    def independent_groups(self) -> int:
        """Distinct author groups reporting the same direction, on non-overlapping cohorts.

        Two papers from one group are one group. Two papers naming the same trial registration are
        one trial reported twice — the replication question is whether someone ELSE, looking at
        someone ELSE's participants, saw the same thing.
        """
        return len(set(self.author_groups))

    @property
    def distinct_cohorts(self) -> int:
        return len({r for r in self.registrations if r})

    @property
    def status(self) -> str:
        """``replicated`` / ``cohorts_unknown`` / ``not_replicated``.

        Three states rather than a boolean, because "two groups reported it and neither paper names
        a trial registration" is not the same claim as "two groups reported it on two different sets
        of participants" (prepush codex 2026-08-13). Observational studies routinely have no
        registration, so an unverifiable cohort is the COMMON case — collapsing it into a boolean
        turned missing data into positive evidence of replication.
        """
        if self.independent_groups < 2:
            return "not_replicated"
        if self.distinct_cohorts >= 2:
            return "replicated"
        if self.distinct_cohorts == 1 and len(self.registrations) == self.reports:
            # EVERY report names it, and they all name the same one. One trial reported twice is one
            # trial. If some report named nothing, overlap is unknown rather than proven — asserting
            # "not replicated" there states a negative the records do not support (prepush codex
            # 2026-08-13, the second half of the same defect).
            return "not_replicated"
        return "cohorts_unknown"

    @property
    def is_replicated(self) -> bool:
        return self.status == "replicated"

    def to_dict(self) -> dict[str, Any]:
        return {
            "finding_key": self.finding_key,
            "endpoint": self.endpoint,
            "direction": self.direction,
            "independent_groups": self.independent_groups,
            "distinct_cohorts": self.distinct_cohorts,
            "unattributed_reports": self.unattributed,
            "status": self.status,
            "replicated": self.is_replicated,
        }


def replication_groups(extractions) -> list[ReplicationGroup]:
    """Group every finding by identity AND direction, across papers.

    Keyed by ``(finding_key, direction)`` rather than ``finding_key`` alone: two papers agreeing on
    what they measured but disagreeing on which way it went are not a replication of each other,
    they are the disagreement the falsification track exists to surface.
    """
    groups: dict[tuple[str, str], ReplicationGroup] = {}
    for extraction in unique_studies(extractions):
        author_group = (getattr(extraction, "author_group", None) or "").strip()
        registration = (getattr(extraction, "trial_registration", None) or "").strip()
        for finding in getattr(extraction, "findings", None) or []:
            if not finding.finding_key:
                continue
            key = (finding.finding_key, finding.direction or "")
            group = groups.setdefault(
                key,
                ReplicationGroup(
                    finding_key=finding.finding_key,
                    endpoint=finding.endpoint or "",
                    direction=finding.direction or "",
                ),
            )
            group.reports += 1
            if author_group:
                group.author_groups.append(author_group)
            else:
                group.unattributed += 1
            if registration:
                group.registrations.append(registration)
    return sorted(groups.values(), key=lambda g: (-g.independent_groups, g.endpoint, g.direction))


def replication_rollup(extractions) -> dict[str, Any]:
    """The step-6 answer: how many findings anyone has actually reproduced."""
    groups = replication_groups(extractions)
    return {
        "findings_total": len(groups),
        "findings_replicated": len([g for g in groups if g.status == "replicated"]),
        # Two or more groups, but no paper names a trial registration — so whether they studied the
        # same participants is unknown, and the episode says exactly that rather than claiming
        # independent confirmation it cannot support.
        "findings_cohorts_unknown": len([g for g in groups if g.status == "cohorts_unknown"]),
        "findings_single_group": len([g for g in groups if g.independent_groups <= 1]),
        # Named separately because "we could not tell" is not "it was not replicated".
        "findings_unattributable": len([g for g in groups if g.independent_groups == 0 and g.unattributed]),
        "groups": [g.to_dict() for g in groups],
    }


def funding_rollup(extractions) -> dict[str, Any]:
    """The step-5 answer: who paid, over the whole extracted set.

    Multi-funder papers are already ``mixed`` by the time they reach here, and mixed is its own
    category — never double-counted into industry and government.
    """
    counts: Counter[str] = Counter()
    api_only = 0
    total = 0
    for extraction in unique_studies(extractions):
        total += 1
        block = getattr(extraction, "funding", None)
        if block is None:
            counts["unknown"] += 1
            continue
        counts[block.funding_category or "unknown"] += 1
        if block.funding_source_type == "api_metadata":
            api_only += 1
    return {
        "studies_total": total,
        "by_category": {category: counts.get(category, 0) for category in FUNDING_CATEGORIES},
        # Reported because these have no locator and cannot be verified against the paper. A rollup
        # that hides how much of itself is unverifiable is a rollup nobody can weigh.
        "from_api_metadata_unverified": api_only,
        "undisclosed": counts.get("undisclosed", 0),
        "unknown": counts.get("unknown", 0),
    }


def bias_rollup(extractions, grade_record: dict[str, Any] | None) -> dict[str, Any]:
    """The step-8 answer: the per-study risk-of-bias spread, and what GRADE actually downgraded for.

    The GRADE half comes from the structured record's ``downgrades``, not from the synthesis prose:
    reasons written into a paragraph cannot be counted, which is why step 8 was not derivable before
    the record existed.
    """
    ratings: Counter[str] = Counter()
    for extraction in unique_studies(extractions):
        rating = (getattr(extraction, "risk_of_bias", None) or "").strip().lower()
        ratings[rating if rating in ("low", "some concerns", "high") else "unclear"] += 1
    downgrades = (grade_record or {}).get("downgrades") or []
    return {
        "studies_total": sum(ratings.values()),
        "risk_of_bias": {name: ratings.get(name, 0) for name in ("low", "some concerns", "high", "unclear")},
        "grade_downgrades": {entry["domain"]: entry["steps"] for entry in downgrades},
        "grade_downgrade_steps": sum(entry["steps"] for entry in downgrades),
    }


def design_rollup(extractions) -> dict[str, Any]:
    """The step-3 answer: what the evidence base is made of, by study design."""
    from dr2_podcast.research.confidence import design_rung, staircase_position

    studies = unique_studies(extractions)
    counts: Counter[str] = Counter()
    for extraction in studies:
        counts[design_rung(getattr(extraction, "study_design", None)) or "unreadable"] += 1
    answer: dict[str, Any] = {name: count for name, count in sorted(counts.items())}
    answer["studies_total"] = sum(counts.values())
    answer["highest_rung"] = staircase_position(studies) or "none_readable"
    return answer
