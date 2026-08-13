"""The nine-step interrogation view of the evidence — DERIVED, never authored.

PLAN.md Step 9b. The SOT keeps its IMRaD shape, because that is the order in which the evidence was
produced; the nine steps are the order in which a skeptic attacks a claim. One document cannot be
optimal on both axes, so the pack is a **projection**: change the SOT, regenerate the pack. There is
no LLM anywhere in this file, and that is the point — a projection that could be written by a model
would be a second source of truth able to disagree with the first.

Step 7 (メディア歪曲チェック) is absent by design: this pipeline reads primary literature rather than
press coverage of it, so the step has no subject and no evidence artifact.

**Every locator resolves into the SOT**, which is the artifact the pack projects. That is a real
provenance claim and a deliberately modest one — it says "this answer was computed from that section
of that document", not "the paper says so". The per-finding locators that reach into the papers
themselves live on the findings, one layer down, where they were verified when the paper was read.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from dr2_podcast.research.confidence import confidence_level
from dr2_podcast.research.rollups import bias_rollup, design_rollup, funding_rollup, replication_rollup

#: The steps this pack can carry, and the question each one asks. Step numbers skip 7.
QUESTIONS_JA: dict[int, str] = {
    1: "事前確率を立てる",
    2: "関連研究を探す",
    3: "各研究のデザインを評価する",
    4: "数字を確認する",
    5: "資金提供者を確認する",
    6: "再現性をチェックする",
    8: "バイアスと誤謬を特定する",
    9: "ベイズ更新する",
    10: "確率の主張にたどり着く",
}

#: Which SOT section each step is projected from, per builder. There are two builders with different
#: numbering (2.x clinical vs 3.x social science), and a domain-blind reference cites the wrong
#: section — the trap PLAN.md names explicitly.
SOT_SECTIONS: dict[str, dict[int, tuple[str, ...]]] = {
    "clinical": {
        2: ("2. Methods",),
        3: ("4.1",),
        4: ("3.3",),
        5: ("4.1",),
        6: ("4.1",),
        8: ("4.1",),
        9: ("Discussion",),
        10: ("Abstract",),
    },
    "social_science": {
        2: ("2. Methods",),
        3: ("3.1",),
        4: ("3.3",),
        5: ("3.1",),
        6: ("3.1",),
        8: ("3.1",),
        9: ("Discussion",),
        10: ("Abstract",),
    },
}


def _locate(sot: str, needles: tuple[str, ...], fields: list[str], artifact_id: str) -> list[dict[str, Any]]:
    """Locators for the SOT sections a step was projected from, skipping any that is not there.

    A section the SOT does not contain yields no locator, which the caller turns into a lower
    ``sufficiency`` — rather than a locator quoting a heading that was never written, which would
    pass verification against nothing.
    """
    locators = []
    for needle in needles:
        match = re.search(re.escape(needle), sot)
        if not match:
            continue
        # The whole heading line, so the span is something a reader can find by eye.
        line_end = sot.find("\n", match.start())
        span = sot[match.start() : line_end if line_end != -1 else len(sot)].rstrip()
        if span:
            locators.append(
                {
                    "fields": fields,
                    "source_artifact_id": artifact_id,
                    "char_offset": match.start(),
                    "quoted_span": span,
                }
            )
    return locators


def _sufficiency(answer: dict[str, Any], locators: list[dict[str, Any]], covered: int, total: int) -> str:
    """complete / partial / absent, from how much of the evidence the answer actually covers.

    ``absent`` is not failure. Where the absence IS the finding — funding undisclosed on 15 of 20
    studies — the step still runs and states it; that is ``partial`` with the numbers, and the
    episode says the numbers. ``absent`` is for a step with nothing to project at all.
    """
    if not answer or not locators or total <= 0:
        return "absent"
    return "complete" if covered >= total else "partial"


@dataclass(frozen=True)
class _Projection:
    """The document a pack is projected from, carried once instead of through every step call."""

    sot: str
    domain: str
    artifact_id: str


def _step(
    number: int,
    answer: dict[str, Any],
    source: _Projection,
    *,
    covered: int,
    total: int,
    verdict: str = "neutral",
) -> dict[str, Any]:
    sot, domain, artifact_id = source.sot, source.domain, source.artifact_id
    sections = SOT_SECTIONS[domain].get(number, ())
    locators = _locate(sot, sections, sorted(answer), artifact_id)
    sufficiency = _sufficiency(answer, locators, covered, total)
    return {
        "step": number,
        "question_ja": QUESTIONS_JA[number],
        "answer": answer or {"stated": False},
        "sot_sections": sorted({s for s in sections if s in sot}),
        # The provenance rule: a non-absent answer with no locator is a fabricated answer that
        # passes a presence check, so an answer we cannot point at is reported as absent.
        "locators": locators if sufficiency != "absent" else [],
        "verdict_contribution": verdict,
        "sufficiency": sufficiency,
    }


def _absent(number: int, reason: str) -> dict[str, Any]:
    """A step this run cannot answer, saying which and why rather than omitting it.

    Steps 1, 4, 9 and 10 are mandatory keys in the schema — prior, numbers, update, verdict — so a
    run that cannot compute one still carries it, marked absent. Silence would read as "not
    applicable" for a step that is always applicable.
    """
    return {
        "step": number,
        "question_ja": QUESTIONS_JA[number],
        "answer": {"unavailable": reason},
        "sot_sections": [],
        "locators": [],
        "verdict_contribution": "neutral",
        "sufficiency": "absent",
    }


def build_step_pack(
    *,
    pipeline_data: dict[str, Any],
    extractions: list,
    sot: str,
    domain: str = "clinical",
    sot_artifact_id: str = "research/source_of_truth.md",
) -> dict[str, Any]:
    """The pack, computed from the same inputs the SOT was built from.

    Nothing here asks a model anything. Where a step's inputs are missing the step is marked absent
    rather than filled in, because a plausible answer and a computed one are indistinguishable once
    written down — which is exactly why ``step_pack_errors`` cannot check derivation and this
    function has to be the thing that guarantees it.
    """
    domain = domain if domain in SOT_SECTIONS else "clinical"
    source = _Projection(sot=sot, domain=domain, artifact_id=sot_artifact_id)
    total = len(extractions or [])
    grade_record = pipeline_data.get("grade_record")
    metrics = pipeline_data.get("metrics") or {}
    steps: dict[str, dict[str, Any]] = {}

    # Step 1 — the frozen prior. Not written by anything yet: PLAN.md assigns it to Claude at stage
    # 1, and there is no Claude-authored stage until Step 2. Present and absent, never invented.
    steps["1"] = _absent(1, "no frozen prior artifact; authored by Claude at stage 1 (PLAN.md Step 2)")

    search = {
        "records_identified": metrics.get("aff_wide_net_total", 0) + metrics.get("fal_wide_net_total", 0),
        "screened_in": metrics.get("aff_screened_in", 0) + metrics.get("fal_screened_in", 0),
        "full_text_retrieved": metrics.get("aff_fulltext_ok", 0) + metrics.get("fal_fulltext_ok", 0),
        "full_text_errors": metrics.get("aff_fulltext_err", 0) + metrics.get("fal_fulltext_err", 0),
        "highest_tier_affirmative": pipeline_data.get("aff_highest_tier"),
        "highest_tier_falsification": pipeline_data.get("fal_highest_tier"),
    }
    steps["2"] = _step(2, search, source, covered=1, total=1)

    designs = design_rollup(extractions)
    steps["3"] = _step(3, designs, source,
        covered=total - designs.get("unreadable", 0), total=total,
    )

    impacts = pipeline_data.get("impacts") or []
    numbers = {
        "findings_with_event_rates": len(impacts),
        "benefit": len([i for i in impacts if getattr(i, "direction", None) == "benefit"]),
        "harm": len([i for i in impacts if getattr(i, "direction", None) == "harm"]),
        "no_effect": len([i for i in impacts if getattr(i, "direction", None) == "no_effect"]),
    }
    steps["4"] = (
        _step(4, numbers, source, covered=1, total=1,
              verdict="support" if numbers["benefit"] > numbers["harm"] else "neutral")
        if impacts
        else _absent(4, "no finding carried both event rates, so there is no ARR/NNT to state")
    )

    funding = funding_rollup(extractions)
    steps["5"] = _step(5, funding, source,
        covered=total - funding["unknown"], total=total,
        verdict="downgrade" if funding["by_category"]["industry"] > total / 2 else "neutral",
    )

    replication = replication_rollup(extractions)
    steps["6"] = _step(6, replication, source,
        covered=replication["findings_total"] - replication["findings_unattributable"],
        total=replication["findings_total"],
        verdict="support" if replication["findings_replicated"] else "neutral",
    )

    bias = bias_rollup(extractions, grade_record)
    steps["8"] = _step(8, bias, source,
        covered=total - bias["risk_of_bias"]["unclear"], total=total,
        verdict="downgrade" if bias["grade_downgrade_steps"] else "neutral",
    )

    # Step 9 needs both ends of the update, and one end does not exist yet.
    steps["9"] = _absent(9, "the update needs a frozen prior, which no stage produces yet")

    if grade_record:
        verdict = {
            "grade_level": grade_record["level"],
            "confidence_ja": confidence_level(grade_record["level"], extractions),
            "grade_downgrade_steps": bias["grade_downgrade_steps"],
        }
        steps["10"] = _step(10, verdict, source, covered=1, total=1)
    else:
        steps["10"] = _absent(10, "no structured GRADE record, so no verdict was computed")

    return {"schema_version": 1, "sot_domain": domain, "steps": steps}
