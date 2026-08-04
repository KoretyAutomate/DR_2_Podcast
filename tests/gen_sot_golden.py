"""Characterization-golden generator for pipeline_sot.build_imrad_sot.

Drives build_imrad_sot across a boundary-driven input grid and writes
tests/golden_sot.json. test_sot_golden.py asserts the current output still
equals that file, byte for byte.

The point is to make a refactor of build_imrad_sot / _build_social_science_sot
provable: regenerate after the refactor and the file must come back identical.

    python -m tests.gen_sot_golden          # writes tests/golden_sot.json
    python -m tests.gen_sot_golden --check  # regenerate and diff, exit 1 on drift

A golden that has never failed proves nothing — see test_sot_golden.py, which
carries a mutation-sensitivity test alongside the equality assertions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

from dr2_podcast.pipeline_sot import build_imrad_sot
from dr2_podcast.research.clinical import (
    DeepExtraction,
    PaperMetadata,
    TierKeywords,
    TieredSearchPlan,
    WideNetRecord,
)
from dr2_podcast.research.clinical_math import ClinicalImpact

GOLDEN_PATH = Path(__file__).parent / "golden_sot.json"


# ---------------------------------------------------------------------------
# Fixture builders — real dataclasses, not mocks, so field access matches prod
# ---------------------------------------------------------------------------


def _tier(n: int) -> TierKeywords:
    return TierKeywords(
        intervention=[f"intervention-t{n}-a", f"intervention-t{n}-b"],
        outcome=[f"outcome-t{n}"],
        population=[f"population-t{n}"] if n != 2 else [],
        rationale=f"Tier {n} rationale.",
    )


def _plan(role: str, *, approved: bool) -> TieredSearchPlan:
    return TieredSearchPlan(
        pico={
            "population": f"{role} population",
            "intervention": f"{role} intervention",
            "comparison": f"{role} comparison",
            "outcome": f"{role} outcome",
        },
        tier1=_tier(1),
        tier2=_tier(2),
        tier3=_tier(3),
        role=role,
        auditor_approved=approved,
        auditor_notes="Auditor notes go here. " * 20,
        revision_count=2 if approved else 0,
    )


def _legacy_plan(role: str) -> SimpleNamespace:
    """Pre-tier strategy object — exercises the mesh_terms backward-compat branch."""
    return SimpleNamespace(
        pico={
            "population": f"{role} population",
            "intervention": f"{role} intervention",
            "comparison": f"{role} comparison",
            "outcome": f"{role} outcome",
        },
        mesh_terms={"intervention": ["Mesh A", "Mesh B"], "outcome": ["Mesh C"]},
    )


def _extraction(i: int, *, with_metadata: bool, sparse: bool = False) -> DeepExtraction:
    if sparse:
        return DeepExtraction(pmid=None, doi=None, title=f"Sparse study {i}", url="")
    meta = None
    if with_metadata:
        meta = PaperMetadata(
            citation_count=10 * i,
            influential_citation_count=i,
            fwci=1.0 + i / 10,
            funding_sources=[f"Funder {i}"],
            is_retracted=(i == 2),
            is_corrected=False,
            has_clinical_trial_number=True,
            clinical_trial_numbers=[f"NCT0000000{i}"],
            enrichment_sources=["openalex", "crossref"],
        )
    return DeepExtraction(
        pmid=f"1000000{i}",
        doi=f"10.1000/test{i}",
        title=f"Study number {i}",
        url=f"https://example.org/study{i}",
        attrition_pct="5",
        effect_size=f"0.{i}5",
        demographics="Adults 40-70",
        follow_up_period=f"{i} years",
        funding_source="Public grant",
        conflicts_of_interest="None declared",
        biological_mechanism="Receptor blockade",
        control_event_rate=0.20,
        experimental_event_rate=0.10,
        outcome_is_adverse=True,
        primary_outcome="All-cause mortality",
        secondary_outcomes=["Hospitalisation"],
        blinding="Double-blind",
        randomization_method="Computer generated",
        intention_to_treat=True,
        sample_size_total=500 * i,
        sample_size_intervention=250 * i,
        sample_size_control=250 * i,
        study_design="RCT",
        risk_of_bias="Low",
        research_tier=(i % 3) + 1,
        raw_facts=f"Raw facts for study {i}.",
        paper_metadata=meta,
    )


def _wide(i: int) -> WideNetRecord:
    return WideNetRecord(
        pmid=f"2000000{i}",
        doi=f"10.2000/wide{i}",
        title=f"Wide net record {i}",
        abstract="Abstract text.",
        study_type="RCT",
        sample_size=str(100 * i),
        primary_objective="Objective text.",
        year=2019 + i,
        journal=f"Wide Journal {i}",
        authors=f"Wide Author {i}",
        url=f"https://example.org/wide{i}",
        source_db="pubmed",
        research_tier=(i % 3) + 1,
        relevance_score=0.5 + i / 10,
    )


def _impact(study_id: str, direction: str) -> ClinicalImpact:
    return ClinicalImpact(
        study_id=study_id,
        cer=0.20,
        eer=0.10 if direction == "benefit" else 0.30,
        arr=0.10 if direction == "benefit" else -0.10,
        rrr=0.5 if direction == "benefit" else -0.5,
        nnt=10.0,
        nnt_interpretation="Treat 10 patients to prevent 1 event",
        direction=direction,
    )


def _report(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        report=text,
        total_summaries=3,
        total_urls_fetched=7,
        duration_seconds=42,
        sources=[],
    )


FULL_AUDIT = (
    "### Overall Certainty\n"
    "Final GRADE: Moderate\n"
    "Verdict: Supported with reservations\n"
    "The body of evidence supports a modest benefit.\n\n"
    "### Risk of Bias\nMostly low.\n\n"
    "### Inconsistency\nSome heterogeneity.\n\n"
    "### Indirectness\nPopulations broadly applicable.\n\n"
    "### Imprecision\nConfidence intervals are wide.\n\n"
    "### Publication Bias\nFunnel plot not assessed.\n"
)


def _pipeline_data(**overrides) -> dict:
    base = {
        "domain": "clinical",
        "aff_strategy": _plan("affirmative", approved=True),
        "fal_strategy": _plan("adversarial", approved=False),
        "aff_extractions": [_extraction(1, with_metadata=True), _extraction(2, with_metadata=True)],
        "fal_extractions": [_extraction(3, with_metadata=False)],
        "aff_top": [_wide(1), _wide(2)],
        "fal_top": [_wide(3)],
        "impacts": [_impact("10000001", "benefit"), _impact("10000003", "harm")],
        "framing_context": "Framing context paragraph for the run.",
        "search_date": "2026-08-03",
        "metrics": {
            "aff_wide_net_total": 120,
            "fal_wide_net_total": 80,
            "aff_screened_in": 20,
            "fal_screened_in": 15,
            "aff_fulltext_ok": 18,
            "fal_fulltext_ok": 12,
            "aff_fulltext_err": 2,
            "fal_fulltext_err": 3,
        },
        "math_report": "## Clinical Math\nARR 10%, NNT 10.\n",
    }
    base.update(overrides)
    return base


def _reports(pd: dict, *, audit: str = FULL_AUDIT) -> dict:
    return {
        "pipeline_data": pd,
        "audit": _report(audit),
        "lead": _report("Affirmative case body.\n<think>hidden</think>\nMore affirmative text."),
        "counter": _report("Falsification case body."),
    }


# ---------------------------------------------------------------------------
# The grid — each entry is a boundary worth pinning
# ---------------------------------------------------------------------------


def _cases() -> list[tuple[str, dict]]:
    cases: list[tuple[str, dict]] = []

    def add(name, *, reports, **kwargs):
        cases.append((name, {"reports": reports, **kwargs}))

    full = _pipeline_data()
    add("clinical_full_en", reports=_reports(full), topic="Statins for primary prevention")
    add("clinical_full_ja", reports=_reports(full), topic="一次予防におけるスタチン", language="ja")

    # Everything empty — the degenerate boundary
    empty = _pipeline_data(
        aff_strategy=None,
        fal_strategy=None,
        aff_extractions=[],
        fal_extractions=[],
        aff_top=[],
        fal_top=[],
        impacts=[],
        framing_context="",
        search_date="",
        metrics={},
        math_report="",
    )
    add("clinical_empty_en", reports=_reports(empty, audit=""), topic="Empty run")
    add("clinical_empty_ja", reports=_reports(empty, audit=""), topic="空の実行", language="ja")

    # Legacy mesh_terms strategy instead of the tiered plan
    add(
        "clinical_legacy_strategy",
        reports=_reports(
            _pipeline_data(aff_strategy=_legacy_plan("affirmative"), fal_strategy=_legacy_plan("adversarial"))
        ),
        topic="Legacy strategy shape",
    )

    # Impacts: harm-only means there is no benefit entry to pick as representative
    add(
        "clinical_harm_only",
        reports=_reports(_pipeline_data(impacts=[_impact("10000003", "harm")])),
        topic="Harm only",
    )
    add("clinical_no_impacts", reports=_reports(_pipeline_data(impacts=[])), topic="No impacts")

    # Extractions with no metadata at all, and sparse extractions
    add(
        "clinical_sparse_extractions",
        reports=_reports(
            _pipeline_data(
                aff_extractions=[_extraction(1, with_metadata=False, sparse=True)],
                fal_extractions=[],
            )
        ),
        topic="Sparse extractions",
    )

    # Audit text with no GRADE header at all
    add(
        "clinical_audit_without_grade",
        reports=_reports(_pipeline_data(), audit="Free-form audit prose with no headers."),
        topic="No GRADE headers",
    )

    # ev_quality / aff_cand boundaries
    add("clinical_zero_candidates", reports=_reports(full), topic="Zero candidates", ev_quality="limited", aff_cand=0)
    add(
        "clinical_high_candidates",
        reports=_reports(full),
        topic="Many candidates",
        ev_quality="sufficient",
        aff_cand=999,
    )

    # social_science dispatch, both explicit and auto-detected from pipeline_data
    ss = _pipeline_data(domain="social_science")
    add("social_science_full_en", reports=_reports(ss), topic="Class size and attainment")
    add("social_science_full_ja", reports=_reports(ss), topic="学級規模と学力", language="ja")
    add(
        "social_science_explicit_arg",
        reports=_reports(_pipeline_data()),
        topic="Explicit social domain",
        domain="social_science",
    )
    ss_empty = _pipeline_data(
        domain="social_science",
        aff_strategy=None,
        fal_strategy=None,
        aff_extractions=[],
        fal_extractions=[],
        aff_top=[],
        fal_top=[],
        impacts=[],
        framing_context="",
        search_date="",
        metrics={},
    )
    add("social_science_empty", reports=_reports(ss_empty, audit=""), topic="Empty social run")

    # reports missing the optional keys entirely
    add("missing_report_keys", reports={"pipeline_data": _pipeline_data()}, topic="Missing report keys")
    add("missing_pipeline_data", reports={}, topic="Missing pipeline data")

    return cases


def generate() -> dict[str, str]:
    out: dict[str, str] = {}
    for name, kwargs in _cases():
        # ev_quality / aff_cand are still carried in the grid: they were
        # parameters of build_imrad_sot that no branch read, and these two cases
        # pin that they never affected the document.
        params = {
            "topic": kwargs.get("topic", "Test topic"),
            "reports": kwargs["reports"],
            "domain": kwargs.get("domain", "clinical"),
            "language": kwargs.get("language", "en"),
        }
        out[name] = build_imrad_sot(**params)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="compare against the stored golden instead of writing")
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
