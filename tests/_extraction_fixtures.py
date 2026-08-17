"""The study text and the model payloads the extraction tests are written against.

Shared because the findings tests and the funding tests read the same paper, and a second copy
of SOURCE would let the two drift until a quote verified in one file and not the other.
"""

from __future__ import annotations

from typing import Any

from dr2_podcast.research.clinical import WideNetRecord

SOURCE = (
    "Methods. Community-dwelling adults aged 50 years or older received 800 IU/day vitamin D or "
    "matching placebo.\n"
    "Results. Absolute risk reduction 5.0% (95% CI 2.0 to 8.0), p=0.03 for hip fracture at 12 months.\n"
    "No significant difference in falls was observed between groups (p=0.41).\n"
    "Funding. Supported by grant R01-AG000000 from the National Institute on Aging.\n"
)


def _raw_finding(**overrides: Any) -> dict:
    base = {
        "population": "adults aged 50 or older",
        "intervention": "vitamin D 800 IU/day",
        "comparator": "placebo",
        "endpoint": "hip fracture",
        "timepoint": "12 months",
        "direction": "decrease",
        "value": 5.0,
        "unit": "%",
        "ci_low": 2.0,
        "ci_high": 8.0,
        "p_value": 0.03,
        "is_primary": True,
        "control_event_rate": 0.15,
        "experimental_event_rate": 0.10,
        "outcome_is_adverse": True,
        "identity_quote": (
            "Community-dwelling adults aged 50 years or older received 800 IU/day vitamin D or "
            "matching placebo."
        ),
        "quote": "Absolute risk reduction 5.0% (95% CI 2.0 to 8.0), p=0.03",
    }
    base.update(overrides)
    return base


def _record() -> WideNetRecord:
    return WideNetRecord(
        pmid="12345678",
        doi=None,
        title="Vitamin D and hip fracture",
        abstract="",
        study_type="rct",
        sample_size=None,
        primary_objective=None,
        year=2026,
        journal=None,
        authors=None,
        url="https://example.org/x",
        source_db="pubmed",
    )
