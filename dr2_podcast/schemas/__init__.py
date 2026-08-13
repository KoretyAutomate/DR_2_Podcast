"""JSON Schema artifacts for the four-role pipeline, plus the checks JSON Schema cannot express.

Why these are files and not prose (PLAN.md, "Where the loop stands"): five rounds of Codex
review on the plan found, in rounds 4 and 5, *only* missing fields in schemas written as
Markdown tables — `outcome_is_adverse` left paper-level while CER/EER moved per-finding,
`intervention`/`comparator` missing from mandatory locator coverage, a funding table that could
not represent its own `undisclosed` state, a GRADE schema that permitted repeated domains while
the formula summed every entry. That defect class keeps appearing until a missing field is a
test failure rather than a reviewer's catch. These files are that move.

Split of responsibility:

* **JSON Schema** owns structure, enums, types, required keys, and the funding legal-combination
  table (expressible as a four-branch ``oneOf``, so the file is self-describing to any consumer).
* **Python, here** owns what JSON Schema genuinely cannot state: ``finding_key`` computation and
  agreement, field-level locator coverage in both directions, at-most-one-entry-per-GRADE-domain,
  CER/EER pairing and its polarity requirement, key↔``step`` agreement in the step pack, literal
  span verification against a source artifact, recomputation of every derived value, and ordinal
  monotonicity.

Everything here is fail-closed by default: ``validate_*`` raises :class:`SchemaValidationError`.
The ``*_errors`` variants return the full list instead, which is what the mutation-matrix tests
assert against.

**``artifacts`` is a required argument, not an optional one.** Every entry point that can contain
a locator takes ``artifacts`` (source_artifact_id -> text) and verifies every span in the instance,
however deeply nested. An optional artifact map is a bypass: the caller who omits it gets a green
result over unverified provenance, which is indistinguishable from a checked one. A caller that
genuinely wants shape alone calls :func:`schema_errors`, which is named for what it does.

Nothing in this package calls an LLM, and nothing in it is authored by one.

The implementation is split across ``_loading``, ``_locators``, ``_derived``, ``_findings`` and
``_records`` to stay under the repo's file-size ceiling. Import from the package root; the module
layout is an implementation detail.
"""

from __future__ import annotations

from dr2_podcast.schemas._derived import (
    CONSTANT_OPERANDS,
    DERIVED_OPERATIONS,
    DERIVED_RESULT_DECIMALS,
    ZERO_THRESHOLD,
    agrees_at_producer_precision,
    recompute_derived,
)
from dr2_podcast.schemas._findings import (
    CLAIM_BEARING_FIELDS,
    FINDING_KEY_FIELDS,
    compute_finding_key,
    extraction_errors,
    finding_errors,
    validate_extraction,
    validate_finding,
)
from dr2_podcast.schemas._loading import (
    EXAMPLE_DIR,
    EXAMPLE_NAMES,
    SCHEMA_DIR,
    SCHEMA_NAMES,
    SCHEMA_VERSION,
    SchemaValidationError,
    example_path,
    load_example,
    load_schema,
    schema_errors,
    schema_path,
)
from dr2_podcast.schemas._locators import iter_locators, span_errors, verify_locator_span
from dr2_podcast.schemas._records import (
    CONFIDENCE_LADDER,
    confidence_index,
    funding_errors,
    grade_errors,
    net_direction,
    ordinal_monotonicity_errors,
    step_pack_errors,
    validate_funding,
    validate_grade,
    validate_ordinal_monotonicity,
    validate_step_pack,
)

__all__ = [
    "CLAIM_BEARING_FIELDS",
    "CONFIDENCE_LADDER",
    "CONSTANT_OPERANDS",
    "DERIVED_OPERATIONS",
    "DERIVED_RESULT_DECIMALS",
    "EXAMPLE_DIR",
    "EXAMPLE_NAMES",
    "FINDING_KEY_FIELDS",
    "SCHEMA_DIR",
    "SCHEMA_NAMES",
    "SCHEMA_VERSION",
    "ZERO_THRESHOLD",
    "SchemaValidationError",
    "agrees_at_producer_precision",
    "compute_finding_key",
    "confidence_index",
    "example_path",
    "extraction_errors",
    "finding_errors",
    "funding_errors",
    "grade_errors",
    "iter_locators",
    "load_example",
    "load_schema",
    "net_direction",
    "ordinal_monotonicity_errors",
    "recompute_derived",
    "schema_errors",
    "schema_path",
    "span_errors",
    "step_pack_errors",
    "validate_extraction",
    "validate_finding",
    "validate_funding",
    "validate_grade",
    "validate_ordinal_monotonicity",
    "validate_step_pack",
    "verify_locator_span",
]
