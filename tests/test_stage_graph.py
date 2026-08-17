"""The stage graph — PLAN.md Step 1.

What each stage consumes and produces, and therefore which stages an answer about staleness
propagates to. No I/O and no execution: this is the contract, tested as a contract.
"""

from __future__ import annotations

import pytest

from dr2_podcast.stages import (
    AVAILABLE_STAGE_NAMES,
    STAGE_NAMES,
    STAGES,
    direct_consumers,
    downstream_of,
    get_stage,
    producer_of,
)


# --------------------------------------------------------------------------- #
# The stage graph
# --------------------------------------------------------------------------- #
def test_the_plan_s_fourteen_stage_names_are_all_declared() -> None:
    """Declared-but-unavailable, not omitted — the target shape stays visible."""
    for name in ("framing", "keywords", "search", "screen", "extract", "synthesize", "grade",
                 "sot", "translate", "blueprint", "draft", "polish", "audit", "audio"):
        assert name in STAGE_NAMES


def test_the_six_phase_one_substages_are_declared_unavailable_with_a_reason() -> None:
    """They cannot be separated before Step 10 splits _run_research_track; saying so beats pretending."""
    for name in ("keywords", "search", "screen", "extract", "synthesize", "grade"):
        stage = get_stage(name)
        assert not stage.available
        assert "Step 10" in stage.unavailable_reason
    assert get_stage("research").available, "the transitional composite has to be usable meanwhile"


def test_sot_is_unavailable_and_says_exactly_why() -> None:
    """It was written and withdrawn: build_imrad_sot runs inside phase 1 on the live reports dict,
    and _serialize_dataclass destroys that dict rather than flattening it."""
    stage = get_stage("sot")
    assert not stage.available
    assert "REPR-STRINGIFIES" in stage.unavailable_reason
    assert producer_of("research/source_of_truth.md") == "research"


def test_every_available_stage_declares_at_least_one_output() -> None:
    for name in AVAILABLE_STAGE_NAMES:
        stage = get_stage(name)
        assert stage.produces or stage.optional_outputs, name


def test_every_consumed_artifact_has_a_declared_producer() -> None:
    """An input nothing writes is a graph that cannot be resolved from disk.

    Except the ones a person writes. Step 10 puts a reviewer between the plan and the search, so
    the strategy approval is deliberately not produced by any stage — a pipeline that could produce
    its own approval would be approving itself.
    """
    from dr2_podcast.stages import REVIEWER_WRITTEN

    for stage in STAGES:
        for artifact in stage.consumes:
            if artifact in REVIEWER_WRITTEN:
                continue
            assert producer_of(artifact) is not None, f"{stage.name} consumes unproduced {artifact}"


def test_the_only_unproduced_input_is_the_one_a_reviewer_writes() -> None:
    """The exemption above has to stay narrow: every other unproduced input is a graph defect."""
    from dr2_podcast.stages import REVIEWER_WRITTEN

    unproduced = {a for stage in STAGES for a in stage.consumes if producer_of(a) is None}
    assert unproduced == set(REVIEWER_WRITTEN)


def test_no_two_stages_claim_the_same_output() -> None:
    seen: dict[str, str] = {}
    for stage in STAGES:
        for artifact in stage.produces + stage.optional_outputs:
            assert artifact not in seen, f"{artifact} claimed by {seen.get(artifact)} and {stage.name}"
            seen[artifact] = stage.name


def test_downstream_is_transitive_and_ordered() -> None:
    # Two consumers, not one: `sot` reads research/domain_classification.json for the framework it
    # renders under, so framing feeds it directly as well as through research.
    # framing feeds research directly, sot and blueprint through domain_classification.json, and
    # draft and polish through meta/session_roles.json.
    # plan_search leads, because Step 10 put it between framing and the search.
    # framing_prior leads now: the prior is written before anything is searched.
    assert direct_consumers("framing") == (
        "framing_prior", "plan_search", "research", "blueprint", "draft", "polish", "audit"
    )
    chain = downstream_of("framing")
    assert {"research", "blueprint", "draft", "polish", "audit", "audio"} <= set(chain)
    assert chain.index("draft") < chain.index("polish") < chain.index("audit")


def test_the_graph_is_acyclic() -> None:
    for name in STAGE_NAMES:
        assert name not in downstream_of(name), f"{name} is downstream of itself"


def test_an_unavailable_stage_must_say_why() -> None:
    from dr2_podcast.stages import Stage

    with pytest.raises(ValueError, match="says nothing about why"):
        Stage("x", (), (), "python", available=False)


def test_unknown_stage_names_list_the_alternatives() -> None:
    with pytest.raises(KeyError, match="known:"):
        get_stage("nonesuch")
