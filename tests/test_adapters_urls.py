"""The url_validation stage and the validated source library it produces.

Split out of test_adapters_scripts.py to stay under the repo's file-size ceiling. An adapter's job
is to reconstruct, from the run directory alone, the state the monolithic runner built in memory;
what is tested here is that reconstruction and the fail-closed behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from dr2_podcast import adapters
from dr2_podcast.adapters import research_stages
from dr2_podcast.artifacts import ArtifactError
from dr2_podcast.stage import write_run_config


@pytest.fixture(autouse=True)
def _no_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never let these tests depend on whether vLLM happens to be up.

    initialise_run_globals probes the backend before building the LLM handles. Left real, this file
    passes or fails according to what is running on the machine — which is how it passed in
    isolation and failed in the suite.
    """
    monkeypatch.setattr("dr2_podcast.pipeline.get_final_model_string", lambda: "test-model")


@pytest.fixture()
def run_dir(tmp_path: Path) -> Path:
    for sub in ("research", "scripts", "audio", "meta"):
        (tmp_path / sub).mkdir()
    write_run_config(tmp_path, topic="ビタミンDと骨折", language="ja", target_length_minutes=25)
    return tmp_path


RUN_CONFIG = {"topic": "ビタミンDと骨折", "language": "ja", "target_length_minutes": 25}


# --------------------------------------------------------------------------- #
# url_validation
# --------------------------------------------------------------------------- #
def test_url_validation_reads_its_input_from_disk(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/research_sources.json").write_text(
        json.dumps(
            {
                "affirmative": [{"url": "https://example.org/a", "title": "A"}],
                "falsification": [{"url": "https://example.org/b"}],
            }
        )
    )
    checked: dict[str, Any] = {}

    def _fake_validate(urls: list[str], max_workers: int = 15) -> dict[str, str]:
        checked["urls"] = urls
        return dict.fromkeys(urls, "Valid")

    monkeypatch.setattr("dr2_podcast.tools.link_validator.validate_multiple_urls_parallel", _fake_validate)
    adapters.url_validation(run_dir, RUN_CONFIG)

    assert checked["urls"] == ["https://example.org/a", "https://example.org/b"]
    results = json.loads((run_dir / "research/url_validation_results.json").read_text())
    assert results["https://example.org/a"] == "Valid"


# prepush codex 2026-08-12: the phase removes broken URLs from the library; dropping that would
# leave staged runs citing sources the pipeline has already determined are unusable.
def test_url_validation_filters_the_broken_sources(run_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (run_dir / "research/research_sources.json").write_text(
        json.dumps(
            {
                "affirmative": [
                    {"url": "https://ok.example/a", "title": "good"},
                    {"url": "https://dead.example/b", "title": "broken"},
                ],
                "falsification": [{"url": "https://err.example/c"}],
            }
        )
    )
    monkeypatch.setattr(
        "dr2_podcast.tools.link_validator.validate_multiple_urls_parallel",
        lambda urls, max_workers=15: {
            "https://ok.example/a": "Valid (200)",
            "https://dead.example/b": "Broken (404)",
            "https://err.example/c": "ERROR: timeout",
        },
    )
    adapters.url_validation(run_dir, RUN_CONFIG)

    from dr2_podcast.pipeline import research_sources_file

    assert research_sources_file(run_dir).name == "research_sources_validated.json", "the stamp matches"
    filtered = json.loads((run_dir / "research/research_sources_validated.json").read_text())
    assert [e["url"] for e in filtered["affirmative"]] == ["https://ok.example/a"]
    assert filtered["falsification"] == []

    untouched = json.loads((run_dir / "research/research_sources.json").read_text())
    assert len(untouched["affirmative"]) == 2, "the producer's own artifact is not edited"


# prepush codex 2026-08-12: LinkValidatorTool._run returns "✗ ERROR: …" with a leading marker, so
# the phase's startswith("ERROR") test misses it and an unusable citation proceeds downstream.
@pytest.mark.parametrize(
    "status",
    ["✗ ERROR: connection reset", "ERROR: timeout", "✗ Broken Link (Status: 404 Not Found)", "✗ Invalid URL: loop"],
)
def test_every_rejected_status_shape_is_filtered(status: str) -> None:
    sources = {"affirmative": [{"url": "https://bad.example/x"}, {"url": "https://good.example/y"}]}
    results = {"https://bad.example/x": status, "https://good.example/y": "✓ Valid (200)"}
    filtered = research_stages._without_broken(sources, results)
    assert [e["url"] for e in filtered["affirmative"]] == ["https://good.example/y"], status


# prepush codex 2026-08-13 [P1]: pipeline.py:1440 SHOWS the agent each entry's stored `index`, and
# read_research_source resolves the number it is handed POSITIONALLY. Dropping a non-final entry
# left a gap between the two, so an agent asking for the source it was shown got a different one.
def test_filtering_renumbers_so_the_listing_and_the_lookup_agree() -> None:
    sources = {
        "affirmative": [
            {"index": 0, "url": "https://ok.example/a", "title": "kept first"},
            {"index": 1, "url": "https://bad.example/x", "title": "dropped"},
            {"index": 2, "url": "https://ok.example/c", "title": "kept second"},
        ]
    }
    results = {
        "https://ok.example/a": "✓ Valid (200)",
        "https://bad.example/x": "✗ Broken Link (Status: 404 Not Found)",
        "https://ok.example/c": "✓ Valid (200)",
    }
    kept = research_stages._without_broken(sources, results)["affirmative"]

    assert [e["title"] for e in kept] == ["kept first", "kept second"]
    for position, entry in enumerate(kept):
        assert entry["index"] == position, "the index shown must be the index that resolves"


def test_filtering_leaves_an_entry_without_an_index_alone() -> None:
    """Not every library shape carries one; inventing the key would change the artifact."""
    sources = {"affirmative": [{"url": "https://ok.example/a"}]}
    kept = research_stages._without_broken(sources, {"https://ok.example/a": "✓ Valid (200)"})
    assert kept["affirmative"] == [{"url": "https://ok.example/a"}]


# prepush codex 2026-08-12: the filtered artifact was written and then read by nobody — the tools
# still opened research_sources.json, so rejected URLs reached the blueprint anyway.
def test_the_agents_read_the_validated_library_when_it_exists(run_dir: Path) -> None:
    import hashlib

    from dr2_podcast.pipeline import research_sources_file

    raw = run_dir / "research/research_sources.json"
    raw.write_text("{}")
    assert research_sources_file(run_dir).name == "research_sources.json"

    (run_dir / "research/research_sources_validated.json").write_text("{}")
    (run_dir / "research/research_sources_validated.sha256").write_text(
        hashlib.sha256(raw.read_bytes()).hexdigest()
    )
    assert research_sources_file(run_dir).name == "research_sources_validated.json"


# prepush codex 2026-08-13, twice. First: the legacy runner regenerates research_sources.json in
# place and knows nothing about the validated copy. Then: comparing mtimes is not a fact about
# derivation — atomic replacement preserves coarse timestamps and restoring a run reorders them —
# so the validated copy is pinned to its source BY HASH.
def test_a_regenerated_library_invalidates_the_validated_copy(run_dir: Path) -> None:
    import hashlib

    from dr2_podcast.pipeline import research_sources_file

    raw = run_dir / "research/research_sources.json"
    raw.write_text('{"affirmative": []}')
    (run_dir / "research/research_sources_validated.json").write_text("{}")
    (run_dir / "research/research_sources_validated.sha256").write_text(
        hashlib.sha256(raw.read_bytes()).hexdigest()
    )
    assert research_sources_file(run_dir).name == "research_sources_validated.json"

    raw.write_text('{"affirmative": [{"url": "https://new.example/x"}]}')
    assert research_sources_file(run_dir).name == "research_sources.json"


def test_a_validated_copy_with_no_stamp_is_not_trusted(run_dir: Path) -> None:
    """The legacy runner writes no stamp, so its directory never matches by accident."""
    from dr2_podcast.pipeline import research_sources_file

    (run_dir / "research/research_sources.json").write_text("{}")
    (run_dir / "research/research_sources_validated.json").write_text("{}")
    assert research_sources_file(run_dir).name == "research_sources.json"


def test_validation_gates_the_blueprint() -> None:
    """Required, not optional: an optional gate lets the blueprint run before validation ever did,
    research_sources_file() falls back to the raw library, and rejected URLs reach the episode."""
    from dr2_podcast.stages import direct_producers, get_stage

    assert "research/research_sources_validated.json" in get_stage("blueprint").consumes
    assert "url_validation" in direct_producers("blueprint")


def test_url_validation_fails_closed_on_a_missing_sources_file(run_dir: Path) -> None:
    with pytest.raises(ArtifactError, match="cannot read"):
        adapters.url_validation(run_dir, RUN_CONFIG)


def test_urls_are_found_at_any_nesting_depth() -> None:
    """The sources document's shape has changed before; a shape-specific reader would miss URLs."""
    found = research_stages._iter_urls(
        {"a": [{"url": "u1"}], "b": {"c": {"d": [{"url": "u2"}]}}, "url": "u3", "n": None}
    )
    assert sorted(found) == ["u1", "u2", "u3"]
