"""Characterization tests for pipeline_flow.phase_1_research and _phase_cache_key.

pipeline_flow.py is the live Prefect orchestrator and had no test coverage at
all. These tests were written BEFORE the phase_1_research complexity refactor so
that the refactor could be shown to preserve behaviour, and they stay as the
first coverage this module has.

The Prefect task is exercised through its undecorated `.fn`, with
`get_run_logger` patched to a plain logger (outside a flow run Prefect's
`get_run_logger` raises). Everything that reaches the network or an LLM is
stubbed; what is under test is the orchestration — which files get written,
what the returned dict contains, and which failures abort versus degrade.
"""

import json
import logging
from types import SimpleNamespace

import pytest

from dr2_podcast import pipeline_flow as pf
from dr2_podcast.pipeline import InsufficientEvidenceError


# ---------------------------------------------------------------------------
# _phase_cache_key — pure, and load-bearing for --resume
# ---------------------------------------------------------------------------


def _ctx(task_name):
    return SimpleNamespace(task=SimpleNamespace(name=task_name))


class TestPhaseCacheKey:
    def test_uses_basename_not_full_path(self):
        """A '/' in the key makes Prefect's LocalFileSystem storage escape its root."""
        key = pf._phase_cache_key(_ctx("phase_1_research"), {"output_dir": "/home/u/out/2026-08-03_12-00-00_ab12"})
        assert "/" not in key
        assert key.startswith("2026-08-03_12-00-00_ab12-")
        assert key.endswith("-phase_1_research")

    def test_same_dir_same_task_is_stable(self):
        params = {"output_dir": "/home/u/out/run-a"}
        assert pf._phase_cache_key(_ctx("phase_5"), params) == pf._phase_cache_key(_ctx("phase_5"), params)

    def test_different_task_names_differ(self):
        params = {"output_dir": "/home/u/out/run-a"}
        assert pf._phase_cache_key(_ctx("phase_5"), params) != pf._phase_cache_key(_ctx("phase_6"), params)

    def test_same_basename_different_parent_differs(self):
        """The path hash is the collision guard that makes this true."""
        a = pf._phase_cache_key(_ctx("p"), {"output_dir": "/one/run"})
        b = pf._phase_cache_key(_ctx("p"), {"output_dir": "/two/run"})
        assert a != b

    def test_falls_back_to_output_dir_str(self):
        key = pf._phase_cache_key(_ctx("p"), {"output_dir_str": "/home/u/out/run-b"})
        assert key.startswith("run-b-")

    def test_missing_output_dir_does_not_raise(self):
        assert pf._phase_cache_key(_ctx("p"), {}).endswith("-p")

    def test_trailing_slash_is_stripped(self):
        a = pf._phase_cache_key(_ctx("p"), {"output_dir": "/home/u/out/run-c"})
        b = pf._phase_cache_key(_ctx("p"), {"output_dir": "/home/u/out/run-c/"})
        # basename matches; the hash differs because it is taken over the raw string
        assert a.split("-")[0] == b.split("-")[0]


# ---------------------------------------------------------------------------
# phase_1_research
# ---------------------------------------------------------------------------


def _report(text, n_sources=2, sources=None):
    return SimpleNamespace(
        report=text,
        total_summaries=n_sources,
        sources=sources if sources is not None else [],
    )


def _source(url="https://example.org/a", summary="Useful summary.", error=None):
    return SimpleNamespace(
        url=url,
        title="A title",
        query="a query",
        goal="a goal",
        summary=summary,
        error=error,
        metadata=None,
    )


@pytest.fixture
def flow_env(monkeypatch, tmp_path):
    """Stub every collaborator phase_1_research reaches out to."""
    monkeypatch.setattr(pf, "get_run_logger", lambda: logging.getLogger("test-flow"))

    state = {"deep_reports": None, "screening": {}, "sot": "## SOT BODY\n", "raise_in_research": None}

    from dr2_podcast import pipeline as _pipeline
    from dr2_podcast.research import clinical as _clinical

    # output_path: flat layout inside tmp_path, directories created on demand
    def _output_path(run_dir, filename):
        p = tmp_path / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    monkeypatch.setattr(_pipeline, "output_path", _output_path)
    monkeypatch.setattr(_pipeline, "build_imrad_sot", lambda **kw: state["sot"])
    monkeypatch.setattr(_pipeline, "summarize_report", lambda *a, **k: "SUMMARY")
    monkeypatch.setattr(_pipeline, "_serialize_dataclass", lambda obj: {"serialized": True})

    written = {}

    def _insufficient(topic, aff_n, neg_n, out_dir):
        written["insufficient"] = (topic, aff_n, neg_n)

    monkeypatch.setattr(_pipeline, "_write_insufficient_evidence_report", _insufficient)

    async def _run_deep_research(**kwargs):
        if state["raise_in_research"]:
            raise state["raise_in_research"]
        for fname, payload in state["screening"].items():
            (tmp_path / fname).write_text(json.dumps(payload))
        return state["deep_reports"]

    monkeypatch.setattr(_clinical, "run_deep_research", _run_deep_research)

    return SimpleNamespace(state=state, tmp_path=tmp_path, written=written)


def _call(flow_env, *, domain="clinical", threshold=5):
    return pf.phase_1_research.fn(
        output_dir=str(flow_env.tmp_path),
        topic_name="Coffee and productivity",
        language="en",
        framing_output="Framing text.",
        research_domain=domain,
        evidence_limited_threshold=threshold,
    )


class TestPhase1Research:
    def test_happy_path_writes_every_artifact(self, flow_env):
        flow_env.state["screening"] = {
            "screening_results_aff.json": {"total_candidates": 20},
            "screening_results_neg.json": {"total_candidates": 12},
        }
        flow_env.state["deep_reports"] = {
            "lead": _report("Affirmative body.", 5, [_source()]),
            "counter": _report("Falsification body.", 3, [_source("https://example.org/b")]),
            "audit": _report("GRADE body.", 1),
        }

        result = _call(flow_env)

        tp = flow_env.tmp_path
        assert (tp / "affirmative_case.md").read_text() == "Affirmative body."
        assert (tp / "falsification_case.md").read_text() == "Falsification body."
        assert (tp / "grade_synthesis.md").read_text() == "GRADE body."
        assert (tp / "source_of_truth.md").read_text() == "## SOT BODY\n"

        sources = json.loads((tp / "research_sources.json").read_text())
        assert [s["url"] for s in sources["lead"]] == ["https://example.org/a"]
        assert [s["url"] for s in sources["counter"]] == ["https://example.org/b"]

        assert result["evidence_quality"] == "sufficient"
        assert result["aff_candidates"] == 20
        assert result["neg_candidates"] == 12
        assert result["sot_content"] == "## SOT BODY\n"
        assert result["sot_summary"] == "SUMMARY"
        assert result["research_domain"] == "clinical"
        assert result["deep_reports_serialized"] == {"serialized": True}

    def test_zero_affirmative_candidates_aborts(self, flow_env):
        flow_env.state["screening"] = {
            "screening_results_aff.json": {"total_candidates": 0},
            "screening_results_neg.json": {"total_candidates": 7},
        }
        flow_env.state["deep_reports"] = {"lead": _report("x"), "counter": _report("y"), "audit": _report("z")}

        with pytest.raises(InsufficientEvidenceError) as exc:
            _call(flow_env)
        assert "0 candidates" in str(exc.value)
        # the operator-facing report is written before the abort
        assert flow_env.written["insufficient"] == ("Coffee and productivity", 0, 7)

    def test_below_threshold_marks_evidence_limited_and_prefixes_sot(self, flow_env):
        flow_env.state["screening"] = {"screening_results_aff.json": {"total_candidates": 3}}
        flow_env.state["deep_reports"] = {"lead": _report("a"), "counter": _report("b"), "audit": _report("c")}

        result = _call(flow_env, threshold=5)

        assert result["evidence_quality"] == "limited"
        assert result["sot_content"].startswith("## Evidence Quality Notice")
        assert "3 candidate studies" in result["sot_content"]
        assert result["sot_content"].endswith("## SOT BODY\n")

    def test_at_threshold_is_not_limited(self, flow_env):
        """Boundary: the comparison is `< threshold`, so equal is still sufficient."""
        flow_env.state["screening"] = {"screening_results_aff.json": {"total_candidates": 5}}
        flow_env.state["deep_reports"] = {"lead": _report("a"), "counter": _report("b"), "audit": _report("c")}
        assert _call(flow_env, threshold=5)["evidence_quality"] == "sufficient"

    def test_research_failure_degrades_instead_of_raising(self, flow_env):
        flow_env.state["raise_in_research"] = RuntimeError("vLLM down")
        result = _call(flow_env)
        assert result["sot_content"] == ""
        assert result["sot_summary"] == ""
        assert result["aff_candidates"] == 0
        assert result["deep_reports_serialized"] is None

    def test_missing_role_report_is_skipped_not_fatal(self, flow_env):
        flow_env.state["screening"] = {"screening_results_aff.json": {"total_candidates": 9}}
        flow_env.state["deep_reports"] = {
            "lead": _report("only lead"),
            "counter": _report("counter body"),
            "audit": None,
        }
        result = _call(flow_env)
        assert not (flow_env.tmp_path / "grade_synthesis.md").exists()
        assert (flow_env.tmp_path / "affirmative_case.md").exists()
        assert result["aff_candidates"] == 9

    def test_errored_and_empty_sources_are_filtered_out(self, flow_env):
        flow_env.state["screening"] = {"screening_results_aff.json": {"total_candidates": 4}}
        flow_env.state["deep_reports"] = {
            "lead": _report(
                "body",
                4,
                [
                    _source("https://ok.example/1"),
                    _source("https://bad.example/2", error="timeout"),
                    _source("https://empty.example/3", summary=""),
                    _source("https://norel.example/4", summary="NO RELEVANT DATA"),
                    _source("", summary="has summary but no url"),
                ],
            ),
            "counter": _report("counter"),
            "audit": _report("audit"),
        }
        _call(flow_env)
        sources = json.loads((flow_env.tmp_path / "research_sources.json").read_text())
        assert [s["url"] for s in sources["lead"]] == ["https://ok.example/1"]

    def test_unknown_domain_falls_back_to_clinical(self, flow_env):
        """The returned research_domain echoes the request; only the search uses the fallback."""
        flow_env.state["screening"] = {"screening_results_aff.json": {"total_candidates": 8}}
        flow_env.state["deep_reports"] = {"lead": _report("a"), "counter": _report("b"), "audit": _report("c")}
        result = _call(flow_env, domain="astrology")
        assert result["research_domain"] == "astrology"
        assert result["aff_candidates"] == 8

    def test_unreadable_screening_json_leaves_count_at_zero(self, flow_env):
        flow_env.state["deep_reports"] = {"lead": _report("a"), "counter": _report("b"), "audit": _report("c")}

        async def _write_garbage(**kwargs):
            (flow_env.tmp_path / "screening_results_aff.json").write_text("{not json")
            return flow_env.state["deep_reports"]

        from dr2_podcast.research import clinical as _clinical

        _clinical.run_deep_research = _write_garbage
        with pytest.raises(InsufficientEvidenceError):
            _call(flow_env)
