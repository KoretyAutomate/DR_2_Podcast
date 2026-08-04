"""Tests for the helpers extracted out of run_pipeline_flow.

run_pipeline_flow was 105 statements. The end-of-run document writers, the
deterministic accuracy gates and the checkpoint writer were lifted into named
functions; these test those units directly, which is what makes the extraction
safe without standing up a whole Prefect run.

There is also a structural test that every phase is still a Prefect task — an
earlier extraction in this module detached a @task decorator by inserting
helpers between it and its function, and nothing but a check like this notices.
"""

import json
import logging
from types import SimpleNamespace

import prefect
import pytest

from dr2_podcast import pipeline_flow as pf


LOG = logging.getLogger("test-flow-helpers")


# ---------------------------------------------------------------------------
# Structural: the Prefect contract
# ---------------------------------------------------------------------------

PHASE_TASKS = [
    "phase_0_framing",
    "phase_1_research",
    "phase_2_url_validation",
    "phase_3_translation",
    "phase_4_blueprint",
    "phase_5_script_draft",
    "phase_6_polish",
    "phase_7_audit",
    "phase_8_audio",
]


@pytest.mark.parametrize("name", PHASE_TASKS)
def test_phase_is_still_a_prefect_task(name):
    assert isinstance(getattr(pf, name), prefect.tasks.Task), f"{name} lost its @task decorator"


def test_run_pipeline_flow_is_still_a_flow():
    assert isinstance(pf.run_pipeline_flow, prefect.flows.Flow)


@pytest.mark.parametrize("name", PHASE_TASKS)
def test_every_phase_uses_the_shared_cache_key(name):
    """--resume depends on all phases keying off the same function."""
    assert getattr(pf, name).cache_key_fn is pf._phase_cache_key


@pytest.mark.parametrize("name", PHASE_TASKS)
def test_every_phase_still_takes_output_dir_by_name(name):
    """_phase_cache_key reads parameters['output_dir'] — the name is load-bearing."""
    params = getattr(pf, name).fn.__code__.co_varnames[: getattr(pf, name).fn.__code__.co_argcount]
    assert "output_dir" in params


# ---------------------------------------------------------------------------
# _augment_audit_for_corrector
# ---------------------------------------------------------------------------


class TestAugmentAuditForCorrector:
    def test_no_issues_returns_audit_unchanged(self):
        assert pf._augment_audit_for_corrector("AUDIT", [], []) == "AUDIT"

    def test_none_audit_becomes_empty_string(self):
        assert pf._augment_audit_for_corrector(None, [], []) == ""

    def test_citation_issues_are_appended_as_bullets(self):
        out = pf._augment_audit_for_corrector("AUDIT", ["fake citation A", "fake citation B"], [])
        assert out.startswith("AUDIT")
        assert "## Deterministic Citation Issues" in out
        assert "- fake citation A" in out and "- fake citation B" in out

    def test_grade_issues_carry_the_script_is_wrong_instruction(self):
        out = pf._augment_audit_for_corrector("AUDIT", [], ["NNT inverted"])
        assert "## Deterministic GRADE/NNT Contradictions" in out
        assert "the SCRIPT is wrong" in out
        assert "Do not raise the certainty level." in out
        assert "- NNT inverted" in out

    def test_both_kinds_appear_citations_first(self):
        out = pf._augment_audit_for_corrector("AUDIT", ["cit"], ["grade"])
        assert out.index("Citation Issues") < out.index("GRADE/NNT Contradictions")


# ---------------------------------------------------------------------------
# _deterministic_gate_issues
# ---------------------------------------------------------------------------


class TestDeterministicGateIssues:
    def test_empty_sot_skips_both_gates(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        called = []
        monkeypatch.setattr(pv, "validate_citations", lambda *a, **k: called.append("cit") or ["x"])
        monkeypatch.setattr(pv, "validate_grade_consistency", lambda *a, **k: called.append("grade") or ["y"])
        assert pf._deterministic_gate_issues("script", "", LOG) == ([], [])
        assert called == []

    def test_returns_both_issue_lists(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        monkeypatch.setattr(pv, "validate_citations", lambda *a, **k: ["cit-1"])
        monkeypatch.setattr(pv, "validate_grade_consistency", lambda *a, **k: ["grade-1", "grade-2"])
        cit, grade = pf._deterministic_gate_issues("script", "SOT", LOG)
        assert cit == ["cit-1"]
        assert grade == ["grade-1", "grade-2"]

    def test_a_raising_validator_degrades_to_empty(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        def boom(*a, **k):
            raise RuntimeError("validator broke")

        monkeypatch.setattr(pv, "validate_citations", boom)
        monkeypatch.setattr(pv, "validate_grade_consistency", lambda *a, **k: ["grade-1"])
        cit, grade = pf._deterministic_gate_issues("script", "SOT", LOG)
        assert cit == []
        assert grade == ["grade-1"], "one gate failing must not suppress the other"


# ---------------------------------------------------------------------------
# _warn_tts_readings
# ---------------------------------------------------------------------------


class TestWarnTtsReadings:
    def test_english_is_skipped_entirely(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        monkeypatch.setattr(pv, "validate_tts_readings", lambda t: pytest.fail("must not run for en"))
        pf._warn_tts_readings("script", "en", LOG)

    def test_japanese_warns_once_per_issue(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        monkeypatch.setattr(pv, "validate_tts_readings", lambda t: ["issue-1", "issue-2"])
        seen = []
        pf._warn_tts_readings("script", "ja", SimpleNamespace(warning=lambda *a: seen.append(a), debug=lambda *a: None))
        assert len(seen) == 2

    def test_validator_failure_is_non_blocking(self, monkeypatch):
        from dr2_podcast import pipeline_validators as pv

        def boom(t):
            raise RuntimeError("nope")

        monkeypatch.setattr(pv, "validate_tts_readings", boom)
        pf._warn_tts_readings("script", "ja", LOG)  # must not raise


# ---------------------------------------------------------------------------
# _write_accuracy_corrections_md / _write_prefect_checkpoint
# ---------------------------------------------------------------------------


@pytest.fixture
def patched_output_path(monkeypatch, tmp_path):
    from dr2_podcast import pipeline as _pipeline

    monkeypatch.setattr(_pipeline, "output_path", lambda run_dir, filename: tmp_path / filename)
    monkeypatch.setattr(_pipeline, "_audit_requires_correction", lambda text: "HIGH" in (text or ""))
    return tmp_path


class TestWriteAccuracyCorrectionsMd:
    def test_records_applied_when_a_correction_came_back(self, patched_output_path):
        pf._write_accuracy_corrections_md(patched_output_path, "HIGH severity", ["cit"], [], "CORRECTED", LOG)
        text = (patched_output_path / "ACCURACY_CORRECTIONS.md").read_text()
        assert "- Correction result: applied" in text
        assert "- Audit verdict trigger: True" in text
        assert "['cit']" in text
        assert "- GRADE/NNT contradiction trigger: none" in text

    def test_records_the_manual_review_warning_when_correction_failed(self, patched_output_path):
        pf._write_accuracy_corrections_md(patched_output_path, "HIGH severity", [], ["grade"], None, LOG)
        text = (patched_output_path / "ACCURACY_CORRECTIONS.md").read_text()
        assert "FAILED — audio uses UNCORRECTED script, manual review needed" in text

    def test_write_failure_is_swallowed(self, monkeypatch, tmp_path):
        from dr2_podcast import pipeline as _pipeline

        monkeypatch.setattr(_pipeline, "_audit_requires_correction", lambda t: False)
        monkeypatch.setattr(_pipeline, "output_path", lambda *a: tmp_path / "no-such-dir" / "x.md")
        pf._write_accuracy_corrections_md(tmp_path, "a", [], [], "c", LOG)  # must not raise


class TestWritePrefectCheckpoint:
    def test_writes_all_nine_phases(self, patched_output_path):
        pf._write_prefect_checkpoint(patched_output_path, "Coffee", "ja", LOG)
        ckpt = json.loads((patched_output_path / "checkpoint.json").read_text())
        assert ckpt["completed_phases"] == list(range(9))
        assert ckpt["topic"] == "Coffee"
        assert ckpt["language"] == "ja"
        assert ckpt["orchestrator"] == "prefect"
        assert ckpt["timestamp"]

    def test_write_failure_is_swallowed(self, monkeypatch, tmp_path):
        from dr2_podcast import pipeline as _pipeline

        monkeypatch.setattr(_pipeline, "output_path", lambda *a: tmp_path / "nope" / "checkpoint.json")
        pf._write_prefect_checkpoint(tmp_path, "t", "en", LOG)  # must not raise


# ---------------------------------------------------------------------------
# _save_run_documents / _generate_run_pdfs
# ---------------------------------------------------------------------------


def _docs(tmp_path, **overrides):
    base = {
        "output_dir": tmp_path,
        "topic_name": "Coffee",
        "language": "en",
        "language_config": {"code": "ja"},
        "session_roles": {},
        "framing_output": "FRAMING",
        "sot_content": "SOT",
        "translated_sot": "",
        "audit_task_ref": "AUDIT",
        "blueprint_task_ref": "BLUEPRINT",
        "script_task_ref": "SCRIPT",
    }
    base.update(overrides)
    return pf.RunDocuments(**base)


class TestSaveRunDocuments:
    @pytest.fixture
    def spy(self, monkeypatch):
        from dr2_podcast import pipeline as _pipeline

        calls = {"outputs": [], "pdfs": [], "metadata": []}
        monkeypatch.setattr(_pipeline, "_save_task_outputs", lambda d, items: calls["outputs"].append(items))
        monkeypatch.setattr(_pipeline, "create_pdf", lambda t, s, f: calls["pdfs"].append(f))
        monkeypatch.setattr(_pipeline, "_save_session_metadata", lambda **kw: calls["metadata"].append(kw))
        return calls

    def test_english_run_writes_three_pdfs(self, spy, tmp_path):
        pf._save_run_documents(_docs(tmp_path), LOG)
        assert spy["pdfs"] == ["research_framing.pdf", "source_of_truth.pdf", "accuracy_audit.pdf"]

    def test_translated_run_adds_the_translated_sot_pdf(self, spy, tmp_path):
        pf._save_run_documents(_docs(tmp_path, translated_sot="翻訳", language="ja"), LOG)
        assert "source_of_truth_ja.pdf" in spy["pdfs"]

    def test_markdown_outputs_and_metadata_are_written(self, spy, tmp_path):
        pf._save_run_documents(_docs(tmp_path), LOG)
        filenames = [item[2] for item in spy["outputs"][0]]
        assert filenames == [
            "research_framing.md",
            "accuracy_audit.md",
            "EPISODE_BLUEPRINT.md",
            "script_draft.md",
        ]
        assert spy["metadata"][0]["topic_name"] == "Coffee"

    def test_one_failing_pdf_does_not_stop_the_others(self, monkeypatch, tmp_path):
        from dr2_podcast import pipeline as _pipeline

        done = []

        def flaky(title, source, filename):
            if filename == "source_of_truth.pdf":
                raise RuntimeError("pdf engine died")
            done.append(filename)

        monkeypatch.setattr(_pipeline, "_save_task_outputs", lambda *a: None)
        monkeypatch.setattr(_pipeline, "_save_session_metadata", lambda **kw: None)
        monkeypatch.setattr(_pipeline, "create_pdf", flaky)
        pf._save_run_documents(_docs(tmp_path), LOG)
        assert done == ["research_framing.pdf", "accuracy_audit.pdf"]
