"""Characterization tests for pipeline.py's phase 5/6 and translation helpers.

_run_sectional_draft, _run_polish_loop and _translate_and_inject_sot had no
tests. These were written before collapsing their parameter lists (10, 16 and
12 arguments) behind context objects.

Everything that reaches an LLM or CrewAI is stubbed, so what is under test is
the orchestration: which guard fires, what falls back to the draft, and what
gets injected into the Crew 3 task descriptions.
"""

from types import SimpleNamespace

import pytest

from dr2_podcast import pipeline as pl


EN_CFG = {
    "name": "English",
    "instruction": "Write in English.",
    "speech_rate": 150,
    "length_unit": "words",
    "prompt_unit": "word",
}

ROLES = {
    "presenter": {"label": "Host 1", "stance": "expert", "personality": "warm"},
    "questioner": {"label": "Host 2", "stance": "skeptic", "personality": "analytical"},
}


def _ctx(**overrides):
    base = {
        "language_config": EN_CFG,
        "session_roles": ROLES,
        "topic_name": "Topic",
        "target_instruction": "ti",
        "sot_content": "SOT",
        "channel_intro": "",
        "target_min": 30,
    }
    base.update(overrides)
    return pl.ScriptRunContext(**base)


def _task(description="base", raw=""):
    return SimpleNamespace(description=description, expected_output="eo", output=SimpleNamespace(raw=raw), context=[])


def _dialogue(turns=10, words=20):
    """Distinct lines per turn — _deduplicate_script is NOT stubbed, and it
    collapses a script whose turns are all identical."""
    return "\n".join(f"Host {1 + i % 2}: turn{i} " + " ".join(f"w{i}x{j}" for j in range(words)) for i in range(turns))


# ---------------------------------------------------------------------------
# _run_sectional_draft
# ---------------------------------------------------------------------------


class TestRunSectionalDraft:
    @pytest.fixture
    def stub_sections(self, monkeypatch):
        """Stub the pipeline_script collaborators _run_sectional_draft imports."""
        from dr2_podcast import pipeline_script as ps

        calls = []

        def fake_allocate(target, cfg, inventory):
            return [
                {"section_id": sid, "word_budget": 100, "length_unit": "words", "pacing": "p", "checklist_items": []}
                for sid in ("opening", "evidence", "synthesis", "closing")
            ]

        def fake_generate(section_config, previous_lines, deps):
            calls.append((section_config["section_id"], list(previous_lines), section_config["word_budget"]))
            text = f"Host 1: {section_config['section_id']} body.\nHost 2: ok."
            return text, section_config["word_budget"], 0

        monkeypatch.setattr(ps, "_allocate_section_budgets", fake_allocate)
        monkeypatch.setattr(ps, "_generate_section", fake_generate)
        return calls

    def test_all_four_sections_are_generated_and_joined(self, stub_sections):
        text, count = pl._run_sectional_draft(
            {"opening": []}, _ctx(target_length_int=400), _call_smart_model=lambda **kw: ""
        )
        assert [c[0] for c in stub_sections] == ["opening", "evidence", "synthesis", "closing"]
        for sid in ("opening", "evidence", "synthesis", "closing"):
            assert f"{sid} body." in text
        assert count > 0

    def test_first_boundary_is_intro_end_and_the_rest_are_transitions(self, stub_sections):
        text, _ = pl._run_sectional_draft({}, _ctx(target_length_int=400), _call_smart_model=lambda **kw: "")
        assert text.count("[INTRO_END]") == 1
        assert text.count("[TRANSITION]") == 2
        assert text.index("[INTRO_END]") < text.index("[TRANSITION]")

    def test_each_section_sees_the_previous_section_tail_as_lead_in(self, stub_sections):
        pl._run_sectional_draft({}, _ctx(target_length_int=400), _call_smart_model=lambda **kw: "")
        assert stub_sections[0][1] == [], "the first section has no lead-in"
        assert stub_sections[1][1], "later sections receive the previous tail"
        assert "opening body." in "\n".join(stub_sections[1][1])


# ---------------------------------------------------------------------------
# _run_polish_loop
# ---------------------------------------------------------------------------


@pytest.fixture
def polish_env(monkeypatch):
    """Stub CrewAI kickoff and script validation for the polish loop."""
    state = {
        "kickoffs": 0,
        "polish_output": _dialogue(10, 20),
        "validations": [],  # queue of pass/fail
        "condense_calls": [],
    }

    class FakeCrew:
        def __init__(self, agents=None, tasks=None, verbose=False):
            self.tasks = tasks

        def kickoff(self):
            state["kickoffs"] += 1
            self.tasks[0].output = SimpleNamespace(raw=state["polish_output"])

    def fake_validate(text, target, tol, cfg, sot, stage="draft"):
        verdict = state["validations"].pop(0) if state["validations"] else True
        return {
            "pass": verdict,
            "word_count": len(text.split()),
            "issues": [] if verdict else ["TOO SHORT"],
            "feedback": "" if verdict else "make it longer",
        }

    def fake_condense(script_text, inventory, target_length, ctx):
        state["condense_calls"].append(target_length)
        return script_text + "\nHost 1: condensed addition."

    def FakeTask(description=None, expected_output=None, agent=None, context=None):
        return SimpleNamespace(
            description=description,
            expected_output=expected_output,
            agent=agent,
            context=list(context or []),
            output=SimpleNamespace(raw=""),
        )

    monkeypatch.setattr(pl, "Crew", FakeCrew)
    monkeypatch.setattr(pl, "Task", FakeTask)
    monkeypatch.setattr(pl, "_validate_script", fake_validate)
    monkeypatch.setattr(pl, "_run_condense_pass", fake_condense)
    return state


def _polish(draft_text, draft_count, polish_env, *, inventory=None, translation_task=None, max_attempts=3):
    return pl._run_polish_loop(
        draft_text,
        draft_count,
        inventory if inventory is not None else {"evidence": [{"q": "a"}]},
        _ctx(target_length_int=1000),
        pl.Crew3Refs(
            script_task=_task("script"),
            polish_task=_task("polish base"),
            translation_task=translation_task,
            editor_agent=SimpleNamespace(role="editor"),
            polish_base_desc="polish base desc",
            polish_expected="polish expected",
        ),
        max_attempts,
    )


class TestRunPolishLoop:
    def test_over_target_draft_triggers_a_condense_pass(self, polish_env):
        _polish(_dialogue(60, 20), 2000, polish_env)
        assert polish_env["condense_calls"] == [1000]

    def test_on_target_draft_does_not_condense(self, polish_env):
        _polish(_dialogue(10, 20), 1000, polish_env)
        assert polish_env["condense_calls"] == []

    def test_condense_is_skipped_when_there_is_no_inventory(self, polish_env):
        """Both conditions are required — over target AND an inventory."""
        _polish(_dialogue(60, 20), 2000, polish_env, inventory={})
        assert polish_env["condense_calls"] == []

    def test_passing_polish_runs_exactly_one_kickoff(self, polish_env):
        _polish(_dialogue(10, 20), 1000, polish_env)
        assert polish_env["kickoffs"] == 1

    def test_failing_polish_retries_up_to_max_attempts(self, polish_env):
        polish_env["validations"] = [False, False, False]
        _polish(_dialogue(10, 20), 1000, polish_env, max_attempts=3)
        assert polish_env["kickoffs"] == 3

    def test_polish_stops_as_soon_as_it_passes(self, polish_env):
        polish_env["validations"] = [False, True]
        _polish(_dialogue(10, 20), 1000, polish_env, max_attempts=3)
        assert polish_env["kickoffs"] == 2

    def test_shrunk_polish_falls_back_to_the_draft(self, polish_env):
        """On-target draft_count so the condense pass does not fire — the
        fallback target is then the draft exactly as passed in."""
        draft = _dialogue(40, 30)
        # Keeps enough speaker labels to clear the LABEL guard (25 of 40 > 50%),
        # so the SHRINKAGE guard is the only one that can fire here.
        polish_env["polish_output"] = "\n".join(f"Host {1 + i % 2}: t{i}" for i in range(25))
        polished, _ = _polish(draft, 1000, polish_env)
        assert polished == draft
        assert "t24" not in polished
        assert polish_env["condense_calls"] == []

    def test_polish_that_strips_speaker_labels_falls_back_to_the_draft(self, polish_env):
        """draft_count is on target so the condense pass does not fire and the
        fallback target is the draft exactly as passed in."""
        draft = _dialogue(20, 60)
        # Long enough to clear the shrinkage guard, but with no Host labels.
        polish_env["polish_output"] = " ".join(f"w{i}" for i in range(1200))
        polished, _ = _polish(draft, 1000, polish_env)
        assert polished == draft
        assert polish_env["condense_calls"] == []

    def test_translation_task_is_added_to_the_polish_context(self, polish_env):
        tt = _task("translation")
        _, final_task = _polish(_dialogue(10, 20), 1000, polish_env, translation_task=tt)
        assert tt in final_task.context

    def test_no_translation_task_leaves_a_single_context_entry(self, polish_env):
        _, final_task = _polish(_dialogue(10, 20), 1000, polish_env, translation_task=None)
        assert len(final_task.context) == 1


# ---------------------------------------------------------------------------
# _translate_and_inject_sot
# ---------------------------------------------------------------------------


@pytest.fixture
def translate_env(monkeypatch, tmp_path):
    state = {"translated": "翻訳された本文", "summary": "要約"}

    monkeypatch.setattr(pl, "_translate_sot_pipelined", lambda sot, lang, cfg: state["translated"])
    monkeypatch.setattr(pl, "summarize_report", lambda text, role, topic: state["summary"])
    monkeypatch.setattr(pl, "output_path", lambda run_dir, filename: tmp_path / filename)
    return state, tmp_path


def _translate(translate_env, tasks):
    _, tmp_path = translate_env
    return pl._translate_and_inject_sot(
        _ctx(sot_content="SOT BODY", language="ja", output_dir=tmp_path),
        tmp_path / "source_of_truth.md",
        "sot summary",
        "GRADE numbers",
        pl.Crew3Refs(
            script_task=tasks["script"],
            audit_task=tasks["audit"],
            blueprint_task=tasks["blueprint"],
            translation_task=tasks["translation"],
        ),
    )


@pytest.fixture
def crew_tasks():
    return {k: _task(f"{k} base") for k in ("blueprint", "script", "audit", "translation")}


class TestTranslateAndInjectSot:
    def test_translated_sot_is_written_to_disk(self, translate_env, crew_tasks):
        _, tmp_path = translate_env
        translated, path, summary = _translate(translate_env, crew_tasks)
        assert translated == "翻訳された本文"
        assert path == tmp_path / "source_of_truth_ja.md"
        assert path.read_text(encoding="utf-8") == "翻訳された本文"
        assert summary == "要約"

    def test_injection_reaches_blueprint_script_and_audit(self, translate_env, crew_tasks):
        _translate(translate_env, crew_tasks)
        for key in ("blueprint", "script", "audit"):
            assert crew_tasks[key].description != f"{key} base", f"{key} task was not injected"
            assert "SOURCE OF TRUTH SUMMARY" in crew_tasks[key].description

    def test_translation_task_gets_a_compact_reference_not_the_full_sot(self, translate_env, crew_tasks):
        """The full SOT as context caused a 36-cycle CrewAI summarizer loop."""
        _translate(translate_env, crew_tasks)
        raw = crew_tasks["translation"].output.raw
        assert "Translation complete" in raw
        assert "source_of_truth_ja.md" in raw
        assert "翻訳された本文" not in raw

    def test_no_translation_writes_nothing_and_injects_nothing(self, translate_env, crew_tasks):
        state, tmp_path = translate_env
        state["translated"] = ""
        translated, path, summary = _translate(translate_env, crew_tasks)
        assert translated == ""
        assert path is None
        assert summary == ""
        assert not (tmp_path / "source_of_truth_ja.md").exists()
        assert crew_tasks["script"].description == "script base"

    def test_empty_summary_skips_injection_but_still_saves_the_file(self, translate_env, crew_tasks):
        state, tmp_path = translate_env
        state["summary"] = ""
        translated, path, summary = _translate(translate_env, crew_tasks)
        assert path.exists()
        assert summary == ""
        assert crew_tasks["script"].description == "script base"
