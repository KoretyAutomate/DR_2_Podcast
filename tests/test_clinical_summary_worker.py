"""Tests for SummaryWorker — the page summarizer retargeted to the Smart model.

File under test: dr2_podcast/research/clinical.py

Why this file exists (2026-08-11): the fast-model removal left SummaryWorker
referencing `self.semaphore` after its constructor stopped creating one, so every
non-empty page raised AttributeError before reaching the exception handler and
summarize_batch() failed through asyncio.gather(). The full suite stayed green —
nothing exercised summarize() on a page with content. Codex caught it in review;
these tests are what should have.
"""

import asyncio
import pathlib
from types import SimpleNamespace

import pytest

from dr2_podcast import utils
from dr2_podcast.research import clinical
from dr2_podcast.research.clinical import FetchedPage, SummaryWorker


def _run(coro):
    return asyncio.run(coro)


def _page(content="Aspirin reduced events from 15% to 10% over 12 months.", error=None):
    return FetchedPage(url="https://example.org/p", title="A trial", content=content, word_count=9, error=error)


class _FakeCompletions:
    """Minimal stand-in for client.chat.completions with a recording call log."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=self.reply, reasoning=None))])


def _client(reply):
    completions = _FakeCompletions(reply)
    client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


_REPLY = (
    "FACTS:\n- Events fell from 15% to 10%\n\n"
    'METADATA:\n{"study_type":"RCT","sample_size":"n=500","publication_year":2020}'
)


class TestSummarize:
    def test_non_empty_page_produces_a_summary(self):
        """The regression: this raised AttributeError on self.semaphore."""
        client, completions = _client(_REPLY)
        worker = SummaryWorker(client, "test-model")
        result = _run(worker.summarize(_page(), goal="does aspirin help", query="aspirin"))
        assert result.error is None
        assert "Events fell from 15% to 10%" in result.summary
        assert len(completions.calls) == 1

    def test_metadata_is_parsed_off_the_reply(self):
        client, _ = _client(_REPLY)
        worker = SummaryWorker(client, "test-model")
        result = _run(worker.summarize(_page(), goal="g", query="q"))
        assert result.metadata is not None
        assert result.metadata.study_type == "RCT"

    def test_no_think_body_is_sent(self):
        """vLLM needs enable_thinking=False or the budget goes to reasoning and
        content comes back null — the 2026-06-13 blank-output bug."""
        client, completions = _client(_REPLY)
        worker = SummaryWorker(client, "test-model")
        _run(worker.summarize(_page(), goal="g", query="q"))
        assert completions.calls[0]["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}

    def test_empty_page_short_circuits_without_calling_the_model(self):
        client, completions = _client(_REPLY)
        worker = SummaryWorker(client, "test-model")
        result = _run(worker.summarize(_page(content="   "), goal="g", query="q"))
        assert result.summary == ""
        assert completions.calls == []

    def test_model_error_becomes_an_error_field_not_an_exception(self):
        class _Boom(_FakeCompletions):
            async def create(self, **kwargs):
                raise RuntimeError("vllm down")

        client = SimpleNamespace(chat=SimpleNamespace(completions=_Boom(None)))
        worker = SummaryWorker(client, "test-model")
        result = _run(worker.summarize(_page(), goal="g", query="q"))
        assert result.summary == ""
        assert "vllm down" in result.error


class TestBatch:
    def test_batch_summarizes_every_page(self):
        client, completions = _client(_REPLY)
        worker = SummaryWorker(client, "test-model")
        pages = [_page(), _page(), _page()]
        results = _run(worker.summarize_batch(pages, goal="g", query="q"))
        assert len(results) == 3
        assert all(r.error is None and r.summary for r in results)
        assert len(completions.calls) == 3


class TestVllmGate:
    def test_no_ungated_completion_call_exists_in_clinical(self):
        """The gate is only global if every call site goes through _gated_create.

        Codex found two escapes on 2026-08-11: deep extraction (plus its
        context-length retry) and the GRADE call each held their own budget, so a
        server sized for VLLM_MAX_CONCURRENCY could see more in flight. A grep is the
        only check that survives someone adding a sixth call site.
        """
        src = pathlib.Path(clinical.__file__).read_text()
        body = src.split("async def _gated_create", 1)[1].split("\n\n\n", 1)[1]
        assert ".chat.completions.create(" not in body, (
            "a direct chat.completions.create escaped the gate — route it through _gated_create()"
        )

    def test_gate_is_shared_across_callers_in_one_loop(self):
        """One gate per loop, not per caller — a per-call semaphore let both research
        tracks admit N each against a server serving VLLM_MAX_CONCURRENCY total."""

        async def two_lookups():
            return utils.vllm_gate(), utils.vllm_gate()

        a, b = _run(two_lookups())
        assert a is b

    def test_gate_is_rebuilt_for_a_new_loop(self):
        """asyncio.run() makes a fresh loop each time; a semaphore bound to a dead one
        must not be handed back."""
        first = _run(_gate())
        second = _run(_gate())
        assert first is not second

    def test_gate_capacity_matches_the_configured_server_limit(self):
        from dr2_podcast.config import VLLM_MAX_CONCURRENCY

        assert _run(_gate())._value == VLLM_MAX_CONCURRENCY


async def _gate():
    return utils.vllm_gate()


class TestAgentDepsRequiresWorker:
    def test_none_summary_worker_is_rejected_at_construction(self):
        """It used to be optional, guarded at every use. The guards went with the fast
        model; a None now surfaces as AttributeError inside a gather, or worse gets
        swallowed by _screen_abstracts' broad except and returns {}."""
        with pytest.raises(ValueError, match="summary_worker"):
            clinical.AgentDeps(
                smart_client=SimpleNamespace(),
                summary_worker=None,
                search_service=SimpleNamespace(),
                fetcher=SimpleNamespace(),
            )
