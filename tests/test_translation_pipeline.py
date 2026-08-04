"""Characterization tests for pipeline_translation._translate_sot_pipelined.

Written before splitting that 88-statement, complexity-28 function. The Smart
Model is replaced by a deterministic fake, so these pin both the reassembled
document and the exact set of chunks sent for translation.

Reassembly is the delicate part: the Discussion section is split at its
sub-headers and its sub-chunks must appear exactly once, in order, and nowhere
else. An existing test asserted that property against a reimplementation of the
logic; these drive the real function.
"""

from dr2_podcast.pipeline_translation import _translate_sot_pipelined


JA_CFG = {"name": "日本語", "code": "ja", "length_unit": "chars"}

SOT = """\
Preamble paragraph before any header, long enough to be translated.

## 1. Introduction
Introduction body text that is comfortably over the ten character floor.

## 2. Methods
Methods body text, also comfortably long enough to translate.

## 3. Results
Results body text with numbers 42% and NNT 10 that must survive.

## 4. Discussion
### 4.1 Affirmative Case
Affirmative discussion body, long enough.

### 4.2 Falsification Case
Falsification discussion body, long enough.

### 4.3 Tiny
x

## 5. References
1. Smith J. *A study*. Journal. (2023).
"""


class FakeTranslator:
    """Marks each body so the reassembly order is visible in the output."""

    def __init__(self, fail_on=None):
        self.calls = []
        self.fail_on = fail_on or set()

    def __call__(self, *, system, user, max_tokens, temperature):
        body = user.split("\n\n", 1)[1]
        self.calls.append({"body": body, "max_tokens": max_tokens, "temperature": temperature})
        if any(marker in body for marker in self.fail_on):
            raise RuntimeError("translator exploded")
        return f"[JA]{body}"


def _run(sot=SOT, **kw):
    fake = FakeTranslator(**kw)
    out = _translate_sot_pipelined(sot, "ja", JA_CFG, _call_smart_model=fake)
    return out, fake


class TestTranslateSotPipelined:
    def test_every_substantive_body_is_translated(self):
        out, fake = _run()
        bodies = [c["body"] for c in fake.calls]
        assert any("Introduction body" in b for b in bodies)
        assert any("Methods body" in b for b in bodies)
        assert any("Results body" in b for b in bodies)
        assert any("Affirmative discussion body" in b for b in bodies)
        assert any("Falsification discussion body" in b for b in bodies)

    def test_references_are_never_sent_for_translation(self):
        _, fake = _run()
        assert not any("Smith J" in c["body"] for c in fake.calls)

    def test_references_survive_verbatim_in_the_output(self):
        out, _ = _run()
        assert "1. Smith J. *A study*. Journal. (2023)." in out
        assert "[JA]1. Smith J" not in out

    def test_a_short_body_under_its_own_header_is_still_translated(self):
        """Passthrough needs BOTH a short body and no header — `### 4.3 Tiny`
        has a header, so its one-character body is still sent."""
        _, fake = _run()
        assert any(c["body"].strip() == "x" for c in fake.calls)

    def test_a_short_top_level_body_under_a_header_is_still_translated(self):
        """Same rule at the top level as in the Discussion sub-chunks."""
        sot = "## 2. Methods\nshort\n\n## 5. References\n1. Ref.\n"
        _, fake = _run(sot=sot)
        assert [c["body"].strip() for c in fake.calls] == ["short"]

    def test_a_headerless_short_body_is_passed_through(self):
        """With no header AND a body under ten characters, nothing is sent."""
        sot = "tiny\n\n## 5. References\n1. Ref.\n"
        out, fake = _run(sot=sot)
        assert fake.calls == []
        assert "tiny" in out

    def test_all_headers_appear_exactly_once(self):
        out, _ = _run()
        for header in (
            "## 1. Introduction",
            "## 2. Methods",
            "## 3. Results",
            "## 4. Discussion",
            "## 5. References",
            "### 4.1 Affirmative Case",
            "### 4.2 Falsification Case",
            "### 4.3 Tiny",
        ):
            assert out.count(header) == 1, f"{header} appears {out.count(header)} times"

    def test_discussion_subsections_appear_exactly_once(self):
        out, _ = _run()
        assert out.count("Affirmative discussion body") == 1
        assert out.count("Falsification discussion body") == 1

    def test_each_discussion_subheader_starts_its_own_line(self):
        """Ordering assertions alone would not notice the sub-parts being
        concatenated without a separator."""
        out, _ = _run()
        # Sub-part bodies already end in a newline, so the join contributes the
        # blank line between them. Assert that separator, not just "on a line".
        assert "\n\n### 4.1 Affirmative Case" in out
        assert "\n\n\n### 4.2 Falsification Case" in out
        assert "\n\n\n### 4.3 Tiny" in out

    def test_section_order_is_preserved(self):
        out, _ = _run()
        positions = [
            out.index("## 1. Introduction"),
            out.index("## 2. Methods"),
            out.index("## 3. Results"),
            out.index("## 4. Discussion"),
            out.index("## 5. References"),
        ]
        assert positions == sorted(positions)

    def test_discussion_subsections_stay_inside_the_discussion_section(self):
        out, _ = _run()
        assert out.index("## 4. Discussion") < out.index("### 4.1 Affirmative Case")
        assert out.index("### 4.1 Affirmative Case") < out.index("### 4.2 Falsification Case")
        assert out.index("### 4.2 Falsification Case") < out.index("## 5. References")

    def test_preamble_is_translated_and_comes_first(self):
        out, fake = _run()
        assert any("Preamble paragraph" in c["body"] for c in fake.calls)
        assert out.index("[JA]Preamble paragraph") < out.index("## 1. Introduction")

    def test_a_failed_chunk_keeps_its_original_text(self):
        out, fake = _run(fail_on={"Methods body"})
        assert "Methods body text, also comfortably long enough to translate." in out
        assert "[JA]Methods body" not in out
        # the rest still translated
        assert "[JA]Introduction body" in out

    def test_a_failed_discussion_chunk_keeps_its_original_text(self):
        out, _ = _run(fail_on={"Affirmative discussion body"})
        assert "Affirmative discussion body, long enough." in out
        assert "[JA]Affirmative discussion body" not in out
        assert "[JA]Falsification discussion body" in out

    def test_translation_uses_low_temperature(self):
        _, fake = _run()
        assert all(c["temperature"] == 0.1 for c in fake.calls)

    def test_max_tokens_scales_with_body_length(self):
        _, fake = _run()
        by_len = sorted(fake.calls, key=lambda c: len(c["body"]))
        assert by_len[0]["max_tokens"] <= by_len[-1]["max_tokens"]

    def test_document_with_nothing_translatable_is_returned_unchanged(self):
        sot = "## 5. References\n1. Only references here.\n"
        out, fake = _run(sot=sot)
        # Reassembly does not re-add the document's trailing newline.
        assert out == sot.rstrip("\n")
        assert fake.calls == []

    def test_empty_document_is_returned_unchanged(self):
        out, fake = _run(sot="")
        assert out == ""
        assert fake.calls == []
