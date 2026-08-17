"""Tests for the RedCircle publish sheet.

The property under test throughout is the one the sheet exists to guarantee:
every field is either read from the run or left visibly blank. A test that only
checked "the sheet was written" would pass on a sheet full of invented titles
and mtime-derived dates, which is the exact failure this module was built to
prevent — so the assertions below are mostly about what does NOT appear.
"""

import json
import struct
import wave

import pytest

from dr2_podcast.tools.publish_sheet import (
    BLANK,
    UNREADABLE,
    PublishSheet,
    PublishSheetError,
    build_publish_sheet,
    main,
    sheet_path,
    write_publish_sheet,
)


def _section(body, heading):
    """The content of one sheet section, bounded by the next heading.

    Splitting on a heading alone is not enough: everything after `### 出典`
    includes the tags, date and episode-number sections, which contain the
    blank marker themselves — so an unbounded `BLANK in ...` assertion passes
    against a sources section that emitted nothing at all.
    """
    after = body.split(heading, 1)[1]
    return after.split("\n## ", 1)[0].split("\n### ", 1)[0].strip()


def _sources_section(body):
    return _section(body, "### 出典")


def _write_wav(path, seconds=1.5, rate=24000):
    """A real, readable wav — duration must come from the header, not a guess."""
    path.parent.mkdir(parents=True, exist_ok=True)
    frames = int(seconds * rate)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(struct.pack("<h", 0) * frames)
    return path


@pytest.fixture
def edu_dir(tmp_path):
    """Educational layout: flat, Ep{NNN}_<topic>, no research/ directory."""
    d = tmp_path / "research_outputs" / "Ep007_認知バイアス_5つの罠と自問"
    _write_wav(d / "audio_mixed.wav", seconds=90)
    _write_wav(d / "audio.wav", seconds=89)
    (d / "script.txt").write_text("Speaker 1: こんにちは\n", encoding="utf-8")
    return d


@pytest.fixture
def edu_root(tmp_path):
    """Project root holding the episode briefs the educational titles come from."""
    brief = tmp_path / "educational_series" / "ep07_source_of_truth.md"
    brief.parent.mkdir(parents=True, exist_ok=True)
    brief.write_text(
        "# Educational Episode Brief — Episode 7: Cognitive Biases\n"
        "\n"
        "## Episode Metadata\n"
        "\n"
        "- **Series:** 科学リテラシー基礎シリーズ (Day 7 of 14)\n"
        "- **Episode Title:** 認知バイアス ― 直感が確率を裏切る5つの罠\n"
        "- **Target Audience:** 全世代\n",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def pipeline_dir(tmp_path):
    """Pipeline layout: research/, scripts/, audio/, meta/ subdirectories."""
    d = tmp_path / "research_outputs" / "2026-08-14_09-00-00"
    _write_wav(d / "audio" / "audio_mixed.wav", seconds=120)
    (d / "meta").mkdir(parents=True, exist_ok=True)
    (d / "meta" / "session_metadata.txt").write_text(
        "PODCAST SESSION METADATA\n"
        "============================================================\n"
        "\n"
        "Topic: 週2回の運動で十分という主張にどれだけ根拠があるのか？\n"
        "\n"
        "Language: 日本語 (Japanese) (ja)\n"
        "\n"
        "Role Assignments:\n"
        "  Host 1: Presenter (enthusiastic)\n",
        encoding="utf-8",
    )
    (d / "research").mkdir(parents=True, exist_ok=True)
    (d / "research" / "research_sources.json").write_text(
        json.dumps(
            {
                "lead": [
                    {
                        "index": 0,
                        "url": "https://pubmed.ncbi.nlm.nih.gov/21448086/",
                        "title": "Low-volume interval training",
                    },
                    {"index": 1, "url": "https://pubmed.ncbi.nlm.nih.gov/99999999/", "title": "Second lead study"},
                ],
                "counter": [
                    {"index": 0, "url": "https://pubmed.ncbi.nlm.nih.gov/12345678/", "title": "A falsifying trial"},
                ],
            }
        ),
        encoding="utf-8",
    )
    return d


# ---------------------------------------------------------------------------
# Layout detection — the two layouts differ, and each must be recognised
# ---------------------------------------------------------------------------


class TestLayoutBranch:
    def test_educational_folder_is_read_as_educational(self, edu_dir, edu_root):
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.layout == "educational"

    def test_timestamped_folder_is_read_as_pipeline(self, pipeline_dir, tmp_path):
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.layout == "pipeline"

    def test_educational_audio_is_found_flat(self, edu_dir, edu_root):
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.audio_path == (edu_dir / "audio_mixed.wav").resolve()

    def test_pipeline_audio_is_found_under_audio_subdir(self, pipeline_dir, tmp_path):
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.audio_path == (pipeline_dir / "audio" / "audio_mixed.wav").resolve()

    def test_an_empty_audio_subdir_does_not_hide_the_flat_wav(self, edu_dir, edu_root):
        """Ep014 has an audio/ dir holding only a backup — the wav is still flat."""
        (edu_dir / "audio").mkdir()
        (edu_dir / "audio" / "pre_rerender_backup").mkdir()
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.audio_path == (edu_dir / "audio_mixed.wav").resolve()

    def test_episode_number_comes_from_the_folder_name(self, edu_dir, edu_root):
        assert build_publish_sheet(edu_dir, project_root=edu_root).episode_number == "7"

    def test_pipeline_runs_have_no_episode_number_to_read(self, pipeline_dir, tmp_path):
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).episode_number == BLANK


# ---------------------------------------------------------------------------
# The load-bearing rule: blank, never invented
# ---------------------------------------------------------------------------


class TestNeverInvents:
    def test_educational_title_is_read_from_the_brief(self, edu_dir, edu_root):
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.title == "認知バイアス ― 直感が確率を裏切る5つの罠"

    def test_a_missing_brief_leaves_the_title_blank(self, edu_dir, tmp_path):
        """No brief on disk means no title exists — not that one may be guessed."""
        sheet = build_publish_sheet(edu_dir, project_root=tmp_path)
        assert sheet.title == BLANK

    def test_the_title_is_never_derived_from_the_folder_slug(self, edu_dir, tmp_path):
        """The slug names the source folder in the header; it must not reach the title."""
        rendered = build_publish_sheet(edu_dir, project_root=tmp_path).render()
        title_section = rendered.split("## 2. タイトル")[1].split("## 3.")[0]
        assert title_section.strip() == BLANK
        assert "認知バイアス" not in title_section

    def test_a_pipeline_topic_is_offered_as_reference_not_as_the_title(self, pipeline_dir, tmp_path):
        """The topic is a research question. Promoting it to a title would be a fabrication."""
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.title == BLANK
        assert sheet.title_reference == "週2回の運動で十分という主張にどれだけ根拠があるのか？"
        body = sheet.render()
        title_section = body.split("## 2. タイトル")[1].split("## 3.")[0]
        assert BLANK in title_section
        assert "参考" in title_section

    def test_description_is_always_blank_because_nothing_records_one(self, edu_dir, pipeline_dir, edu_root, tmp_path):
        assert build_publish_sheet(edu_dir, project_root=edu_root).description == BLANK
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).description == BLANK

    def test_tags_and_publish_date_are_blank_in_both_layouts(self, edu_dir, pipeline_dir, edu_root, tmp_path):
        for sheet in (
            build_publish_sheet(edu_dir, project_root=edu_root),
            build_publish_sheet(pipeline_dir, project_root=tmp_path),
        ):
            assert sheet.tags == BLANK
            assert sheet.publish_date == BLANK

    def test_the_publish_date_is_not_taken_from_the_audio_mtime(self, pipeline_dir, tmp_path):
        """An mtime is a render date. Printing it under 公開日 would be a wrong fact."""
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        date_section = sheet.render().split("## 5. 公開日")[1].split("## 6.")[0]
        assert date_section.strip() == BLANK
        assert "2026" not in date_section

    def test_a_topic_line_that_is_absent_leaves_no_reference(self, pipeline_dir, tmp_path):
        (pipeline_dir / "meta" / "session_metadata.txt").write_text("PODCAST SESSION METADATA\n", encoding="utf-8")
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.title_reference is None
        assert "参考" not in sheet.render()

    def test_the_reuse_header_variant_is_still_parsed(self, pipeline_dir, tmp_path):
        (pipeline_dir / "meta" / "session_metadata.txt").write_text(
            "PODCAST SESSION METADATA (REUSE: crew3_only)\n"
            "============================================================\n"
            "\n"
            "Topic: コーヒーと寿命\n"
            "Language: 日本語 (Japanese) (ja)\n"
            "Reused from: /somewhere\n",
            encoding="utf-8",
        )
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).title_reference == "コーヒーと寿命"


# ---------------------------------------------------------------------------
# Sources — the show notes of a research show must cite something
# ---------------------------------------------------------------------------


class TestSources:
    def test_both_tracks_are_collected(self, pipeline_dir, tmp_path):
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert len(sheet.sources) == 3
        assert ("A falsifying trial", "https://pubmed.ncbi.nlm.nih.gov/12345678/") in sheet.sources

    def test_sources_are_rendered_into_the_show_notes(self, pipeline_dir, tmp_path):
        body = build_publish_sheet(pipeline_dir, project_root=tmp_path).render()
        notes = body.split("### 出典")[1]
        assert "https://pubmed.ncbi.nlm.nih.gov/21448086/" in notes
        assert "Low-volume interval training" in notes

    def test_duplicate_urls_are_collapsed(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").write_text(
            json.dumps(
                {
                    "lead": [{"url": "https://x.test/1", "title": "A"}],
                    "counter": [{"url": "https://x.test/1", "title": "A again"}],
                }
            ),
            encoding="utf-8",
        )
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).sources == [("A", "https://x.test/1")]

    def test_an_entry_with_no_url_is_dropped_rather_than_cited_bare(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").write_text(
            json.dumps({"lead": [{"title": "Uncitable"}, {"url": "https://x.test/2", "title": "Citable"}]}),
            encoding="utf-8",
        )
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).sources == [("Citable", "https://x.test/2")]

    def test_an_entry_with_no_title_is_marked_not_invented(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").write_text(
            json.dumps({"lead": [{"url": "https://x.test/3"}]}), encoding="utf-8"
        )
        assert build_publish_sheet(pipeline_dir, project_root=tmp_path).sources == [
            ("(タイトル不明)", "https://x.test/3")
        ]

    def test_a_missing_sources_file_leaves_a_blank_not_an_empty_list(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").unlink()
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.sources == []
        assert _sources_section(sheet.render()) == BLANK

    def test_a_corrupt_sources_file_says_so_instead_of_showing_nothing(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").write_text("{not json", encoding="utf-8")
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert UNREADABLE in _sources_section(sheet.render())

    def test_a_sources_file_of_an_unknown_shape_refuses(self, pipeline_dir, tmp_path):
        (pipeline_dir / "research" / "research_sources.json").write_text('"just a string"', encoding="utf-8")
        with pytest.raises(PublishSheetError, match="unexpected shape"):
            build_publish_sheet(pipeline_dir, project_root=tmp_path)

    def test_educational_sheets_say_why_they_have_no_sources(self, edu_dir, edu_root):
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.sources == []
        assert "出典ファイルがありません" in sheet.render()


# ---------------------------------------------------------------------------
# Audio and duration — the one field that may not be blank
# ---------------------------------------------------------------------------


class TestAudioAndDuration:
    def test_duration_is_read_from_the_wav_header(self, pipeline_dir, tmp_path):
        sheet = build_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert sheet.duration_text.startswith("2分00秒")

    def test_the_audio_path_is_absolute(self, edu_dir, edu_root):
        assert build_publish_sheet(edu_dir, project_root=edu_root).audio_path.is_absolute()

    def test_a_missing_mixed_master_is_refused_not_substituted(self, edu_dir, edu_root):
        """audio.wav is the raw TTS. Falling back to it would ship an episode with no BGM."""
        (edu_dir / "audio_mixed.wav").unlink()
        with pytest.raises(PublishSheetError, match="audio_mixed.wav"):
            build_publish_sheet(edu_dir, project_root=edu_root)

    def test_the_refusal_names_what_was_actually_present(self, edu_dir, edu_root):
        (edu_dir / "audio_mixed.wav").unlink()
        with pytest.raises(PublishSheetError, match="audio.wav"):
            build_publish_sheet(edu_dir, project_root=edu_root)

    def test_a_corrupt_wav_reports_unreadable_rather_than_zero(self, edu_dir, edu_root):
        """A duration of 0分00秒 would read as a fact. It is not one."""
        (edu_dir / "audio_mixed.wav").write_bytes(b"not a wav at all")
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        assert sheet.duration_text == UNREADABLE

    def test_a_directory_that_is_not_a_run_is_refused(self, tmp_path):
        empty = tmp_path / "empty"
        empty.mkdir()
        with pytest.raises(PublishSheetError):
            build_publish_sheet(empty, project_root=tmp_path)

    def test_a_stray_folder_holding_audio_is_refused_not_called_a_pipeline_run(self, tmp_path):
        """Having a wav is not being a run. A backup folder must not get a sheet
        that labels it 'pipeline' — that describes it as something it is not."""
        stray = tmp_path / "pre_rerender_backup_20260813_231822"
        _write_wav(stray / "audio_mixed.wav", seconds=10)
        with pytest.raises(PublishSheetError, match="neither layout"):
            build_publish_sheet(stray, project_root=tmp_path)

    def test_a_timestamped_run_with_no_subdirs_is_still_accepted(self, tmp_path):
        """Legacy runs are flat but genuinely runs — the name carries the proof."""
        legacy = tmp_path / "2026-04-30_22-54-23"
        _write_wav(legacy / "audio_mixed.wav", seconds=10)
        assert build_publish_sheet(legacy, project_root=tmp_path).layout == "pipeline"

    def test_a_marker_subdir_is_enough_without_a_timestamp_name(self, tmp_path):
        odd = tmp_path / "renamed_run"
        _write_wav(odd / "audio" / "audio_mixed.wav", seconds=10)
        (odd / "meta").mkdir()
        assert build_publish_sheet(odd, project_root=tmp_path).layout == "pipeline"

    def test_a_nonexistent_directory_is_refused(self, tmp_path):
        with pytest.raises(PublishSheetError, match="not a directory"):
            build_publish_sheet(tmp_path / "nope", project_root=tmp_path)


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


class TestWriting:
    def test_pipeline_sheets_land_beside_the_other_metadata(self, pipeline_dir, tmp_path):
        target = write_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert target == pipeline_dir / "meta" / "publish_sheet.md"
        assert target.is_file()

    def test_educational_sheets_land_in_the_flat_folder(self, edu_dir, edu_root):
        target = write_publish_sheet(edu_dir, project_root=edu_root)
        assert target == edu_dir / "publish_sheet.md"
        assert target.is_file()

    def test_an_existing_sheet_is_not_overwritten(self, edu_dir, edu_root):
        """The blanks get filled in by hand — a re-render must not wipe that work."""
        target = sheet_path(edu_dir)
        target.write_text("タイトル: 人が書いた内容\n", encoding="utf-8")
        write_publish_sheet(edu_dir, project_root=edu_root)
        assert target.read_text(encoding="utf-8") == "タイトル: 人が書いた内容\n"

    def test_force_regenerates(self, edu_dir, edu_root):
        target = sheet_path(edu_dir)
        target.write_text("stale", encoding="utf-8")
        write_publish_sheet(edu_dir, force=True, project_root=edu_root)
        assert "公開シート" in target.read_text(encoding="utf-8")

    def test_the_sheet_is_written_in_redcircle_form_order(self, pipeline_dir, tmp_path):
        body = write_publish_sheet(pipeline_dir, project_root=tmp_path).read_text(encoding="utf-8")
        headings = [line for line in body.split("\n") if line.startswith("## ")]
        assert headings == [
            "## 1. 音声ファイル",
            "## 2. タイトル",
            "## 3. 説明文 / ショーノート",
            "## 4. タグ",
            "## 5. 公開日",
            "## 6. エピソード番号",
        ]

    def test_every_field_reaches_the_rendered_sheet(self, edu_dir, edu_root):
        """A field that is computed but never rendered is a silently missing field.

        Uses the educational fixture deliberately: there every field has a
        distinct real value, so each assertion can only be satisfied by that
        field's own content. On a pipeline run most fields are the same blank
        marker, and "the marker appears somewhere" proves nothing.
        """
        sheet = build_publish_sheet(edu_dir, project_root=edu_root)
        body = sheet.render()
        assert str(sheet.audio_path) in _section(body, "## 1. 音声ファイル")
        assert sheet.duration_text in _section(body, "## 1. 音声ファイル")
        assert _section(body, "## 2. タイトル") == sheet.title
        assert _section(body, "## 6. エピソード番号") == "7"

    def test_a_refused_run_writes_no_partial_sheet(self, edu_dir, edu_root):
        (edu_dir / "audio_mixed.wav").unlink()
        with pytest.raises(PublishSheetError):
            write_publish_sheet(edu_dir, project_root=edu_root)
        assert not sheet_path(edu_dir).exists()


class TestCli:
    def test_it_writes_sheets_for_the_directories_given(self, pipeline_dir, capsys):
        assert main([str(pipeline_dir)]) == 0
        assert (pipeline_dir / "meta" / "publish_sheet.md").is_file()
        assert "OK" in capsys.readouterr().out

    def test_a_bad_directory_is_reported_and_exits_nonzero(self, tmp_path, capsys):
        bad = tmp_path / "bad"
        bad.mkdir()
        assert main([str(bad)]) == 1
        assert "FAILED" in capsys.readouterr().err

    def test_one_failure_does_not_stop_the_others(self, pipeline_dir, tmp_path, capsys):
        bad = tmp_path / "bad"
        bad.mkdir()
        assert main([str(bad), str(pipeline_dir)]) == 1
        assert (pipeline_dir / "meta" / "publish_sheet.md").is_file()


class TestRenderDefaults:
    def test_a_sheet_built_with_no_readable_fields_is_all_blanks(self, tmp_path):
        """Constructed directly: the dataclass defaults must themselves be blanks."""
        sheet = PublishSheet(
            run_dir=tmp_path,
            layout="pipeline",
            audio_path=tmp_path / "audio_mixed.wav",
            duration_text=UNREADABLE,
        )
        body = sheet.render()
        for heading in ("## 2. タイトル", "## 4. タグ", "## 5. 公開日", "## 6. エピソード番号"):
            section = body.split(heading)[1].split("\n##")[0]
            assert BLANK in section


class TestReuseDoesNotInheritASheet:
    """A TTS-only reuse copies the previous run's meta/ wholesale. The sheet must
    not ride along: it names the previous episode's absolute wav path and that
    wav's duration, and because an existing sheet is never overwritten, the
    stale one would be the one that survived into the new run.
    """

    def test_the_copy_helper_excludes_the_sheet(self, tmp_path):
        import shutil

        from dr2_podcast.tools.publish_sheet import SHEET_FILENAME

        old_meta = tmp_path / "old" / "meta"
        old_meta.mkdir(parents=True)
        (old_meta / SHEET_FILENAME).write_text("previous episode's sheet", encoding="utf-8")
        (old_meta / "session_metadata.txt").write_text("Topic: X\n", encoding="utf-8")

        new_meta = tmp_path / "new" / "meta"
        shutil.copytree(old_meta, new_meta, dirs_exist_ok=True, ignore=shutil.ignore_patterns(SHEET_FILENAME))

        assert (new_meta / "session_metadata.txt").is_file(), "the rest of meta/ must still be copied"
        assert not (new_meta / SHEET_FILENAME).exists()

    def test_a_fresh_run_then_gets_its_own_sheet(self, pipeline_dir, tmp_path):
        """With no inherited sheet present, the new run writes one naming its own audio."""
        target = write_publish_sheet(pipeline_dir, project_root=tmp_path)
        assert str((pipeline_dir / "audio" / "audio_mixed.wav").resolve()) in target.read_text(encoding="utf-8")


# prepush codex 2026-08-14: url_validation removes URLs that failed a HEAD request into a SEPARATE
# file rather than editing the raw library, so reading the raw one copied dead links straight into
# the show notes a human pastes into RedCircle.
def test_the_sheet_reads_the_validated_library_when_there_is_one(tmp_path) -> None:
    import hashlib
    import json as _json

    from dr2_podcast.tools.publish_sheet import _read_sources

    research = tmp_path / "research"
    research.mkdir()
    raw = research / "research_sources.json"
    raw.write_text(_json.dumps({"lead": [{"url": "https://dead.example/x", "title": "broken"}]}))
    (research / "research_sources_validated.json").write_text(
        _json.dumps({"lead": [{"url": "https://ok.example/y", "title": "kept"}]})
    )
    (research / "research_sources_validated.sha256").write_text(
        hashlib.sha256(raw.read_bytes()).hexdigest()
    )

    sources, _note = _read_sources(tmp_path)
    assert all("dead.example" not in url for url, _title in sources), sources


def test_the_sheet_falls_back_to_the_raw_library(tmp_path) -> None:
    """A run that never validated still has show notes."""
    import json as _json

    from dr2_podcast.tools.publish_sheet import _read_sources

    research = tmp_path / "research"
    research.mkdir()
    (research / "research_sources.json").write_text(
        _json.dumps({"lead": [{"url": "https://ok.example/y", "title": "kept"}]})
    )
    sources, _note = _read_sources(tmp_path)
    assert sources
