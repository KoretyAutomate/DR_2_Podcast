"""Test M9: research_outputs/ per-phase subdirectories.

Tests that:
1. output_path() maps known filenames to correct subdirectories
2. output_path() falls back to root for unknown filenames
3. output_path() falls back to flat layout for legacy directories (no subdirs)
4. create_timestamped_output_dir() creates the subdirectories
5. _find_artifact() finds files in both flat and subdirectory layouts
"""
import os
import tempfile
from pathlib import Path

from dr2_podcast.pipeline import (
    output_path, OUTPUT_SUBDIRS, _FILE_SUBDIR_MAP,
    create_timestamped_output_dir, _find_artifact,
)


def test_output_path_known_files():
    """Known filenames map to their designated subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        for subdir in OUTPUT_SUBDIRS:
            (run_dir / subdir).mkdir()

        # Research files
        assert output_path(run_dir, "source_of_truth.md") == run_dir / "research" / "source_of_truth.md"
        assert output_path(run_dir, "research_sources.json") == run_dir / "research" / "research_sources.json"
        assert output_path(run_dir, "grade_synthesis.md") == run_dir / "research" / "grade_synthesis.md"
        assert output_path(run_dir, "affirmative_case.md") == run_dir / "research" / "affirmative_case.md"
        assert output_path(run_dir, "domain_classification.json") == run_dir / "research" / "domain_classification.json"

        # Script files
        assert output_path(run_dir, "script_draft.md") == run_dir / "scripts" / "script_draft.md"
        assert output_path(run_dir, "script_final.md") == run_dir / "scripts" / "script_final.md"
        assert output_path(run_dir, "script.txt") == run_dir / "scripts" / "script.txt"

        # Audio files
        assert output_path(run_dir, "audio.wav") == run_dir / "audio" / "audio.wav"
        assert output_path(run_dir, "audio_mixed.wav") == run_dir / "audio" / "audio_mixed.wav"

        # Meta files
        assert output_path(run_dir, "session_metadata.txt") == run_dir / "meta" / "session_metadata.txt"
        assert output_path(run_dir, "podcast_generation.log") == run_dir / "meta" / "podcast_generation.log"
        assert output_path(run_dir, "checkpoint.json") == run_dir / "meta" / "checkpoint.json"

    print("PASS: Known files map to correct subdirectories")


def test_output_path_dynamic_files():
    """Dynamic filenames (source_of_truth_ja.md, podcast_*.wav) resolve correctly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        for subdir in OUTPUT_SUBDIRS:
            (run_dir / subdir).mkdir()

        # Translated SOT
        assert output_path(run_dir, "source_of_truth_ja.md") == run_dir / "research" / "source_of_truth_ja.md"
        assert output_path(run_dir, "source_of_truth_ko.md") == run_dir / "research" / "source_of_truth_ko.md"

        # Translated SOT PDF
        assert output_path(run_dir, "source_of_truth_ja.pdf") == run_dir / "meta" / "source_of_truth_ja.pdf"

        # Podcast audio files
        assert output_path(run_dir, "podcast_mixed.wav") == run_dir / "audio" / "podcast_mixed.wav"
        assert output_path(run_dir, "podcast_final.mp3") == run_dir / "audio" / "podcast_final.mp3"

    print("PASS: Dynamic filenames resolve correctly")


def test_output_path_unknown_files():
    """Unknown filenames fall back to run_dir root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        for subdir in OUTPUT_SUBDIRS:
            (run_dir / subdir).mkdir()

        assert output_path(run_dir, "random_file.xyz") == run_dir / "random_file.xyz"

    print("PASS: Unknown files fall back to root")


def test_output_path_legacy_fallback():
    """When subdirectories don't exist (legacy run), paths fall back to flat layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "legacy_run"
        run_dir.mkdir()
        # No subdirectories created

        assert output_path(run_dir, "source_of_truth.md") == run_dir / "source_of_truth.md"
        assert output_path(run_dir, "script_final.md") == run_dir / "script_final.md"
        assert output_path(run_dir, "audio.wav") == run_dir / "audio.wav"
        assert output_path(run_dir, "session_metadata.txt") == run_dir / "session_metadata.txt"

    print("PASS: Legacy (flat) directories fall back correctly")


def test_create_timestamped_output_dir():
    """create_timestamped_output_dir creates all 4 subdirectories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        base_dir = Path(tmpdir) / "research_outputs"
        base_dir.mkdir()
        run_dir = create_timestamped_output_dir(base_dir)

        for subdir in OUTPUT_SUBDIRS:
            sub_path = run_dir / subdir
            assert sub_path.is_dir(), f"Expected {sub_path} to be a directory"

    print("PASS: create_timestamped_output_dir creates subdirectories")


def test_find_artifact_subdirectory():
    """_find_artifact finds files in subdirectory layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        for subdir in OUTPUT_SUBDIRS:
            (run_dir / subdir).mkdir()

        # Write a file in the research subdirectory
        test_file = run_dir / "research" / "source_of_truth.md"
        test_file.write_text("test content")

        found = _find_artifact(run_dir, "source_of_truth.md")
        assert found.exists(), f"Expected to find {test_file}"
        assert found == test_file

    print("PASS: _find_artifact finds files in subdirectory layout")


def test_find_artifact_flat():
    """_find_artifact finds files in flat layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        run_dir = Path(tmpdir) / "test_run"
        run_dir.mkdir()
        # No subdirectories

        test_file = run_dir / "source_of_truth.md"
        test_file.write_text("test content")

        found = _find_artifact(run_dir, "source_of_truth.md")
        assert found.exists(), f"Expected to find {test_file}"
        assert found == test_file

    print("PASS: _find_artifact finds files in flat layout")


if __name__ == "__main__":
    test_output_path_known_files()
    test_output_path_dynamic_files()
    test_output_path_unknown_files()
    test_output_path_legacy_fallback()
    test_create_timestamped_output_dir()
    test_find_artifact_subdirectory()
    test_find_artifact_flat()
    print()
    print("All M9 tests PASSED")
