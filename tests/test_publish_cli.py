"""The four commands, end to end, against a synthetic run directory.

These run the real `add` → `stage` → `release` → `sync` path: a real WAV is
really encoded, the real manifest code writes a real file, and only R2 is a
stand-in. What is worth pinning is the ordering the whole design rests on —
audio is uploaded while the episode is still a draft, so the preview feed can
carry it before the public feed can — and the refusals that stop a placeholder
reaching a listener.
"""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path

import pytest
from PIL import Image

from dr2_podcast.publish import cli
from dr2_podcast.publish.manifest import Manifest, Show, load_manifest, save_manifest
from dr2_podcast.publish.storage import R2Config, R2Storage, StorageError
from tests.test_publish_storage import ENV, FakeClient


@pytest.fixture
def workspace(tmp_path, monkeypatch):
    """A manifest, a build tree, one run directory, and a fake bucket."""
    build = tmp_path / "build"
    monkeypatch.setattr(cli, "BUILD_DIR", build)

    manifest_path = tmp_path / "episodes.json"
    save_manifest(
        Manifest(
            show=Show(
                title="テスト番組",
                description="番組説明",
                author="作者",
                owner_name="作者",
                owner_email="owner@example.com",
                subcategory="Life Sciences",
                base_url="https://media.test.example",
            )
        ),
        manifest_path,
    )

    Image.new("RGB", (1500, 1500), (10, 80, 120)).save(tmp_path / "logo.png")
    cli.artwork_mod.build_artwork(tmp_path / "logo.png", build / cli.ARTWORK_KEY)

    client = FakeClient()
    monkeypatch.setattr(cli, "_storage_for", lambda manifest: R2Storage(R2Config.from_env(ENV), client=client))

    return {
        "root": tmp_path,
        "manifest": manifest_path,
        "build": build,
        "client": client,
        "run_dir": _make_run_dir(tmp_path / "run_2026-08-20_06-00-00", seconds=2.0),
    }


def _make_run_dir(path, *, seconds: float = 2.0, topic: str = "テストのトピック"):
    """A pipeline-layout run: meta/, and the BGM-mixed master under audio/."""
    (path / "meta").mkdir(parents=True)
    (path / "audio").mkdir(parents=True)
    (path / "meta" / "session_metadata.txt").write_text(f"Topic: {topic}\n", encoding="utf-8")

    rate = 44100
    with wave.open(str(path / "audio" / "audio_mixed.wav"), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(
                struct.pack("<h", int(9000 * math.sin(2 * math.pi * 330 * t / rate)))
                for t in range(int(rate * seconds))
            )
        )
    return path


def _run(workspace, *argv) -> int:
    return cli.main(["--manifest", str(workspace["manifest"]), *argv])


def _manifest(workspace) -> Manifest:
    return load_manifest(workspace["manifest"])


# --- add ---------------------------------------------------------------------


def test_add_registers_a_run_and_mints_one_guid(workspace):
    assert _run(workspace, "add", str(workspace["run_dir"]), "--season", "2", "--episode", "7") == 0
    episodes = _manifest(workspace).episodes
    assert len(episodes) == 1
    assert episodes[0].audio_key == "audio/s2e007.mp3"
    assert episodes[0].state == "draft"
    assert episodes[0].guid


def test_add_refuses_the_same_run_twice(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    assert _run(workspace, "add", str(workspace["run_dir"]), "--episode", "2") == 1
    assert "already in the manifest" in capsys.readouterr().err
    assert len(_manifest(workspace).episodes) == 1


def test_a_publish_sheet_sentinel_never_becomes_a_title(workspace):
    """A pipeline run has no human title on disk, so publish_sheet returns
    `(要記入)`. That string in a live feed IS the episode's name in every app."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    episode = _manifest(workspace).episodes[0]
    assert episode.title == ""
    assert "要記入" not in json.dumps(_manifest(workspace).to_json(), ensure_ascii=False)


def test_add_refuses_to_guess_a_missing_episode_number(workspace, capsys):
    assert _run(workspace, "add", str(workspace["run_dir"])) == 1
    assert "--episode" in capsys.readouterr().err


def test_add_refuses_a_directory_with_no_mixed_master(workspace, capsys):
    """The published artifact is the BGM-mixed master; raw TTS is not a substitute."""
    bare = workspace["root"] / "empty_run"
    (bare / "meta").mkdir(parents=True)
    assert _run(workspace, "add", str(bare), "--episode", "1") == 1
    assert "audio_mixed.wav" in capsys.readouterr().err


def test_a_run_added_by_relative_path_still_stages_from_another_directory(workspace, monkeypatch):
    """The manifest outlives the shell that registered the run.

    `add` today, `stage` next week from wherever the terminal happens to be:
    storing the relative string as typed makes the second command look for the
    run under the wrong parent and fail on a run that registered fine.
    """
    monkeypatch.chdir(workspace["root"])
    assert _run(workspace, "add", workspace["run_dir"].name, "--episode", "1") == 0

    stored = _manifest(workspace).episodes[0].run_dir
    assert Path(stored).is_absolute()

    elsewhere = workspace["root"] / "somewhere_else"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert _run(workspace, "stage", "--all-draft") == 0
    assert _manifest(workspace).episodes[0].is_staged


def test_add_refuses_the_same_run_spelled_a_different_way(workspace, monkeypatch, capsys):
    """One run, one GUID — including when the second `add` types a relative path."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    monkeypatch.chdir(workspace["root"])
    assert _run(workspace, "add", workspace["run_dir"].name, "--episode", "2") == 1
    assert "already in the manifest" in capsys.readouterr().err
    assert len(_manifest(workspace).episodes) == 1


# --- stage --------------------------------------------------------------------


def test_stage_encodes_uploads_and_writes_back_real_numbers(workspace):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    assert _run(workspace, "stage", "--all-draft") == 0

    episode = _manifest(workspace).episodes[0]
    mp3 = workspace["build"] / episode.audio_key
    assert episode.bytes == mp3.stat().st_size
    assert episode.duration_seconds == 2
    assert workspace["client"].objects[episode.audio_key] == mp3.read_bytes()
    assert cli.ARTWORK_KEY in workspace["client"].objects


def test_stage_leaves_the_episode_a_draft(workspace):
    """The ordering the preview feed depends on: uploaded, still unpublished."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    assert _manifest(workspace).episodes[0].state == "draft"


def test_stage_is_idempotent_and_does_not_re_encode(workspace, monkeypatch):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    first = _manifest(workspace).episodes[0]
    monkeypatch.setattr(cli.encode_mod, "encode_mp3", _refuse_to_encode)

    assert _run(workspace, "stage", first.guid) == 0
    assert _manifest(workspace).episodes[0].bytes == first.bytes
    assert _manifest(workspace).episodes[0].duration_seconds == first.duration_seconds


def test_restaging_refreshes_the_tags_of_an_mp3_it_reuses(workspace, monkeypatch):
    """The title is usually empty at `add` — a pipeline run carries none on disk.

    So the normal sequence is stage, write the title into the manifest, stage
    again. If the second stage returned the existing file untouched, the
    enclosure every listener downloads would still be tagged `s2e001`, with no
    error and nothing on screen to say so — and correcting it would mean knowing
    to pass `--force-encode` and waiting out a full re-encode.
    """
    from mutagen.id3 import ID3

    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    mp3 = workspace["build"] / "audio/s2e001.mp3"
    assert str(ID3(str(mp3))["TIT2"].text[0]) == "s2e001"

    _fill_notes(workspace)
    monkeypatch.setattr(cli.encode_mod, "encode_mp3", _refuse_to_encode)
    assert _run(workspace, "stage", "--all-draft") == 0

    assert str(ID3(str(mp3))["TIT2"].text[0]) == "第1回 テスト"
    episode = _manifest(workspace).episodes[0]
    assert episode.bytes == mp3.stat().st_size, "<enclosure length> must be the retagged size"
    assert workspace["client"].objects[episode.audio_key] == mp3.read_bytes()


def _refuse_to_encode(*args, **kwargs):
    raise AssertionError("re-encoded an MP3 that was already in the build tree")


def test_stage_can_run_without_credentials(workspace):
    """Everything before the upload is buildable offline while .env is pending."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    assert _run(workspace, "stage", "--all-draft", "--no-upload") == 0
    assert workspace["client"].objects == {}
    assert _manifest(workspace).episodes[0].is_staged


# --- release --------------------------------------------------------------------


def test_release_refuses_an_episode_with_no_title_or_notes(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    guid = _manifest(workspace).episodes[0].guid

    assert _run(workspace, "release", guid) == 1
    assert "title" in capsys.readouterr().err
    assert _manifest(workspace).episodes[0].state == "draft"


def test_release_refuses_an_episode_that_was_never_staged(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _fill_notes(workspace)
    assert _run(workspace, "release", _manifest(workspace).episodes[0].guid) == 1
    assert "not staged" in capsys.readouterr().err


def test_release_publishes_once_the_metadata_is_there(workspace):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    guid = _manifest(workspace).episodes[0].guid

    assert _run(workspace, "release", guid) == 0
    assert _manifest(workspace).episodes[0].state == "published"


# --- sync -----------------------------------------------------------------------


def test_dry_run_prints_a_feed_and_uploads_nothing(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft", "--no-upload")
    capsys.readouterr()  # drop what add/stage printed
    assert _run(workspace, "sync", "--dry-run") == 0

    out = capsys.readouterr().out
    assert out.startswith('<?xml version="1.0" encoding="UTF-8"?>')
    assert workspace["client"].objects == {}


def test_sync_writes_both_feeds_with_the_right_content_types(workspace):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    assert _run(workspace, "sync") == 0

    token = _manifest(workspace).show.preview_token
    objects = workspace["client"].objects
    assert cli.FEED_KEY in objects
    assert f"preview-{token}.xml" in objects
    feed_puts = [put for put in workspace["client"].puts if put["key"].endswith(".xml")]
    assert {put["content_type"] for put in feed_puts} == {cli.CONTENT_TYPE_RSS}


def test_a_staged_draft_is_listenable_in_preview_while_the_public_feed_is_empty(workspace):
    """The two facts that prove draft and published are genuinely separate."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    assert _run(workspace, "sync") == 0

    objects = workspace["client"].objects
    token = _manifest(workspace).show.preview_token
    assert b"<item>" not in objects[cli.FEED_KEY]
    assert b"<item>" in objects[f"preview-{token}.xml"]
    assert "【試聴】".encode() in objects[f"preview-{token}.xml"]


def test_sync_refuses_when_the_enclosure_is_not_in_the_bucket(workspace, capsys):
    """A listener would see a new episode that 404s when they tap it."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    workspace["client"].objects.pop("audio/s2e001.mp3")

    assert _run(workspace, "sync") == 1
    assert "not in the bucket" in capsys.readouterr().err


def test_sync_refuses_a_preview_enclosure_that_was_never_uploaded(workspace, capsys):
    """`stage --no-upload` is a local-only operation, but it still writes real
    bytes and duration back into the manifest — which is all it takes for the
    draft to appear in the preview feed. Uploading that feed would hand out a
    private URL whose one episode 404s, and the public feed is empty here, so
    guarding only the public items catches nothing.
    """
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft", "--no-upload")

    assert _run(workspace, "sync") == 1
    assert "not in the bucket" in capsys.readouterr().err
    assert workspace["client"].objects == {}, "no feed may be uploaded once a guard has failed"


def test_sync_checks_an_episode_in_both_feeds_only_once(workspace):
    """A released episode is in the public feed AND the preview feed, at the same
    size. Guarding both must not double the HEAD requests against the bucket."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    workspace["client"].heads.clear()

    assert _run(workspace, "sync") == 0
    assert workspace["client"].heads.count("audio/s2e001.mp3") == 1


def test_sync_refuses_when_the_bucket_object_is_a_different_size(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    workspace["client"].objects["audio/s2e001.mp3"] = b"truncated"

    assert _run(workspace, "sync") == 1
    assert "stalls downloads" in capsys.readouterr().err


def test_sync_keeps_the_previous_feed_for_rollback(workspace):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _run(workspace, "sync")
    first = workspace["client"].objects[cli.FEED_KEY]

    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    _run(workspace, "sync")

    assert workspace["client"].objects[cli.FEED_PREV_KEY] == first
    assert workspace["client"].objects[cli.FEED_KEY] != first


def test_sync_refuses_to_drop_an_episode_already_in_the_live_feed(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _fill_notes(workspace)
    _run(workspace, "release", _manifest(workspace).episodes[0].guid)
    _run(workspace, "sync")

    manifest = _manifest(workspace)
    manifest.episodes[0].state = "draft"
    save_manifest(manifest, workspace["manifest"])

    assert _run(workspace, "sync") == 1
    assert "would disappear" in capsys.readouterr().err
    assert _run(workspace, "sync", "--allow-removal") == 0


def test_sync_mints_the_preview_token_and_show_guid_exactly_once(workspace):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    _run(workspace, "sync")
    first = _manifest(workspace).show

    _run(workspace, "sync")
    second = _manifest(workspace).show
    assert (second.preview_token, second.podcast_guid) == (first.preview_token, first.podcast_guid)
    assert second.podcast_guid, "a real base_url should have produced a show guid"


# --- check --------------------------------------------------------------------


def test_check_passes_when_the_live_url_answers_a_range_request(workspace, monkeypatch, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    asked = []

    def fake_probe(url, *, timeout):
        asked.append(url)
        return 206, "bytes 0-1023/900000"

    monkeypatch.setattr(cli, "check_byte_range", fake_probe)
    capsys.readouterr()
    assert _run(workspace, "check") == 0
    assert asked == ["https://media.test.example/audio/s2e001.mp3"]
    assert "ok" in capsys.readouterr().out


def test_check_fails_when_the_range_is_ignored(workspace, monkeypatch, capsys):
    """A 200 means the whole file came back: playback works, seeking does not,
    and Apple's validation refuses the feed. Silence here would hide all three."""
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    _run(workspace, "stage", "--all-draft")
    monkeypatch.setattr(cli, "check_byte_range", lambda url, *, timeout: (200, ""))

    assert _run(workspace, "check") == 1
    assert "Seeking is broken" in capsys.readouterr().err


def test_check_has_nothing_to_say_about_an_unstaged_episode(workspace, capsys):
    _run(workspace, "add", str(workspace["run_dir"]), "--episode", "1")
    assert _run(workspace, "check") == 1
    assert "nothing staged" in capsys.readouterr().err


# --- ship -----------------------------------------------------------------------


def test_ship_finishes_on_the_second_pass_once_the_title_is_written(workspace, capsys):
    """The ordinary path, not an edge case: a pipeline run carries no title on
    disk, so the first `ship` gets as far as `release` and stops there. The human
    then writes the title into the manifest and runs the same command again —
    which is the whole point of `ship` being four re-runnable commands.
    """
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 1
    assert "title" in capsys.readouterr().err
    staged = _manifest(workspace).episodes[0]
    assert staged.is_staged and staged.state == "draft"

    _fill_notes(workspace)
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 0

    episodes = _manifest(workspace).episodes
    assert len(episodes) == 1
    assert episodes[0].guid == staged.guid
    assert episodes[0].state == "published"


def test_ship_resumes_a_run_it_already_registered(workspace, monkeypatch, capsys):
    """`ship` is the four commands in sequence, and each of the four exists to be
    re-runnable. If the encode, the upload or the feed sync fails, the episode is
    already in the manifest — and a second `ship` that called `add` again would
    be refused there, for minting a second GUID, before it ever reached the step
    that actually failed. Nothing else re-runs the remaining three for you.
    """
    real_encode_and_tag = cli.encode_mod.encode_and_tag
    attempts = {"n": 0}

    def fails_the_first_time(*args, **kwargs):
        attempts["n"] += 1
        if attempts["n"] == 1:
            raise cli.encode_mod.EncodeError("interrupted part-way through the encode")
        return real_encode_and_tag(*args, **kwargs)

    monkeypatch.setattr(cli.encode_mod, "encode_and_tag", fails_the_first_time)

    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 1
    assert "interrupted part-way" in capsys.readouterr().err
    first_guid = _manifest(workspace).episodes[0].guid
    assert workspace["client"].objects == {}

    _fill_notes(workspace)  # what a human does once the episode exists in the manifest
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 0
    assert "already in the manifest" not in capsys.readouterr().err

    episodes = _manifest(workspace).episodes
    assert len(episodes) == 1
    assert episodes[0].guid == first_guid, "a resumed ship must not mint a second GUID"
    assert episodes[0].state == "published"
    assert episodes[0].audio_key in workspace["client"].objects
    assert cli.FEED_KEY in workspace["client"].objects


# --- credentials ----------------------------------------------------------------


def test_the_cli_loads_the_r2_credentials_from_the_project_env_file(workspace, monkeypatch):
    """`.env` is where the README puts the four R2 variables.

    Nothing else in `dr2_podcast.publish` reads that file, so without a load at
    the CLI boundary every upload reports all four missing on a machine that is
    configured exactly as documented.
    """
    for name in ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        monkeypatch.setenv(name, "recorded-so-monkeypatch-restores-it")
        monkeypatch.delenv(name)

    env_file = workspace["root"] / ".env"
    env_file.write_text(
        "R2_ACCOUNT_ID=from-dotenv\nR2_ACCESS_KEY_ID=key\nR2_SECRET_ACCESS_KEY=secret\nR2_BUCKET=bucket\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "ENV_FILE", env_file)

    with pytest.raises(StorageError):
        R2Config.from_env()

    assert _run(workspace, "sync", "--dry-run") == 0
    assert R2Config.from_env().account_id == "from-dotenv"


def _fill_notes(workspace):
    """Do by hand what a human does: write the title and the show notes."""
    manifest = _manifest(workspace)
    manifest.episodes[0].title = "第1回 テスト"
    manifest.episodes[0].description = "<p>ショーノート</p>"
    save_manifest(manifest, workspace["manifest"])


def test_ship_no_upload_needs_no_credentials(workspace, monkeypatch, capsys):
    """Codex review 2026-08-16, verified: `--no-upload` advertises "encode
    locally only; needs no credentials" and `stage` honoured it, but `ship` then
    called `sync`, which reached for R2 unconditionally. The flag failed on
    exactly the credentials it had promised not to need.

    `_storage_for` raises here, which is what a machine without R2 configured
    actually does — so this fails if anything on the path asks for storage.
    """

    def no_credentials(manifest):
        raise StorageError("R2 credentials are not configured on this machine")

    monkeypatch.setattr(cli, "_storage_for", no_credentials)

    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1", "--no-upload") == 0
    assert "Nothing was uploaded" in capsys.readouterr().err
    assert workspace["client"].objects == {}

    # and the episode stays a DRAFT: an episode nobody can download is not
    # released, and claiming otherwise would make a later ordinary sync publish
    # a feed pointing at an enclosure that was never uploaded
    episode = _manifest(workspace).episodes[0]
    assert episode.state == "draft"
    assert episode.is_staged


def test_ship_without_no_upload_still_uploads(workspace):
    """The guard above must not have quietly turned every ship into a dry run."""
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 1
    _fill_notes(workspace)
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 0
    assert workspace["client"].objects, "ship should have uploaded the audio and the feeds"


def test_sync_records_minted_identities_before_it_uploads(workspace, monkeypatch, capsys):
    """Codex review 2026-08-16, verified. podcast:guid and the preview token are
    minted in memory at the top of sync, and the manifest used to be saved only
    after BOTH feeds had uploaded. A public upload that succeeded followed by any
    later failure lost them; the retry minted different ones and overwrote the
    live public feed with a different podcast:guid — to a subscribed client, a
    different show — while orphaning the first preview URL in the bucket.

    Both values are permanent by design, which is precisely why they must not
    live only inside a process that might not finish.
    """
    assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") == 1
    _fill_notes(workspace)

    real_put = workspace["client"].put_object
    calls = {"n": 0}

    def fail_after_the_public_feed(*a, **kw):
        calls["n"] += 1
        if calls["n"] > 2:  # audio, public feed, then die on the preview feed
            raise RuntimeError("network died mid-sync")
        return real_put(*a, **kw)

    monkeypatch.setattr(workspace["client"], "put_object", fail_after_the_public_feed)
    try:
        assert _run(workspace, "ship", str(workspace["run_dir"]), "--episode", "1") != 0
    except RuntimeError as exc:  # the CLI may let it through rather than translate it
        assert "network died" in str(exc)

    show = _manifest(workspace).show
    assert show.podcast_guid or show.preview_token, "identities must survive the failure"
    first = (show.preview_token, show.podcast_guid)

    monkeypatch.setattr(workspace["client"], "put_object", real_put)
    assert _run(workspace, "sync") == 0
    after = _manifest(workspace).show
    assert (after.preview_token, after.podcast_guid) == first, "a retry must not re-mint identity"
