"""The manifest's refusals.

Almost every test here asserts that something is *rejected*. That is the point
of the module: the manifest holds values other people's software has already
stored, and the failure mode of getting one wrong is silent — a podcast app
re-downloads a back catalogue, or an episode quietly disappears from a feed, and
nothing anywhere reports an error. The refusals are the only alarm there is.
"""

from __future__ import annotations

import json

import pytest

from dr2_podcast.publish.manifest import (
    PLACEHOLDER_BASE_URL,
    STATE_PUBLISHED,
    Episode,
    Manifest,
    ManifestError,
    Show,
    derive_podcast_guid,
    load_manifest,
    mint_guid,
    parse_publish_at,
    save_manifest,
)


def _show(**overrides) -> Show:
    defaults = {
        "title": "テスト番組",
        "description": "説明",
        "owner_email": "owner@example.com",
        "author": "作者",
    }
    return Show(**{**defaults, **overrides})


def _episode(number: int = 1, **overrides) -> Episode:
    defaults = {
        "guid": mint_guid(),
        "season": 1,
        "episode": number,
        "title": f"第{number}回",
        "audio_key": f"audio/s1e{number:03d}.mp3",
        "run_dir": f"research_outputs/Ep{number:03d}_topic",
        "publish_at": "2026-08-01T06:00:00+09:00",
    }
    return Episode(**{**defaults, **overrides})


def _manifest(count: int = 2) -> Manifest:
    manifest = Manifest(show=_show())
    for n in range(1, count + 1):
        manifest.add_episode(_episode(n))
    return manifest


# --- GUIDs -----------------------------------------------------------------


def test_guids_are_random_not_derived():
    """Two calls never collide, and nothing about the episode feeds into them.

    A GUID derived from the URL, title or file hash would change on a re-render
    or a title fix — both routine here — and re-notify every subscriber.
    """
    assert mint_guid() != mint_guid()


def test_adding_the_same_run_dir_twice_is_refused():
    manifest = _manifest(1)
    duplicate = _episode(2, run_dir=manifest.episodes[0].run_dir)
    with pytest.raises(ManifestError, match="already in the manifest"):
        manifest.add_episode(duplicate)


def test_duplicate_guid_is_refused():
    manifest = _manifest(1)
    with pytest.raises(ManifestError, match="already used by"):
        manifest.add_episode(_episode(2, guid=manifest.episodes[0].guid))


def test_duplicate_audio_key_is_refused():
    manifest = _manifest(1)
    with pytest.raises(ManifestError, match="audio_key"):
        manifest.add_episode(_episode(2, audio_key=manifest.episodes[0].audio_key))


def test_duplicate_season_episode_is_refused():
    manifest = _manifest(1)
    with pytest.raises(ManifestError, match="season 1 episode 1"):
        manifest.add_episode(_episode(1, audio_key="audio/other.mp3", run_dir="research_outputs/other"))


def test_saving_a_manifest_that_lost_a_guid_is_refused(tmp_path):
    """The single highest-consequence invariant, enforced at the write."""
    path = tmp_path / "episodes.json"
    original = _manifest(3)
    save_manifest(original, path)

    trimmed = load_manifest(path)
    trimmed.episodes.pop()
    with pytest.raises(ManifestError, match="GUID"):
        save_manifest(trimmed, path)

    # And the file on disk is untouched by the refused write.
    assert len(load_manifest(path).episodes) == 3


def test_saving_a_manifest_that_rewrote_a_guid_is_refused(tmp_path):
    path = tmp_path / "episodes.json"
    save_manifest(_manifest(2), path)
    tampered = load_manifest(path)
    tampered.episodes[0].guid = mint_guid()
    with pytest.raises(ManifestError, match="GUID"):
        save_manifest(tampered, path)


# --- the preview token and the show GUID ------------------------------------


def test_preview_token_is_minted_once_and_never_rerolled(tmp_path):
    manifest = _manifest(1)
    first = manifest.ensure_preview_token()
    assert manifest.ensure_preview_token() == first
    assert len(first) == 32

    path = tmp_path / "episodes.json"
    save_manifest(manifest, path)
    reloaded = load_manifest(path)
    reloaded.show.preview_token = "0" * 32
    with pytest.raises(ManifestError, match="preview_token"):
        save_manifest(reloaded, path)


def test_podcast_guid_waits_for_a_real_domain():
    """Deriving it from the placeholder would freeze the wrong identity forever."""
    manifest = _manifest(1)
    assert manifest.show.base_url == PLACEHOLDER_BASE_URL
    assert manifest.ensure_podcast_guid() is None

    manifest.show.base_url = "https://media.example.org"
    minted = manifest.ensure_podcast_guid()
    assert minted == derive_podcast_guid("https://media.example.org/feed.xml")
    # Frozen once minted, even if the feed URL later moves — that is its purpose.
    manifest.show.base_url = "https://media.elsewhere.org"
    assert manifest.ensure_podcast_guid() == minted


def test_podcast_guid_ignores_scheme_and_trailing_slash():
    assert derive_podcast_guid("https://media.example.org/feed.xml") == derive_podcast_guid(
        "http://media.example.org/feed.xml/"
    )


def test_changing_the_show_guid_is_refused(tmp_path):
    path = tmp_path / "episodes.json"
    manifest = _manifest(1)
    manifest.show.base_url = "https://media.example.org"
    manifest.ensure_podcast_guid()
    save_manifest(manifest, path)

    reloaded = load_manifest(path)
    reloaded.show.podcast_guid = derive_podcast_guid("https://other.example.org/feed.xml")
    with pytest.raises(ManifestError, match="podcast_guid"):
        save_manifest(reloaded, path)


# --- validation -------------------------------------------------------------


def test_non_ascii_audio_key_is_refused():
    """The episode folders are Japanese; Apple restricts enclosure URLs to ASCII."""
    manifest = Manifest(show=_show())
    manifest.episodes.append(_episode(1, audio_key="audio/第1回.mp3"))
    with pytest.raises(ManifestError, match="ASCII-safe"):
        manifest.validate()


def test_naive_publish_at_is_refused():
    with pytest.raises(ManifestError, match="no timezone"):
        parse_publish_at("2026-08-01T06:00:00")


def test_unknown_state_is_refused():
    manifest = Manifest(show=_show())
    manifest.episodes.append(_episode(1, state="scheduled"))
    with pytest.raises(ManifestError, match="state"):
        manifest.validate()


def test_unknown_key_in_json_is_refused(tmp_path):
    """A typo in a hand-edited manifest must not be silently dropped.

    Show notes and publish dates are edited by hand in this file. A misspelled
    key that vanished into **kwargs would publish an episode missing the field
    someone thought they had just written.
    """
    path = tmp_path / "episodes.json"
    save_manifest(_manifest(1), path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["episodes"][0]["descripton"] = "typo"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ManifestError, match="unknown key"):
        load_manifest(path)


# --- selection --------------------------------------------------------------


def test_public_selection_needs_both_published_and_due():
    manifest = _manifest(3)
    now = parse_publish_at("2026-08-02T00:00:00+09:00")
    for episode, publish_at in zip(
        manifest.episodes,
        ["2026-08-01T06:00:00+09:00", "2026-08-01T06:00:00+09:00", "2026-08-05T06:00:00+09:00"],
        strict=True,
    ):
        episode.publish_at = publish_at
    manifest.episodes[0].state = STATE_PUBLISHED  # published and due
    manifest.episodes[2].state = STATE_PUBLISHED  # published but not yet due

    assert [ep.episode for ep in manifest.public_at(now)] == [1]


def test_is_staged_needs_both_bytes_and_duration():
    episode = _episode(1)
    assert not episode.is_staged
    episode.bytes = 1000
    assert not episode.is_staged
    episode.duration_seconds = 60
    assert episode.is_staged


def test_round_trip_preserves_every_field(tmp_path):
    path = tmp_path / "episodes.json"
    manifest = _manifest(2)
    manifest.ensure_preview_token()
    manifest.episodes[0].bytes = 24_000_000
    manifest.episodes[0].duration_seconds = 1495
    manifest.episodes[0].description = "<p>ショーノート</p>"
    save_manifest(manifest, path)

    assert load_manifest(path).to_json() == manifest.to_json()
    # Japanese is stored readably, not as \uXXXX escapes — this file is edited by hand.
    assert "テスト番組" in path.read_text(encoding="utf-8")
