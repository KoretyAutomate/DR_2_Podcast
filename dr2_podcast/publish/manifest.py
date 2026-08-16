"""`podcast/episodes.json` — the source of truth for the feed.

Everything the RSS feed says about the show comes from this file. Pipeline
output folders are where the audio lives; the publish sheet is a human-facing
crib; this is the machine record, and it is committed to git.

The invariant that shapes the whole module: **a GUID is minted exactly once and
never changes.** Podcast apps identify an episode by its GUID and by nothing
else — not the URL, not the title, not the file. Change a GUID and every
subscriber's app re-downloads the episode and re-notifies them; change all of
them and it re-downloads the entire back catalogue. Because a re-render is
routine in this project (`rerender-podcast` exists, and the episode folders hold
`pre_rerender_backup_*` directories), a GUID must not be derivable from the
audio, the path, or the title — all three change for cosmetic reasons.

So this module refuses, rather than warns, in three places:

* :meth:`Manifest.add_episode` will not overwrite an episode that already has a
  GUID.
* :func:`save_manifest` will not write a file that has lost or altered a GUID
  present in the file it is replacing.
* The show's ``podcast_guid`` and ``preview_token``, once set, are frozen the
  same way. Regenerating the preview token silently breaks the subscription in
  the user's podcast app, which looks like nothing at all until an episode fails
  to appear.
"""

from __future__ import annotations

import json
import re
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

#: Episode lifecycle. `draft` is staged-but-invisible; only `published` reaches
#: the public feed, and even then only once `publish_at` has passed.
STATE_DRAFT = "draft"
STATE_PUBLISHED = "published"
STATES = (STATE_DRAFT, STATE_PUBLISHED)

#: What `show.base_url` holds before a domain has been registered. Feed
#: generation works against it so the XML can be inspected offline; anything
#: that would make the show reachable by a listener refuses while it is in place.
PLACEHOLDER_BASE_URL = "https://media.example.com"

#: Podcasting 2.0's namespace UUID for `podcast:guid` (podcast-namespace 1.0).
PODCAST_NAMESPACE_UUID = uuid.UUID("ead4c236-bf58-58c6-a2c6-a6b28d128cb6")

#: Apple restricts enclosure URLs to `a-z A-Z 0-9`. The episode folders are
#: named in Japanese, so object keys are derived from season/episode numbers and
#: never from a folder name. This is the check that keeps that true.
_ASCII_KEY_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9/_.-]*$")

DEFAULT_MANIFEST_PATH = Path(__file__).resolve().parent.parent.parent / "podcast" / "episodes.json"


class ManifestError(Exception):
    """The manifest is inconsistent, or an operation would violate an invariant."""


def mint_guid() -> str:
    """A fresh episode GUID.

    Random, not derived. Deriving from the URL, the title, the path or a file
    hash would couple episode identity to values this project changes routinely.
    """
    return str(uuid.uuid4())


def mint_preview_token() -> str:
    """32 hex characters of unguessability for the private preview feed URL.

    Not a password — it authorises nothing and protects nothing valuable. It
    keeps the draft feed out of crawlers and directories, which is the right
    strength for an audience of one.
    """
    return secrets.token_hex(16)


def derive_podcast_guid(feed_url: str) -> str:
    """The show-level `podcast:guid`: UUIDv5 over the feed URL, scheme stripped.

    Per the Podcasting 2.0 spec the protocol and any trailing slash come off
    first, so that the identity survives an http→https change or a feed move.
    """
    trimmed = re.sub(r"^https?://", "", feed_url.strip()).rstrip("/")
    return str(uuid.uuid5(PODCAST_NAMESPACE_UUID, trimmed))


def parse_publish_at(value: str) -> datetime:
    """Parse an ISO 8601 `publish_at`, demanding a timezone.

    A naive datetime formats as `-0000` in RFC 2822, which Apple reads as
    "unknown local time" — the episode then sorts unpredictably against
    everything else in the feed. Requiring the offset here means the feed
    generator can never emit one.
    """
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError) as exc:
        raise ManifestError(f"publish_at is not ISO 8601: {value!r}") from exc
    if parsed.tzinfo is None:
        raise ManifestError(
            f"publish_at has no timezone offset: {value!r}. "
            "Write it as e.g. 2026-08-18T06:00:00+09:00 — a naive time becomes '-0000' in RFC 2822."
        )
    return parsed


@dataclass
class Show:
    """Channel-level metadata. One per manifest."""

    title: str
    description: str
    language: str = "ja"
    author: str = ""
    owner_name: str = ""
    owner_email: str = ""
    category: str = "Science"
    subcategory: str = ""
    explicit: bool = False
    type: str = "episodic"
    base_url: str = PLACEHOLDER_BASE_URL
    feed_path: str = "/feed.xml"
    artwork_path: str = "/artwork.jpg"
    podcast_guid: str | None = None
    preview_token: str | None = None
    copyright: str = ""

    @property
    def base(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def feed_url(self) -> str:
        return f"{self.base}{self.feed_path}"

    @property
    def artwork_url(self) -> str:
        return f"{self.base}{self.artwork_path}"

    @property
    def preview_path(self) -> str:
        if not self.preview_token:
            raise ManifestError("show.preview_token is not set — cannot address the preview feed")
        return f"/preview-{self.preview_token}.xml"

    @property
    def preview_url(self) -> str:
        return f"{self.base}{self.preview_path}"

    @property
    def has_real_domain(self) -> bool:
        """False while `base_url` is still the placeholder from before Step 2."""
        return self.base != PLACEHOLDER_BASE_URL.rstrip("/")

    def to_json(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "description": self.description,
            "language": self.language,
            "author": self.author,
            "owner_name": self.owner_name,
            "owner_email": self.owner_email,
            "category": self.category,
            "subcategory": self.subcategory,
            "explicit": self.explicit,
            "type": self.type,
            "base_url": self.base_url,
            "feed_path": self.feed_path,
            "artwork_path": self.artwork_path,
            "podcast_guid": self.podcast_guid,
            "preview_token": self.preview_token,
            "copyright": self.copyright,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Show:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ManifestError(f"unknown key(s) in show block: {', '.join(sorted(unknown))}")
        return cls(**payload)


@dataclass
class Episode:
    """One item in the feed.

    `bytes` and `duration_seconds` are written by `stage` from the encoded file
    and are never hand-typed: `<enclosure length>` has to be the exact byte count
    of what is actually served, or downloads stall near the end in some apps.
    A zero in either field means "not staged yet".
    """

    guid: str
    season: int
    episode: int
    title: str
    audio_key: str
    run_dir: str
    publish_at: str
    subtitle: str = ""
    description: str = ""
    bytes: int = 0
    duration_seconds: int = 0
    explicit: bool = False
    state: str = STATE_DRAFT

    @property
    def is_staged(self) -> bool:
        """True once the MP3 is uploaded and its real size and length are known."""
        return self.bytes > 0 and self.duration_seconds > 0

    @property
    def publish_at_dt(self) -> datetime:
        return parse_publish_at(self.publish_at)

    def is_public_at(self, now: datetime) -> bool:
        """Whether the public feed should carry this episode at `now`."""
        return self.state == STATE_PUBLISHED and self.publish_at_dt <= now

    def enclosure_url(self, show: Show) -> str:
        return f"{show.base}/{self.audio_key.lstrip('/')}"

    def to_json(self) -> dict[str, Any]:
        return {
            "guid": self.guid,
            "season": self.season,
            "episode": self.episode,
            "title": self.title,
            "subtitle": self.subtitle,
            "description": self.description,
            "run_dir": self.run_dir,
            "audio_key": self.audio_key,
            "bytes": self.bytes,
            "duration_seconds": self.duration_seconds,
            "publish_at": self.publish_at,
            "explicit": self.explicit,
            "state": self.state,
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Episode:
        known = {f for f in cls.__dataclass_fields__}
        unknown = set(payload) - known
        if unknown:
            raise ManifestError(f"unknown key(s) in episode block: {', '.join(sorted(unknown))}")
        return cls(**payload)


@dataclass
class Manifest:
    show: Show
    episodes: list[Episode] = field(default_factory=list)

    # -- lookup ------------------------------------------------------------

    def by_guid(self, guid: str) -> Episode:
        for ep in self.episodes:
            if ep.guid == guid:
                return ep
        raise ManifestError(f"no episode with guid {guid}")

    def by_run_dir(self, run_dir: str) -> Episode | None:
        normalised = str(run_dir).rstrip("/")
        for ep in self.episodes:
            if ep.run_dir.rstrip("/") == normalised:
                return ep
        return None

    def drafts(self) -> list[Episode]:
        return [ep for ep in self.episodes if ep.state == STATE_DRAFT]

    def staged(self) -> list[Episode]:
        return [ep for ep in self.episodes if ep.is_staged]

    def public_at(self, now: datetime) -> list[Episode]:
        return [ep for ep in self.episodes if ep.is_public_at(now)]

    # -- mutation ----------------------------------------------------------

    def add_episode(self, episode: Episode) -> Episode:
        """Append an episode, refusing anything that would collide.

        The GUID guard is the important one and it is an assertion, not a
        convention: an `add` that silently re-registered an existing episode
        under a second GUID would put the same audio in the feed twice, and a
        re-render that re-ran `add` would do exactly that.
        """
        already = self.by_run_dir(episode.run_dir)
        if already is not None:
            raise ManifestError(
                f"{episode.run_dir} is already in the manifest (guid {already.guid}). "
                "Refusing to mint a second GUID for it — that would publish the same episode twice."
            )
        for existing in self.episodes:
            if existing.guid == episode.guid:
                raise ManifestError(f"guid {episode.guid} is already used by {existing.run_dir}")
            if existing.audio_key == episode.audio_key:
                raise ManifestError(f"audio_key {episode.audio_key} is already used by {existing.run_dir}")
            if (existing.season, existing.episode) == (episode.season, episode.episode):
                raise ManifestError(
                    f"season {episode.season} episode {episode.episode} is already used by {existing.run_dir}"
                )
        self.episodes.append(episode)
        return episode

    def ensure_preview_token(self) -> str:
        """Mint the preview token if it has never been minted. Never re-mint."""
        if not self.show.preview_token:
            self.show.preview_token = mint_preview_token()
        return self.show.preview_token

    def ensure_podcast_guid(self) -> str | None:
        """Derive `podcast:guid` from the feed URL, once, once a real domain exists.

        Returns None while the base URL is still the placeholder — deriving it
        from `media.example.com` would freeze the wrong identity forever, and
        the whole point of this value is that it never changes.
        """
        if self.show.podcast_guid:
            return self.show.podcast_guid
        if not self.show.has_real_domain:
            return None
        self.show.podcast_guid = derive_podcast_guid(self.show.feed_url)
        return self.show.podcast_guid

    # -- validation --------------------------------------------------------

    def validate(self) -> None:
        """Check every invariant that a later step would otherwise discover live."""
        if not self.show.title.strip():
            raise ManifestError("show.title is empty")
        if not self.show.language.strip():
            raise ManifestError("show.language is empty")
        if self.show.type not in ("episodic", "serial"):
            raise ManifestError(f"show.type must be 'episodic' or 'serial', not {self.show.type!r}")

        seen_guids: set[str] = set()
        seen_keys: set[str] = set()
        seen_numbers: set[tuple[int, int]] = set()
        for ep in self.episodes:
            if not ep.guid:
                raise ManifestError(f"episode {ep.run_dir} has no guid")
            if ep.guid in seen_guids:
                raise ManifestError(f"duplicate guid {ep.guid}")
            seen_guids.add(ep.guid)

            if not _ASCII_KEY_RE.match(ep.audio_key):
                raise ManifestError(
                    f"audio_key {ep.audio_key!r} is not ASCII-safe. Apple restricts enclosure URLs to "
                    "a-z A-Z 0-9; derive the key from season/episode numbers, never from the folder name."
                )
            if ep.audio_key in seen_keys:
                raise ManifestError(f"duplicate audio_key {ep.audio_key}")
            seen_keys.add(ep.audio_key)

            if (ep.season, ep.episode) in seen_numbers:
                raise ManifestError(f"duplicate season/episode s{ep.season}e{ep.episode}")
            seen_numbers.add((ep.season, ep.episode))

            if ep.state not in STATES:
                raise ManifestError(f"episode {ep.guid} has state {ep.state!r}, expected one of {STATES}")
            if ep.bytes < 0 or ep.duration_seconds < 0:
                raise ManifestError(f"episode {ep.guid} has negative bytes/duration")
            # Raises on a naive or malformed value, which is the point of the call.
            parse_publish_at(ep.publish_at)

    # -- serialisation -----------------------------------------------------

    def to_json(self) -> dict[str, Any]:
        return {
            "show": self.show.to_json(),
            "episodes": [ep.to_json() for ep in self.episodes],
        }

    @classmethod
    def from_json(cls, payload: dict[str, Any]) -> Manifest:
        if not isinstance(payload, dict) or "show" not in payload:
            raise ManifestError("manifest has no 'show' block")
        episodes = payload.get("episodes") or []
        if not isinstance(episodes, list):
            raise ManifestError("manifest 'episodes' is not a list")
        return cls(
            show=Show.from_json(payload["show"]),
            episodes=[Episode.from_json(ep) for ep in episodes],
        )


def load_manifest(path: Path | str = DEFAULT_MANIFEST_PATH) -> Manifest:
    path = Path(path)
    if not path.is_file():
        raise ManifestError(f"no manifest at {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ManifestError(f"could not read {path}: {exc}") from exc
    manifest = Manifest.from_json(payload)
    manifest.validate()
    return manifest


def _assert_identity_preserved(previous: Manifest, current: Manifest) -> None:
    """Refuse a write that drops or rewrites an already-published identity.

    Everything checked here is a value that other people's software has already
    stored. A GUID that vanishes takes its episode out of every subscriber's
    app; a changed preview token silently unsubscribes the one person the
    preview feed exists for; a changed `podcast:guid` splits the show's identity
    in Podcast Index. None of these fail loudly on their own, which is why they
    fail loudly here.
    """
    current_guids = {ep.guid for ep in current.episodes}
    lost = sorted({ep.guid for ep in previous.episodes} - current_guids)
    if lost:
        raise ManifestError(
            f"refusing to write: {len(lost)} GUID(s) present in the existing manifest would be lost "
            f"({', '.join(lost[:3])}{'…' if len(lost) > 3 else ''}). A GUID is permanent — an episode that "
            "loses its GUID is a new episode to every podcast app."
        )
    if previous.show.podcast_guid and previous.show.podcast_guid != current.show.podcast_guid:
        raise ManifestError(
            "refusing to write: show.podcast_guid would change. It is minted once and is the show's "
            "identity across feed-URL changes."
        )
    if previous.show.preview_token and previous.show.preview_token != current.show.preview_token:
        raise ManifestError(
            "refusing to write: show.preview_token would change. Regenerating it changes the preview "
            "feed URL and silently breaks the subscription in the podcast app on the phone."
        )


def save_manifest(manifest: Manifest, path: Path | str = DEFAULT_MANIFEST_PATH) -> Path:
    """Validate, compare against what is on disk, then write atomically."""
    path = Path(path)
    manifest.validate()
    if path.is_file():
        _assert_identity_preserved(load_manifest(path), manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(
        json.dumps(manifest.to_json(), ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    tmp.replace(path)
    return path
