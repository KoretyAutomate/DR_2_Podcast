"""`python -m dr2_podcast.publish` — add, stage, release, sync.

Four commands rather than one, on purpose. A single opaque `ship` that
half-fails leaves you guessing whether the audio uploaded, whether the manifest
was written, and whether the feed went out; four commands each say where they
stopped and each can be re-run. `ship` exists as a convenience wrapper once the
pieces are known to work, and is nothing more than the four in sequence.

    add <run_dir>       register a finished run, mint its GUID exactly once
    stage <guid|--all-draft>
                        encode → tag → upload; write back real bytes/duration
    release <guid>      flip state to published
    sync                rebuild feed.xml + the preview feed and upload both
    check [guid]        ask the live URLs whether they serve byte ranges
    ship <run_dir>      the four above, in order

Everything up to `stage` runs without credentials, so the whole thing can be
built, encoded and inspected offline; only the uploads need R2.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv

from dr2_podcast.publish import artwork as artwork_mod
from dr2_podcast.publish import encode as encode_mod
from dr2_podcast.publish import feed as feed_mod
from dr2_podcast.publish.manifest import (
    DEFAULT_MANIFEST_PATH,
    STATE_DRAFT,
    STATE_PUBLISHED,
    Episode,
    Manifest,
    ManifestError,
    load_manifest,
    mint_guid,
    save_manifest,
)
from dr2_podcast.publish.storage import (
    CONTENT_TYPE_JPEG,
    CONTENT_TYPE_MP3,
    CONTENT_TYPE_RSS,
    R2Storage,
    StorageError,
    check_byte_range,
)
from dr2_podcast.tools.publish_sheet import BLANK, UNREADABLE, PublishSheetError, build_publish_sheet

logger = logging.getLogger(__name__)

JST = timezone(timedelta(hours=9))

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
#: Encoded MP3s and artwork land here, laid out exactly like the bucket, so
#: `stage` is a straight upload and the local tree can be inspected first.
BUILD_DIR = PROJECT_ROOT / "podcast" / "build"
#: Where the R2 credentials live, per README and `.env.example`. Loaded at the
#: CLI boundary rather than on import, so importing this module has no effect on
#: the environment of anything that only wants to call one of its functions.
ENV_FILE = PROJECT_ROOT / ".env"

FEED_KEY = "feed.xml"
FEED_PREV_KEY = "feed.xml.prev"
ARTWORK_KEY = "artwork.jpg"

#: publish_sheet's "I could not read this" markers. Neither may reach a feed.
_SENTINELS = (BLANK, UNREADABLE)


class CliError(Exception):
    """Something the user needs to fix before the command can do its job."""


def _clean(value: str | None) -> str:
    """Return a seeded field, or empty if it is a publish_sheet sentinel.

    publish_sheet never guesses: an unreadable title comes back as `(要記入)`.
    That string is useful in a Markdown sheet a human fills in and catastrophic
    in an RSS feed, where it would ship as the episode's actual title. So it is
    translated to "not set" here and refused at `release`.
    """
    if value is None:
        return ""
    text = value.strip()
    if any(sentinel in text for sentinel in _SENTINELS):
        return ""
    return text


def _now() -> datetime:
    return datetime.now(JST)


def _local_mp3(episode: Episode) -> Path:
    return BUILD_DIR / episode.audio_key


def _run_dir_arg(raw: str) -> Path:
    """A run directory as typed on the command line.

    Relative to the current directory, the way every other CLI treats a path —
    which is *not* how a relative string already sitting in the manifest is read
    (`manifest.resolve_run_dir` explains that one). Both agree from the repository
    root, and this is the only place a human types the path.
    """
    return Path(raw).expanduser().resolve()


# ---------------------------------------------------------------------------
# add
# ---------------------------------------------------------------------------


def cmd_add(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    # Absolute, because the manifest outlives the shell that wrote it: `stage`
    # resolves this string later, from wherever it happens to be run.
    run_dir = _run_dir_arg(args.run_dir)

    try:
        sheet = build_publish_sheet(run_dir)
    except PublishSheetError as exc:
        raise CliError(f"cannot register {run_dir}: {exc}") from exc

    season = args.season
    episode_number = args.episode
    if episode_number is None:
        seeded = _clean(sheet.episode_number)
        if not seeded.isdigit():
            raise CliError(
                f"{run_dir} does not state its episode number and --episode was not passed. "
                "Pass it explicitly rather than letting the feed guess."
            )
        episode_number = int(seeded)

    publish_at = args.publish_at or _now().isoformat(timespec="seconds")
    episode = Episode(
        guid=mint_guid(),
        season=season,
        episode=episode_number,
        title=_clean(sheet.title),
        audio_key=f"audio/s{season}e{episode_number:03d}.mp3",
        run_dir=str(run_dir),
        publish_at=publish_at,
        description=_clean(sheet.description),
        state=STATE_DRAFT,
    )
    manifest.add_episode(episode)
    save_manifest(manifest, args.manifest)

    print(f"added {episode.guid}  s{season}e{episode_number:03d}  {episode.title or '(no title yet)'}")
    for field_name in ("title", "description"):
        if not getattr(episode, field_name):
            print(
                f"  NOTE: {field_name} is empty — {run_dir.name} does not carry one on disk. "
                f"Fill it in {args.manifest} before `release`; that command refuses without it."
            )
    return 0


# ---------------------------------------------------------------------------
# stage
# ---------------------------------------------------------------------------


def _episodes_for(manifest: Manifest, args: argparse.Namespace) -> list[Episode]:
    if getattr(args, "all_draft", False):
        return manifest.drafts()
    if not args.guid:
        raise CliError("pass a guid, or --all-draft")
    return [manifest.by_guid(args.guid)]


def _encode_one(episode: Episode, show_title: str, *, cover: Path | None, force: bool) -> tuple[int, int]:
    """Encode this episode's mixed master to the build tree. Idempotent.

    An MP3 already in the build tree is reused rather than re-encoded, but it is
    still re-tagged from the current manifest. The usual reason to re-run `stage`
    is that the title was empty when the episode was added, or the show metadata
    or cover art has since changed — and reusing the file untouched would upload
    an enclosure whose embedded tags still say `s2e001` with last month's cover,
    with nothing to indicate it. Tagging is a few hundred milliseconds against a
    re-encode's minutes, so the caller never has to reach for `--force-encode`
    just to correct a title.
    """
    sheet = build_publish_sheet(episode.run_path)
    mp3_path = _local_mp3(episode)
    tags = encode_mod.Id3Tags(
        title=episode.title or f"s{episode.season}e{episode.episode:03d}",
        artist=show_title,
        album=show_title,
        track=episode.episode,
        year=str(episode.publish_at_dt.year),
        cover_jpeg=cover,
    )
    if mp3_path.is_file() and not force:
        encode_mod.verify_encode(sheet.audio_path, mp3_path)
        encode_mod.tag_mp3(mp3_path, tags)
        # Re-verified after tagging, and the size read last, for the same reason
        # `encode_and_tag` does both: an APIC frame changes the byte count that
        # `<enclosure length>` has to state exactly.
        duration = encode_mod.verify_encode(sheet.audio_path, mp3_path)
        return mp3_path.stat().st_size, int(round(duration))
    return encode_mod.encode_and_tag(sheet.audio_path, mp3_path, tags)


def cmd_stage(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    targets = _episodes_for(manifest, args)
    if not targets:
        print("nothing to stage")
        return 0

    cover = BUILD_DIR / ARTWORK_KEY
    if not cover.is_file():
        if not args.artwork_source:
            raise CliError(
                f"no cover art at {cover} and --artwork-source was not passed. "
                "Run `publish artwork <source.png>` first."
            )
        artwork_mod.build_artwork(args.artwork_source, cover)

    storage = None if args.no_upload else _storage_for(manifest)

    for episode in targets:
        size, duration = _encode_one(episode, manifest.show.title, cover=cover, force=args.force_encode)
        episode.bytes = size
        episode.duration_seconds = duration
        if storage is not None:
            uploaded = storage.put_file(episode.audio_key, _local_mp3(episode), CONTENT_TYPE_MP3)
            if uploaded != size:
                raise CliError(f"uploaded {uploaded} bytes for {episode.audio_key} but the file is {size}")
        print(f"staged {episode.guid}  {episode.audio_key}  {size} bytes  {duration}s")

    if storage is not None:
        storage.put_file(ARTWORK_KEY, cover, CONTENT_TYPE_JPEG)
    save_manifest(manifest, args.manifest)
    return 0


# ---------------------------------------------------------------------------
# artwork
# ---------------------------------------------------------------------------


def cmd_artwork(args: argparse.Namespace) -> int:
    target = BUILD_DIR / ARTWORK_KEY
    edge, size = artwork_mod.build_artwork(args.source, target, edge=args.edge)
    fmt, dimensions, mode, actual = artwork_mod.describe_artwork(target)
    print(f"{target}: {fmt} {dimensions[0]}x{dimensions[1]} {mode} {actual} bytes (edge {edge}, {size} bytes)")
    return 0


# ---------------------------------------------------------------------------
# release
# ---------------------------------------------------------------------------


def cmd_release(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    episode = manifest.by_guid(args.guid)

    missing = [name for name in ("title", "description") if not getattr(episode, name).strip()]
    if missing:
        raise CliError(
            f"refusing to release {episode.guid}: {', '.join(missing)} is empty. "
            f"Write it into {args.manifest} first — an episode published with a placeholder title "
            "is a placeholder title in every podcast app."
        )
    if not episode.is_staged:
        raise CliError(f"refusing to release {episode.guid}: not staged yet (bytes/duration are 0). Run `stage` first.")

    if args.publish_at:
        episode.publish_at = args.publish_at
    episode.state = STATE_PUBLISHED
    save_manifest(manifest, args.manifest)
    print(f"released {episode.guid}  publish_at={episode.publish_at}")
    return 0


# ---------------------------------------------------------------------------
# sync
# ---------------------------------------------------------------------------


def _storage_for(manifest: Manifest) -> R2Storage:
    if not manifest.show.has_real_domain:
        raise CliError(
            "show.base_url is still the placeholder. Register the domain, put it in the manifest, "
            "and re-run. (`sync --dry-run` works without one.)"
        )
    storage = R2Storage.from_env()
    storage.check_bucket()
    return storage


def _guard_enclosures(manifest: Manifest, storage: R2Storage, items: list[dict[str, str]]) -> None:
    """Every enclosure must already exist in the bucket, at the stated size.

    Called with the items of BOTH feeds, not just the public one. `stage
    --no-upload` writes real bytes and duration back into the manifest without
    uploading anything, and that alone is enough to put an episode in the
    preview feed — so guarding only the public feed ships a private feed whose
    entire promise, that a draft is listenable before release, is a 404.

    The two feeds share GUIDs and share each episode's recorded `bytes`, so a
    GUID seen twice is one object and is checked once.
    """
    by_guid = {ep.guid: ep for ep in manifest.episodes}
    seen: set[str] = set()
    for item in items:
        if item["guid"] in seen:
            continue
        seen.add(item["guid"])
        episode = by_guid[item["guid"]]
        actual = storage.object_size(episode.audio_key)
        if actual is None:
            raise CliError(
                f"{episode.audio_key} is in the feed but not in the bucket. "
                "A listener would see a new episode that 404s. Run `stage` for it."
            )
        if actual != int(item["length"]):
            raise CliError(
                f"{episode.audio_key} is {actual} bytes in the bucket but the feed says {item['length']}. "
                "Re-run `stage`; a wrong <enclosure length> stalls downloads near the end."
            )


def cmd_sync(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest)
    manifest.ensure_preview_token()
    manifest.ensure_podcast_guid()
    now = _now()

    public = feed_mod.build_feed(manifest, preview=False, now=now)
    public_items = feed_mod.validate_feed_bytes(public)
    feed_mod.cross_check_manifest(manifest, {ep.audio_key: _local_mp3(ep) for ep in manifest.episodes})

    preview_token = manifest.show.preview_token
    preview = feed_mod.build_feed(manifest, preview=True, now=now)
    preview_items = feed_mod.validate_feed_bytes(preview)

    if args.dry_run:
        print(public.decode("utf-8"))
        print(
            f"\n--- dry run: {len(public_items)} public item(s), {len(preview_items)} preview item(s); "
            f"nothing uploaded ---",
            file=sys.stderr,
        )
        return 0

    storage = _storage_for(manifest)
    _guard_enclosures(manifest, storage, public_items + preview_items)

    live = storage.get_bytes(FEED_KEY)
    if live is not None and not args.allow_removal:
        feed_mod.assert_no_episode_disappears(live, public)
    if live is not None:
        # The 30-second rollback: the previous feed stays one copy away.
        storage.copy(FEED_KEY, FEED_PREV_KEY)

    storage.put_bytes(FEED_KEY, public, CONTENT_TYPE_RSS)
    storage.put_bytes(f"preview-{preview_token}.xml", preview, CONTENT_TYPE_RSS)
    save_manifest(manifest, args.manifest)

    print(f"synced: {len(public_items)} public, {len(preview_items)} preview")
    print(f"  public  {manifest.show.feed_url}")
    print(f"  preview {manifest.show.preview_url}")
    return 0


# ---------------------------------------------------------------------------
# check
# ---------------------------------------------------------------------------


def cmd_check(args: argparse.Namespace) -> int:
    """Ask the live URLs for their first kilobyte and report what came back.

    Byte-range support is a property of the deployment — the custom domain in
    front of the bucket — not of anything in this repository, so it can only be
    established by asking. A 200 means the range was ignored: playback from the
    start works, nobody can scrub, and Apple's validation fails at submission.
    """
    manifest = load_manifest(args.manifest)
    targets = [manifest.by_guid(args.guid)] if args.guid else manifest.staged()
    if not targets:
        raise CliError("nothing staged to check — run `stage` first")

    failed = []
    for episode in targets:
        url = episode.enclosure_url(manifest.show)
        status, content_range = check_byte_range(url, timeout=args.timeout)
        ok = status == 206 and content_range.startswith("bytes ")
        if not ok:
            failed.append(episode.audio_key)
        print(f"{'ok  ' if ok else 'FAIL'} {status} {content_range or '(no Content-Range)'}  {url}")

    if failed:
        raise CliError(
            f"{len(failed)} enclosure(s) answered a range request without a 206: {', '.join(failed)}. "
            "Seeking is broken for every listener and Apple's validation will refuse the feed."
        )
    return 0


# ---------------------------------------------------------------------------
# ship
# ---------------------------------------------------------------------------


def cmd_ship(args: argparse.Namespace) -> int:
    """add → stage → release → sync for one run directory.

    `add` runs only for a run that is not registered yet. The four commands
    exist separately so that a half-finished publish can be re-run, and `ship`
    keeps that property only if it tolerates the work its own previous attempt
    completed: an encode, an upload or a feed sync that fails leaves the episode
    in the manifest, and a second `ship` that called `add` again would be
    refused there — correctly, since a second GUID for one run publishes the
    episode twice — without ever reaching the step that actually failed.
    """
    run_dir = _run_dir_arg(args.run_dir)
    episode = load_manifest(args.manifest).by_run_dir(run_dir)
    if episode is None:
        cmd_add(args)
        episode = load_manifest(args.manifest).by_run_dir(run_dir)
        if episode is None:  # pragma: no cover — add() would have raised
            raise CliError(f"{args.run_dir} is not in the manifest after add")
    else:
        print(f"already registered {episode.guid}  s{episode.season}e{episode.episode:03d} — reusing its GUID")

    args.guid = episode.guid
    args.all_draft = False
    cmd_stage(args)
    cmd_release(args)
    return cmd_sync(args)


# ---------------------------------------------------------------------------
# argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m dr2_podcast.publish", description=__doc__.split("\n")[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH, help="path to episodes.json")
    subparsers = parser.add_subparsers(dest="command", required=True)

    add = subparsers.add_parser("add", help="register a finished run in the manifest")
    add.add_argument("run_dir", help="episode or pipeline run directory")
    add.add_argument("--season", type=int, default=2, help="season number (default 2, the 実践編)")
    add.add_argument("--episode", type=int, default=None, help="episode number within the season")
    add.add_argument("--publish-at", default=None, help="ISO 8601 with offset, e.g. 2026-09-01T06:00:00+09:00")
    add.set_defaults(func=cmd_add)

    art = subparsers.add_parser("artwork", help="build the show cover from a source image")
    art.add_argument("source", help="square source image, at least 1400px")
    art.add_argument("--edge", type=int, default=artwork_mod.MAX_EDGE, help="target edge in px")
    art.set_defaults(func=cmd_artwork)

    stage = subparsers.add_parser("stage", help="encode, tag and upload audio")
    stage.add_argument("guid", nargs="?", default=None)
    stage.add_argument("--all-draft", action="store_true", help="stage every draft episode")
    stage.add_argument("--no-upload", action="store_true", help="encode locally only; needs no credentials")
    stage.add_argument("--force-encode", action="store_true", help="re-encode even if the MP3 already exists")
    stage.add_argument("--artwork-source", default=None, help="build the cover from this image if it is missing")
    stage.set_defaults(func=cmd_stage)

    release = subparsers.add_parser("release", help="mark an episode published")
    release.add_argument("guid")
    release.add_argument("--publish-at", default=None, help="override publish_at while releasing")
    release.set_defaults(func=cmd_release)

    sync = subparsers.add_parser("sync", help="rebuild and upload feed.xml and the preview feed")
    sync.add_argument("--dry-run", action="store_true", help="print the public feed; upload nothing")
    sync.add_argument("--allow-removal", action="store_true", help="permit dropping an episode already in the feed")
    sync.set_defaults(func=cmd_sync)

    check = subparsers.add_parser("check", help="prove the live URLs answer byte-range requests")
    check.add_argument("guid", nargs="?", default=None, help="one episode; default every staged one")
    check.add_argument("--timeout", type=float, default=20.0)
    check.set_defaults(func=cmd_check)

    ship = subparsers.add_parser("ship", help="add + stage + release + sync for one run")
    ship.add_argument("run_dir")
    ship.add_argument("--season", type=int, default=2)
    ship.add_argument("--episode", type=int, default=None)
    ship.add_argument("--publish-at", default=None)
    ship.add_argument("--no-upload", action="store_true")
    ship.add_argument("--force-encode", action="store_true")
    ship.add_argument("--artwork-source", default=None)
    ship.add_argument("--dry-run", action="store_true")
    ship.add_argument("--allow-removal", action="store_true")
    ship.set_defaults(func=cmd_ship)

    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # The README says the R2 credentials go in `.env`; nothing else in this
    # package reads that file, so without this every upload reports all four
    # variables missing unless the caller exported them by hand. `override` is
    # left at its default, so an explicitly exported value still wins.
    load_dotenv(ENV_FILE)
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except (CliError, ManifestError, StorageError, feed_mod.FeedError, encode_mod.EncodeError,
            artwork_mod.ArtworkError, PublishSheetError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
