"""The RSS feed, built by hand from the manifest.

This is the highest-risk artifact in the project, because a broken feed is worse
than no feed: directories and apps cache what they fetch, so a malformed
document can take days to flush, and a wrong `<enclosure length>` produces
downloads that stall near the end with no error anywhere.

Built with `xml.etree.ElementTree` rather than `feedgen`, for two reasons. It
would be a new dependency, and its iTunes extension does not cover the
`podcast:` namespace that Podcast Index keys on. Feed correctness is worth
owning outright rather than delegating to a library whose failure modes have to
be learned anyway.

Two feeds come out of the same manifest:

``feed.xml``
    Public. Carries an episode when it is `published` **and** its `publish_at`
    has passed. Filtering here rather than future-dating `<pubDate>` and hoping
    apps hide it is deliberate — client behaviour varies, and some show it
    immediately.

``preview-<token>.xml``
    Private, unguessable URL. Carries every **staged** episode, drafts included,
    so the show can be listened to on a phone before anyone else can see it. It
    is marked `<itunes:block>Yes</itunes:block>` so a directory refuses it if the
    URL ever leaks, and both its channel title and every item title are visibly
    marked, so there is never a question about which of the two is playing.

Free text is escaped by ElementTree rather than CDATA-wrapped. Both are valid
per Apple's guidance ("escape or CDATA-wrap"), and escaping is what the stdlib
serialiser does correctly and consistently; hand-assembling CDATA sections
around Japanese show notes would be a hand-rolled escape path in the one file
that must not have one.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import datetime
from email.utils import format_datetime
from pathlib import Path
from xml.etree.ElementTree import Element

import defusedxml.ElementTree as DefusedET

from dr2_podcast.publish.manifest import Episode, Manifest, ManifestError, Show

ITUNES_NS = "http://www.itunes.com/dtds/podcast-1.0.dtd"
CONTENT_NS = "http://purl.org/rss/1.0/modules/content/"
ATOM_NS = "http://www.w3.org/2005/Atom"
PODCAST_NS = "https://podcastindex.org/namespace/1.0"

# Without these, ElementTree invents `ns0:`/`ns1:` prefixes for the qualified
# tag names below. The prefixes are semantically irrelevant to a conforming
# parser and highly relevant to anyone reading the feed — and to the handful of
# validators that still pattern-match on `itunes:`.
for _prefix, _uri in (("itunes", ITUNES_NS), ("content", CONTENT_NS), ("atom", ATOM_NS), ("podcast", PODCAST_NS)):
    ET.register_namespace(_prefix, _uri)

#: `audio/mp3` is not a registered MIME type and some clients reject it.
ENCLOSURE_TYPE = "audio/mpeg"

#: Marks on the private feed, so a listener always knows which one they are in.
PREVIEW_TITLE_SUFFIX = "（試聴用・未公開）"
PREVIEW_ITEM_PREFIX = "【試聴】"


class FeedError(Exception):
    """The feed could not be built, or the built feed failed its own checks."""


def _sub(parent: Element, tag: str, text: str | None = None, **attrs: str) -> Element:
    child = ET.SubElement(parent, tag, {k: v for k, v in attrs.items() if v is not None})
    if text is not None:
        child.text = text
    return child


def _itunes(tag: str) -> str:
    return f"{{{ITUNES_NS}}}{tag}"


def _channel_head(show: Show, *, preview: bool, self_url: str, now: datetime) -> tuple[Element, Element]:
    # The xmlns declarations are emitted by the serialiser for every namespace
    # the document actually uses, from the registrations above.
    rss = ET.Element("rss", {"version": "2.0"})
    channel = _sub(rss, "channel")

    title = show.title + (PREVIEW_TITLE_SUFFIX if preview else "")
    _sub(channel, "title", title)
    _sub(channel, "description", show.description)
    _sub(channel, "language", show.language)
    _sub(channel, "link", self_url)
    _sub(channel, "lastBuildDate", format_datetime(now))
    if show.copyright:
        _sub(channel, "copyright", show.copyright)

    ET.SubElement(
        channel,
        f"{{{ATOM_NS}}}link",
        {"rel": "self", "type": "application/rss+xml", "href": self_url},
    )

    _sub(channel, _itunes("image"), href=show.artwork_url)
    # The category name is an ATTRIBUTE named `text`, not element content — and
    # its casing is enforced by Apple: "Science", not "science". Built with
    # SubElement directly because `_sub`'s own `text` parameter would otherwise
    # capture the attribute and write it as the element's body.
    category = ET.SubElement(channel, _itunes("category"), {"text": show.category})
    if show.subcategory:
        ET.SubElement(category, _itunes("category"), {"text": show.subcategory})
    _sub(channel, _itunes("explicit"), "true" if show.explicit else "false")
    if show.author:
        _sub(channel, _itunes("author"), show.author)
    if show.owner_email or show.owner_name:
        # Load-bearing: Apple and Spotify mail ownership verification here.
        owner = _sub(channel, _itunes("owner"))
        _sub(owner, _itunes("name"), show.owner_name or show.author)
        _sub(owner, _itunes("email"), show.owner_email)
    _sub(channel, _itunes("type"), show.type)

    if preview:
        # If this URL ever leaks into a directory, the directory refuses it.
        _sub(channel, _itunes("block"), "Yes")
    else:
        if show.podcast_guid:
            _sub(channel, f"{{{PODCAST_NS}}}guid", show.podcast_guid)
        if show.owner_email:
            # Tells other hosts to refuse an import of this feed unless the
            # owner address verifies — cheap insurance on a public feed URL.
            _sub(channel, f"{{{PODCAST_NS}}}locked", "yes", owner=show.owner_email)

    return rss, channel


def _append_item(channel: Element, episode: Episode, show: Show, *, preview: bool) -> None:
    item = _sub(channel, "item")
    _sub(item, "title", (PREVIEW_ITEM_PREFIX if preview else "") + episode.title)
    if episode.subtitle:
        _sub(item, _itunes("subtitle"), episode.subtitle)
    if episode.description:
        _sub(item, "description", episode.description)
        _sub(item, f"{{{CONTENT_NS}}}encoded", episode.description)

    ET.SubElement(
        item,
        "enclosure",
        {
            "url": episode.enclosure_url(show),
            "type": ENCLOSURE_TYPE,
            "length": str(episode.bytes),
        },
    )
    # isPermaLink="false" is not optional: without it the value is read as a URL
    # and episode identity becomes coupled to the domain.
    _sub(item, "guid", episode.guid, isPermaLink="false")
    _sub(item, "pubDate", format_datetime(episode.publish_at_dt))
    _sub(item, _itunes("duration"), str(episode.duration_seconds))
    _sub(item, _itunes("season"), str(episode.season))
    _sub(item, _itunes("episode"), str(episode.episode))
    _sub(item, _itunes("explicit"), "true" if episode.explicit else "false")


def select_episodes(manifest: Manifest, *, preview: bool, now: datetime) -> list[Episode]:
    """Which episodes belong in this feed, newest first.

    The public feed shows what has been released and is due. The preview shows
    everything that has been staged — an episode whose audio is uploaded is
    listenable, and being listenable before release is the entire purpose.
    """
    chosen = manifest.staged() if preview else [ep for ep in manifest.public_at(now) if ep.is_staged]
    return sorted(chosen, key=lambda ep: (ep.publish_at_dt, ep.season, ep.episode), reverse=True)


def build_feed(manifest: Manifest, *, preview: bool = False, now: datetime | None = None) -> bytes:
    """Render the manifest as an RSS document. Returns UTF-8 XML bytes."""
    if now is None:
        raise FeedError("build_feed needs an explicit `now` — the feed's contents depend on it")
    show = manifest.show
    self_url = show.preview_url if preview else show.feed_url

    rss, channel = _channel_head(show, preview=preview, self_url=self_url, now=now)
    for episode in select_episodes(manifest, preview=preview, now=now):
        _append_item(channel, episode, show, preview=preview)

    ET.indent(rss, space="  ")
    body = ET.tostring(rss, encoding="unicode")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n' + body + "\n").encode("utf-8")


# ---------------------------------------------------------------------------
# Checks run against the rendered bytes, before anything is uploaded
# ---------------------------------------------------------------------------


def validate_feed_bytes(xml_bytes: bytes) -> list[dict[str, str]]:
    """Parse and structurally check a rendered feed. Returns its items' facts.

    Deliberately re-parses the serialised bytes rather than inspecting the tree
    that produced them: what gets uploaded is the bytes, and this is the last
    point at which a serialisation problem is still cheap.

    Parsed through `defusedxml` because the same function is pointed at the feed
    fetched back out of the bucket (see :func:`assert_no_episode_disappears`),
    which arrives over the network.
    """
    try:
        root = DefusedET.fromstring(xml_bytes)
    except ET.ParseError as exc:
        raise FeedError(f"generated feed does not parse: {exc}") from exc

    channel = root.find("channel")
    if channel is None:
        raise FeedError("generated feed has no <channel>")
    for required in ("title", "description", "language", "link"):
        node = channel.find(required)
        if node is None or not (node.text or "").strip():
            raise FeedError(f"channel is missing <{required}>")
    image = channel.find(_itunes("image"))
    if image is None or not (image.get("href") or "").startswith("https://"):
        raise FeedError("channel needs an <itunes:image> with an absolute HTTPS href")
    category = channel.find(_itunes("category"))
    if category is None or not (category.get("text") or "").strip():
        # The name is an attribute, not element content. Getting this wrong
        # produces a feed that parses and validates as XML but categorises the
        # show as nothing at all, which Apple rejects at submission.
        raise FeedError("channel needs an <itunes:category> carrying a text attribute")

    items: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in channel.findall("item"):
        guid_node = item.find("guid")
        guid = (guid_node.text or "").strip() if guid_node is not None else ""
        if not guid:
            raise FeedError("an <item> has no <guid>")
        if guid_node is not None and guid_node.get("isPermaLink") != "false":
            raise FeedError(f"<guid> {guid} is missing isPermaLink=\"false\"")
        if guid in seen:
            raise FeedError(f"duplicate <guid> in feed: {guid}")
        seen.add(guid)

        enclosure = item.find("enclosure")
        if enclosure is None:
            raise FeedError(f"item {guid} has no <enclosure>")
        url = enclosure.get("url") or ""
        length = enclosure.get("length") or "0"
        if enclosure.get("type") != ENCLOSURE_TYPE:
            raise FeedError(f"item {guid} has enclosure type {enclosure.get('type')!r}, expected {ENCLOSURE_TYPE}")
        if not url.startswith("https://"):
            raise FeedError(f"item {guid} has a non-HTTPS enclosure URL: {url}")
        if not url.isascii():
            raise FeedError(f"item {guid} has a non-ASCII enclosure URL: {url}")
        if not length.isdigit() or int(length) <= 0:
            raise FeedError(f"item {guid} has enclosure length {length!r}")
        items.append({"guid": guid, "url": url, "length": length})

    return items


def assert_no_episode_disappears(previous_xml: bytes, current_xml: bytes) -> None:
    """Refuse a feed that drops a GUID the last published feed carried.

    An episode silently vanishing from the feed makes it vanish from every app
    that has not already downloaded it, and nothing anywhere reports it.
    """
    before = {item["guid"] for item in validate_feed_bytes(previous_xml)}
    after = {item["guid"] for item in validate_feed_bytes(current_xml)}
    lost = sorted(before - after)
    if lost:
        raise FeedError(
            f"{len(lost)} episode(s) present in the live feed would disappear: {', '.join(lost[:3])}"
            f"{'…' if len(lost) > 3 else ''}. Pass --allow-removal if that is genuinely intended."
        )


def cross_check_manifest(manifest: Manifest, local_files: dict[str, Path]) -> None:
    """Compare each staged episode's recorded `bytes` against the file on disk.

    The manifest's cached numbers are convenient but they are a copy; the file is
    the thing being served. A divergence means an encode ran after the manifest
    was written, and shipping the stale number produces downloads that stall.
    """
    for episode in manifest.staged():
        path = local_files.get(episode.audio_key)
        if path is None or not path.is_file():
            continue
        actual = path.stat().st_size
        if actual != episode.bytes:
            raise ManifestError(
                f"{episode.audio_key} is {actual} bytes on disk but the manifest says {episode.bytes}. "
                "Re-run `stage` for this episode; a wrong <enclosure length> stalls downloads."
            )
