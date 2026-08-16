"""What the feed must say, and what it must never say.

A feed is the one artifact in this project that other people's software caches.
A malformed document can sit in a directory for days after it is fixed, and a
wrong `<enclosure length>` produces downloads that stall near the end with no
error surfaced anywhere. So these tests read the *serialised bytes* rather than
the tree that produced them — a bug in serialisation is invisible to a test that
inspects the tree, and the bytes are what gets uploaded.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import defusedxml.ElementTree as DefusedET
import pytest

from dr2_podcast.publish.feed import (
    ATOM_NS,
    ENCLOSURE_TYPE,
    ITUNES_NS,
    PODCAST_NS,
    PREVIEW_ITEM_PREFIX,
    PREVIEW_TITLE_SUFFIX,
    FeedError,
    assert_no_episode_disappears,
    build_feed,
    validate_feed_bytes,
)
from dr2_podcast.publish.manifest import STATE_PUBLISHED, Episode, Manifest, Show, mint_guid, parse_publish_at

NOW = parse_publish_at("2026-08-20T12:00:00+09:00")


def _manifest(count: int = 3, *, staged: bool = True, published: bool = True) -> Manifest:
    manifest = Manifest(
        show=Show(
            title="仕組み化パパの若返りラボ",
            description="説明 & 記号 <b>",
            author="仕組み化パパ",
            owner_name="仕組み化パパ",
            owner_email="owner@example.com",
            subcategory="Life Sciences",
            base_url="https://media.example.org",
            copyright="© 2026",
        )
    )
    manifest.ensure_preview_token()
    for n in range(1, count + 1):
        manifest.add_episode(
            Episode(
                guid=mint_guid(),
                season=1,
                episode=n,
                title=f"第{n}回 テスト",
                description=f"<p>第{n}回のショーノート</p>",
                audio_key=f"audio/s1e{n:03d}.mp3",
                run_dir=f"research_outputs/Ep{n:03d}_topic",
                publish_at=f"2026-08-{n:02d}T06:00:00+09:00",
                bytes=24_000_000 + n if staged else 0,
                duration_seconds=1500 + n if staged else 0,
                state=STATE_PUBLISHED if published else "draft",
            )
        )
    return manifest


def _parse(xml_bytes: bytes) -> ET.Element:
    return DefusedET.fromstring(xml_bytes)


def _channel(xml_bytes: bytes) -> ET.Element:
    channel = _parse(xml_bytes).find("channel")
    assert channel is not None
    return channel


# --- structure ---------------------------------------------------------------


def test_feed_declares_utf8_and_survives_a_round_trip():
    xml_bytes = build_feed(_manifest(), now=NOW)
    assert xml_bytes.startswith(b'<?xml version="1.0" encoding="UTF-8"?>')
    # Japanese content is the whole show; a mis-declared encoding is mojibake
    # in every app, and the failure is visible only to a listener.
    assert "仕組み化パパの若返りラボ" in xml_bytes.decode("utf-8")
    assert _channel(xml_bytes).findtext("title") == "仕組み化パパの若返りラボ"


def test_free_text_with_markup_is_escaped_not_injected():
    """`&` and `<` in show notes must not break out of their element."""
    channel = _channel(build_feed(_manifest(), now=NOW))
    assert channel.findtext("description") == "説明 & 記号 <b>"


def test_channel_carries_every_element_apple_requires():
    channel = _channel(build_feed(_manifest(), now=NOW))
    assert channel.findtext("language") == "ja"
    assert channel.findtext("link")
    assert channel.find(f"{{{ITUNES_NS}}}image").get("href") == "https://media.example.org/artwork.jpg"
    assert channel.find(f"{{{ITUNES_NS}}}explicit").text == "false"
    assert channel.find(f"{{{ATOM_NS}}}link").get("href") == "https://media.example.org/feed.xml"


def test_category_name_is_an_attribute_not_element_content():
    """Regression: `text=` as element content produces an uncategorised show.

    The XML still parses and still validates as XML — Apple simply sees no
    category and rejects the submission.
    """
    channel = _channel(build_feed(_manifest(), now=NOW))
    category = channel.find(f"{{{ITUNES_NS}}}category")
    assert category.get("text") == "Science"
    assert (category.text or "").strip() == ""
    assert category.find(f"{{{ITUNES_NS}}}category").get("text") == "Life Sciences"


def test_owner_email_is_present_because_three_directories_mail_it():
    owner = _channel(build_feed(_manifest(), now=NOW)).find(f"{{{ITUNES_NS}}}owner")
    assert owner.find(f"{{{ITUNES_NS}}}email").text == "owner@example.com"


def test_public_feed_is_locked_to_the_owner_address():
    locked = _channel(build_feed(_manifest(), now=NOW)).find(f"{{{PODCAST_NS}}}locked")
    assert locked.text == "yes"
    assert locked.get("owner") == "owner@example.com"


# --- items -------------------------------------------------------------------


def test_enclosure_carries_all_three_required_attributes():
    manifest = _manifest(1)
    item = _channel(build_feed(manifest, now=NOW)).find("item")
    enclosure = item.find("enclosure")
    assert enclosure.get("type") == ENCLOSURE_TYPE
    assert enclosure.get("length") == str(manifest.episodes[0].bytes)
    assert enclosure.get("url") == "https://media.example.org/audio/s1e001.mp3"


def test_guid_is_marked_not_a_permalink():
    """Without isPermaLink="false" the GUID is read as a URL and couples
    episode identity to the domain."""
    item = _channel(build_feed(_manifest(1), now=NOW)).find("item")
    assert item.find("guid").get("isPermaLink") == "false"


def test_pubdate_is_rfc2822_with_a_real_offset():
    item = _channel(build_feed(_manifest(1), now=NOW)).find("item")
    pub_date = item.findtext("pubDate")
    assert pub_date.startswith("Sat, 01 Aug 2026 06:00:00 ")
    # A naive datetime serialises as -0000, which Apple reads as unknown local time.
    assert pub_date.endswith("+0900")


def test_items_are_newest_first():
    numbers = [
        int(item.find(f"{{{ITUNES_NS}}}episode").text)
        for item in _channel(build_feed(_manifest(3), now=NOW)).findall("item")
    ]
    assert numbers == [3, 2, 1]


# --- which episodes appear ----------------------------------------------------


def test_draft_episodes_never_reach_the_public_feed():
    manifest = _manifest(3, published=False)
    assert _channel(build_feed(manifest, now=NOW)).findall("item") == []


def test_an_episode_not_yet_due_is_withheld_rather_than_future_dated():
    """Client behaviour on a future pubDate varies; filtering here does not."""
    manifest = _manifest(2)
    manifest.episodes[1].publish_at = "2026-09-01T06:00:00+09:00"
    items = _channel(build_feed(manifest, now=NOW)).findall("item")
    assert len(items) == 1


def test_a_published_but_unstaged_episode_is_withheld():
    """Better absent than present with a 404 enclosure a listener taps."""
    manifest = _manifest(2)
    manifest.episodes[0].bytes = 0
    manifest.episodes[0].duration_seconds = 0
    items = _channel(build_feed(manifest, now=NOW)).findall("item")
    assert len(items) == 1


# --- the preview feed ---------------------------------------------------------


def test_preview_carries_drafts_while_the_public_feed_is_empty():
    """The two facts that together prove draft and published are separate."""
    manifest = _manifest(14, published=False)
    assert _channel(build_feed(manifest, preview=False, now=NOW)).findall("item") == []
    assert len(_channel(build_feed(manifest, preview=True, now=NOW)).findall("item")) == 14


def test_preview_is_visibly_marked_in_channel_and_item_titles():
    channel = _channel(build_feed(_manifest(1, published=False), preview=True, now=NOW))
    assert channel.findtext("title").endswith(PREVIEW_TITLE_SUFFIX)
    assert channel.find("item").findtext("title").startswith(PREVIEW_ITEM_PREFIX)


def test_preview_is_blocked_from_directories_and_carries_no_show_guid():
    channel = _channel(build_feed(_manifest(1, published=False), preview=True, now=NOW))
    assert channel.find(f"{{{ITUNES_NS}}}block").text == "Yes"
    assert channel.find(f"{{{PODCAST_NS}}}guid") is None


def test_preview_self_link_is_the_token_url_not_the_public_one():
    manifest = _manifest(1)
    channel = _channel(build_feed(manifest, preview=True, now=NOW))
    assert channel.find(f"{{{ATOM_NS}}}link").get("href") == manifest.show.preview_url
    assert manifest.show.preview_token in manifest.show.preview_url


def test_preview_and_public_share_guids():
    """Two feed URLs are two subscriptions, so identical GUIDs cannot collide —
    and keeping them identical keeps one GUID per episode in the manifest."""
    manifest = _manifest(2)

    def guids(preview: bool) -> set[str]:
        channel = _channel(build_feed(manifest, preview=preview, now=NOW))
        return {item.findtext("guid", "") for item in channel.findall("item")}

    preview_guids = guids(preview=True)
    # The empty-string default is what `findtext` returns for a missing <guid>.
    # Asserting it away first stops "both feeds are equally broken" reading as a pass.
    assert "" not in preview_guids
    assert len(preview_guids) == 2
    assert preview_guids == guids(preview=False)


# --- the checks that run before an upload -------------------------------------


def test_validate_accepts_a_feed_this_module_produced():
    items = validate_feed_bytes(build_feed(_manifest(3), now=NOW))
    assert len(items) == 3


def test_validate_rejects_a_zero_length_enclosure():
    broken = build_feed(_manifest(1), now=NOW).replace(b'length="24000001"', b'length="0"')
    with pytest.raises(FeedError, match="enclosure length"):
        validate_feed_bytes(broken)


def test_validate_rejects_the_wrong_enclosure_mime_type():
    """`audio/mp3` is not a registered MIME type and some clients reject it."""
    broken = build_feed(_manifest(1), now=NOW).replace(b'type="audio/mpeg"', b'type="audio/mp3"')
    with pytest.raises(FeedError, match="enclosure type"):
        validate_feed_bytes(broken)


def test_validate_rejects_a_non_https_enclosure():
    broken = build_feed(_manifest(1), now=NOW).replace(b"https://media.example.org/audio", b"http://media.example.org/audio")
    with pytest.raises(FeedError, match="non-HTTPS"):
        validate_feed_bytes(broken)


def test_validate_rejects_a_duplicate_guid():
    manifest = _manifest(2)
    xml_bytes = build_feed(manifest, now=NOW)
    broken = xml_bytes.replace(manifest.episodes[0].guid.encode(), manifest.episodes[1].guid.encode())
    with pytest.raises(FeedError, match="duplicate"):
        validate_feed_bytes(broken)


def test_validate_rejects_unparseable_bytes():
    with pytest.raises(FeedError, match="does not parse"):
        validate_feed_bytes(b"<rss><channel></rss>")


def test_an_episode_disappearing_from_the_live_feed_is_refused():
    """It would vanish from every app that had not already downloaded it."""
    before = build_feed(_manifest(3), now=NOW)
    after = build_feed(_manifest(2), now=NOW)
    with pytest.raises(FeedError, match="would disappear"):
        assert_no_episode_disappears(before, after)


def test_adding_an_episode_is_not_a_disappearance():
    manifest = _manifest(2)
    before = build_feed(manifest, now=NOW)
    manifest.add_episode(
        Episode(
            guid=mint_guid(),
            season=1,
            episode=3,
            title="第3回",
            description="notes",
            audio_key="audio/s1e003.mp3",
            run_dir="research_outputs/Ep003_topic",
            publish_at="2026-08-03T06:00:00+09:00",
            bytes=1000,
            duration_seconds=100,
            state=STATE_PUBLISHED,
        )
    )
    assert_no_episode_disappears(before, build_feed(manifest, now=NOW))


def test_build_feed_demands_an_explicit_now():
    """The feed's contents depend on the clock; an implicit one hides that."""
    with pytest.raises(FeedError, match="explicit"):
        build_feed(_manifest(1))
