"""Self-hosted podcast publishing — manifest, encode, storage, feed.

The show is served from a single Cloudflare R2 bucket behind a custom domain:
MP3 enclosures, the cover art, and `feed.xml` are all objects in that bucket.
There is no server and no build step. Publishing an episode is three writes to
object storage.

`podcast/episodes.json` is the source of truth, not the pipeline output folders
and not the publish sheet. It is committed to git because it holds the GUIDs,
and a GUID is the one value in this system that can never be regenerated: change
one and every subscribed app treats the episode as new and re-downloads it.

The command flow is deliberately four steps rather than one, so a half-failure
says where it stopped::

    add    — register a finished run in the manifest, mint its GUID (once)
    stage  — encode to MP3, tag it, upload it; write back bytes + duration
    release— flip state to published
    sync   — rebuild feed.xml (and the private preview feed) and upload

`stage` precedes `release`, which is what makes the preview feed possible: the
audio is already at a public URL while the episode is still a draft, so a second
XML document over the same objects lets the show be listened to on a phone
before anyone else can see it.
"""

from __future__ import annotations

from dr2_podcast.publish.manifest import (
    DEFAULT_MANIFEST_PATH,
    Episode,
    Manifest,
    ManifestError,
    Show,
    load_manifest,
    save_manifest,
)

__all__ = [
    "DEFAULT_MANIFEST_PATH",
    "Episode",
    "Manifest",
    "ManifestError",
    "Show",
    "load_manifest",
    "save_manifest",
]
