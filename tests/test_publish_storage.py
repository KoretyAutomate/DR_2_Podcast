"""The R2 wrapper, against a stand-in client.

No network, no credentials. What is worth pinning here is not that boto3 works
but that this module always states the things R2 will not infer — above all the
`ContentType` on every upload, because a feed served as
`application/octet-stream` is mis-parsed or refused by clients and nothing in
the pipeline notices.
"""

from __future__ import annotations

import io

import pytest

from dr2_podcast.publish import storage as storage_mod
from dr2_podcast.publish.storage import (
    CONTENT_TYPE_JPEG,
    CONTENT_TYPE_MP3,
    CONTENT_TYPE_RSS,
    R2Config,
    R2Storage,
    StorageError,
    check_byte_range,
)

ENV = {
    "R2_ACCOUNT_ID": "acct",
    "R2_ACCESS_KEY_ID": "key",
    "R2_SECRET_ACCESS_KEY": "secret",
    "R2_BUCKET": "bucket",
}


class FakeClient:
    """Records calls; raises the shape of error botocore raises for a 404."""

    def __init__(self, objects: dict[str, bytes] | None = None):
        self.objects = dict(objects or {})
        self.puts: list[dict] = []
        self.heads: list[str] = []

    def head_bucket(self, Bucket):
        return {}

    def put_object(self, Bucket, Key, Body, ContentType):
        payload = Body if isinstance(Body, bytes) else Body.read()
        self.objects[Key] = payload
        self.puts.append({"key": Key, "content_type": ContentType, "size": len(payload)})

    def head_object(self, Bucket, Key):
        self.heads.append(Key)
        if Key not in self.objects:
            raise _not_found()
        return {"ContentLength": len(self.objects[Key])}

    def get_object(self, Bucket, Key):
        if Key not in self.objects:
            raise _not_found()
        return {"Body": io.BytesIO(self.objects[Key])}

    def copy_object(self, Bucket, Key, CopySource):
        source = CopySource["Key"]
        if source not in self.objects:
            raise _not_found()
        self.objects[Key] = self.objects[source]


def _not_found() -> Exception:
    exc = Exception("Not Found")
    exc.response = {"Error": {"Code": "404"}, "ResponseMetadata": {"HTTPStatusCode": 404}}
    return exc


def _storage(objects=None) -> tuple[R2Storage, FakeClient]:
    client = FakeClient(objects)
    return R2Storage(R2Config.from_env(ENV), client=client), client


def test_missing_credentials_name_every_one_that_is_missing():
    """An agent cannot write .env here, so the error has to be actionable by a human."""
    with pytest.raises(StorageError) as excinfo:
        R2Config.from_env({"R2_ACCOUNT_ID": "acct"})
    message = str(excinfo.value)
    for name in ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET"):
        assert name in message


def test_blank_credentials_count_as_missing():
    with pytest.raises(StorageError, match="R2_BUCKET"):
        R2Config.from_env({**ENV, "R2_BUCKET": "   "})


def test_endpoint_is_derived_from_the_account_id():
    assert R2Config.from_env(ENV).endpoint_url == "https://acct.r2.cloudflarestorage.com"


def test_every_upload_states_its_content_type(tmp_path):
    storage, client = _storage()
    mp3 = tmp_path / "s1e001.mp3"
    mp3.write_bytes(b"\xff\xfb" + b"0" * 500)
    jpeg = tmp_path / "artwork.jpg"
    jpeg.write_bytes(b"\xff\xd8" + b"0" * 100)

    storage.put_file("audio/s1e001.mp3", mp3, CONTENT_TYPE_MP3)
    storage.put_file("artwork.jpg", jpeg, CONTENT_TYPE_JPEG)
    storage.put_bytes("feed.xml", b"<rss/>", CONTENT_TYPE_RSS)

    assert [put["content_type"] for put in client.puts] == [CONTENT_TYPE_MP3, CONTENT_TYPE_JPEG, CONTENT_TYPE_RSS]
    # The feed's charset is part of the type; without it Japanese can be misread.
    assert "charset=utf-8" in CONTENT_TYPE_RSS


def test_put_file_returns_and_uploads_the_real_byte_count(tmp_path):
    storage, client = _storage()
    path = tmp_path / "a.mp3"
    path.write_bytes(b"x" * 1234)
    assert storage.put_file("audio/a.mp3", path, CONTENT_TYPE_MP3) == 1234
    assert client.puts[0]["size"] == 1234


def test_uploading_something_that_is_not_there_is_refused(tmp_path):
    storage, _ = _storage()
    with pytest.raises(StorageError, match="nothing to upload"):
        storage.put_file("audio/a.mp3", tmp_path / "missing.mp3", CONTENT_TYPE_MP3)


def test_a_missing_object_reads_as_absent_not_as_an_error():
    storage, _ = _storage({"feed.xml": b"<rss/>"})
    assert storage.object_size("feed.xml") == 6
    assert storage.object_size("audio/nope.mp3") is None
    assert storage.get_bytes("feed.xml") == b"<rss/>"
    assert storage.get_bytes("feed.xml.prev") is None


def test_a_real_failure_is_not_mistaken_for_absence():
    """Only a 404 means "not there"; anything else must surface."""

    class Failing(FakeClient):
        def head_object(self, Bucket, Key):
            raise RuntimeError("connection reset")

    storage = R2Storage(R2Config.from_env(ENV), client=Failing())
    with pytest.raises(StorageError, match="HEAD"):
        storage.head("feed.xml")


def test_the_previous_feed_is_kept_before_it_is_overwritten():
    """`feed.xml.prev` is the 30-second rollback after a bad sync."""
    storage, client = _storage({"feed.xml": b"<rss>old</rss>"})
    storage.copy("feed.xml", "feed.xml.prev")
    assert client.objects["feed.xml.prev"] == b"<rss>old</rss>"


def test_the_first_publish_has_nothing_to_keep_and_that_is_not_an_error():
    storage, client = _storage()
    storage.copy("feed.xml", "feed.xml.prev")
    assert "feed.xml.prev" not in client.objects


# --- byte-range probe ---------------------------------------------------------


class _FakeStream:
    """What `httpx.stream(...)` yields: a response whose body is never read."""

    def __init__(self, status_code: int, headers: dict[str, str]):
        self.status_code = status_code
        self.headers = headers

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_a_206_with_a_content_range_is_what_a_seekable_enclosure_looks_like(monkeypatch):
    monkeypatch.setattr(
        storage_mod.httpx,
        "stream",
        lambda *a, **k: _FakeStream(206, {"Content-Range": "bytes 0-1023/5000000"}),
    )
    assert check_byte_range("https://media.example/audio/a.mp3") == (206, "bytes 0-1023/5000000")


def test_a_200_reports_itself_rather_than_raising():
    """A server that ignores the range answers 200 with the whole file. That is a
    finding to report, not a transport failure — the caller decides what it means."""
    seen = {}

    def fake_stream(method, url, **kwargs):
        seen.update(kwargs)
        return _FakeStream(200, {})

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(storage_mod.httpx, "stream", fake_stream)
        assert check_byte_range("https://media.example/audio/a.mp3") == (200, "")
    assert seen["headers"] == {"Range": "bytes=0-1023"}


def test_a_non_http_url_cannot_read_the_local_disk(tmp_path):
    """`show.base_url` is configuration. urlopen would serve a `file:` one and
    report a healthy read of the local filesystem; httpx refuses the scheme."""
    local = tmp_path / "secret.mp3"
    local.write_bytes(b"not an enclosure")
    with pytest.raises(StorageError, match="could not fetch"):
        check_byte_range(local.as_uri())


def test_an_unreachable_host_surfaces_as_a_storage_error(monkeypatch):
    def boom(*a, **k):
        raise storage_mod.httpx.ConnectError("no route to host")

    monkeypatch.setattr(storage_mod.httpx, "stream", boom)
    with pytest.raises(StorageError, match="could not fetch"):
        check_byte_range("https://media.example/audio/a.mp3")
