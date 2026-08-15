"""Cloudflare R2, over its S3-compatible API, via boto3.

One bucket holds everything: the MP3 enclosures, the cover art, and the feed
itself. No `wrangler`, no `rclone`, no extra CLI — boto3 is already installed and
R2 speaks S3.

Two things R2 will not guess and that matter to listeners:

* **`ContentType`.** A feed served as `application/octet-stream` is refused or
  mis-parsed by some clients, and an MP3 served without `audio/mpeg` will not
  play inline in others. Every upload here states its type; there is no
  `mimetypes.guess_type` fallback, because the failure it produces is invisible
  until a directory rejects the feed.
* **Byte-range support.** Apple's validation checks it and every app relies on
  it to seek. R2 behind a custom domain supports single-range requests, which is
  all that is needed — but that is a property of the deployment, not of this
  code, so :func:`check_byte_range` exists to prove it against the live URL
  rather than assume it.

Credentials come from the environment and are never written to the manifest;
`show.base_url` is public and lives there instead.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

CONTENT_TYPE_MP3 = "audio/mpeg"
CONTENT_TYPE_JPEG = "image/jpeg"
CONTENT_TYPE_RSS = "application/rss+xml; charset=utf-8"

_ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET")


class StorageError(Exception):
    """The bucket could not be reached, or an object is not what it should be."""


@dataclass(frozen=True)
class R2Config:
    account_id: str
    access_key_id: str
    secret_access_key: str
    bucket: str

    @property
    def endpoint_url(self) -> str:
        return f"https://{self.account_id}.r2.cloudflarestorage.com"

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> R2Config:
        """Read the four R2 variables, naming every one that is missing.

        These live in `.env`, which an agent may not write in this environment —
        so the error has to be actionable by a human reading it, which means
        listing all the missing names at once rather than the first.
        """
        source = os.environ if env is None else env
        values = {name: (source.get(name) or "").strip() for name in _ENV_VARS}
        missing = [name for name, value in values.items() if not value]
        if missing:
            raise StorageError(
                f"missing R2 credentials in the environment: {', '.join(missing)}. "
                "Add them to .env (see .env.example); they are the only thing this step needs."
            )
        return cls(
            account_id=values["R2_ACCOUNT_ID"],
            access_key_id=values["R2_ACCESS_KEY_ID"],
            secret_access_key=values["R2_SECRET_ACCESS_KEY"],
            bucket=values["R2_BUCKET"],
        )


class R2Storage:
    """Thin, explicit wrapper over the handful of S3 calls this project makes."""

    def __init__(self, config: R2Config, client: Any = None):
        self.config = config
        self._client = client

    @property
    def client(self) -> Any:
        if self._client is None:
            import boto3  # imported lazily: the manifest and feed steps need no credentials

            self._client = boto3.client(
                "s3",
                endpoint_url=self.config.endpoint_url,
                aws_access_key_id=self.config.access_key_id,
                aws_secret_access_key=self.config.secret_access_key,
                region_name="auto",
            )
        return self._client

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> R2Storage:
        return cls(R2Config.from_env(env))

    def check_bucket(self) -> None:
        """Fail now, with a clear message, rather than on the first upload."""
        try:
            self.client.head_bucket(Bucket=self.config.bucket)
        except Exception as exc:  # botocore raises a family of client errors
            raise StorageError(f"cannot reach bucket {self.config.bucket}: {exc}") from exc

    def put_file(self, key: str, path: Path | str, content_type: str) -> int:
        """Upload a file and return its byte count."""
        path = Path(path)
        if not path.is_file():
            raise StorageError(f"nothing to upload at {path}")
        size = path.stat().st_size
        with path.open("rb") as handle:
            self._put(key, handle, content_type)
        logger.info("uploaded %s (%.1f MB) → %s", path.name, size / 1e6, key)
        return size

    def put_bytes(self, key: str, payload: bytes, content_type: str) -> int:
        self._put(key, payload, content_type)
        logger.info("uploaded %d bytes → %s", len(payload), key)
        return len(payload)

    def _put(self, key: str, body: Any, content_type: str) -> None:
        try:
            self.client.put_object(
                Bucket=self.config.bucket,
                Key=key,
                Body=body,
                ContentType=content_type,
            )
        except Exception as exc:
            raise StorageError(f"upload of {key} failed: {exc}") from exc

    def head(self, key: str) -> dict[str, Any] | None:
        """Object metadata, or None if it is not there."""
        try:
            return self.client.head_object(Bucket=self.config.bucket, Key=key)
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise StorageError(f"HEAD {key} failed: {exc}") from exc

    def get_bytes(self, key: str) -> bytes | None:
        """Fetch an object's body, or None if it is not there.

        Used to read back the feed that is currently live, so a new one can be
        checked against it before it replaces it.
        """
        try:
            return self.client.get_object(Bucket=self.config.bucket, Key=key)["Body"].read()
        except Exception as exc:
            if _is_not_found(exc):
                return None
            raise StorageError(f"GET {key} failed: {exc}") from exc

    def object_size(self, key: str) -> int | None:
        meta = self.head(key)
        return None if meta is None else int(meta["ContentLength"])

    def copy(self, source_key: str, dest_key: str) -> None:
        """Server-side copy — used to keep `feed.xml.prev` before overwriting."""
        try:
            self.client.copy_object(
                Bucket=self.config.bucket,
                Key=dest_key,
                CopySource={"Bucket": self.config.bucket, "Key": source_key},
            )
        except Exception as exc:
            if _is_not_found(exc):
                logger.info("no %s to keep a copy of — first publish", source_key)
                return
            raise StorageError(f"copy {source_key} → {dest_key} failed: {exc}") from exc


def _is_not_found(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return False
    code = str(response.get("Error", {}).get("Code", ""))
    status = response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def check_byte_range(url: str, *, timeout: float = 20.0) -> tuple[int, str]:
    """Ask the live URL for the first kilobyte. Returns `(status, Content-Range)`.

    Apple's validation checks byte-range support and every podcast app uses it to
    seek, so a 200 here — the whole file, range ignored — means scrubbing is
    broken for every listener even though playback from the start works.
    """
    request = Request(url, method="GET", headers={"Range": "bytes=0-1023"})
    try:
        with urlopen(request, timeout=timeout) as response:
            return response.status, response.headers.get("Content-Range", "")
    except HTTPError as exc:
        return exc.code, exc.headers.get("Content-Range", "") if exc.headers else ""
    except URLError as exc:
        raise StorageError(f"could not fetch {url}: {exc}") from exc
