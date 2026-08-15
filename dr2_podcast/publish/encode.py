"""WAV → MP3, and the ID3 tags that go on the result.

The pipeline's deliverable is a 44.1 kHz mono 16-bit WAV of 100–150 MB. That is
not shippable as a podcast enclosure: Spotify's delivery spec accepts MP3 only,
at 96–320 kbps, and recommends staying under 200 MB per file. 128 kbps mono is
the choice — comfortably inside the accepted window rather than sitting on the
96 boundary, no wasted bitrate on a second channel the source does not have,
and ~24 MB for a 25-minute episode.

There is no `ffmpeg` binary on this machine and no `sudo` to install one, so the
encode goes through PyAV, which bundles its own libraries and has `libmp3lame`.
`pydub` is installed but shells out to ffmpeg, so it is unusable here.

The failure this module is built to catch is a **truncated encode**. It produces
a perfectly valid, perfectly playable, short MP3 — nothing raises, the file
exists, and the episode simply stops early for every listener. So the exit
criterion is never "the file exists": it is "the decoded duration matches the
source within a second", checked by decoding the output.
"""

from __future__ import annotations

import logging
import wave
from dataclasses import dataclass
from pathlib import Path

import av

logger = logging.getLogger(__name__)

#: 128 kbps mono. See the module docstring for why not 96 and why not stereo.
DEFAULT_BITRATE = 128_000
SAMPLE_RATE = 44_100

#: How far the encoded MP3's decoded duration may sit from the source WAV's.
#: A real truncation is off by minutes; MP3 frame padding is off by milliseconds.
DURATION_TOLERANCE_SECONDS = 1.0


class EncodeError(Exception):
    """The encode did not produce a file that can be shipped."""


@dataclass(frozen=True)
class Id3Tags:
    """What a listener sees on a car head unit or in a downloaded file.

    Cosmetic — no directory requires any of it — but free, and the alternative
    is an episode that shows up as `s1e001.mp3` with no artist.
    """

    title: str
    artist: str
    album: str
    track: int
    year: str
    genre: str = "Podcast"
    cover_jpeg: Path | None = None


def wav_duration_seconds(wav_path: Path | str) -> float:
    """Source duration from the WAV header — frames over frame rate."""
    with wave.open(str(wav_path), "rb") as handle:
        rate = handle.getframerate()
        if rate <= 0:
            raise EncodeError(f"{wav_path} reports a frame rate of {rate}")
        return handle.getnframes() / float(rate)


def media_duration_seconds(path: Path | str) -> float:
    """Decoded duration of any media file PyAV can open.

    Prefers the container's declared duration and falls back to decoding to the
    last frame, because a container written by a crashed encoder can carry a
    duration that its packets do not back up.
    """
    with av.open(str(path)) as container:
        if container.duration:
            return container.duration / av.time_base
        last = 0.0
        for frame in container.decode(audio=0):
            if frame.time is not None:
                last = frame.time + (frame.samples / float(frame.sample_rate or SAMPLE_RATE))
        return last


def encode_mp3(wav_path: Path | str, mp3_path: Path | str, *, bitrate: int = DEFAULT_BITRATE) -> Path:
    """Encode one WAV to a mono MP3 at `bitrate`, overwriting any existing file.

    The resampler to `s16p`/mono is load-bearing rather than defensive: libmp3lame
    will not accept the decoded frames' native format directly.
    """
    wav_path, mp3_path = Path(wav_path), Path(mp3_path)
    if not wav_path.is_file():
        raise EncodeError(f"no such WAV: {wav_path}")
    mp3_path.parent.mkdir(parents=True, exist_ok=True)

    # Encode beside the target and move into place, so an interrupted run leaves
    # no half-file that a later `stage` would happily upload.
    partial = mp3_path.with_suffix(mp3_path.suffix + ".partial")
    try:
        # `format="mp3"` is explicit because the partial file's extension is
        # `.mp3.partial`, from which PyAV cannot infer the container.
        with av.open(str(wav_path)) as inp, av.open(str(partial), "w", format="mp3") as out:
            stream = out.add_stream("libmp3lame", rate=SAMPLE_RATE)
            stream.bit_rate = bitrate
            stream.layout = "mono"
            resampler = av.audio.resampler.AudioResampler(format="s16p", layout="mono", rate=SAMPLE_RATE)
            for frame in inp.decode(audio=0):
                frame.pts = None
                for resampled in resampler.resample(frame):
                    for packet in stream.encode(resampled):
                        out.mux(packet)
            for packet in stream.encode(None):
                out.mux(packet)
    except av.FFmpegError as exc:
        partial.unlink(missing_ok=True)
        raise EncodeError(f"encoding {wav_path} failed: {exc}") from exc

    partial.replace(mp3_path)
    return mp3_path


def verify_encode(wav_path: Path | str, mp3_path: Path | str) -> float:
    """Confirm the MP3 is the whole episode. Returns its duration in seconds.

    This is the check that catches the silent failure mode; see the module
    docstring. It decodes the output rather than trusting that the encode loop
    ran to completion.
    """
    mp3_path = Path(mp3_path)
    if not mp3_path.is_file() or mp3_path.stat().st_size == 0:
        raise EncodeError(f"{mp3_path} is missing or empty")
    source = wav_duration_seconds(wav_path)
    encoded = media_duration_seconds(mp3_path)
    drift = abs(encoded - source)
    if drift > DURATION_TOLERANCE_SECONDS:
        raise EncodeError(
            f"{mp3_path.name} is {encoded:.1f}s but {Path(wav_path).name} is {source:.1f}s "
            f"({drift:.1f}s apart, tolerance {DURATION_TOLERANCE_SECONDS}s). "
            "A truncated encode plays fine and stops early — not shipping this."
        )
    return encoded


def tag_mp3(mp3_path: Path | str, tags: Id3Tags) -> None:
    """Write ID3v2 tags, including cover art, onto an already-encoded MP3.

    Call this **before** reading the byte count for `<enclosure length>`: an
    APIC frame is a few hundred kilobytes, and a length taken before tagging is
    a length that does not match the file being served.
    """
    from mutagen.id3 import APIC, ID3, TALB, TCON, TDRC, TIT2, TPE1, TRCK, ID3NoHeaderError

    mp3_path = Path(mp3_path)
    try:
        id3 = ID3(str(mp3_path))
    except ID3NoHeaderError:
        # A freshly encoded MP3 has no tag block yet; that is the normal path.
        id3 = ID3()

    id3.setall("TIT2", [TIT2(encoding=3, text=[tags.title])])
    id3.setall("TPE1", [TPE1(encoding=3, text=[tags.artist])])
    id3.setall("TALB", [TALB(encoding=3, text=[tags.album])])
    id3.setall("TRCK", [TRCK(encoding=3, text=[str(tags.track)])])
    id3.setall("TDRC", [TDRC(encoding=3, text=[tags.year])])
    id3.setall("TCON", [TCON(encoding=3, text=[tags.genre])])
    if tags.cover_jpeg is not None:
        cover = Path(tags.cover_jpeg)
        if not cover.is_file():
            raise EncodeError(f"cover art not found: {cover}")
        id3.setall(
            "APIC",
            [APIC(encoding=3, mime="image/jpeg", type=3, desc="Cover", data=cover.read_bytes())],
        )
    id3.save(str(mp3_path), v2_version=3)


def encode_and_tag(
    wav_path: Path | str,
    mp3_path: Path | str,
    tags: Id3Tags | None = None,
    *,
    bitrate: int = DEFAULT_BITRATE,
) -> tuple[int, int]:
    """Encode, tag, verify. Returns `(bytes, duration_seconds)` for the manifest.

    The byte count is taken last, after tagging, because that is the number the
    feed has to state and the tags change it.
    """
    mp3_path = Path(mp3_path)
    encode_mp3(wav_path, mp3_path, bitrate=bitrate)
    duration = verify_encode(wav_path, mp3_path)
    if tags is not None:
        tag_mp3(mp3_path, tags)
        # Tagging rewrites the file; re-confirm it still decodes to full length.
        duration = verify_encode(wav_path, mp3_path)
    size = mp3_path.stat().st_size
    logger.info("encoded %s → %s (%.1f MB, %.1f s)", Path(wav_path).name, mp3_path.name, size / 1e6, duration)
    return size, int(round(duration))
