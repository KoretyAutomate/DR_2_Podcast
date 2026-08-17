"""The encoder, and the one failure it exists to catch.

A truncated MP3 is a valid MP3. It opens, it plays, it is simply short — so
"the file exists and is non-empty" proves nothing, and every check here is
against the *decoded* output instead.

These tests really encode. PyAV is in-process and a few seconds of tone takes
milliseconds, so there is no fixture-vs-reality gap to worry about: if
`libmp3lame` disappeared from the installed PyAV, these fail, which is exactly
what should happen.
"""

from __future__ import annotations

import math
import struct
import wave

import pytest

from dr2_podcast.publish.encode import (
    DEFAULT_BITRATE,
    SAMPLE_RATE,
    EncodeError,
    Id3Tags,
    encode_and_tag,
    encode_mp3,
    media_duration_seconds,
    tag_mp3,
    verify_encode,
    wav_duration_seconds,
)


def _write_tone(path, seconds: float = 2.0, rate: int = SAMPLE_RATE):
    """A mono 16-bit WAV, shaped like the pipeline's own output."""
    frames = int(rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(
            b"".join(struct.pack("<h", int(12000 * math.sin(2 * math.pi * 440 * t / rate))) for t in range(frames))
        )
    return path


def test_wav_duration_comes_from_the_header(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav", seconds=2.5)
    assert wav_duration_seconds(wav) == pytest.approx(2.5, abs=0.01)


def test_encode_produces_a_mono_mp3_at_the_requested_bitrate(tmp_path):
    from mutagen.mp3 import MP3

    wav = _write_tone(tmp_path / "tone.wav", seconds=3.0)
    mp3 = encode_mp3(wav, tmp_path / "out.mp3")

    info = MP3(str(mp3)).info
    assert info.channels == 1
    assert info.sample_rate == SAMPLE_RATE
    # Encoding stereo would double the size for a source that has one channel.
    assert info.bitrate == pytest.approx(DEFAULT_BITRATE, rel=0.05)


def test_encoded_duration_matches_the_source(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav", seconds=3.0)
    mp3 = encode_mp3(wav, tmp_path / "out.mp3")
    assert media_duration_seconds(mp3) == pytest.approx(wav_duration_seconds(wav), abs=0.1)
    assert verify_encode(wav, mp3) == pytest.approx(3.0, abs=0.1)


def test_a_truncated_encode_is_refused_even_though_it_plays(tmp_path):
    """The whole reason `verify_encode` decodes rather than stats the file."""
    long_wav = _write_tone(tmp_path / "long.wav", seconds=6.0)
    short_wav = _write_tone(tmp_path / "short.wav", seconds=2.0)
    truncated = encode_mp3(short_wav, tmp_path / "out.mp3")

    # It is a perfectly good MP3 — that is the trap.
    assert media_duration_seconds(truncated) == pytest.approx(2.0, abs=0.1)
    with pytest.raises(EncodeError, match="truncated"):
        verify_encode(long_wav, truncated)


def test_an_empty_output_is_refused(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav", seconds=1.0)
    empty = tmp_path / "empty.mp3"
    empty.write_bytes(b"")
    with pytest.raises(EncodeError, match="missing or empty"):
        verify_encode(wav, empty)


def test_a_missing_source_is_refused(tmp_path):
    with pytest.raises(EncodeError, match="no such WAV"):
        encode_mp3(tmp_path / "nope.wav", tmp_path / "out.mp3")


def test_an_interrupted_encode_leaves_no_usable_partial(tmp_path):
    """A half-file at the target path is one a later `stage` would upload."""
    not_audio = tmp_path / "broken.wav"
    not_audio.write_bytes(b"RIFF" + b"\x00" * 64)
    with pytest.raises(EncodeError):
        encode_mp3(not_audio, tmp_path / "out.mp3")
    assert not (tmp_path / "out.mp3").exists()
    assert not (tmp_path / "out.mp3.partial").exists()


def test_tags_round_trip_including_japanese_and_cover_art(tmp_path):
    from mutagen.id3 import ID3

    wav = _write_tone(tmp_path / "tone.wav", seconds=1.0)
    mp3 = encode_mp3(wav, tmp_path / "out.mp3")
    cover = tmp_path / "cover.jpg"
    _write_jpeg(cover)

    tag_mp3(
        mp3,
        Id3Tags(title="第1回 テスト", artist="仕組み化パパ", album="季節1", track=1, year="2026", cover_jpeg=cover),
    )

    id3 = ID3(str(mp3))
    assert id3.getall("TIT2")[0].text == ["第1回 テスト"]
    assert id3.getall("TPE1")[0].text == ["仕組み化パパ"]
    assert id3.getall("TCON")[0].text == ["Podcast"]
    assert id3.getall("APIC")[0].mime == "image/jpeg"


def test_byte_count_is_taken_after_tagging(tmp_path):
    """A length measured before the APIC frame does not match the file served."""
    wav = _write_tone(tmp_path / "tone.wav", seconds=1.0)
    cover = tmp_path / "cover.jpg"
    _write_jpeg(cover, edge=600)

    untagged = encode_mp3(wav, tmp_path / "plain.mp3").stat().st_size
    size, duration = encode_and_tag(
        wav,
        tmp_path / "tagged.mp3",
        Id3Tags(title="t", artist="a", album="b", track=1, year="2026", cover_jpeg=cover),
    )
    assert size > untagged
    assert size == (tmp_path / "tagged.mp3").stat().st_size
    assert duration == 1


def test_a_missing_cover_is_refused_rather_than_skipped(tmp_path):
    wav = _write_tone(tmp_path / "tone.wav", seconds=1.0)
    mp3 = encode_mp3(wav, tmp_path / "out.mp3")
    with pytest.raises(EncodeError, match="cover art not found"):
        tag_mp3(mp3, Id3Tags(title="t", artist="a", album="b", track=1, year="2026", cover_jpeg=tmp_path / "nope.jpg"))


def test_reencoding_is_skippable_because_the_output_is_deterministic_enough(tmp_path):
    """Two encodes of the same source agree on duration, which is what `stage`
    re-checks when it finds an MP3 already built."""
    wav = _write_tone(tmp_path / "tone.wav", seconds=2.0)
    first = media_duration_seconds(encode_mp3(wav, tmp_path / "a.mp3"))
    second = media_duration_seconds(encode_mp3(wav, tmp_path / "b.mp3"))
    assert first == pytest.approx(second, abs=0.05)


def _write_jpeg(path, edge: int = 200):
    from PIL import Image

    Image.new("RGB", (edge, edge), (30, 60, 90)).save(path, format="JPEG", quality=80)
    return path


def test_a_file_chopped_after_encoding_is_refused_despite_its_header(tmp_path):
    """Codex review 2026-08-16, verified. The test above encodes a SHORTER
    source, so the MP3's header honestly describes it — which is not the
    dangerous case. This is: encode the full episode, then lose the tail. The
    Xing/container duration was written before the truncation, so the header
    still claims the whole episode while the packets no longer carry it.

    media_duration_seconds used to return that declared duration without
    decoding, so verify_encode compared the stale header against the WAV, found
    them equal, and passed a file that stops early — the exact failure its
    docstring promises to catch.
    """
    wav = _write_tone(tmp_path / "full.wav", seconds=6.0)
    mp3 = encode_mp3(wav, tmp_path / "out.mp3")
    assert verify_encode(wav, mp3) == pytest.approx(6.0, abs=0.2)  # intact: accepted

    whole = mp3.read_bytes()
    mp3.write_bytes(whole[: len(whole) // 3])  # header intact, audio tail gone

    decoded = media_duration_seconds(mp3)
    assert decoded < 5.0, f"duration must come from the packets, got {decoded:.2f}s"
    with pytest.raises(EncodeError, match="truncated"):
        verify_encode(wav, mp3)
