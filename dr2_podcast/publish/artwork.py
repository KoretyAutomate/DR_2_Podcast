"""Show cover art, sized and compressed to what the directories accept.

Apple's window is 1400×1400 to 3000×3000, square, RGB, JPEG or PNG. The byte
budget — around 512 KB — is the one figure here that comes from hosting-provider
documentation rather than from Apple directly (Apple's own spec page is
JavaScript-rendered and serves nothing to a fetch), so it is treated as a target
to stay under rather than a quoted limit. The validators run in Step 7 are what
actually decide.

`Logo.png` is 1536×1536 RGB at 2376 KB — the right shape, roughly five times the
byte budget. So this re-encodes rather than copies, walking JPEG quality down
until the file fits and refusing to ship something that never got there.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image, ImageFile

logger = logging.getLogger(__name__)

#: Third-party-sourced; see the module docstring.
TARGET_MAX_BYTES = 512 * 1024

#: Apple's accepted square range.
MIN_EDGE = 1400
MAX_EDGE = 3000

#: Quality is stepped down from the first value. Below the last, the plan's
#: instruction is to keep the native 1536 px rather than keep crushing a 3000 px
#: upscale — an upscale that needs quality 75 to fit looks worse than a native
#: image at 88 and is no more compliant.
_QUALITY_LADDER = (88, 85, 82, 80)


class ArtworkError(Exception):
    """The cover art could not be produced within the directories' constraints."""


def _encode_at(image: Image.Image, target: Path, quality: int) -> int:
    target.parent.mkdir(parents=True, exist_ok=True)
    # With `optimize=True` libjpeg needs the whole image in one buffer, and
    # Pillow only guesses `width × height` bytes for it. A detailed image can
    # exceed that guess, and when it does the save dies partway through with
    # "broken data stream when writing image file" — leaving a truncated JPEG at
    # the target path, which is the shape of failure this module exists to
    # prevent. Four bytes per pixel is the same headroom Pillow itself uses for
    # CMYK, and 36 MB at 3000×3000 is not worth economising on.
    previous_maxblock = ImageFile.MAXBLOCK
    ImageFile.MAXBLOCK = max(previous_maxblock, image.width * image.height * 4)
    try:
        image.save(target, format="JPEG", quality=quality, optimize=True, progressive=False, subsampling=0)
    except OSError as exc:
        target.unlink(missing_ok=True)
        raise ArtworkError(
            f"JPEG encode of a {image.width}×{image.height} image at quality {quality} failed: {exc}"
        ) from exc
    finally:
        ImageFile.MAXBLOCK = previous_maxblock
    return target.stat().st_size


def build_artwork(
    source: Path | str,
    target: Path | str,
    *,
    edge: int = MAX_EDGE,
    max_bytes: int = TARGET_MAX_BYTES,
) -> tuple[int, int]:
    """Produce `target` as a square RGB JPEG. Returns `(edge_px, bytes)`.

    Tries `edge` first and falls back to the source's native size if the upscale
    cannot get under `max_bytes` at a quality worth shipping.
    """
    source, target = Path(source), Path(target)
    if not source.is_file():
        raise ArtworkError(f"no such image: {source}")

    with Image.open(source) as opened:
        original = opened.convert("RGB")
        width, height = original.size
        if width != height:
            raise ArtworkError(
                f"{source} is {width}×{height}, not square. Apple requires square artwork; "
                "use the square logo, not the banner crop."
            )
        if width < MIN_EDGE:
            raise ArtworkError(
                f"{source} is {width}px — below Apple's {MIN_EDGE}px minimum, "
                "and upscaling that far would show."
            )

        for candidate_edge in dict.fromkeys((min(edge, MAX_EDGE), width)):
            if candidate_edge == width:
                resized = original
            else:
                resized = original.resize((candidate_edge, candidate_edge), Image.LANCZOS)
            for quality in _QUALITY_LADDER:
                size = _encode_at(resized, target, quality)
                if size <= max_bytes:
                    logger.info("artwork: %dpx @ q%d → %d bytes", candidate_edge, quality, size)
                    return candidate_edge, size
                logger.info("artwork: %dpx @ q%d is %d bytes, over budget", candidate_edge, quality, size)

    target.unlink(missing_ok=True)
    raise ArtworkError(
        f"could not get {source.name} under {max_bytes} bytes at quality {_QUALITY_LADDER[-1]} or better, "
        f"at either {edge}px or its native {width}px. Simplify the image or raise max_bytes deliberately."
    )


def describe_artwork(path: Path | str) -> tuple[str, tuple[int, int], str, int]:
    """`(format, size, mode, bytes)` — for asserting what was actually written."""
    path = Path(path)
    with Image.open(path) as img:
        return img.format or "?", img.size, img.mode, path.stat().st_size
