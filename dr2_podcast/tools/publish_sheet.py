"""Per-episode publish sheet for the manual RedCircle upload.

RedCircle has no public API, so every episode is published by hand through its
web form. This module writes one Markdown sheet per episode holding what that
form asks for, in the order it asks for it, so publishing is a copy across
rather than a hunt through the run directory.

The one rule that shapes everything here: **a field is either read from the run
or left blank.** Nothing is inferred, guessed or synthesised. A title this code
cannot find is written as ``(要記入)`` for a human to fill in; it is never
approximated from a topic string or a folder name, and a publish date is never
taken from a file's mtime. A blank costs someone thirty seconds. A plausible
fabrication ships wrong metadata and is not visible as an error.

Two run layouts exist and both are supported:

* **Educational** — ``research_outputs/Ep{NNN}_<topic>/`` with the wavs and
  ``script.txt`` flat in the folder. The episode number is in the folder name and
  the published title lives in ``educational_series/ep{NN}_source_of_truth.md``.
* **Pipeline (実践編)** — a timestamped folder with ``research/``, ``scripts/``,
  ``audio/`` and ``meta/``. The topic comes from ``meta/session_metadata.txt``
  and the show-note sources from ``research/research_sources.json``.

Standalone use, which does not re-run anything::

    python -m dr2_podcast.tools.publish_sheet research_outputs/Ep001_*
    python -m dr2_podcast.tools.publish_sheet --force <run_dir>
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Written wherever a value could not be read. Deliberately conspicuous, and in
#: Japanese because the person filling the RedCircle form works in Japanese.
BLANK = "(要記入)"

#: Written where a value exists on disk but could not be decoded. Distinct from
#: BLANK: BLANK means "nobody has written this yet", UNREADABLE means "something
#: is here and it is broken", and the two want different responses.
UNREADABLE = "(取得できませんでした)"

SHEET_FILENAME = "publish_sheet.md"

#: The published artifact is the BGM-mixed master. `audio.wav` is the raw TTS
#: and is NOT an acceptable substitute — uploading it would ship an episode with
#: no music, silently. Missing `audio_mixed.wav` is therefore refused, not
#: worked around.
_MIXED_AUDIO_RELPATHS = ("audio/audio_mixed.wav", "audio_mixed.wav")

_EP_DIR_RE = re.compile(r"^Ep(\d{3})_(.+)$")
#: A pipeline run's directory name is its start timestamp.
_RUN_DIR_RE = re.compile(r"^\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")
#: Any one of these marks a directory as a pipeline run. Checked so that a
#: directory which is neither layout — a backup folder, a stray copy — is
#: refused rather than handed a sheet labelled "pipeline" that describes it.
_PIPELINE_MARKER_DIRS = ("meta", "research", "scripts")
_EPISODE_TITLE_RE = re.compile(r"^-\s*\*\*Episode Title:\*\*\s*(.+?)\s*$", re.MULTILINE)
_TOPIC_RE = re.compile(r"^Topic:\s*(.+?)\s*$", re.MULTILINE)


class PublishSheetError(Exception):
    """The sheet could not be produced from this directory.

    Raised rather than writing a sheet with an invented or empty key field —
    a sheet that names no audio file is worse than no sheet, because it looks
    like a deliverable.
    """


@dataclass
class PublishSheet:
    """The fields of one episode's RedCircle form, in the order it asks them."""

    run_dir: Path
    layout: str
    audio_path: Path
    duration_text: str
    title: str = BLANK
    title_reference: str | None = None
    description: str = BLANK
    sources: list[tuple[str, str]] = field(default_factory=list)
    sources_note: str | None = None
    tags: str = BLANK
    publish_date: str = BLANK
    episode_number: str = BLANK

    def render(self) -> str:
        """Render the sheet as Markdown, top to bottom in RedCircle form order."""
        lines: list[str] = [
            f"# 公開シート — {self.run_dir.name}",
            "",
            "RedCircle の投稿フォームに、上から順にそのまま転記してください。",
            "",
            f"`{BLANK}` は、この回のデータからは読み取れなかった項目です。"
            "推測で埋めると誤った情報を公開することになるため、自動生成はしていません。"
            "人が記入してください。",
            "",
            f"- レイアウト: {self.layout}",
            f"- 生成元: `{self.run_dir}`",
            "",
            "---",
            "",
            "## 1. 音声ファイル",
            "",
            f"- パス: `{self.audio_path}`",
            f"- 長さ: {self.duration_text}",
            "",
            "## 2. タイトル",
            "",
            self.title,
        ]
        if self.title_reference:
            lines += [
                "",
                f"> 参考（この回のトピック。タイトルそのものではありません）: {self.title_reference}",
            ]
        lines += [
            "",
            "## 3. 説明文 / ショーノート",
            "",
            self.description,
        ]
        lines += ["", "### 出典", ""]
        if self.sources:
            for i, (title, url) in enumerate(self.sources, start=1):
                lines.append(f"{i}. {title}")
                lines.append(f"   {url}")
        else:
            lines.append(self.sources_note or BLANK)
        lines += [
            "",
            "## 4. タグ",
            "",
            self.tags,
            "",
            "## 5. 公開日",
            "",
            self.publish_date,
            "",
            "## 6. エピソード番号",
            "",
            self.episode_number,
            "",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Readers — each returns a value or None. None becomes a blank, never a guess.
# ---------------------------------------------------------------------------


def _resolve_mixed_audio(run_dir: Path) -> Path:
    """Return the BGM-mixed master, or refuse.

    Checks `.exists()` per candidate rather than trusting the directory shape:
    Ep014 has an `audio/` directory holding only an old backup, so "the subdir
    exists" does not imply "the wav is in it".
    """
    for rel in _MIXED_AUDIO_RELPATHS:
        candidate = run_dir / rel
        if candidate.is_file():
            return candidate.resolve()
    found = sorted(p.name for p in run_dir.rglob("*.wav"))
    raise PublishSheetError(
        f"no audio_mixed.wav under {run_dir} (looked at {', '.join(_MIXED_AUDIO_RELPATHS)}). "
        f"wav files present: {', '.join(found) if found else 'none'}. "
        "The published artifact is the BGM-mixed master; audio.wav is the raw TTS and is not a substitute."
    )


def _format_duration(audio_path: Path) -> str:
    """Read the wav header for a real duration, or say it could not be read."""
    try:
        with wave.open(str(audio_path), "rb") as wav:
            rate = wav.getframerate()
            if rate <= 0:
                return UNREADABLE
            seconds = wav.getnframes() / float(rate)
    except (wave.Error, OSError) as exc:
        logger.warning("could not read duration from %s: %s", audio_path, exc)
        return UNREADABLE
    minutes, secs = divmod(int(round(seconds)), 60)
    return f"{minutes}分{secs:02d}秒 ({seconds:.1f}秒)"


def _read_educational_title(project_root: Path, episode_number: int) -> str | None:
    """Read the published Japanese title from the episode brief.

    The brief is the only place a human-authored title exists for the
    educational series; the folder slug is an abbreviation of it, not the title.
    """
    brief = project_root / "educational_series" / f"ep{episode_number:02d}_source_of_truth.md"
    if not brief.is_file():
        return None
    try:
        text = brief.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", brief, exc)
        return None
    match = _EPISODE_TITLE_RE.search(text)
    return match.group(1) if match else None


def _read_topic(run_dir: Path) -> str | None:
    """Read the run's topic from meta/session_metadata.txt.

    Tolerates both header variants — the normal run and the `(REUSE: ...)` one,
    which differ in spacing but share the `Topic:` line.
    """
    meta = run_dir / "meta" / "session_metadata.txt"
    if not meta.is_file():
        return None
    try:
        text = meta.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("could not read %s: %s", meta, exc)
        return None
    match = _TOPIC_RE.search(text)
    if not match:
        return None
    topic = match.group(1).strip()
    return topic or None


def _iter_source_records(payload: object) -> list[dict]:
    """Flatten research_sources.json into a list of entries.

    The file is a dict of track name -> list ({"lead": [...], "counter": [...]}).
    A bare list is accepted too. Anything else is not understood, and is
    reported as unreadable rather than silently yielding zero sources — an
    empty citation list on a research show is a claim, not an absence.
    """
    if isinstance(payload, dict):
        records: list[dict] = []
        for value in payload.values():
            if isinstance(value, list):
                records.extend(entry for entry in value if isinstance(entry, dict))
        return records
    if isinstance(payload, list):
        return [entry for entry in payload if isinstance(entry, dict)]
    raise PublishSheetError(f"research_sources.json has an unexpected shape: {type(payload).__name__}")


def _read_sources(run_dir: Path) -> tuple[list[tuple[str, str]], str | None]:
    """Return (sources, note). A note is set only when the list is empty."""
    # The VALIDATED library when there is a current one. url_validation removes the URLs that
    # failed a HEAD request and writes a separate file rather than editing the raw one; reading the
    # raw library here copied dead links straight into the show notes a human pastes into RedCircle
    # (prepush codex 2026-08-14). research_sources_file() is the same selector the agents use, and
    # it prefers the validated copy only while its stamp still matches the raw library.
    from dr2_podcast.pipeline import research_sources_file

    path = Path(research_sources_file(run_dir))
    if not path.is_file():
        return [], BLANK
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("could not read %s: %s", path, exc)
        return [], UNREADABLE

    seen: set[str] = set()
    sources: list[tuple[str, str]] = []
    for entry in _iter_source_records(payload):
        url = str(entry.get("url") or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        title = str(entry.get("title") or "").strip()
        sources.append((title or "(タイトル不明)", url))
    if not sources:
        return [], f"{UNREADABLE} — research_sources.json は存在しますが、引用できる出典がありませんでした。"
    return sources, None


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Repository root — this file is dr2_podcast/tools/publish_sheet.py."""
    return Path(__file__).resolve().parent.parent.parent


def build_publish_sheet(run_dir: Path, project_root: Path | None = None) -> PublishSheet:
    """Assemble the sheet for one episode directory.

    Raises PublishSheetError if the directory is not a run, or has no
    BGM-mixed audio to publish.
    """
    run_dir = Path(run_dir)
    if not run_dir.is_dir():
        raise PublishSheetError(f"not a directory: {run_dir}")
    root = project_root if project_root is not None else _project_root()

    audio_path = _resolve_mixed_audio(run_dir)
    match = _EP_DIR_RE.match(run_dir.name)

    if match:
        number = int(match.group(1))
        title = _read_educational_title(root, number)
        # The educational series has no description, tags or date anywhere on
        # disk — every one of those is human-only, so every one stays blank.
        return PublishSheet(
            run_dir=run_dir,
            layout="educational",
            audio_path=audio_path,
            duration_text=_format_duration(audio_path),
            title=title or BLANK,
            description=BLANK,
            sources=[],
            sources_note=f"{BLANK} — 教育シリーズの回には出典ファイルがありません。",
            episode_number=str(number),
        )

    if not _RUN_DIR_RE.match(run_dir.name) and not any((run_dir / d).is_dir() for d in _PIPELINE_MARKER_DIRS):
        raise PublishSheetError(
            f"{run_dir} holds an audio_mixed.wav but is neither layout: its name is not Ep{{NNN}}_... "
            f"nor a run timestamp, and it has none of {'/'.join(_PIPELINE_MARKER_DIRS)}. "
            "Refusing rather than emitting a sheet that would describe it as a pipeline run."
        )

    topic = _read_topic(run_dir)
    sources, note = _read_sources(run_dir)
    return PublishSheet(
        run_dir=run_dir,
        layout="pipeline",
        audio_path=audio_path,
        duration_text=_format_duration(audio_path),
        # The topic is a research question, not an episode title. It is offered
        # as reference material underneath the blank, never promoted into it.
        title=BLANK,
        title_reference=topic,
        description=BLANK,
        sources=sources,
        sources_note=note,
    )


def sheet_path(run_dir: Path) -> Path:
    """Where the sheet goes: beside the run's other metadata."""
    run_dir = Path(run_dir)
    meta = run_dir / "meta"
    return (meta if meta.is_dir() else run_dir) / SHEET_FILENAME


def write_publish_sheet(run_dir: Path, *, force: bool = False, project_root: Path | None = None) -> Path:
    """Write the sheet and return its path.

    An existing sheet is left alone unless `force` is passed. That default is
    load-bearing: the blanks in a sheet are meant to be filled in by hand, and a
    re-render of the audio must not silently wipe the title and show notes
    someone already wrote.
    """
    target = sheet_path(run_dir)
    if target.exists() and not force:
        logger.info("publish sheet already exists, leaving it untouched: %s", target)
        return target
    sheet = build_publish_sheet(run_dir, project_root=project_root)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(sheet.render(), encoding="utf-8")
    logger.info("publish sheet written: %s", target)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dr2_podcast.tools.publish_sheet",
        description="Write the RedCircle publish sheet for one or more finished episode directories.",
    )
    parser.add_argument("run_dir", nargs="+", type=Path, help="episode or run directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="overwrite an existing sheet (default: keep it, so hand-filled fields survive)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    failures = 0
    for run_dir in args.run_dir:
        try:
            target = write_publish_sheet(run_dir, force=args.force)
        except (PublishSheetError, OSError) as exc:
            print(f"FAILED {run_dir}: {exc}", file=sys.stderr)
            failures += 1
        else:
            print(f"OK {target}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
