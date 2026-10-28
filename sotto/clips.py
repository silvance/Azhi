"""Save a passage of a recording as an audio file of its own.

For briefing on a lead, or handing it on, without sending the whole recording.
The passage is cut from the original recording with ffmpeg, at the quality it
was recorded (16-bit PCM WAV, the source's own rate and channels; nothing is
filtered), with a little room either side so the first and last words are not
clipped. Beside it goes a record, ``<clip>.wav.clip.json``, of where it came
from: the source file and its SHA-256, the time range, the clip's own SHA-256
and format, and the transcript lines it covers. Only what was observed is
written; a value that could not be read is left out, never guessed.
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Union

from .atomicio import write_json_atomic
from .hashing import sha256_file_or_none
from .microphone import wav_format
from .output_names import clean
from .resources import find_ffmpeg
from .transcription import Segment, clock_time
from .version import __version__

PathLike = Union[str, Path]
SpeakerLabel = Optional[Callable[[str], str]]

# Seconds kept either side of the passage: word times are close, not exact,
# and a clip that starts mid-syllable is harder to hear and to trust.
MARGIN_SECONDS = 0.25

RECORD_SUFFIX = ".clip.json"


class ClipError(RuntimeError):
    """The clip could not be cut, in words an operator can act on."""


def clip_name(source: PathLike, start: float, end: float) -> str:
    """``<source stem> 00-12-34 to 00-12-50.wav``: what and where, at a glance."""
    span = (
        f"{clock_time(start).replace(':', '-')} to {clock_time(end).replace(':', '-')}"
    )
    return clean(f"{Path(source).stem} {span}") + ".wav"


def record_path(clip: PathLike) -> Path:
    path = Path(clip)
    return path.with_name(path.name + RECORD_SUFFIX)


def cut(
    source: PathLike,
    start: float,
    end: float,
    out: PathLike,
    *,
    margin: float = MARGIN_SECONDS,
) -> "tuple[float, float]":
    """Cut ``[start, end]`` (plus ``margin`` either side) from ``source``.

    Returns the span actually cut, in the source's time. Raises
    :class:`ClipError` if the source is gone or ffmpeg cannot cut it.
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        raise ClipError("ffmpeg was not found, so the passage can't be cut.")
    src = Path(source)
    if not src.is_file():
        raise ClipError(
            f"The recording is not where the transcript says it is ({src}), so "
            "the passage can't be cut from it."
        )
    if end <= start:
        raise ClipError("That passage has no length to save.")
    first = max(0.0, start - margin)
    last = end + margin
    result = subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            # Seeking before -i is exact when the audio is decoded, as it is
            # here (written out as PCM), and quick on a long recording.
            "-ss",
            f"{first:.3f}",
            "-i",
            str(src),
            "-t",
            f"{last - first:.3f}",
            "-vn",
            "-c:a",
            "pcm_s16le",
            str(out),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not Path(out).is_file():
        detail = (result.stderr or "").strip().splitlines()
        raise ClipError(
            "The passage could not be cut" + (f": {detail[-1]}" if detail else ".")
        )
    return first, last


def write_clip_record(
    clip: PathLike,
    *,
    source: PathLike,
    start: float,
    end: float,
    lines: Sequence[Segment],
    speaker_label: SpeakerLabel = None,
) -> Path:
    """Write what is known about ``clip`` beside it; returns the record's path.

    ``start``/``end`` are the span cut, in the source's time. ``lines`` are the
    transcript lines the passage covers; ``speaker_label`` turns a speaker id
    into the name shown (unconfirmed-lead marker included).
    """
    path, src = Path(clip), Path(source)
    record: Dict[str, object] = {
        "kind": "audio clip",
        "file": path.name,
        "file_size": _size(path),
        "sha256": sha256_file_or_none(path),
        "source_file": src.name,
        "source_path": str(src),
        "source_sha256": sha256_file_or_none(src),
        "start": clock_time(start, millis=True),
        "end": clock_time(end, millis=True),
        "start_seconds": round(start, 3),
        "end_seconds": round(end, 3),
        "lines": [_line(segment, speaker_label) for segment in lines],
        "created_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "sotto_version": __version__,
    }
    record.update(wav_format(path))
    return write_json_atomic(record_path(path), record)


def lines_within(
    segments: Sequence[Segment], start: float, end: float
) -> List[Segment]:
    """The transcript lines that overlap ``[start, end]``."""
    return [s for s in segments if s.end > start and s.start < end]


def _line(segment: Segment, speaker_label: SpeakerLabel) -> Dict[str, object]:
    line: Dict[str, object] = {
        "start": clock_time(segment.start, millis=True),
        "end": clock_time(segment.end, millis=True),
        "text": segment.text.strip(),
    }
    if segment.speaker:
        line["speaker"] = (
            speaker_label(segment.speaker) if speaker_label else segment.speaker
        )
    return line


def _size(path: Path) -> Optional[int]:
    try:
        return path.stat().st_size
    except OSError:
        return None
