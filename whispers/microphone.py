"""A microphone on this machine as a live source, through the bundled ffmpeg.

ffmpeg already reads every live stream; on Windows it can also open a
microphone (DirectShow, compiled into the ffmpeg a bundle ships), and on Linux
the system's default capture device (ALSA). So a microphone needs no new
library: only the right input options ahead of ``-i``, and a list of what is
plugged in.

Unlike a stream, a microphone's audio exists nowhere else. The Live page
therefore always keeps the whole recording, in a folder the analyst chose, and
when capture ends writes a small record beside it: the device, when it ran,
the file's size and SHA-256. That record holds only what was observed; nothing
in it is guessed.
"""

from __future__ import annotations

import array
import json
import re
import subprocess
import sys
import wave
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Union

from .atomicio import write_json_atomic
from .hashing import sha256_file_or_none
from .output_names import clean
from .version import __version__

PathLike = Union[str, Path]

# Milliseconds of audio DirectShow gathers before handing it on. Its default
# is about half a second; shorter keeps the live chunks close to real time.
DSHOW_BUFFER_MS = 50

# Seconds of nothing but exact zeros before saying so. A microphone that
# Windows has blocked, or one muted in hardware, opens normally and then
# delivers digital silence; a real room is never exactly zero for this long.
SILENT_WARNING_SECONDS = 10.0

RECORD_SUFFIX = ".capture.json"


@dataclass(frozen=True)
class Microphone:
    """One capture device: what to show, and what ffmpeg opens."""

    name: str
    # ffmpeg's input options, ``-i`` and its argument included.
    input_args: "tuple[str, ...]"


class MicrophoneError(RuntimeError):
    """The device list could not be read."""


def system() -> str:
    """``windows``, ``linux``, or ``other`` (no microphone support)."""
    if sys.platform.startswith("win"):
        return "windows"
    if sys.platform.startswith("linux"):
        return "linux"
    return "other"


def list_microphones(ffmpeg: PathLike, *, timeout: float = 15.0) -> List[Microphone]:
    """The microphones ffmpeg can open on this machine, in its order.

    Raises :class:`MicrophoneError` with a plain reason when ffmpeg cannot be
    asked. An empty list means it was asked and found none.
    """
    kind = system()
    if kind == "windows":
        args = [str(ffmpeg), "-hide_banner", "-list_devices", "true"]
        args += ["-f", "dshow", "-i", "dummy"]
    elif kind == "linux":
        args = [str(ffmpeg), "-hide_banner", "-sources", "alsa"]
    else:
        raise MicrophoneError("Microphone capture is available on Windows and Linux.")
    try:
        result = subprocess.run(args, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise MicrophoneError("Listing the microphones took too long.") from exc
    except OSError as exc:
        raise MicrophoneError(f"Couldn't run ffmpeg: {exc}") from exc
    # Listing ends in an error by design on Windows ("dummy" is not a device),
    # so the exit code says nothing; the list is in the output.
    output = (result.stderr + result.stdout).decode("utf-8", "replace")
    if kind == "windows":
        return parse_dshow(output)
    return parse_alsa(output)


_LOG_PREFIX = re.compile(r"^\[[^\]]*\]\s?")
_QUOTED = re.compile(r'^\s*"(?P<name>[^"]+)"\s*(?:\((?P<kinds>[^)]*)\))?\s*$')
_ALTERNATIVE = re.compile(r'^\s*Alternative name\s+"(?P<id>[^"]+)"')


def parse_dshow(output: str) -> List[Microphone]:
    """Audio devices from ``ffmpeg -list_devices true -f dshow -i dummy``.

    ffmpeg 5 and later mark each device ``"Name" (audio)`` or ``(video)``;
    older builds list them under "DirectShow audio devices" headings instead.
    Each is followed by an alternative name, which is unique even when two
    devices share a friendly one, so that is what is opened.
    """
    found: List[Dict[str, str]] = []
    section = ""
    last: Optional[Dict[str, str]] = None
    for raw in output.splitlines():
        line = _LOG_PREFIX.sub("", raw.rstrip())
        lowered = line.lower()
        if "directshow audio devices" in lowered:
            section = "audio"
            continue
        if "directshow video devices" in lowered:
            section = "video"
            continue
        alternative = _ALTERNATIVE.match(line)
        if alternative:
            if last is not None:
                last["id"] = alternative.group("id")
            continue
        quoted = _QUOTED.match(line)
        if not quoted:
            continue
        kinds = quoted.group("kinds")
        audio = "audio" in kinds if kinds is not None else section == "audio"
        last = {"name": quoted.group("name")} if audio else None
        if last is not None:
            found.append(last)
    return [
        Microphone(
            name,
            (
                "-f",
                "dshow",
                "-audio_buffer_size",
                str(DSHOW_BUFFER_MS),
                "-i",
                f"audio={device.get('id') or device['name']}",
            ),
        )
        for name, device in zip(_distinct([d["name"] for d in found]), found)
    ]


_ALSA_SOURCE = re.compile(r"^\s*(?P<default>\*)?\s*(?P<id>\S+)\s+\[(?P<label>.*)\]")


def parse_alsa(output: str) -> List[Microphone]:
    """Capture devices from ``ffmpeg -sources alsa``, the default first.

    ALSA's ``default`` device (which on a desktop goes through the sound
    server, so follows the system's chosen microphone) is always offered, even
    where ffmpeg does not list it. ``null`` only ever gives silence.
    """
    devices = [("System default microphone", "default")]
    for line in output.splitlines():
        match = _ALSA_SOURCE.match(line)
        if not match or match.group("id") in ("null", "default"):
            continue
        devices.append(
            (match.group("label").strip() or match.group("id"), match.group("id"))
        )
    names = _distinct([name for name, _ in devices])
    return [
        Microphone(name, ("-f", "alsa", "-i", device))
        for name, (_, device) in zip(names, devices)
    ]


def _distinct(names: List[str]) -> List[str]:
    """Names as shown: a second device with the same name becomes "Name (2)"."""
    seen: Dict[str, int] = {}
    shown = []
    for name in names:
        seen[name] = seen.get(name, 0) + 1
        shown.append(name if seen[name] == 1 else f"{name} ({seen[name]})")
    return shown


def explain_capture_error(detail: str) -> Optional[str]:
    """A plain reason for an ffmpeg failure to open a microphone, if known."""
    lowered = detail.lower()
    if "could not find audio only device" in lowered or "no such device" in lowered:
        return (
            "That microphone was not found. It may have been unplugged: "
            "press Refresh and pick it again."
        )
    if any(
        sign in lowered
        for sign in ("i/o error", "could not run graph", "device or resource busy")
    ):
        return (
            "The microphone could not be opened. Check that Windows lets desktop "
            "apps use the microphone (Settings, Privacy & security, Microphone) "
            "and that no other program is holding it."
            if system() == "windows"
            else "The microphone could not be opened, or another program is holding it."
        )
    return None


SILENT_MESSAGE = (
    "The microphone is sending only silence. It may be muted, or Windows may be "
    "blocking it (Settings, Privacy & security, Microphone). The recording "
    "carries on."
)


def is_digital_silence(wav_path: PathLike) -> bool:
    """True when a 16-bit WAV holds samples and every one of them is zero."""
    with wave.open(str(wav_path), "rb") as chunk:
        if chunk.getsampwidth() != 2:
            return False
        frames = chunk.readframes(chunk.getnframes())
    samples = array.array("h")
    samples.frombytes(frames[: len(frames) - len(frames) % 2])
    return len(samples) > 0 and samples.count(0) == len(samples)


# -- The recording ------------------------------------------------------------


def recording_path(folder: PathLike, started: datetime) -> Path:
    """A new file name in ``folder`` for a capture starting at ``started``."""
    stem = clean(f"Microphone {started.strftime('%Y-%m-%d %H-%M-%S')}")
    target = Path(folder) / f"{stem}.wav"
    n = 2
    while target.exists() or record_path(target).exists():
        target = Path(folder) / f"{stem} ({n}).wav"
        n += 1
    return target


def record_path(recording: PathLike) -> Path:
    """Where the record of ``recording`` is kept: beside it."""
    path = Path(recording)
    return path.with_name(path.name + RECORD_SUFFIX)


def recording_args(path: PathLike) -> List[str]:
    """ffmpeg output options keeping everything the device sent, unresampled.

    16-bit PCM WAV at the device's own rate and channels; past 4 GB (about six
    hours of 48 kHz stereo) the file becomes RF64 rather than breaking.
    """
    return ["-vn", "-c:a", "pcm_s16le", "-rf64", "auto", str(path)]


def write_capture_record(
    recording: PathLike,
    *,
    device: str,
    started: datetime,
    stopped: datetime,
    transcription_model: str,
) -> Path:
    """Hash the finished recording and write what is known about it beside it."""
    path = Path(recording)
    size: Optional[int]
    try:
        size = path.stat().st_size
    except OSError:
        size = None
    record: Dict[str, object] = {
        "kind": "microphone recording",
        "file": path.name,
        "file_size": size,
        "sha256": sha256_file_or_none(path),
        "device": device,
        "started_utc": _utc(started),
        "stopped_utc": _utc(stopped),
        "live_transcription_model": transcription_model,
        "whispers_version": __version__,
    }
    record.update(_wav_format(path))
    return write_json_atomic(record_path(path), record)


def read_capture_record(recording: PathLike) -> Dict[str, object]:
    with open(record_path(recording), encoding="utf-8") as fh:
        data = json.load(fh)
    return data if isinstance(data, dict) else {}


def _utc(moment: datetime) -> str:
    if moment.tzinfo is None:
        moment = moment.astimezone()
    return moment.astimezone(timezone.utc).replace(microsecond=0).isoformat()


def _wav_format(path: Path) -> Dict[str, object]:
    """Rate, channels and length, from the file itself; nothing if unreadable."""
    try:
        with wave.open(str(path), "rb") as recording:
            rate = recording.getframerate()
            return {
                "sample_rate": rate,
                "channels": recording.getnchannels(),
                "duration_seconds": round(recording.getnframes() / rate, 2),
            }
    except (OSError, EOFError, wave.Error, ZeroDivisionError):
        return {}
