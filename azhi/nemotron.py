"""Speaker separation with NVIDIA's Nemotron 3 Diarization, through NeMo-Speech.cpp.

The two engines the application already has work the same way: cut the audio
into pieces, measure each piece's voice, group pieces that sound alike. A turn
of a few tenths of a second gives that measurement too little to go on, which
is where quick replies go to the wrong person. Nemotron 3 Diarization is one
model that decides, every 10 ms, how likely each of up to eight people is to
be speaking - overlapping speech included - with no separate grouping step.

The turns are built here from those probabilities, not taken from NVIDIA's
runtime. Its turns are cut for a benchmark that scores speech *time*: each is
widened by about 0.3 s, gaps under 0.3 s within one person's speech are filled
in, and turns under half a second are dropped. A one-word "Yes." between two
pieces of the other person's speech came out inside that person's widened
turn, and went to them. Here every 10 ms frame in which anyone is more likely
than not to be speaking belongs to whoever is most likely: nothing is widened,
bridged or dropped for being short.

Because each frame says how likely *each* speaker is, a speaker count can be
honoured too: told there are two people when the model heard three voices, the
two heard longest are kept and each moment of the third goes to whichever of
them is more likely there. A model that heard fewer voices than it was told
cannot split one, and says so.

The runtime's C library (Apache-2.0) is run by :mod:`azhi.nemotron_worker`
in a process of its own, from ``azhi_assets/nemotron`` beside the model.
It is given the bundled file's path and nothing else; it has no means of
downloading anything.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

MAX_SPEAKERS = 8

# A frame is speech when the most likely speaker is more likely than not.
THRESHOLD = 0.5


def describe_method() -> str:
    """The configuration and the turn rule, as a report states them."""
    return (
        "chunk 340, right context 40, FIFO 40, speaker cache 264, "
        "update period 300 (80 ms frames); turns: the most probable speaker "
        f"in each 10 ms frame where one is above {THRESHOLD:g}"
    )


@dataclass
class Activity:
    """Turns built from the model's probabilities, and what was heard."""

    turns: List[Tuple[float, float, int]]
    voices_heard: int
    voices_kept: int


def turns_from_activity(
    probs: Any,
    seconds_per_frame: float,
    *,
    num_speakers: Optional[int] = None,
    threshold: float = THRESHOLD,
) -> Activity:
    """Speaker turns from frames x speakers probabilities.

    Each frame where some speaker is above ``threshold`` goes to the most
    probable one; runs of frames with the same speaker become a turn. Speakers
    are numbered from 0 in the order they first speak.
    """
    import numpy as np

    probs = np.asarray(probs, dtype=np.float32)
    if probs.ndim != 2 or not probs.size:
        return Activity(turns=[], voices_heard=0, voices_kept=0)
    speech = probs.max(axis=1) > threshold
    who = probs.argmax(axis=1)
    heard = {
        int(s): int(np.count_nonzero(speech & (who == s)))
        for s in np.unique(who[speech])
    }
    kept = sorted(heard)
    if num_speakers and len(heard) > num_speakers:
        kept = sorted(sorted(heard, key=lambda s: -heard[s])[:num_speakers])
        who = np.asarray(kept)[probs[:, kept].argmax(axis=1)]
    label = np.where(speech, who, -1)
    edges = np.flatnonzero(np.diff(label)) + 1
    starts = np.concatenate(([0], edges))
    ends = np.concatenate((edges, [len(label)]))
    raw = [
        (int(a), int(b), int(label[a])) for a, b in zip(starts, ends) if label[a] >= 0
    ]
    order: dict = {}
    for _a, _b, channel in raw:
        order.setdefault(channel, len(order))
    turns = [
        (a * seconds_per_frame, b * seconds_per_frame, order[channel])
        for a, b, channel in raw
    ]
    return Activity(turns=turns, voices_heard=len(heard), voices_kept=len(order))


def plain_path(path: PathLike, folder: Path, name: str) -> str:
    """``path`` as the library can open it: unchanged when it is plain ASCII.

    The library takes the model's path as a narrow string, opened in the
    Windows code page, so a copy under ``C:\\Users\\José`` would not open. Such
    a file is linked - or, across drives, copied - into the work folder and
    named there.
    """
    text = str(path)
    if text.isascii():
        return text
    target = folder / name
    try:
        os.link(text, target)
    except OSError:
        shutil.copyfile(text, target)
    return name


def worker_command(
    library: PathLike, model: str, wav: PathLike, out: PathLike
) -> List[str]:
    """How to start :mod:`azhi.nemotron_worker`, from source or in a bundle."""
    args = [str(library), model, str(wav), str(out)]
    if getattr(sys, "frozen", False):
        return [sys.executable, "--nemotron-worker", *args]
    return [sys.executable, "-m", "azhi.nemotron_worker", *args]


def _environment() -> "dict":
    env = dict(os.environ)
    # From a source checkout the worker runs in the work folder, so it needs
    # telling where the package is.
    package_root = str(Path(__file__).resolve().parent.parent)
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (package_root, env.get("PYTHONPATH", "")) if part
    )
    return env


def _read_progress(path: Path) -> Optional[float]:
    try:
        return float(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def run(
    wav_path: PathLike,
    library: PathLike,
    model: PathLike,
    *,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
    poll_seconds: float = 0.25,
) -> Tuple[Any, float]:
    """Run the worker on a WAV: ``(frames x speakers probabilities, seconds per frame)``."""
    import numpy as np

    for path, what in ((library, "library"), (model, "model")):
        if not Path(path).is_file():
            raise RuntimeError(
                f"The Nemotron speaker-separation {what} is missing from this "
                "copy, so that method cannot run. Choose another Method under "
                "Advanced options, or run Azhi --verify."
            )
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="azhi-nemotron-") as work:
        folder = Path(work)
        model_arg = plain_path(model, folder, "model.gguf")
        process = subprocess.Popen(
            worker_command(library, model_arg, wav_path, folder),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            cwd=work,
            env=_environment(),
            creationflags=flags,
        )
        while True:
            try:
                code = process.wait(timeout=poll_seconds)
                break
            except subprocess.TimeoutExpired:
                if cancelled is not None and cancelled():
                    process.kill()
                    process.wait()
                    raise CancelledError("Diarization cancelled.") from None
                fraction = _read_progress(folder / "progress")
                if on_progress is not None and fraction is not None:
                    on_progress(fraction)
        if code != 0:
            try:
                detail = (folder / "error.txt").read_text(encoding="utf-8").strip()
            except OSError:
                detail = ""
            raise RuntimeError(
                "Nemotron speaker separation failed"
                + (f": {detail}" if detail else f" (it stopped with exit code {code}).")
            )
        try:
            result = json.loads((folder / "result.json").read_text(encoding="utf-8"))
            probs = np.load(folder / "probs.npy")
            step = float(result["seconds_per_frame"])
        except (OSError, ValueError, KeyError) as exc:
            raise RuntimeError(
                "Nemotron speaker separation finished without a readable result. "
                "This copy may be damaged; run Azhi --verify."
            ) from exc
    return probs, step


def diarize(
    wav_path: PathLike,
    *,
    num_speakers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> "List[Tuple[float, float, str]]":
    """Diarize a 16 kHz mono WAV with the bundled model.

    Returns ``(start, end, "SPEAKER_NN")`` turns, at most ``num_speakers``
    speakers when a number is given.
    """
    from .resources import bundled_nemotron

    bundled = bundled_nemotron()
    if bundled is None:
        raise RuntimeError(
            "This copy does not include the Nemotron speaker-separation model, "
            "so that method cannot run here. Choose another Method under "
            "Advanced options."
        )
    library, model = bundled
    if cancelled is not None and cancelled():
        raise CancelledError("Diarization cancelled.")
    if progress is not None:
        progress("Identifying speakers with Nemotron…")
    if on_progress is not None:
        on_progress(0.0)
    probs, step = run(
        wav_path, library, model, on_progress=on_progress, cancelled=cancelled
    )
    activity = turns_from_activity(probs, step, num_speakers=num_speakers)
    if on_progress is not None:
        on_progress(1.0)
    for line in _count_notes(activity, num_speakers):
        if progress is not None:
            progress(line)
    return [(a, b, f"SPEAKER_{speaker:02d}") for a, b, speaker in activity.turns]


def _count_notes(activity: Activity, num_speakers: Optional[int]) -> Sequence[str]:
    if not num_speakers or activity.voices_heard == num_speakers:
        return []
    if activity.voices_heard > num_speakers:
        return [
            f"Nemotron heard {activity.voices_heard} voices; as {num_speakers} "
            f"were expected, each moment of the other "
            f"{activity.voices_heard - num_speakers} went to whichever of the "
            f"{num_speakers} is more likely there."
        ]
    return [
        f"Nemotron heard {activity.voices_heard} voice(s); {num_speakers} were "
        "expected. It cannot split a voice it heard as one: if two people are "
        "merged, Redo speaker separation with another Method."
    ]
