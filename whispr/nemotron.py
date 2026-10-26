"""Speaker separation with NVIDIA's Nemotron 3 Diarization, through NeMo-Speech.cpp.

The two engines the application already has work the same way: cut the audio
into pieces, measure each piece's voice, group pieces that sound alike. A turn
of a few tenths of a second gives that measurement too little to go on, which
is where quick replies go to the wrong person. Nemotron 3 Diarization is one
model that decides, every 10 ms, which of up to eight people is speaking -
overlapping speech included - with no separate grouping step.

It is run as NVIDIA's own command-line program (``nemo-speech``, Apache-2.0),
shipped in ``whispr_assets/nemotron`` beside the model. Nothing about it may
reach the network:

- the model is always passed as the path of the bundled file, checked to exist
  first; the program only downloads when given a repository name instead;
- its download address is pointed at a port on this machine that nothing
  listens on, so if that ever changed, the download would fail here;
- its model cache is a temporary folder, removed afterwards.

The model works out how many people are speaking by itself, and cannot be told.
When the operator has given a number and the model finds another, the status
says so; the turns are not forced to match.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

MAX_SPEAKERS = 8

# The configuration NVIDIA's published accuracy figures were measured with
# (the model card's "very high latency (offline)" row), in 80 ms frames: a
# 27.2 s chunk with 3.2 s of look-ahead. The runtime's own "v3-offline" preset
# is a different configuration, and has no published figures.
GEOMETRY: Dict[str, int] = {
    "--diar-chunk": 340,
    "--diar-rc": 40,
    "--diar-fifo": 40,
    "--diar-spkcache": 264,
    "--diar-update-period": 300,
}

# Where the runtime would download from, were it ever to try: a closed port on
# this machine. The runtime accepts plain HTTP only for loopback addresses.
_NOWHERE = "http://127.0.0.1:9"


def describe_geometry() -> str:
    """The configuration, as a report states it."""
    return (
        "chunk 340, right context 40, FIFO 40, speaker cache 264, "
        "update period 300 (80 ms frames)"
    )


def command(
    binary: PathLike, model: PathLike, wav: PathLike, output: PathLike
) -> List[str]:
    """The ``nemo-speech diarize`` invocation for one recording."""
    args = [
        str(binary),
        "--quiet",
        "diarize",
        str(wav),
        "--model",
        str(model),
        "--device",
        "cpu",
        "--format",
        "json",
        "--output",
        str(output),
    ]
    for option, value in GEOMETRY.items():
        args += [option, str(value)]
    return args


def environment(cache: PathLike) -> Dict[str, str]:
    """The program's environment: this one, with every download route closed."""
    env = dict(os.environ)
    env["NEMO_SPEECH_HF_BASE_URL"] = _NOWHERE
    env["NEMO_SPEECH_MODEL_DIR"] = str(cache)
    env.pop("NEMO_SPEECH_MODEL_INDEX", None)
    return env


def parse_output(text: str) -> List[Tuple[float, float, int]]:
    """``(start, end, speaker)`` turns from the program's JSON, speakers from 0.

    The program numbers speakers from 1, in the order they first speak.
    """
    try:
        data = json.loads(text)
        segments = data["segments"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(
            "Nemotron speaker separation finished but its output could not be "
            "read. This copy may be damaged; run whispr --verify."
        ) from exc
    turns = []
    for item in segments:
        try:
            start, end, speaker = (
                float(item["start"]),
                float(item["end"]),
                int(item["speaker"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Nemotron speaker separation returned a turn without a start, "
                "an end or a speaker."
            ) from exc
        if end > start and speaker >= 1:
            turns.append((start, end, speaker - 1))
    return sorted(turns)


def plain_path(path: PathLike, folder: Path, name: str) -> str:
    """``path`` as the program can open it: unchanged when it is plain ASCII.

    The program reads its arguments, and opens files, in the Windows code page,
    so a recording under ``C:\\Users\\José`` would not open. Such a file is
    linked - or, across drives, copied - into the work folder and named there.
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


def _last_lines(path: Path, count: int = 4) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").strip().splitlines()
    except OSError:
        return ""
    return " ".join(line.strip() for line in lines[-count:] if line.strip())


def run(
    wav_path: PathLike,
    binary: PathLike,
    model: PathLike,
    *,
    cancelled: Optional[CancelCallback] = None,
    poll_seconds: float = 0.25,
) -> List[Tuple[float, float, int]]:
    """Run the program on a WAV and return its turns.

    Output goes to files, not pipes: a long recording produces more than a
    pipe holds, and a program blocked writing to one never finishes.
    """
    for path, what in ((binary, "program"), (model, "model")):
        if not Path(path).is_file():
            raise RuntimeError(
                f"The Nemotron speaker-separation {what} is missing from this "
                "copy, so that method cannot run. Choose another Method under "
                "Advanced options, or run whispr --verify."
            )
    if os.name != "nt" and not os.access(binary, os.X_OK):
        # Copied as data, the program can lose its execute bit on Linux.
        try:
            os.chmod(binary, os.stat(binary).st_mode | 0o111)
        except OSError:
            pass
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
    with tempfile.TemporaryDirectory(prefix="whispr-nemotron-") as work:
        folder = Path(work)
        output = folder / "turns.json"
        errors = folder / "errors.txt"
        cache = folder / "cache"
        cache.mkdir()
        # Relative to the work folder, which is where the program runs: its own
        # path may not be plain ASCII either.
        wav_arg = plain_path(wav_path, folder, "input.wav")
        model_arg = plain_path(model, folder, "model.gguf")
        with errors.open("wb") as stderr:
            process = subprocess.Popen(
                command(binary, model_arg, wav_arg, output.name),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=stderr,
                cwd=work,
                env=environment(cache),
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
        if code != 0:
            detail = _last_lines(errors)
            raise RuntimeError(
                f"Nemotron speaker separation failed (exit code {code})."
                + (f" It said: {detail}" if detail else "")
            )
        try:
            text = output.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(
                "Nemotron speaker separation finished without writing its result."
            ) from exc
    return parse_output(text)


def diarize(
    wav_path: PathLike,
    *,
    num_speakers: Optional[int] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> "List[Tuple[float, float, str]]":
    """Diarize a 16 kHz mono WAV with the bundled model.

    Returns ``(start, end, "SPEAKER_NN")`` turns. ``num_speakers`` is not
    imposed - the model cannot take it - but a different count is reported.
    """
    from .resources import bundled_nemotron

    bundled = bundled_nemotron()
    if bundled is None:
        raise RuntimeError(
            "This copy does not include the Nemotron speaker-separation model, "
            "so that method cannot run here. Choose another Method under "
            "Advanced options."
        )
    binary, model = bundled
    if cancelled is not None and cancelled():
        raise CancelledError("Diarization cancelled.")
    if progress is not None:
        progress(
            "Identifying speakers with Nemotron… (this method does not report "
            "how far it has got; Cancel stops it)"
        )
    if on_progress is not None:
        on_progress(0.0)
    turns = run(wav_path, binary, model, cancelled=cancelled)
    if on_progress is not None:
        on_progress(1.0)
    found = len({speaker for _start, _end, speaker in turns})
    if progress is not None and num_speakers and found != num_speakers:
        progress(
            f"Nemotron found {found} speaker(s); {num_speakers} were expected. "
            "It works out the number itself and cannot be given one. If the "
            "split looks wrong, Redo speaker separation with another Method, "
            "which uses the number you give."
        )
    return [(start, end, f"SPEAKER_{speaker:02d}") for start, end, speaker in turns]
