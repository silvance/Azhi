"""Run Nemotron 3 Diarization through NVIDIA's C library, in a process of its own.

:mod:`sotto.nemotron` starts this as a separate process and reads back what
it writes. Native code that fails takes this process with it, not the
application, and Cancel can stop it outright.

It asks the library for the model's own output - the probability, every
10 ms, that each of up to eight people is speaking - rather than for speaker
turns. The library's turns are cut with settings tuned for a benchmark that
scores speech time: turns shorter than half a second are dropped and short
gaps filled, which is what lost one-word replies.

    python -m sotto.nemotron_worker LIBRARY MODEL WAV OUT_DIR

(``Sotto --nemotron-worker ...`` in a shipped copy.) Writes into ``OUT_DIR``:

- ``progress``: the share of the recording processed so far, rewritten as it goes;
- ``probs.npy``: frames x speakers, float32;
- ``result.json``: seconds per frame, frames and speakers;
- ``error.txt``, only if it failed: what went wrong.

Imports nothing but the standard library before the C library is loaded, so
that the library's own runtime DLLs are the ones the process loads.
"""

from __future__ import annotations

import ctypes
import json
import os
import sys
import wave
from pathlib import Path
from typing import Any, List, Optional

OK = 0
SAMPLE_RATE = 16000
PUSH_SECONDS = 15

# The configuration NVIDIA's published figures were measured with, in 80 ms
# frames (see sotto.nemotron.GEOMETRY). Left context is left at the model's.
CHUNK, RIGHT_CONTEXT, FIFO, SPEAKER_CACHE, UPDATE_PERIOD = 340, 40, 40, 264, 300

# Probabilities older than this are re-read after every push: the newest
# frames can still be revised while later audio is processed.
REREAD_SECONDS = 60.0


class ModelConfig(ctypes.Structure):
    """``nemo_speech_diar_model_config`` (include/nemo_speech/diar.h, 0.2.0)."""

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("model_path", ctypes.c_char_p),
        ("gpu", ctypes.c_int32),
        ("preset", ctypes.c_char_p),
        ("chunk_frames", ctypes.c_int32),
        ("right_context_frames", ctypes.c_int32),
        ("left_context_frames", ctypes.c_int32),
        ("fifo_frames", ctypes.c_int32),
        ("spkcache_frames", ctypes.c_int32),
        ("update_period_frames", ctypes.c_int32),
    ]


def load(library: Path) -> Any:
    """The C library, with every function this uses given its signature."""
    if sys.platform == "win32":
        os.add_dll_directory(str(library.parent))
    lib = ctypes.CDLL(str(library))
    handle = ctypes.c_void_p
    out_handle = ctypes.POINTER(ctypes.c_void_p)
    signatures = {
        "nemo_speech_diar_create": (
            ctypes.c_int,
            [ctypes.POINTER(ModelConfig), out_handle],
        ),
        "nemo_speech_diar_destroy": (None, [handle]),
        "nemo_speech_diar_num_speakers": (ctypes.c_int32, [handle]),
        "nemo_speech_diar_seconds_per_frame": (ctypes.c_double, [handle]),
        "nemo_speech_diar_stream_open": (ctypes.c_int, [handle, out_handle]),
        "nemo_speech_diar_stream_push_f32": (
            ctypes.c_int,
            [handle, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t, ctypes.c_int32],
        ),
        "nemo_speech_diar_stream_finish": (ctypes.c_int, [handle]),
        "nemo_speech_diar_stream_close": (None, [handle]),
        "nemo_speech_diar_frame_count": (ctypes.c_int64, [handle]),
        "nemo_speech_diar_frame_probs_start": (ctypes.c_int64, [handle]),
        "nemo_speech_diar_frame_probs": (
            ctypes.c_int,
            [handle, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t],
        ),
        "nemo_speech_asr_last_error": (ctypes.c_char_p, []),
    }
    for name, (restype, argtypes) in signatures.items():
        function = getattr(lib, name)
        function.restype = restype
        function.argtypes = argtypes
    return lib


class LibraryError(RuntimeError):
    pass


def _check(lib: Any, status: int, what: str) -> None:
    if status != OK:
        detail = lib.nemo_speech_asr_last_error()
        message = detail.decode("utf-8", "replace") if detail else f"status {status}"
        raise LibraryError(f"{what} failed: {message}")


def read_wav(path: Path) -> "Any":
    """A 16 kHz mono 16-bit WAV as float32 samples."""
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        channels, rate, width = (
            wav.getnchannels(),
            wav.getframerate(),
            wav.getsampwidth(),
        )
        raw = wav.readframes(wav.getnframes())
    if width != 2 or rate != SAMPLE_RATE:
        raise LibraryError(
            f"expected 16 kHz 16-bit audio, got {rate} Hz {8 * width}-bit"
        )
    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return np.ascontiguousarray(samples, dtype=np.float32)


def _write(path: Path, text: str) -> None:
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _progress(out: Path, fraction: float) -> None:
    """Best effort: on Windows the replace fails while the parent is reading it."""
    try:
        _write(out / "progress", f"{min(1.0, fraction):.4f}")
    except OSError:
        pass


def diarize(lib: Any, model_path: str, wav: Path, out: Path) -> None:
    import numpy as np

    config = ModelConfig(
        size=ctypes.sizeof(ModelConfig),
        model_path=model_path.encode("ascii"),
        gpu=-1,
        preset=None,
        chunk_frames=CHUNK,
        right_context_frames=RIGHT_CONTEXT,
        left_context_frames=-1,
        fifo_frames=FIFO,
        spkcache_frames=SPEAKER_CACHE,
        update_period_frames=UPDATE_PERIOD,
    )
    model = ctypes.c_void_p()
    _check(
        lib,
        lib.nemo_speech_diar_create(ctypes.byref(config), ctypes.byref(model)),
        "loading the model",
    )
    stream = ctypes.c_void_p()
    try:
        speakers = int(lib.nemo_speech_diar_num_speakers(model))
        step = float(lib.nemo_speech_diar_seconds_per_frame(model))
        _check(
            lib,
            lib.nemo_speech_diar_stream_open(model, ctypes.byref(stream)),
            "opening a stream",
        )
        samples = read_wav(wav)
        expected = int(len(samples) / SAMPLE_RATE / step) + 1
        probs = np.zeros((max(expected, 1), speakers), dtype=np.float32)
        copied = 0
        reread = int(REREAD_SECONDS / step)

        def collect() -> int:
            nonlocal probs
            count = int(lib.nemo_speech_diar_frame_count(stream))
            first = int(lib.nemo_speech_diar_frame_probs_start(stream))
            if count <= first:
                return count
            retained = np.empty((count - first, speakers), dtype=np.float32)
            _check(
                lib,
                lib.nemo_speech_diar_frame_probs(
                    stream,
                    retained.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    retained.size,
                ),
                "reading probabilities",
            )
            if count > len(probs):
                grown = np.zeros((count, speakers), dtype=np.float32)
                grown[: len(probs)] = probs
                probs = grown
            # Frames older than the re-read window were final when last read,
            # and may since have been dropped from what the library retains.
            start = max(first, copied - reread)
            probs[start:count] = retained[start - first :]
            return count

        chunk = SAMPLE_RATE * PUSH_SECONDS
        for offset in range(0, len(samples), chunk):
            piece = samples[offset : offset + chunk]
            _check(
                lib,
                lib.nemo_speech_diar_stream_push_f32(
                    stream,
                    piece.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    len(piece),
                    SAMPLE_RATE,
                ),
                "processing audio",
            )
            copied = collect()
            _progress(out, (offset + len(piece)) / len(samples))
        _check(lib, lib.nemo_speech_diar_stream_finish(stream), "finishing")
        copied = collect()
        np.save(out / "probs.npy", probs[:copied])
        _write(
            out / "result.json",
            json.dumps(
                {"seconds_per_frame": step, "frames": copied, "speakers": speakers}
            ),
        )
    finally:
        if stream:
            lib.nemo_speech_diar_stream_close(stream)
        lib.nemo_speech_diar_destroy(model)


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 4:
        return 2
    library, model, wav, out = Path(args[0]), args[1], Path(args[2]), Path(args[3])
    try:
        lib = load(library)
        diarize(lib, model, wav, out)
    except Exception as exc:  # noqa: BLE001 - reported to the parent, which explains
        _write(out / "error.txt", f"{type(exc).__name__}: {exc}")
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - run as a process
    raise SystemExit(main())
