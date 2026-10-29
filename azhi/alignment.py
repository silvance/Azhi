"""Sharper word timings for English, from a second model that spells.

Whisper times each word by where its attention was looking as it wrote it.
That is close, but on a short word it can be a few tenths of a second out, and
the error has a direction: a reply's first word is put a little early, inside
the end of the turn before it. "Who" then goes to the person who asked the
question, and the speaker separation was right all along.

This re-times the words Whisper heard; it never changes them. A wav2vec 2.0
model trained to spell English (facebook/wav2vec2-base-960h, Apache-2.0) says,
every 20 ms, how likely each letter is. A forced alignment finds where
Whisper's words, letter by letter, fit those likelihoods best, and each word
takes the start and end of its first and last letter.

What it cannot do, it leaves alone:

- English only. The model knows the 26 letters and the apostrophe; another
  language's words would be fitted to sounds they do not make. Nor a
  translation, whose English words were never spoken.
- A word with nothing it can spell ("2019", "%") keeps Whisper's time, kept
  between the words either side of it.
- A segment it cannot fit - more letters than the audio has frames for, or a
  model error - keeps Whisper's timings throughout.

The model is part of the bundle (``azhi_assets/alignment``), run with
onnxruntime, which the application already ships. Nothing is downloaded.
"""

from __future__ import annotations

import json
import unicodedata
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from .transcription import CancelCallback, CancelledError, ProgressCallback, Segment

PathLike = Union[str, Path]

SAMPLE_RATE = 16000
# Audio either side of a segment that its words may be fitted into: Whisper's
# segment edges are no more exact than its word times.
PAD_SECONDS = 0.2
BLANK = "<pad>"
WORD_BREAK = "|"

_NEG = float("-inf")


def applies(language: Optional[str], task: str = "transcribe") -> bool:
    """Whether re-timing makes sense: English speech, transcribed."""
    return task == "transcribe" and (language or "").lower().split("-")[0] == "en"


def available() -> bool:
    """Whether this build carries the aligner and can run it."""
    from .resources import bundled_aligner

    if bundled_aligner() is None:
        return False
    try:
        import onnxruntime  # noqa: F401
    except Exception:  # noqa: BLE001 - an absent library is an answer, not a fault
        return False
    return True


def describe_method() -> str:
    """The model and the rule, as a report states them."""
    from .resources import bundled_aligner_info

    model = bundled_aligner_info().get("model")
    repo = model.get("repo") if isinstance(model, dict) else None
    revision = model.get("revision") if isinstance(model, dict) else None
    name = (
        f"{repo}@{revision}"
        if repo and revision
        else f"{repo} (revision not recorded)"
        if repo
        else "wav2vec2-base-960h (revision not recorded)"
    )
    return (
        f"{name}, CTC forced alignment of the transcribed letters (20 ms "
        "frames); words it cannot spell keep the transcription's times"
    )


def model_sha256() -> Optional[str]:
    """SHA-256 of the bundled model file, read now; None when there is none."""
    from .hashing import sha256_file_or_none
    from .resources import bundled_aligner

    directory = bundled_aligner()
    return sha256_file_or_none(directory / "model.onnx") if directory else None


# -- Forced alignment ----------------------------------------------------------


def forced_align(
    log_probs: Any, tokens: Sequence[int], blank: int = 0
) -> Optional[List[Tuple[int, int]]]:
    """The frames each token occupies on the best CTC path through ``log_probs``.

    ``log_probs`` is frames x vocabulary (log-softmax). Returns one
    ``(first frame, last frame + 1)`` per token, in order, or ``None`` when the
    tokens cannot fit in the frames. Standard CTC: a blank may sit between any
    two tokens, and must between two equal ones.
    """
    import numpy as np

    frames = int(log_probs.shape[0])
    count = len(tokens)
    if count == 0:
        return []
    repeats = sum(1 for a, b in zip(tokens, tokens[1:]) if a == b)
    if frames < count + repeats:
        return None
    states = 2 * count + 1
    labels = np.full(states, blank, dtype=np.int64)
    labels[1::2] = np.asarray(tokens, dtype=np.int64)
    # A token may follow the previous token directly, skipping the blank
    # between them, unless the two are the same letter.
    skip = np.zeros(states, dtype=bool)
    skip[3::2] = labels[3::2] != labels[1:-2:2]

    emit = np.asarray(log_probs, dtype=np.float64)[:, labels]
    alpha = np.full(states, _NEG)
    alpha[0] = emit[0, 0]
    alpha[1] = emit[0, 1]
    back = np.zeros((frames, states), dtype=np.int8)
    candidates = np.empty((3, states))
    columns = np.arange(states)
    for t in range(1, frames):
        candidates[0] = alpha
        candidates[1, 0] = _NEG
        candidates[1, 1:] = alpha[:-1]
        candidates[2, :2] = _NEG
        candidates[2, 2:] = alpha[:-2]
        candidates[2, ~skip] = _NEG
        step = np.argmax(candidates, axis=0)
        alpha = candidates[step, columns] + emit[t]
        back[t] = step

    end = states - 1 if alpha[states - 1] >= alpha[states - 2] else states - 2
    if not np.isfinite(alpha[end]):
        return None
    path = np.empty(frames, dtype=np.int64)
    state = end
    for t in range(frames - 1, -1, -1):
        path[t] = state
        state -= int(back[t, state])

    spans: List[Tuple[int, int]] = []
    for index in range(count):
        hits = np.flatnonzero(path == 2 * index + 1)
        if hits.size == 0:  # cannot happen on a complete path; be safe
            return None
        spans.append((int(hits[0]), int(hits[-1]) + 1))
    return spans


def spellable(word: str, vocabulary: Dict[str, int]) -> str:
    """The letters of ``word`` the model can spell, upper-cased."""
    text = word.replace("’", "'").replace("‘", "'")
    text = unicodedata.normalize("NFKD", text)
    return "".join(c for c in text.upper() if c in vocabulary and c != WORD_BREAK)


# -- The model -----------------------------------------------------------------


class Aligner:
    """The bundled letter model, loaded once per run."""

    def __init__(self, directory: Path, threads: int = 2):
        import onnxruntime

        vocabulary = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
        self.vocabulary: Dict[str, int] = {
            str(k): int(v) for k, v in vocabulary.items()
        }
        self.blank = self.vocabulary.get(BLANK, 0)
        self.word_break = self.vocabulary.get(WORD_BREAK)
        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self._session = onnxruntime.InferenceSession(
            str(directory / "model.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        inputs = [i.name for i in self._session.get_inputs()]
        self._input = inputs[0]
        # Some exports ask for a mask as well; all of the audio is real.
        self._mask = "attention_mask" if "attention_mask" in inputs else None

    def log_probs(self, audio: Any) -> Any:
        """Frames x vocabulary log-probabilities for 16 kHz mono ``audio``."""
        import numpy as np

        samples = np.asarray(audio, dtype=np.float32)
        # The model was trained on audio scaled to zero mean, unit variance.
        samples = (samples - samples.mean()) / np.sqrt(samples.var() + 1e-7)
        feeds = {self._input: samples[np.newaxis, :]}
        if self._mask:
            feeds[self._mask] = np.ones((1, samples.size), dtype=np.int64)
        logits = self._session.run(None, feeds)[0]
        logits = np.asarray(logits[0], dtype=np.float64)
        peak = logits.max(axis=1, keepdims=True)
        return logits - peak - np.log(np.exp(logits - peak).sum(axis=1, keepdims=True))


def load() -> Aligner:
    """The bundled aligner, or a plain refusal to look anywhere else."""
    from .resources import bundled_aligner

    directory = bundled_aligner()
    if directory is None:
        raise RuntimeError(
            "This copy of the application does not include the English "
            "word-timing model. Word times stay as transcribed."
        )
    return Aligner(directory)


@dataclass
class AlignmentSummary:
    """What re-timing did to one transcript."""

    words: int = 0
    retimed: int = 0
    segments_kept: int = 0


def _read_wav(path: PathLike) -> Any:
    import numpy as np

    with wave.open(str(path), "rb") as wav:
        if wav.getframerate() != SAMPLE_RATE or wav.getsampwidth() != 2:
            raise RuntimeError("word re-timing needs a 16 kHz, 16-bit WAV")
        channels = wav.getnchannels()
        raw = wav.readframes(wav.getnframes())
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        data = data.reshape(-1, channels).mean(axis=1)
    return data


def _align_segment(segment: Segment, audio: Any, aligner: Aligner) -> int:
    """Re-time one segment's words in place; how many were re-timed."""
    words = segment.words
    letters = [spellable(w.word, aligner.vocabulary) for w in words]
    spelled = [i for i, text in enumerate(letters) if text]
    if not spelled:
        return 0
    duration = len(audio) / SAMPLE_RATE
    lo = max(0.0, min(segment.start, words[0].start) - PAD_SECONDS)
    hi = min(duration, max(segment.end, words[-1].end) + PAD_SECONDS)
    first, last = int(lo * SAMPLE_RATE), int(hi * SAMPLE_RATE)
    if last - first < SAMPLE_RATE // 10:
        return 0
    log_probs = aligner.log_probs(audio[first:last])
    seconds_per_frame = (last - first) / SAMPLE_RATE / log_probs.shape[0]

    tokens: List[int] = []
    owner: List[int] = []  # the word each token spells; -1 for a word break
    for n, index in enumerate(spelled):
        if n and aligner.word_break is not None:
            tokens.append(aligner.word_break)
            owner.append(-1)
        for letter in letters[index]:
            tokens.append(aligner.vocabulary[letter])
            owner.append(index)
    spans = forced_align(log_probs, tokens, blank=aligner.blank)
    if spans is None:
        return 0

    times: Dict[int, Tuple[float, float]] = {}
    for (start, end), index in zip(spans, owner):
        if index < 0:
            continue
        begin, finish = lo + start * seconds_per_frame, lo + end * seconds_per_frame
        if index in times:
            begin = times[index][0]
        times[index] = (begin, finish)
    for index, (begin, finish) in times.items():
        words[index].start = round(begin, 3)
        words[index].end = round(finish, 3)
    # A word it could not spell stays where Whisper put it, but not out of
    # order with the re-timed words either side.
    previous_end = lo
    for index, word in enumerate(words):
        if index in times:
            previous_end = word.end
            continue
        following = [words[j].start for j in range(index + 1, len(words)) if j in times]
        ceiling = following[0] if following else hi
        word.start = min(max(word.start, previous_end), ceiling)
        word.end = min(max(word.end, word.start), ceiling)
        previous_end = word.end
    segment.start = min(segment.start, words[0].start)
    segment.end = max(segment.end, words[-1].end)
    return len(times)


def align_segments(
    segments: Sequence[Segment],
    wav_path: PathLike,
    *,
    aligner: Optional[Aligner] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Any] = None,
    cancelled: Optional[CancelCallback] = None,
) -> AlignmentSummary:
    """Re-time every segment's words in place from the 16 kHz WAV they came from."""
    aligner = aligner or load()
    audio = _read_wav(wav_path)
    summary = AlignmentSummary()
    with_words = [s for s in segments if s.words]
    for done, segment in enumerate(with_words):
        if cancelled is not None and cancelled():
            raise CancelledError("Cancelled while re-timing words.")
        summary.words += len(segment.words)
        try:
            retimed = _align_segment(segment, audio, aligner)
        except Exception:  # noqa: BLE001 - one segment never costs the transcript
            retimed = 0
        if not retimed:
            summary.segments_kept += 1
        summary.retimed += retimed
        if on_progress is not None:
            on_progress((done + 1) / len(with_words))
    if progress is not None:
        progress(
            f"Word timings: {summary.retimed} of {summary.words} words re-timed "
            "by the English word-timing model."
        )
    return summary
