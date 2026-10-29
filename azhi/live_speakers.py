"""Who is speaking in a live stream, worked out a little behind the words.

Speaker separation needs to hear a voice for a while before it can tell it
from another, so it cannot label each few-second chunk as it is transcribed.
This keeps the stream's audio as it arrives and, whenever about
:data:`REVIEW_EVERY_SECONDS` more has come in and the last pass has finished,
separates the speakers again over the most recent :data:`WINDOW_SECONDS` and
gives every transcribed word to a speaker. Earlier lines can change speaker
on a later pass, because a later pass has heard more of each voice. When the
stream stops, the end is gone over once more.

A pass labels its speakers from scratch, so its SPEAKER_00 is not the last
pass's SPEAKER_00. Each new speaker takes the name of the earlier speaker it
shares the most time with, where the two passes overlap; one that matches
nobody gets the next unused number. Lines older than the window keep the
speaker they last had. Someone silent for longer than the window who then
speaks again has no time in common with either pass to be matched on, and
comes back under a new number.

The audio is kept in the session's temporary folder, as the chunks it is
built from already were, and deleted when the session ends.
"""

from __future__ import annotations

import copy
import shutil
import tempfile
import threading
import wave
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence

from .diarization import SpeakerSegment, assign_speakers, diarize
from .speaker_names import cluster_index
from .transcription import CancelledError, Segment

SAMPLE_RATE = 16000
# New audio, in seconds, before the speakers are gone over again.
REVIEW_EVERY_SECONDS = 20.0
# How far back each pass looks. Longer hears each voice for longer; shorter
# keeps a pass quick on a long stream.
WINDOW_SECONDS = 600.0

# (lines given to speakers, up to ``until`` seconds of stream; lines after it,
# not yet gone over). Called with the reviewer's lock held, in order, after
# every chunk and every pass: it should only hand the lists on.
ChangeCallback = Callable[[List[Segment], float, List[Segment]], None]


def stitch_labels(
    previous: Sequence[SpeakerSegment], new: Sequence[SpeakerSegment], since: float
) -> Dict[str, str]:
    """Rename a pass's speakers after the earlier speakers they overlap most.

    Only time from ``since`` (where the new pass starts) counts. A new speaker
    overlapping no earlier one is given the next unused ``SPEAKER_nn``, in the
    order they first speak.
    """
    shared: Dict[tuple, float] = {}
    for turn in new:
        for old in previous:
            overlap = min(turn.end, old.end) - max(turn.start, old.start, since)
            if overlap > 0:
                key = (turn.speaker, old.speaker)
                shared[key] = shared.get(key, 0.0) + overlap
    mapping: Dict[str, str] = {}
    for (label, old_label), _seconds in sorted(
        shared.items(), key=lambda item: -item[1]
    ):
        if label not in mapping and old_label not in mapping.values():
            mapping[label] = old_label
    numbers = [cluster_index(t.speaker) for t in previous]
    following = max([n for n in numbers if n is not None], default=-1) + 1
    for turn in sorted(new, key=lambda t: t.start):
        if turn.speaker not in mapping:
            mapping[turn.speaker] = f"SPEAKER_{following:02d}"
            following += 1
    return mapping


def merge_pass(
    previous: Sequence[SpeakerSegment], new: Sequence[SpeakerSegment], since: float
) -> List[SpeakerSegment]:
    """Earlier turns up to ``since``, then the new pass's, renamed to match."""
    mapping = stitch_labels(previous, new, since)
    kept = [
        SpeakerSegment(start=t.start, end=min(t.end, since), speaker=t.speaker)
        for t in previous
        if t.start < since
    ]
    renamed = [
        SpeakerSegment(start=t.start, end=t.end, speaker=mapping[t.speaker])
        for t in new
    ]
    return sorted(kept + renamed, key=lambda t: (t.start, t.end))


class SpeakerReviewer:
    """Keeps a stream's audio and words, and labels the words in passes."""

    def __init__(
        self,
        *,
        on_change: ChangeCallback,
        backend: str = "auto",
        num_speakers: Optional[int] = None,
        on_status: Optional[Callable[[str], None]] = None,
        on_error: Optional[Callable[[Exception], None]] = None,
        on_finished: Optional[Callable[[], None]] = None,
        review_every: float = REVIEW_EVERY_SECONDS,
        window: float = WINDOW_SECONDS,
        separate: Callable[..., List[SpeakerSegment]] = diarize,
    ) -> None:
        self._on_change = on_change
        self._on_status = on_status
        self._on_error = on_error
        self._on_finished = on_finished
        self._backend = backend
        self._num_speakers = num_speakers
        self._review_every = review_every
        self._window = window
        self._separate = separate

        self._folder = Path(tempfile.mkdtemp(prefix="azhi-live-speakers-"))
        self._audio = self._folder / "stream.pcm"
        self._audio.touch()
        self._lock = threading.Lock()
        self._wake = threading.Condition(self._lock)
        self._samples = 0
        self._segments: List[Segment] = []
        self._turns: List[SpeakerSegment] = []
        self._labelled: List[Segment] = []
        self._reviewed_until = 0.0
        self._failed_at = 0.0
        self._finishing = False
        self._cancelled = threading.Event()
        self._error_reported = False
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    @property
    def seconds(self) -> float:
        """Seconds of stream heard so far."""
        with self._lock:
            return self._samples / SAMPLE_RATE

    def add(self, wav_path: Path, segments: Sequence[Segment]) -> float:
        """Keep a finished chunk's audio and its transcript, together.

        ``segments`` are timed from the chunk's start (none if it could not be
        transcribed); returns where the chunk starts in the stream. Audio and
        words go in at once, so a pass never hears audio whose words it lacks.
        """
        with wave.open(str(wav_path), "rb") as chunk:
            if (
                chunk.getframerate() != SAMPLE_RATE
                or chunk.getsampwidth() != 2
                or chunk.getnchannels() != 1
            ):
                raise RuntimeError("live chunks must be 16 kHz, 16-bit mono WAV")
            frames = chunk.readframes(chunk.getnframes())
        with self._lock:
            offset = self._samples / SAMPLE_RATE
            with self._audio.open("ab") as out:
                out.write(frames)
            self._samples += len(frames) // 2
            for segment in segments:
                item = copy.deepcopy(segment)
                item.start += offset
                item.end += offset
                for word in item.words:
                    word.start += offset
                    word.end += offset
                self._segments.append(item)
            self._wake.notify_all()
            self._report()
        return offset

    def _report(self) -> None:
        """Hand on what is labelled and what is not yet. Lock held."""
        pending = [
            copy.deepcopy(s) for s in self._segments if s.start >= self._reviewed_until
        ]
        self._on_change(list(self._labelled), self._reviewed_until, pending)

    def finish(self) -> None:
        """No more audio: go over the end once more, then stop."""
        with self._lock:
            self._finishing = True
            self._wake.notify_all()

    def cancel(self) -> None:
        """Stop now, without a last pass, and delete the audio."""
        self._cancelled.set()
        with self._lock:
            self._finishing = True
            self._wake.notify_all()

    def join(self, timeout: Optional[float] = None) -> None:
        self._thread.join(timeout)

    # -- The passes ------------------------------------------------------------

    def _due(self) -> bool:
        heard = self._samples / SAMPLE_RATE
        last = max(self._reviewed_until, self._failed_at)
        if self._finishing:
            return heard > last
        return heard - last >= self._review_every

    def _run(self) -> None:
        try:
            while not self._cancelled.is_set():
                with self._lock:
                    while not self._due() and not self._finishing:
                        self._wake.wait(1.0)
                        if self._cancelled.is_set():
                            return
                    if not self._due():
                        return  # finishing, and nothing new since the last pass
                    finishing = self._finishing
                    # At most half a window past the last pass, so the two
                    # always share time to match names on, however far
                    # behind a slow engine has fallen. It catches up in steps.
                    last = max(self._reviewed_until, self._failed_at)
                    until = min(self._samples / SAMPLE_RATE, last + self._window / 2)
                try:
                    self._review(until)
                except CancelledError:
                    return
                except Exception as exc:  # noqa: BLE001 - surfaced, words unaffected
                    if self._on_error is not None and not self._error_reported:
                        self._error_reported = True
                        self._on_error(exc)
                    # The words stay as they are, unlabelled; the next pass
                    # waits for more audio rather than failing in a loop.
                    with self._lock:
                        self._failed_at = until
                if finishing:
                    with self._lock:
                        if not self._due():
                            return
        finally:
            shutil.rmtree(self._folder, ignore_errors=True)
            if self._on_finished is not None and not self._cancelled.is_set():
                self._on_finished()

    def _review(self, until: float) -> None:
        since = max(0.0, until - self._window)
        clip = self._folder / "pass.wav"
        first = int(since * SAMPLE_RATE)
        last = int(until * SAMPLE_RATE)
        with self._audio.open("rb") as source:
            source.seek(first * 2)
            frames = source.read((last - first) * 2)
        with wave.open(str(clip), "wb") as out:
            out.setnchannels(1)
            out.setsampwidth(2)
            out.setframerate(SAMPLE_RATE)
            out.writeframes(frames)
        if self._on_status is not None:
            self._on_status("Working out who is speaking…")
        found = self._separate(
            clip,
            backend=self._backend,
            num_speakers=self._num_speakers,
            cancelled=self._cancelled.is_set,
        )
        moved = [
            SpeakerSegment(start=t.start + since, end=t.end + since, speaker=t.speaker)
            for t in found
        ]
        with self._lock:
            self._turns = merge_pass(self._turns, moved, since)
            words = [copy.deepcopy(s) for s in self._segments if s.start < until]
            self._labelled = assign_speakers(words, self._turns)
            self._reviewed_until = until
            self._report()
        if self._on_status is not None:
            self._on_status(f"Speakers checked up to {_clock(until)}.")


def _clock(seconds: float) -> str:
    whole = int(seconds)
    hours, rest = divmod(whole, 3600)
    minutes, secs = divmod(rest, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}" if hours else f"{minutes}:{secs:02d}"
