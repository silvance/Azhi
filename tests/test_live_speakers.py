"""Speakers in a live stream, worked out in passes behind the words.

The speaker separation is a stand-in that answers from a known timeline, so
these check the bookkeeping: chunks placed in stream time, names kept steady
from pass to pass, a last pass when the stream stops, and failures that leave
the words alone. How well the real engines separate voices is measured with a
test set (docs/test-set.md), not here.
"""

import threading
import wave

import numpy as np

from sotto.diarization import SpeakerSegment
from sotto.live_speakers import SpeakerReviewer, merge_pass, stitch_labels
from sotto.transcription import Segment, Word


def turn(start, end, speaker):
    return SpeakerSegment(start=start, end=end, speaker=speaker)


def _chunk(path, seconds):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.zeros(int(seconds * 16000), dtype=np.int16).tobytes())
    return path


def _line(start, end, text):
    return Segment(start, end, text, words=[Word(start, end, f" {text}")])


# -- Keeping names steady --------------------------------------------------------


def test_a_new_pass_takes_the_names_it_overlaps_most():
    previous = [turn(0, 10, "SPEAKER_00"), turn(10, 20, "SPEAKER_01")]
    # The new pass called them the other way round.
    new = [turn(5, 10, "SPEAKER_01"), turn(10, 25, "SPEAKER_00")]
    assert stitch_labels(previous, new, since=5) == {
        "SPEAKER_01": "SPEAKER_00",
        "SPEAKER_00": "SPEAKER_01",
    }


def test_someone_new_gets_the_next_number_in_order_of_speaking():
    previous = [turn(0, 10, "SPEAKER_00"), turn(10, 20, "SPEAKER_01")]
    new = [
        turn(10, 20, "SPEAKER_02"),
        turn(30, 31, "SPEAKER_00"),
        turn(25, 26, "SPEAKER_01"),
    ]
    mapping = stitch_labels(previous, new, since=10)
    assert mapping["SPEAKER_02"] == "SPEAKER_01"
    # Neither of the others overlaps anyone earlier: new people, numbered in
    # the order they speak, never folded into someone already named.
    assert mapping["SPEAKER_01"] == "SPEAKER_02"
    assert mapping["SPEAKER_00"] == "SPEAKER_03"


def test_turns_before_the_window_are_kept_and_cut_where_it_starts():
    previous = [turn(0, 8, "SPEAKER_00"), turn(8, 12, "SPEAKER_01")]
    new = [turn(10, 15, "SPEAKER_00")]
    merged = merge_pass(previous, new, since=10)
    assert merged == [
        turn(0, 8, "SPEAKER_00"),
        turn(8, 10, "SPEAKER_01"),
        turn(10, 15, "SPEAKER_01"),
    ]


# -- The reviewer ----------------------------------------------------------------


class Timeline:
    """Answers each pass from the true turns inside its window."""

    def __init__(self, truth, rename=None):
        self.truth = truth
        self.rename = rename or {}
        self.windows = []

    def __call__(self, wav, *, backend, num_speakers, cancelled):
        with wave.open(str(wav), "rb") as clip:
            seconds = clip.getnframes() / clip.getframerate()
        # The window's start is the total heard minus this clip.
        since = self.heard() - seconds
        self.windows.append((round(since, 3), round(since + seconds, 3)))
        found = []
        for t in self.truth:
            start, end = max(t.start, since), min(t.end, since + seconds)
            if end > start:
                name = self.rename.get(len(self.windows), {}).get(t.speaker, t.speaker)
                found.append(turn(start - since, end - since, name))
        return found


def _reviewer(separate, **kwargs):
    changes = []
    done = threading.Event()

    def on_change(labelled, until, pending):
        changes.append((labelled, until, pending))

    reviewer = SpeakerReviewer(
        on_change=on_change, separate=separate, review_every=10.0, **kwargs
    )
    separate.heard = lambda: reviewer._samples / 16000
    return reviewer, changes, done


def test_chunks_are_placed_in_stream_time_and_labelled_behind_them(tmp_path):
    truth = [turn(0, 5, "A"), turn(5, 6, "B"), turn(6, 12, "A")]
    separate = Timeline(truth)
    reviewer, changes, _ = _reviewer(separate)
    # Two 6 s chunks; the second's line is timed from its own start.
    reviewer.add(
        _chunk(tmp_path / "1.wav", 6),
        [_line(0.5, 4.5, "hello"), _line(5.1, 5.8, "yes")],
    )
    reviewer.add(_chunk(tmp_path / "2.wav", 6), [_line(1.0, 5.0, "right")])
    reviewer.finish()
    reviewer.join(10)
    labelled, until, pending = changes[-1]
    assert until == 12.0 and pending == []
    assert [(s.text, s.speaker) for s in labelled] == [
        ("hello", "SPEAKER_00"),
        ("yes", "SPEAKER_01"),
        ("right", "SPEAKER_00"),
    ]
    # The second chunk's words were moved to where it sits in the stream.
    assert labelled[-1].start == 7.0
    # Every chunk was shown at once, before any pass had labelled it.
    first_labelled, first_until, first_pending = changes[0]
    assert first_labelled == [] and first_until == 0.0
    assert [s.text for s in first_pending] == ["hello", "yes"]


def test_names_stay_put_when_a_later_pass_numbers_them_differently(tmp_path):
    truth = [turn(0, 10, "A"), turn(10, 20, "B"), turn(20, 30, "A")]
    # The second pass calls A "B" and B "A".
    separate = Timeline(truth, rename={2: {"A": "B", "B": "A"}})
    reviewer, changes, _ = _reviewer(separate)
    for n in range(3):
        reviewer.add(_chunk(tmp_path / f"{n}.wav", 10), [_line(1.0, 9.0, f"line{n}")])
        # Let each pass happen before the next chunk arrives.
        for _ in range(100):
            if reviewer._reviewed_until >= 10.0 * (n + 1) - 1e-9:
                break
            threading.Event().wait(0.02)
    reviewer.finish()
    reviewer.join(10)
    labelled = changes[-1][0]
    speakers = [s.speaker for s in labelled]
    assert speakers[0] == speakers[2] != speakers[1]


def test_a_long_stream_is_gone_over_only_as_far_back_as_the_window(tmp_path):
    truth = [turn(0, 40, "A")]
    separate = Timeline(truth)
    reviewer, changes, _ = _reviewer(separate, window=15.0)
    for n in range(4):
        reviewer.add(_chunk(tmp_path / f"{n}.wav", 10), [_line(1.0, 9.0, "x")])
    reviewer.finish()
    reviewer.join(10)
    assert all(end - start <= 15.0 + 1e-6 for start, end in separate.windows)
    # Lines older than the window kept the speaker they had.
    words = [(w.word, s.speaker) for s in changes[-1][0] for w in s.words]
    assert words == [(" x", "SPEAKER_00")] * 4


def test_a_failed_pass_leaves_the_words_unlabelled_and_says_so(tmp_path):
    errors = []

    def broken(wav, **_kwargs):
        raise RuntimeError("no engine")

    changes = []
    reviewer = SpeakerReviewer(
        on_change=lambda *a: changes.append(a),
        on_error=errors.append,
        separate=broken,
        review_every=5.0,
    )
    reviewer.add(_chunk(tmp_path / "1.wav", 6), [_line(1.0, 2.0, "hello")])
    reviewer.finish()
    reviewer.join(10)
    assert [str(e) for e in errors] == ["no engine"]
    labelled, _until, pending = changes[-1]
    assert labelled == [] and [s.text for s in pending] == ["hello"]


def test_cancel_stops_without_a_last_pass_and_deletes_the_audio(tmp_path):
    separate = Timeline([turn(0, 3, "A")])
    reviewer, _changes, _ = _reviewer(separate)
    reviewer.add(_chunk(tmp_path / "1.wav", 3), [_line(0.5, 2.0, "hi")])
    folder = reviewer._folder
    assert folder.is_dir()
    reviewer.cancel()
    reviewer.join(10)
    assert separate.windows == []
    assert not folder.exists()
