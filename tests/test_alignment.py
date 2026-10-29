"""English word re-timing: the forced alignment, and what it leaves alone.

The letter probabilities here are made up, frame by frame, so each test knows
where every letter is. That checks the alignment and the bookkeeping around it;
it says nothing about how well the real model hears speech, which only a
labelled test set can (docs/test-set.md).
"""

import hashlib
import json
import wave

import numpy as np
import pytest

from azhi import alignment
from azhi.alignment import AlignmentSummary, align_segments, forced_align
from azhi.provenance import TranscriptionProvenance
from azhi.transcription import CancelledError, Segment, Word

# facebook/wav2vec2-base-960h's vocabulary.
VOCAB = {
    "<pad>": 0, "<s>": 1, "</s>": 2, "<unk>": 3, "|": 4, "E": 5, "T": 6, "A": 7,
    "O": 8, "N": 9, "I": 10, "H": 11, "S": 12, "R": 13, "D": 14, "L": 15,
    "U": 16, "M": 17, "W": 18, "C": 19, "F": 20, "G": 21, "Y": 22, "P": 23,
    "B": 24, "V": 25, "K": 26, "'": 27, "X": 28, "J": 29, "Q": 30, "Z": 31,
}  # fmt: skip


def _emissions(frames, marks):
    """Log-probabilities with ``marks[frame] = token`` likely, blank elsewhere."""
    probs = np.full((frames, len(VOCAB)), 0.001)
    probs[:, 0] = 0.97
    for frame, token in marks.items():
        probs[frame, :] = 0.001
        probs[frame, token] = 0.97
    probs /= probs.sum(axis=1, keepdims=True)
    return np.log(probs)


def _marks(spec):
    """``{"HI": 5, ...}``: each letter of a word on consecutive frames from there."""
    marks = {}
    for text, frame in spec:
        for offset, letter in enumerate(text):
            marks[frame + offset] = VOCAB[letter]
    return marks


class FakeAligner:
    """Stands in for the model: fixed letter frames, whatever the audio."""

    vocabulary = VOCAB
    blank = 0
    word_break = 4

    def __init__(self, spec, fail=False):
        self.spec = spec
        self.fail = fail
        self.calls = 0

    def log_probs(self, audio):
        self.calls += 1
        if self.fail:
            raise RuntimeError("model fell over")
        return _emissions(len(audio) // 320, _marks(self.spec))


def _wav(path, seconds=3.0):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(np.zeros(int(seconds * 16000), dtype=np.int16).tobytes())
    return path


# -- forced_align --------------------------------------------------------------


def test_each_letter_gets_the_frames_it_is_likely_in():
    log_probs = _emissions(20, {3: 7, 4: 7, 9: 24, 10: 4, 11: 4, 14: 7, 15: 7})
    spans = forced_align(log_probs, [7, 24, 4, 7])
    assert spans == [(3, 5), (9, 10), (10, 12), (14, 16)]


def test_a_doubled_letter_needs_a_blank_between():
    log_probs = _emissions(3, {0: 15, 2: 15})
    assert forced_align(log_probs[:2], [15, 15]) is None
    assert forced_align(log_probs, [15, 15]) == [(0, 1), (2, 3)]


def test_more_letters_than_frames_cannot_be_fitted():
    assert forced_align(_emissions(3, {}), [5, 6, 7, 8]) is None


def test_nothing_to_fit_is_fitted_trivially():
    assert forced_align(_emissions(5, {}), []) == []


def test_only_letters_the_model_knows_are_spelled():
    assert alignment.spellable("Don’t,", VOCAB) == "DON'T"
    assert alignment.spellable(" café.", VOCAB) == "CAFE"
    assert alignment.spellable("2019", VOCAB) == ""
    assert alignment.spellable("a|b", VOCAB) == "AB"


def test_only_english_transcriptions_are_retimed():
    assert alignment.applies("en")
    assert alignment.applies("en-GB")
    assert not alignment.applies("ru")
    assert not alignment.applies(None)
    # A translation's English words were never spoken.
    assert not alignment.applies("en", task="translate")


# -- Re-timing a transcript ----------------------------------------------------


def test_words_take_the_times_of_their_letters(tmp_path):
    # Whisper put "Yes" straight after "Hi"; the letters say it came later.
    # Window: 0.8 s to 2.2 s (0.2 s either side), 20 ms frames.
    segment = Segment(
        start=1.0,
        end=2.0,
        text="Hi. Yes.",
        words=[Word(1.0, 1.4, " Hi."), Word(1.4, 2.0, " Yes.")],
    )
    fake = FakeAligner([("HI", 5), ("YES", 40)])
    summary = align_segments([segment], _wav(tmp_path / "a.wav"), aligner=fake)
    hi, yes = segment.words
    assert (hi.start, hi.end) == (pytest.approx(0.9), pytest.approx(0.94))
    assert (yes.start, yes.end) == (pytest.approx(1.6), pytest.approx(1.66))
    # The words themselves are untouched.
    assert [w.word for w in segment.words] == [" Hi.", " Yes."]
    assert summary == AlignmentSummary(words=2, retimed=2, segments_kept=0)


def test_a_word_it_cannot_spell_stays_between_its_neighbours(tmp_path):
    segment = Segment(
        start=1.0,
        end=2.0,
        text="In 2019 yes",
        words=[Word(1.0, 1.2, " In"), Word(1.0, 1.9, " 2019"), Word(1.6, 2.0, " yes")],
    )
    fake = FakeAligner([("IN", 10), ("YES", 50)])
    summary = align_segments([segment], _wav(tmp_path / "a.wav"), aligner=fake)
    first, number, last = segment.words
    assert first.end <= number.start <= number.end <= last.start
    assert summary.retimed == 2 and summary.words == 3


def test_a_segment_it_cannot_fit_keeps_the_transcription_times(tmp_path):
    words = [Word(1.0, 1.01, " supercalifragilisticexpialidocious")]
    segment = Segment(start=1.0, end=1.01, text="x", words=list(words))
    # 0.41 s of window is 20 frames: too few for 34 letters.
    fake = FakeAligner([])
    summary = align_segments([segment], _wav(tmp_path / "a.wav"), aligner=fake)
    assert (segment.words[0].start, segment.words[0].end) == (1.0, 1.01)
    assert summary.segments_kept == 1 and summary.retimed == 0


def test_a_model_failure_costs_that_segment_only(tmp_path):
    segment = Segment(start=1.0, end=2.0, text="Hi", words=[Word(1.0, 1.4, " Hi")])
    fake = FakeAligner([], fail=True)
    summary = align_segments([segment], _wav(tmp_path / "a.wav"), aligner=fake)
    assert (segment.words[0].start, segment.words[0].end) == (1.0, 1.4)
    assert summary.segments_kept == 1


def test_segments_without_words_are_passed_over(tmp_path):
    fake = FakeAligner([])
    summary = align_segments(
        [Segment(start=0.0, end=1.0, text="")], _wav(tmp_path / "a.wav"), aligner=fake
    )
    assert fake.calls == 0 and summary == AlignmentSummary()


def test_cancel_stops_it(tmp_path):
    segment = Segment(start=1.0, end=2.0, text="Hi", words=[Word(1.0, 1.4, " Hi")])
    with pytest.raises(CancelledError):
        align_segments(
            [segment],
            _wav(tmp_path / "a.wav"),
            aligner=FakeAligner([]),
            cancelled=lambda: True,
        )


def test_it_says_how_many_words_it_retimed(tmp_path):
    lines = []
    segment = Segment(start=1.0, end=2.0, text="Hi", words=[Word(1.0, 1.4, " Hi")])
    align_segments(
        [segment],
        _wav(tmp_path / "a.wav"),
        aligner=FakeAligner([("HI", 12)]),
        progress=lines.append,
    )
    assert lines == [
        "Word timings: 1 of 1 words re-timed by the English word-timing model."
    ]


# -- Without the model ---------------------------------------------------------


def test_without_the_model_it_is_not_offered_and_says_why(tmp_path, monkeypatch):
    monkeypatch.setenv("AZHI_ASSETS", str(tmp_path))
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("azhi.resources.asset_dirs", lambda: [tmp_path])
    assert not alignment.available()
    with pytest.raises(RuntimeError, match="does not include the English"):
        alignment.load()


def test_the_build_record_names_the_model(tmp_path, monkeypatch):
    folder = tmp_path / "alignment"
    folder.mkdir()
    (folder / "model.onnx").write_bytes(b"weights")
    (folder / "vocab.json").write_text(json.dumps(VOCAB))
    monkeypatch.setattr("azhi.resources.asset_dirs", lambda: [tmp_path])
    assert "revision not recorded" in alignment.describe_method()
    (folder / "alignment.json").write_text(
        json.dumps({"model": {"repo": "onnx-community/x", "revision": "abc123"}})
    )
    assert alignment.describe_method().startswith("onnx-community/x@abc123, CTC")
    assert alignment.model_sha256() == hashlib.sha256(b"weights").hexdigest()


# -- The record ----------------------------------------------------------------


def test_the_record_keeps_how_the_words_were_timed():
    record = TranscriptionProvenance(
        model_name="medium",
        word_timing="onnx-community/x@abc, CTC forced alignment",
        word_timing_model_sha256="f" * 64,
        words_retimed=410,
        words_total=430,
    )
    again = TranscriptionProvenance.from_dict(record.to_dict())
    assert again == record


def test_an_older_record_says_nothing_about_word_timing():
    from azhi.provenance import AnalysisProvenance

    record = TranscriptionProvenance.from_dict({"model_name": "medium"})
    assert record.word_timing == "" and record.words_retimed is None
    lines = AnalysisProvenance(transcription=record).describe()
    assert not any("Word tim" in line for line in lines)


def test_the_report_names_the_word_timing_model():
    from azhi.provenance import AnalysisProvenance

    record = TranscriptionProvenance(
        model_name="medium",
        word_timing="onnx-community/x@abc, CTC forced alignment",
        word_timing_model_sha256="f" * 64,
        words_retimed=410,
        words_total=430,
    )
    lines = AnalysisProvenance(transcription=record).describe()
    assert (
        "Word timings re-fitted by: onnx-community/x@abc, CTC forced alignment "
        "(410 of 430 words)"
    ) in lines
    assert f"Word-timing model SHA-256: {'f' * 64}" in lines


# -- The runtime, when it is installed -----------------------------------------


def test_the_model_runs_through_onnxruntime(tmp_path, monkeypatch):
    # A one-layer stand-in with the real model's input and output: raw audio
    # in, 32 letter scores per 320 samples out.
    onnx = pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from onnx import TensorProto, helper

    weights = np.zeros((320, len(VOCAB)), dtype=np.float32)
    weights[:, 0] = 1.0
    graph = helper.make_graph(
        [
            helper.make_node("Reshape", ["input_values", "shape"], ["frames"]),
            helper.make_node("MatMul", ["frames", "w"], ["logits"]),
        ],
        "stand-in",
        [helper.make_tensor_value_info("input_values", TensorProto.FLOAT, [1, None])],
        [helper.make_tensor_value_info("logits", TensorProto.FLOAT, None)],
        initializer=[
            helper.make_tensor("shape", TensorProto.INT64, [3], [1, -1, 320]),
            helper.make_tensor(
                "w", TensorProto.FLOAT, weights.shape, weights.flatten().tolist()
            ),
        ],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8  # readable by every onnxruntime the build allows
    folder = tmp_path / "alignment"
    folder.mkdir()
    onnx.save(model, str(folder / "model.onnx"))
    (folder / "vocab.json").write_text(json.dumps(VOCAB))
    monkeypatch.setattr("azhi.resources.asset_dirs", lambda: [tmp_path])

    assert alignment.available()
    aligner = alignment.load()
    log_probs = aligner.log_probs(np.random.default_rng(0).normal(size=3200))
    assert log_probs.shape == (10, len(VOCAB))
    assert np.allclose(np.exp(log_probs).sum(axis=1), 1.0)
