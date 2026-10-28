import io
from pathlib import Path

import pytest

from whispers import live
from whispers.live import LiveTranscriber, LiveTranscriptionError


class _FakeResult:
    def __init__(self, returncode, stderr=b""):
        self.returncode = returncode
        self.stderr = stderr


class _FakePopen:
    """Stands in for a finished ffmpeg process so start()/stop() run without one."""

    def __init__(self, args, **_kwargs):
        self.args = args
        self.stdin = io.BytesIO()
        self.stderr = io.BytesIO(b"")

    def poll(self):
        return 0  # already exited -> the watch thread drains and stops promptly

    def wait(self, timeout=None):
        return 0

    def terminate(self):
        pass

    def kill(self):
        pass


def test_start_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(live, "find_ffmpeg", lambda: None)
    lt = LiveTranscriber("rtmp://localhost/live/stream")
    with pytest.raises(LiveTranscriptionError, match="ffmpeg"):
        lt.start(on_text=lambda _t: None)


def test_segment_seconds_is_clamped():
    lt = LiveTranscriber("rtmp://x", segment_seconds=0)
    assert lt._segment_seconds >= 2


def test_ffmpeg_args_pull_mode(monkeypatch):
    captured = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakePopen(args, **kwargs)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "Popen", _fake_popen)

    lt = LiveTranscriber("rtmp://localhost/live/stream", segment_seconds=8)
    lt.start(on_text=lambda _t: None)
    lt.stop()

    args = captured["args"]
    assert "rtmp://localhost/live/stream" in args
    assert "-vn" in args  # audio only
    assert "-listen" not in args  # pull mode
    # 16 kHz mono segmented output at the requested length
    assert args[args.index("-segment_time") + 1] == "8"
    assert args[args.index("-ar") + 1] == str(live.LIVE_SAMPLE_RATE)
    # temp dir is cleaned up on stop
    assert lt._tmpdir is None


def test_ffmpeg_args_listen_mode(monkeypatch):
    captured = {}

    def _fake_popen(args, **kwargs):
        captured["args"] = args
        return _FakePopen(args, **kwargs)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "Popen", _fake_popen)

    lt = LiveTranscriber("rtmp://0.0.0.0/live", listen=True)
    lt.start(on_text=lambda _t: None)
    lt.stop()

    args = captured["args"]
    assert args.index("-listen") < args.index("-i")  # -listen precedes the input


def test_test_connection_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(live, "find_ffmpeg", lambda: None)
    ok, message = live.test_connection("rtmp://x")
    assert ok is False
    assert "ffmpeg" in message


def test_test_connection_success(monkeypatch):
    captured = {}

    def _fake_run(args, **kwargs):
        captured["args"] = args
        return _FakeResult(returncode=0)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "run", _fake_run)
    ok, message = live.test_connection("rtmp://localhost/live/stream")
    assert ok is True
    args = captured["args"]
    # Probe reads a bounded slice and discards output.
    assert "-t" in args and args[args.index("-f") + 1] == "null"
    assert "rtmp://localhost/live/stream" in args


def test_test_connection_failure_reports_stderr(monkeypatch):
    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(
        live.subprocess,
        "run",
        lambda args, **kw: _FakeResult(returncode=1, stderr=b"Connection refused"),
    )
    ok, message = live.test_connection("rtmp://nope")
    assert ok is False
    assert "Connection refused" in message


def test_test_connection_timeout(monkeypatch):
    import subprocess

    def _raise_timeout(args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=args, timeout=1)

    monkeypatch.setattr(live, "find_ffmpeg", lambda: Path("ffmpeg"))
    monkeypatch.setattr(live.subprocess, "run", _raise_timeout)
    ok, message = live.test_connection("rtmp://slow", listen=True)
    assert ok is False
    assert "timeout" in message.lower()


def test_each_chunk_goes_to_the_speaker_reviewer_with_word_times(tmp_path, monkeypatch):
    from whispers.transcription import Segment, TranscriptionResult, Word

    asked = []

    def fake_transcribe(path, **kwargs):
        asked.append(kwargs["word_timestamps"])
        return TranscriptionResult(
            text=f"text {Path(path).stem}",
            language="en",
            language_probability=1.0,
            duration=1.0,
            segments=[Segment(0.1, 0.9, "x", words=[Word(0.1, 0.9, " x")])],
        )

    class Reviewer:
        def __init__(self):
            self.chunks = []
            self.finished = False

        def add(self, wav, segments):
            self.chunks.append((wav.name, [s.text for s in segments]))
            return 0.0

        def finish(self):
            self.finished = True

    class Done:
        """ffmpeg, already finished: every chunk on disk is final."""

        def poll(self):
            return 0

    monkeypatch.setattr(live, "transcribe_audio", fake_transcribe)
    reviewer = Reviewer()
    transcriber = live.LiveTranscriber("rtmp://x", speakers=reviewer)
    transcriber._tmpdir = tmp_path
    transcriber._proc = Done()
    for name in ("seg_00000.wav", "seg_00001.wav"):
        (tmp_path / name).write_bytes(b"")
    texts = []
    transcriber._run_watch(texts.append, None, None)
    assert asked == [True, True]
    assert reviewer.chunks == [("seg_00000.wav", ["x"]), ("seg_00001.wav", ["x"])]
    assert reviewer.finished
    assert texts == ["text seg_00000", "text seg_00001"]


@pytest.mark.parametrize("model, primed", [("base.en", True), ("turbo", False)])
def test_earlier_chunks_prime_the_next_only_where_the_model_carries_context(
    monkeypatch, tmp_path, model, primed
):
    from whispers.transcription import TranscriptionResult

    prompts = []

    def fake_transcribe(path, **kwargs):
        prompts.append(kwargs["initial_prompt"])
        return TranscriptionResult(
            text="WORDS", language="en", language_probability=1.0, duration=1.0
        )

    monkeypatch.setattr(live, "transcribe_audio", fake_transcribe)
    transcriber = live.LiveTranscriber("rtmp://x", model_size=model)
    transcriber._transcribe_segment(tmp_path / "a.wav", "")
    transcriber._transcribe_segment(tmp_path / "b.wav", "WORDS")
    assert prompts == [None, "WORDS" if primed else None]
