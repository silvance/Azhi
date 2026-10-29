"""Saving a passage of a recording as audio, with a record of where it came from."""

import json
import math
import struct
import wave

import pytest

from azhi import clips
from azhi.hashing import sha256_file
from azhi.transcription import Segment


def _ffmpeg():
    from azhi.resources import find_ffmpeg

    return find_ffmpeg()


def _recording(path, seconds=5.0, rate=48000, channels=2):
    """A tone, as a stand-in for an original recording."""
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        frames = bytearray()
        for n in range(int(seconds * rate)):
            sample = int(8000 * math.sin(2 * math.pi * 220 * n / rate))
            frames += struct.pack("<h", sample) * channels
        out.writeframes(bytes(frames))
    return path


def test_a_clip_is_named_after_its_recording_and_time_range():
    name = clips.clip_name("/cases/interview.m4a", 754.5, 770.0)
    assert name == "interview 00-12-34 to 00-12-50.wav"


def test_the_lines_a_passage_covers():
    lines = [Segment(0, 2, "a"), Segment(2, 4, "b"), Segment(4, 6, "c")]
    assert [s.text for s in clips.lines_within(lines, 1.5, 4.0)] == ["a", "b"]


def test_a_missing_recording_or_empty_passage_is_said_plainly(tmp_path, monkeypatch):
    monkeypatch.setattr(clips, "find_ffmpeg", lambda: "ffmpeg")
    with pytest.raises(clips.ClipError, match="not where the transcript says"):
        clips.cut(tmp_path / "gone.wav", 1.0, 2.0, tmp_path / "clip.wav")
    source = tmp_path / "here.wav"
    source.write_bytes(b"")
    with pytest.raises(clips.ClipError, match="no length"):
        clips.cut(source, 2.0, 2.0, tmp_path / "clip.wav")


@pytest.mark.skipif(_ffmpeg() is None, reason="needs ffmpeg")
def test_a_passage_is_cut_at_the_recordings_own_quality_with_room_either_side(
    tmp_path,
):
    source = _recording(tmp_path / "interview.wav")
    out = tmp_path / "clip.wav"
    first, last = clips.cut(source, 1.0, 2.0, out)
    assert (first, last) == (0.75, 2.25)
    with wave.open(str(out), "rb") as clip:
        assert clip.getframerate() == 48000 and clip.getnchannels() == 2
        assert abs(clip.getnframes() / 48000 - 1.5) < 0.02
    # The start is never before the recording.
    first, _ = clips.cut(source, 0.1, 0.5, tmp_path / "early.wav")
    assert first == 0.0


@pytest.mark.skipif(_ffmpeg() is None, reason="needs ffmpeg")
def test_the_record_says_where_the_clip_came_from(tmp_path):
    source = _recording(tmp_path / "interview.wav")
    out = tmp_path / "clip.wav"
    first, last = clips.cut(source, 1.0, 2.0, out)
    lines = [Segment(0.9, 2.1, " Meet at the bridge.", speaker="SPEAKER_00")]
    path = clips.write_clip_record(
        out,
        source=source,
        start=first,
        end=last,
        lines=lines,
        speaker_label=lambda sid: {"SPEAKER_00": "Ann"}[sid],
    )
    assert path.name == "clip.wav.clip.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["source_file"] == "interview.wav"
    assert record["source_sha256"] == sha256_file(source)
    assert record["sha256"] == sha256_file(out)
    assert (record["start"], record["end"]) == ("00:00:00.750", "00:00:02.250")
    assert (record["sample_rate"], record["channels"]) == (48000, 2)
    assert record["lines"] == [
        {
            "start": "00:00:00.900",
            "end": "00:00:02.100",
            "text": "Meet at the bridge.",
            "speaker": "Ann",
        }
    ]
