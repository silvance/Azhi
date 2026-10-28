from pathlib import Path

import pytest

from sotto import playback
from sotto.playback import (
    PlaybackError,
    SegmentPlayer,
    _player_args,
    playback_available,
)


def test_playback_available_false_without_ffmpeg(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: None)
    assert playback_available() is False


def test_playback_available_with_ffmpeg_and_player(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(playback, "_player_command", lambda: "/usr/bin/aplay")
    assert playback_available() is True


def test_play_segment_without_ffmpeg_raises(monkeypatch):
    monkeypatch.setattr(playback, "find_ffmpeg", lambda: None)
    with pytest.raises(PlaybackError, match="ffmpeg"):
        SegmentPlayer().play_segment("clip.wav", 0.0, 1.0)


def test_player_args_ffplay_uses_headless_flags():
    args = _player_args("/usr/bin/ffplay", Path("/tmp/x.wav"))
    assert "-autoexit" in args and "-nodisp" in args
    assert args[-1] == "/tmp/x.wav"


def test_player_args_simple_player_is_command_and_file():
    assert _player_args("/usr/bin/aplay", Path("/tmp/x.wav")) == [
        "/usr/bin/aplay",
        "/tmp/x.wav",
    ]


def test_stop_is_safe_when_nothing_playing():
    # No exception when there's no temp file or process.
    SegmentPlayer().stop()


@pytest.mark.parametrize(
    "speed, tempo", [(1.0, None), (0.75, "atempo=0.750"), (0.5, "atempo=0.500")]
)
def test_a_slower_speed_slows_the_line_and_keeps_its_pitch(
    monkeypatch, tmp_path, speed, tempo
):
    import subprocess

    calls = []

    def fake_run(args, **_kwargs):
        calls.append(args)
        Path(args[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(playback, "find_ffmpeg", lambda: Path("/usr/bin/ffmpeg"))
    monkeypatch.setattr(playback.subprocess, "run", fake_run)
    player = SegmentPlayer()
    monkeypatch.setattr(player, "_start_playback", lambda wav: None)
    player.play_segment("in.wav", 1.0, 3.0, speed=speed)
    args = calls[0]
    if tempo is None:
        assert "-af" not in args
    else:
        assert args[args.index("-af") + 1] == tempo
    # Still the same two seconds of the recording, however slowly played.
    assert args[args.index("-t") + 1] == "2.000"
    player.stop()


def test_every_offered_speed_is_within_what_ffmpeg_accepts():
    assert all(0.5 <= speed <= 2.0 for speed in playback.SPEEDS.values())
    assert next(iter(playback.SPEEDS.values())) == 1.0  # normal comes first


def _real_ffmpeg():
    from sotto.resources import find_ffmpeg

    return find_ffmpeg()


@pytest.mark.skipif(_real_ffmpeg() is None, reason="needs ffmpeg")
def test_half_speed_really_takes_twice_as_long(tmp_path, monkeypatch):
    import wave

    source = tmp_path / "in.wav"
    with wave.open(str(source), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(16000)
        out.writeframes(b"\x10\x00" * 16000 * 4)
    played = []
    player = SegmentPlayer()
    monkeypatch.setattr(player, "_start_playback", played.append)
    player.play_segment(source, 1.0, 3.0, speed=0.5)
    with wave.open(str(played[0]), "rb") as heard:
        assert abs(heard.getnframes() / heard.getframerate() - 4.0) < 0.1
    player.stop()
