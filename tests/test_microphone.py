"""Microphones as a live source: finding them, opening them, keeping the recording."""

import json
import subprocess
import wave
from datetime import datetime, timezone

import pytest

from whispers import microphone
from whispers.hashing import sha256_file

# What ffmpeg 7.1 prints on Windows for -list_devices (abridged).
DSHOW_NEW = r"""
[in#0 @ 000001c6] "Integrated Camera" (video)
[in#0 @ 000001c6]   Alternative name "@device_pnp_\\?\usb#vid_04f2&pid_b6d9"
[in#0 @ 000001c6] "Microphone Array (Realtek(R) Audio)" (audio)
[in#0 @ 000001c6]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{AAA}"
[in#0 @ 000001c6] "USB Audio Device" (audio)
[in#0 @ 000001c6]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{BBB}"
[in#0 @ 000001c6] "USB Audio Device" (audio)
[in#0 @ 000001c6]   Alternative name "@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\wave_{CCC}"
[in#0 @ 000001c6] "Capture card" (audio, video)
[in#0 @ 000001c6]   Alternative name "@device_pnp_capture"
[in#0 @ 000001c6] "Virtual camera" (none)
[in#0 @ 000001c6]   Alternative name "@device_sw_none"
dummy: Immediate exit requested
"""

# ffmpeg 4.x, under section headings instead.
DSHOW_OLD = r"""
[dshow @ 0000020b] DirectShow video devices (some may be both video and audio devices)
[dshow @ 0000020b]  "Integrated Camera"
[dshow @ 0000020b]     Alternative name "@device_pnp_cam"
[dshow @ 0000020b] DirectShow audio devices
[dshow @ 0000020b]  "Microphone (Realtek High Definition Audio)"
[dshow @ 0000020b]     Alternative name "@device_cm_{X}\wave_{DDD}"
dummy: Immediate exit requested
"""


def test_windows_microphones_are_found_and_opened_by_their_unique_name():
    found = microphone.parse_dshow(DSHOW_NEW)
    assert [m.name for m in found] == [
        "Microphone Array (Realtek(R) Audio)",
        "USB Audio Device",
        "USB Audio Device (2)",
        "Capture card",
    ]
    first = found[0].input_args
    assert first[:2] == ("-f", "dshow")
    assert first[-2:] == (
        "-i",
        "audio=@device_cm_{33D9A762-90C8-11D0-BD43-00A0C911CE86}\\wave_{AAA}",
    )
    # Two devices with one name are still two devices.
    assert found[1].input_args[-1] != found[2].input_args[-1]


def test_older_ffmpeg_lists_under_headings():
    found = microphone.parse_dshow(DSHOW_OLD)
    assert [m.name for m in found] == ["Microphone (Realtek High Definition Audio)"]
    assert found[0].input_args[-1] == "audio=@device_cm_{X}\\wave_{DDD}"


def test_no_microphones_is_an_empty_list():
    assert microphone.parse_dshow("dummy: Immediate exit requested\n") == []


def test_linux_offers_the_system_default_first_and_skips_null():
    output = (
        "Auto-detected sources for alsa:\n"
        "  null [Discard all samples (playback) or generate zero samples (capture)] (none)\n"
        "* default [Default ALSA Output (currently PulseAudio Sound Server)] (none)\n"
        "  hw:CARD=PCH,DEV=0 [HDA Intel PCH, ALC3246 Analog] (none)\n"
    )
    found = microphone.parse_alsa(output)
    assert [m.name for m in found] == [
        "System default microphone",
        "HDA Intel PCH, ALC3246 Analog",
    ]
    assert found[0].input_args == ("-f", "alsa", "-i", "default")
    assert found[1].input_args == ("-f", "alsa", "-i", "hw:CARD=PCH,DEV=0")


def test_listing_asks_ffmpeg_the_platforms_way(monkeypatch):
    asked = []

    def fake_run(args, **_kwargs):
        asked.append(args)
        return subprocess.CompletedProcess(args, 1, b"", DSHOW_NEW.encode())

    monkeypatch.setattr(microphone.subprocess, "run", fake_run)
    monkeypatch.setattr(microphone, "system", lambda: "windows")
    found = microphone.list_microphones("ffmpeg")
    assert asked[0][-6:] == ["-list_devices", "true", "-f", "dshow", "-i", "dummy"]
    assert len(found) == 4  # the exit code is an error by design, and ignored


def test_listing_fails_plainly(monkeypatch):
    def fake_run(args, **_kwargs):
        raise subprocess.TimeoutExpired(args, 15)

    monkeypatch.setattr(microphone.subprocess, "run", fake_run)
    monkeypatch.setattr(microphone, "system", lambda: "linux")
    with pytest.raises(microphone.MicrophoneError, match="too long"):
        microphone.list_microphones("ffmpeg")
    monkeypatch.setattr(microphone, "system", lambda: "other")
    with pytest.raises(microphone.MicrophoneError, match="Windows and Linux"):
        microphone.list_microphones("ffmpeg")


@pytest.mark.parametrize(
    "detail, says",
    [
        (
            "Could not find audio only device with name [x] among source devices",
            "unplugged",
        ),
        ("audio=@device_cm_{X}: I/O error", "could not be opened"),
        (
            "Could not run graph (sometimes caused by a device already in use)",
            "holding it",
        ),
        ("something else entirely", None),
    ],
)
def test_capture_failures_are_explained(detail, says):
    explained = microphone.explain_capture_error(detail)
    assert (explained is None) if says is None else (says in explained)


def _wav(path, samples, rate=16000, channels=1):
    with wave.open(str(path), "wb") as out:
        out.setnchannels(channels)
        out.setsampwidth(2)
        out.setframerate(rate)
        out.writeframes(
            b"".join(int(s).to_bytes(2, "little", signed=True) for s in samples)
        )


def test_digital_silence_is_exact_zeros_only(tmp_path):
    _wav(tmp_path / "zeros.wav", [0] * 1600)
    _wav(tmp_path / "quiet.wav", [0] * 1599 + [1])
    _wav(tmp_path / "empty.wav", [])
    assert microphone.is_digital_silence(tmp_path / "zeros.wav")
    assert not microphone.is_digital_silence(tmp_path / "quiet.wav")
    assert not microphone.is_digital_silence(tmp_path / "empty.wav")


def test_a_new_recording_never_takes_an_existing_name(tmp_path):
    started = datetime(2026, 10, 28, 12, 30, 5)
    first = microphone.recording_path(tmp_path, started)
    assert first.name == "Microphone 2026-10-28 12-30-05.wav"
    first.write_bytes(b"")
    second = microphone.recording_path(tmp_path, started)
    assert second.name == "Microphone 2026-10-28 12-30-05 (2).wav"
    # A record left without its recording still holds the name.
    microphone.record_path(second).write_text("{}")
    assert microphone.recording_path(tmp_path, started).name.endswith("(3).wav")


def test_the_record_says_what_was_observed_and_nothing_more(tmp_path):
    recording = tmp_path / "Microphone.wav"
    _wav(recording, [5] * 96000, rate=48000, channels=2)
    started = datetime(2026, 10, 28, 12, 30, 5, tzinfo=timezone.utc)
    stopped = datetime(2026, 10, 28, 12, 30, 6, tzinfo=timezone.utc)
    path = microphone.write_capture_record(
        recording,
        device="USB Audio Device",
        started=started,
        stopped=stopped,
        transcription_model="base.en",
    )
    assert path.name == "Microphone.wav.capture.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    assert record["sha256"] == sha256_file(recording)
    assert record["file_size"] == recording.stat().st_size
    assert record["device"] == "USB Audio Device"
    assert record["started_utc"] == "2026-10-28T12:30:05+00:00"
    assert (record["sample_rate"], record["channels"]) == (48000, 2)
    assert record["duration_seconds"] == 1.0
    assert microphone.read_capture_record(recording) == record


def test_an_unreadable_recording_gets_no_invented_format(tmp_path):
    recording = tmp_path / "Microphone.wav"
    recording.write_bytes(b"RIFF....not really")
    now = datetime.now(timezone.utc)
    microphone.write_capture_record(
        recording, device="x", started=now, stopped=now, transcription_model="base"
    )
    record = microphone.read_capture_record(recording)
    assert "sample_rate" not in record and "duration_seconds" not in record
    assert record["sha256"] == sha256_file(recording)
