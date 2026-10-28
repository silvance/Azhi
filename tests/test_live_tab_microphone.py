"""The Live page with a microphone: choosing one, the recording line, the record.

Builds the real page, so it needs a display; skipped without one. The device
list and the capture itself are stand-ins: no microphone or ffmpeg is needed.
"""

import json
import os
import time

import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="needs a display",
)


class FakeLive:
    """Stands in for LiveTranscriber: writes the recording it was asked for."""

    made = []

    def __init__(self, source, **kwargs):
        self.kwargs = kwargs
        self._running = False
        FakeLive.made.append(self)

    def start(self, *, on_text, on_status=None, on_error=None):
        self._running = True
        if self.kwargs["record_to"] is not None:
            self.kwargs["record_to"].write_bytes(b"RIFF fake recording")

    def stop(self):
        self._running = False

    @property
    def running(self):
        return self._running


class InlineThread:
    def __init__(self, target, daemon=None, args=()):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def page(tmp_path, monkeypatch):
    from sotto.gui import live_tab
    from sotto.gui.theme import init_theme
    from sotto.microphone import Microphone

    FakeLive.made = []
    mics = [
        Microphone("Desk microphone", ("-f", "dshow", "-i", "audio=@desk")),
        Microphone("Headset", ("-f", "dshow", "-i", "audio=@headset")),
    ]
    monkeypatch.setattr(live_tab, "find_ffmpeg", lambda: "ffmpeg")
    monkeypatch.setattr(live_tab, "list_microphones", lambda _ffmpeg: mics)
    monkeypatch.setattr(live_tab, "LiveTranscriber", FakeLive)
    monkeypatch.setattr(live_tab, "diarization_available", lambda: False)
    # The page's workers hand back through root.after, which Tk allows from
    # another thread only inside a running main loop; run them in line here.
    monkeypatch.setattr(live_tab.threading, "Thread", InlineThread)
    root = tk.Tk()
    init_theme(root)
    try:
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        tab = live_tab.LiveTab(frame, root)
        tab.record_dir_var.set(str(tmp_path / "recordings"))
        yield tab, root, tmp_path
    finally:
        tab.close()
        root.destroy()


def _settle(root, seconds=0.3):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def test_choosing_a_microphone_shows_its_settings_and_lists_devices(page):
    tab, root, _ = page
    assert not tab.mic_combo.winfo_ismapped() or not tab.microphone_chosen
    tab.source_kind_var.set("microphone")
    tab._show_source_kind()
    _settle(root)
    assert tab.mic_combo.cget("values") == ("Desk microphone", "Headset")
    assert tab.mic_var.get() == "Desk microphone"
    assert tab.status_var.get() == "Found 2 microphones."
    assert not tab.test_button.winfo_ismapped()  # nothing to test a connection to


def test_a_microphone_session_is_recorded_shown_and_recorded_on_stop(page):
    tab, root, tmp_path = page
    tab.source_kind_var.set("microphone")
    tab._show_source_kind()
    _settle(root)
    tab.mic_var.set("Headset")
    tab._start()
    _settle(root)
    live = FakeLive.made[-1]
    assert live.kwargs["input_args"] == ("-f", "dshow", "-i", "audio=@headset")
    assert live.kwargs["watch_silence"] is True
    recording = live.kwargs["record_to"]
    assert recording.parent == tmp_path / "recordings"  # made for it
    assert tab.recording_var.get().startswith("● Recording from Headset")
    assert tab.recording_label.winfo_ismapped()
    assert str(tab._kind_buttons[0].cget("state")) == "disabled"

    tab._stop()
    _settle(root, 1.0)
    record = json.loads(
        (recording.parent / (recording.name + ".capture.json")).read_text("utf-8")
    )
    assert record["device"] == "Headset"
    assert len(record["sha256"]) == 64
    assert tab.status_var.get().startswith(f"Stopped. Recording kept: {recording.name}")
    assert not tab.recording_label.winfo_ismapped()
    assert str(tab._kind_buttons[0].cget("state")) == "normal"


def test_without_a_microphone_nothing_starts(page, monkeypatch):
    from sotto.gui import live_tab

    tab, root, _ = page
    monkeypatch.setattr(live_tab, "list_microphones", lambda _ffmpeg: [])
    tab.source_kind_var.set("microphone")
    tab._show_source_kind()
    _settle(root)
    assert "No microphone found" in tab.status_var.get()
    tab._start()
    assert not FakeLive.made
    assert tab.status_var.get().startswith("Pick a microphone first")


def test_a_stream_is_still_a_stream(page):
    tab, root, _ = page
    tab.source_var.set("rtmp://localhost/live/stream")
    tab._start()
    _settle(root)
    live = FakeLive.made[-1]
    assert live.kwargs["input_args"] is None
    assert live.kwargs["record_to"] is None
    assert not tab.recording_label.winfo_ismapped()
    tab._stop()
    _settle(root, 0.5)
    assert tab.status_var.get() == "Stopped."
