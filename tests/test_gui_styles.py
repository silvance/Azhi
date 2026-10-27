"""The shipped theme keeps the application's own widget styles (QA-18).

ttkbootstrap rewrites any ``style=`` it has not registered, so 170 labels and
144 frames were built as plain TLabel/TFrame: titles at body size, muted text
not muted, and low-emphasis buttons drawn as the filled primary button. This
builds the real window under ttkbootstrap and checks what each widget got.
Needs a display, so it is skipped on a headless runner.
"""

import os
from collections import Counter
from pathlib import Path

import pytest

tk = pytest.importorskip("tkinter")
tb = pytest.importorskip("ttkbootstrap")

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="needs a display",
)


@pytest.fixture
def app(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import whispers.gui.app as app_mod

    # Build the speaker pages too, without needing the model file.
    monkeypatch.setattr(app_mod, "bundled_embedding_model", lambda: Path("e.onnx"))
    root = tb.Window(themename="darkly")
    try:
        yield app_mod.WhispersApp(root, drag_and_drop=False, bootstrap=True)
    finally:
        root.destroy()


def _styles(widget):
    seen = Counter()
    for child in widget.winfo_children():
        try:
            seen[child.cget("style")] += 1
        except (tk.TclError, AttributeError):
            pass
        seen.update(_styles(child))
    return seen


def test_custom_styles_survive_ttkbootstrap(app):
    from whispers.gui.theme import Style

    seen = _styles(app.root)
    for name in (
        Style.BODY,
        Style.META,
        Style.MUTED,
        Style.SECTION_TITLE,
        Style.PAGE,
        Style.CARD,
        Style.CARD_INNER,
        Style.SUBTLE,
    ):
        assert seen[name] > 0, f"no widget kept {name}"
    # The ones that were stripped to ttkbootstrap's defaults.
    assert seen["TLabel"] == 0
    assert seen["TFrame"] == 0


def test_translate_with_a_model_that_cannot_is_refused_before_it_runs(app):
    # turbo writes the original language when asked to translate: a Chinese
    # recording came back in Chinese, with nothing on screen to say so.
    page = app.transcribe
    page.diarize_var.set(False)
    page.model_var.set("turbo")
    page.task_var.set("translate")
    assert "cannot translate" in page.model_status_var.get()
    page.run_in_thread()
    assert page._busy is False
    assert page.progress_label_var.get() == "Choose a model that can translate"
    page.task_var.set("transcribe")
    assert "cannot translate" not in page.model_status_var.get()


def test_word_timings_are_sharpened_for_english_only(tmp_path, monkeypatch):
    from whispers import alignment
    from whispers.provenance import TranscriptionProvenance
    from whispers.transcription import Segment, TranscriptionResult, Word

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setattr(alignment, "available", lambda: True)
    monkeypatch.setattr(alignment, "describe_method", lambda: "aligner@rev, rule")
    monkeypatch.setattr(alignment, "model_sha256", lambda: "a" * 64)
    calls = []

    def align_segments(segments, wav, **_kwargs):
        calls.append(wav)
        return alignment.AlignmentSummary(words=3, retimed=2)

    monkeypatch.setattr(alignment, "align_segments", align_segments)
    import whispers.gui.app as app_mod

    root = tb.Window(themename="darkly")
    try:
        page = app_mod.WhispersApp(root, drag_and_drop=False, bootstrap=True).transcribe
        # Offered, and on, where the build carries the model.
        assert page.retime_var.get() is True
        assert "retime_words" in page.get_settings()

        def result(language):
            words = [Word(0.0, 0.5, " Hi")]
            return TranscriptionResult(
                text="Hi",
                language=language,
                language_probability=1.0,
                duration=1.0,
                segments=[Segment(0.0, 0.5, "Hi", words=words)],
            )

        record = TranscriptionProvenance()
        page._retime_words(result("ru"), tmp_path / "a.wav", "transcribe", record)
        page._retime_words(result("en"), tmp_path / "a.wav", "translate", record)
        assert calls == [] and record.word_timing == ""
        page._retime_words(result("en"), tmp_path / "a.wav", "transcribe", record)
        assert calls == [tmp_path / "a.wav"]
        assert (record.word_timing, record.words_retimed, record.words_total) == (
            "aligner@rev, rule",
            2,
            3,
        )
        page.retime_var.set(False)
        page._retime_words(result("en"), tmp_path / "a.wav", "transcribe", record)
        assert len(calls) == 1
    finally:
        root.destroy()


def test_live_speakers_relabel_the_transcript_and_rewrite_the_file(
    tmp_path, monkeypatch
):
    from whispers.transcription import Segment

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import whispers.gui.app as app_mod
    import whispers.gui.live_tab as live_mod

    monkeypatch.setattr(live_mod, "diarization_available", lambda: True)
    # The Live page is only built where ffmpeg can read a stream.
    monkeypatch.setattr(app_mod, "find_ffmpeg", lambda: Path("ffmpeg"))
    root = tb.Window(themename="darkly")
    try:
        page = app_mod.WhispersApp(root, drag_and_drop=False, bootstrap=True).live
        assert page.speakers_var.get() is False  # a second model: asked for

        saved = tmp_path / "live.txt"
        saved.write_text("from last time\n", encoding="utf-8")
        page._save_path, page._save_prefix = saved, saved.read_text(encoding="utf-8")
        reviewer = object()
        page._reviewer = reviewer
        labelled = [
            Segment(0.0, 2.0, "Who is it?", speaker="SPEAKER_00"),
            Segment(2.1, 2.5, "Me.", speaker="SPEAKER_01"),
        ]
        pending = [Segment(9.0, 12.0, "Come in.")]
        page._show_speakers(reviewer, labelled, pending)
        shown = page.transcript.get("1.0", "end-1c")
        assert shown == ("[Speaker 1] Who is it?\n[Speaker 2] Me.\n[…] Come in.\n")
        assert saved.read_text(encoding="utf-8") == "from last time\n" + shown

        # A later pass changes its mind; the screen and the file follow.
        labelled[1].speaker = "SPEAKER_00"
        page._show_speakers(reviewer, labelled, [])
        assert "[Speaker 1] Me." in page.transcript.get("1.0", "end-1c")
        assert "[Speaker 1] Me." in saved.read_text(encoding="utf-8")

        # Lines from a session that has been replaced are not drawn.
        page._show_speakers(object(), [], pending)
        assert "Come in." not in page.transcript.get("1.0", "end-1c")

        # Cleared lines stay cleared when the next pass redraws.
        page._hide_before = 2.05
        page._show_speakers(reviewer, labelled, pending)
        assert page.transcript.get("1.0", "end-1c") == (
            "[Speaker 1] Me.\n[…] Come in.\n"
        )
        page._reviewer = None
    finally:
        root.destroy()
