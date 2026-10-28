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


@pytest.fixture
def app_with_profiles(app, monkeypatch):
    # The Transcribe page asks for itself whether voices can be measured.
    import whispers.gui.transcribe_tab as transcribe_mod

    monkeypatch.setattr(
        transcribe_mod, "bundled_embedding_model", lambda: Path("e.onnx")
    )
    import whispers.gui.app as app_mod

    root = app.root
    for child in list(root.winfo_children()):
        child.destroy()
    return app_mod.WhispersApp(root, drag_and_drop=False, bootstrap=True)


def test_a_transcript_speaker_can_be_searched_against_every_profile(
    app_with_profiles, tmp_path
):
    from whispers.transcription import Segment, TranscriptionResult

    app = app_with_profiles
    recording = tmp_path / "call.wav"
    recording.write_bytes(b"RIFF")
    page = app.transcribe
    result = TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=10.0,
        segments=[
            Segment(0.0, 2.0, "Who is it?", speaker="SPEAKER_00"),
            Segment(2.5, 3.0, "Me.", speaker="SPEAKER_01"),
            Segment(4.0, 6.5, "Come in.", speaker="SPEAKER_00"),
        ],
    )
    page._result, page._result_source = result, recording
    page._speaker_names = {"SPEAKER_00": "Anna"}
    page.transcript_view.set_result(result, page._speaker_names)

    menus = []
    page.transcript_view._popup = lambda menu, _event: menus.append(menu)
    searched = []
    compare = app.speaker_compare
    compare._search_gallery = lambda: searched.append(
        (compare.questioned_var.get(), compare.ranges_var.get())
    )
    page.transcript_view._speaker_menu_handler(0)(None)
    menu = menus[-1]
    labels = {
        menu.entrycget(i, "label"): i
        for i in range(menu.index("end") + 1)
        if menu.type(i) == "command"
    }
    menu.invoke(labels["Compare ‘Anna’ to speaker profiles…"])

    # Anna's two lines, from the recording, on the Compare page.
    assert searched == [(str(recording), "0:00.00-0:02.00, 0:04.00-0:06.50")]
    assert app._current_page == "compare"
    description = compare._selection_description([(0.0, 2.0), (4.0, 6.5)])
    assert "the lines given to Anna in the transcript" in description

    # Edited by hand, the ranges are no longer "Anna's lines".
    compare.ranges_var.set("0:00-0:02")
    assert "Anna" not in compare._selection_description([(0.0, 2.0)])


def test_hesitations_are_kept_by_default_and_the_choice_is_remembered(app):
    page = app.transcribe
    assert page.verbatim_var.get() is True
    page.verbatim_var.set(False)
    saved = page.get_settings()
    assert saved["verbatim"] is False
    page.verbatim_var.set(True)
    page.apply_settings(saved)
    assert page.verbatim_var.get() is False


def test_a_method_saved_under_an_old_label_keeps_its_choice(app):
    page = app.transcribe
    page.apply_settings({"engine": "pyannote - most accurate"})
    assert page.engine_var.get() == "pyannote (strong on noisy audio)"
    page.apply_settings({"engine": "Auto (pyannote if available)"})
    assert page.engine_var.get() == "Auto (recommended)"
