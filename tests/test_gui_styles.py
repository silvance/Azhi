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
    import sotto.gui.app as app_mod

    # Build the speaker pages too, without needing the model file.
    monkeypatch.setattr(app_mod, "bundled_embedding_model", lambda: Path("e.onnx"))
    root = tb.Window(themename="darkly")
    try:
        yield app_mod.SottoApp(root, drag_and_drop=False, bootstrap=True)
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
    from sotto.gui.theme import Style

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
    from sotto import alignment
    from sotto.provenance import TranscriptionProvenance
    from sotto.transcription import Segment, TranscriptionResult, Word

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
    import sotto.gui.app as app_mod

    root = tb.Window(themename="darkly")
    try:
        page = app_mod.SottoApp(root, drag_and_drop=False, bootstrap=True).transcribe
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
    from sotto.transcription import Segment

    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("APPDATA", str(tmp_path))
    import sotto.gui.app as app_mod
    import sotto.gui.live_tab as live_mod

    monkeypatch.setattr(live_mod, "diarization_available", lambda: True)
    # The Live page is only built where ffmpeg can read a stream.
    monkeypatch.setattr(app_mod, "find_ffmpeg", lambda: Path("ffmpeg"))
    root = tb.Window(themename="darkly")
    try:
        page = app_mod.SottoApp(root, drag_and_drop=False, bootstrap=True).live
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
    import sotto.gui.transcribe_tab as transcribe_mod

    monkeypatch.setattr(
        transcribe_mod, "bundled_embedding_model", lambda: Path("e.onnx")
    )
    import sotto.gui.app as app_mod

    root = app.root
    for child in list(root.winfo_children()):
        child.destroy()
    return app_mod.SottoApp(root, drag_and_drop=False, bootstrap=True)


def test_a_transcript_speaker_can_be_searched_against_every_profile(
    app_with_profiles, tmp_path
):
    from sotto.transcription import Segment, TranscriptionResult

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


def test_the_task_box_shows_words_and_runs_on_the_value(app):
    page = app.transcribe
    page.task_var.set("translate")
    assert page.task_choice_var.get() == "Translate into English"
    page.task_var.set("transcribe")
    assert page.task_choice_var.get() == "Transcribe (keep the language)"


def test_grouping_sensitivity_shows_only_where_it_does_something(app):
    page = app.transcribe
    page.num_speakers_var.set("")
    page.engine_var.set("sherpa (lightest, for clean audio)")
    assert page._sensitivity_row.winfo_manager() == "grid"
    page.num_speakers_var.set("2")  # a count given: sherpa does not group
    assert page._sensitivity_row.winfo_manager() == ""
    page.num_speakers_var.set("")
    page.engine_var.set("pyannote (strong on noisy audio)")
    assert page._sensitivity_row.winfo_manager() == ""


def _page_with_transcript(app, tmp_path):
    from sotto.transcription import Segment, TranscriptionResult

    page = app.transcribe
    result = TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=80.0,
        segments=[Segment(75.0, 76.0, "Come in.", speaker="SPEAKER_00")],
    )
    page._result, page._result_source = result, tmp_path / "call.wav"
    page._speaker_names = {"SPEAKER_00": "Anna"}
    page.transcript_view.set_result(result, page._speaker_names)
    return page


@pytest.mark.parametrize(
    "kind, ending, expect",
    [
        ("Spreadsheet (CSV)", ".csv", "00:01:15.000,00:01:16.000,75.000"),
        ("Text file", ".txt", "[00:01:15] [Anna] Come in."),
    ],
)
def test_save_as_writes_the_type_picked_even_without_an_ending(
    app, tmp_path, monkeypatch, kind, ending, expect
):
    import sotto.gui.transcribe_tab as transcribe_mod

    page = _page_with_transcript(app, tmp_path)
    page.times_var.set(True)

    def dialog(**options):
        options["typevariable"].set(kind)  # what the analyst picked
        return str(tmp_path / "typed-without-an-ending")

    monkeypatch.setattr(transcribe_mod.filedialog, "asksaveasfilename", dialog)
    page._save_transcript_as()
    written = tmp_path / f"typed-without-an-ending{ending}"
    assert expect in written.read_text(encoding="utf-8-sig")


def test_the_playback_speed_is_remembered_and_checked(app):
    page = app.transcribe
    page.playback_speed_var.set("0.5× speed")
    assert page.get_settings()["playback_speed"] == "0.5× speed"
    page.apply_settings({"playback_speed": "9× speed"})  # not on offer
    assert page.playback_speed_var.get() == "Normal speed"


def test_the_watch_list_is_highlighted_saved_and_carried_by_profiles(app, tmp_path):
    page = _page_with_transcript(app, tmp_path)
    page.watch_list_var.set("come in\nNobody")
    for _ in range(5):
        app.root.update()
    assert page.watch_status_var.get() == "Watch list: come in ×1."
    assert page.get_settings()["watch_list"] == "come in\nNobody"
    assert page._profile_settings()["watch_list"] == "come in\nNobody"
    page.apply_settings({"watch_list": ""})
    for _ in range(5):
        app.root.update()
    assert page.watch_matcher is None
    assert page.watch_status_var.get() == ""


def test_the_live_transcript_follows_the_same_watch_list(app, tmp_path):
    from sotto.transcription import Segment

    live = getattr(app, "live", None)
    if live is None:
        pytest.skip("this copy has no Live page")
    app.transcribe.watch_list_var.set("bridge")
    live._reviewer = object()  # as if speakers were on
    live._show_speakers(
        live._reviewer,
        [Segment(0, 2, "Meet at the bridge.", speaker="SPEAKER_00")],
        [Segment(2, 4, "Which bridge?")],
    )
    w = live.transcript
    marked = w.tag_ranges("watch")
    assert [w.get(a, b) for a, b in zip(marked[0::2], marked[1::2])] == [
        "bridge",
        "bridge",
    ]
    live._reviewer = None


def test_copy_carries_the_lead_legend_once_with_times_and_notes(app, tmp_path):
    from sotto.speaker_names import LEAD_LEGEND, as_lead
    from sotto.transcription import Note

    page = _page_with_transcript(app, tmp_path)
    page._speaker_names = {"SPEAKER_00": as_lead("Anna")}
    page.transcript_view.set_result(page._result, page._speaker_names)
    page._result.notes = [Note(75.0, "Door slams first.")]
    page.times_var.set(True)
    for _ in range(5):
        app.root.update()
    page._copy_transcript()
    copied = app.root.clipboard_get()
    assert copied.count(LEAD_LEGEND) == 1  # it used to be there twice
    assert "[00:01:15]" in copied
    assert "    Analyst note: Door slams first." in copied
    page.notes_var.set(False)
    page._copy_transcript()
    assert "Analyst note" not in app.root.clipboard_get()
    assert page.get_settings()["notes"] is False


def test_a_comparison_shows_its_windows_and_how_the_recordings_were_made(app):
    import math

    from sotto.matching import compare_questioned_to_profile
    from sotto.quality import GOOD, QualityReport
    from sotto.questioned import MeasuredWindow, QuestionedSpeaker
    from sotto.speaker_profiles import (
        EmbeddingModelIdentity,
        EnrollmentSample,
        SpeakerProfile,
    )

    model = EmbeddingModelIdentity(name="m", sha256="a" * 64, vector_dimension=2)
    profile = SpeakerProfile(display_name="Subject A", embedding_model=model)
    profile.add_reference_sample(
        EnrollmentSample(
            embedding=[1.0, 0.0],
            speech_duration=20.0,
            quality={"assessment": GOOD, "voiced_seconds": 20.0, "bandwidth_hz": 7000},
        )
    )

    def window(spans, cosine):
        return MeasuredWindow(
            spans=spans,
            report=QualityReport(
                voiced_seconds=7.0, assessment=GOOD, bandwidth_hz=3750.0
            ),
            embedding=[cosine, math.sqrt(1 - cosine * cosine)],
        )

    measured = QuestionedSpeaker(
        embedding=[1.0, 0.0],
        speech_seconds=28.0,
        quality=GOOD,
        window_count=4,
        windows=[
            window([(0.0, 8.0)], 0.9),
            window([(8.0, 16.0)], 0.85),
            window([(16.0, 18.0), (30.0, 36.0)], 0.3),
            window([(40.0, 48.0)], 0.88),
        ],
    )
    result = compare_questioned_to_profile(measured, profile, questioned_model=model)

    played = []

    class Listener:
        def play(self, span, _on_error):
            played.append(span)

        def stop(self):
            played.append("stop")

    page = app.speaker_compare
    page._show_comparison(result, Listener())
    app.root.update()
    assert page._windows_disclosure.winfo_manager() == "pack"
    assert page._conditions_disclosure.winfo_manager() == "pack"

    strip = page.window_strip
    assert len(strip.canvas.find_withtag("hit")) == 4
    assert "0:16-0:36 (joined from 2 turns) scores 0.30" in strip.notes_var.get()
    # Clicking the bar plays its longest stretch and says what it is.
    strip._play(2)
    assert played == [(30.0, 36.0)]
    assert strip.detail_var.get().startswith("0:16-0:36 (joined from 2 turns): 0.30")

    notes = page.conditions_table.notes_var.get()
    assert "questioned audio reaches only 3.8 kHz against 7.0 kHz" in notes

    # A new run takes both off the card, and stops the audio.
    page._start_measuring(Path("next.wav"), "Comparing")
    assert played[-1] == "stop"
    assert page._windows_disclosure.winfo_manager() == ""
    assert page._conditions_disclosure.winfo_manager() == ""
