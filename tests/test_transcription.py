import pytest

from sotto import transcription
from sotto.transcription import (
    Segment,
    TranscriptionResult,
    Word,
    _format_timestamp,
    convert_to_wav,
    is_low_confidence_segment,
    is_low_confidence_word,
    is_supported_media,
    is_video,
    transcribe_audio,
)


@pytest.mark.parametrize(
    "prob,expected",
    [(0.95, False), (0.55, False), (0.4, True), (None, False)],
)
def test_is_low_confidence_word(prob, expected):
    word = Word(start=0.0, end=1.0, word="x", probability=prob)
    assert is_low_confidence_word(word) is expected


@pytest.mark.parametrize(
    "logprob,expected",
    [(-0.1, False), (-0.7, False), (-1.2, True), (None, False)],
)
def test_is_low_confidence_segment(logprob, expected):
    seg = Segment(start=0.0, end=1.0, text="x", avg_logprob=logprob)
    assert is_low_confidence_segment(seg) is expected


@pytest.mark.parametrize(
    "seconds,expected",
    [
        (0, "00:00:00,000"),
        (1.5, "00:00:01,500"),
        (61.25, "00:01:01,250"),
        (3661.007, "01:01:01,007"),
        (-5, "00:00:00,000"),
    ],
)
def test_format_timestamp(seconds, expected):
    assert _format_timestamp(seconds) == expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("clip.mp3", True),
        ("clip.WAV", True),
        ("recording.mp4", True),
        ("notes.txt", False),
        ("archive.zip", False),
    ],
)
def test_is_supported_media(name, expected):
    assert is_supported_media(name) is expected


@pytest.mark.parametrize(
    "name,expected",
    [
        ("movie.mp4", True),
        ("clip.MKV", True),
        ("recording.mov", True),
        ("audio.mp3", False),
        ("audio.wav", False),
    ],
)
def test_is_video(name, expected):
    assert is_video(name) is expected


def test_convert_to_wav_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        convert_to_wav(tmp_path / "missing.mp4")


def test_convert_to_wav_without_ffmpeg(tmp_path, monkeypatch):
    monkeypatch.setattr(transcription, "find_ffmpeg", lambda: None)
    media = tmp_path / "movie.mp4"
    media.write_bytes(b"\x00")
    with pytest.raises(RuntimeError, match="ffmpeg was not found"):
        convert_to_wav(media)


def test_to_srt_formats_segments():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=0.99,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello"),
            Segment(start=1.0, end=2.0, text="world"),
        ],
    )
    srt = result.to_srt()
    assert "1\n00:00:00,000 --> 00:00:01,000\nhello" in srt
    assert "2\n00:00:01,000 --> 00:00:02,000\nworld" in srt


def test_to_txt_returns_text():
    result = TranscriptionResult(
        text="some text", language="en", language_probability=1.0, duration=1.0
    )
    assert result.to_txt() == "some text"


def test_to_txt_with_speakers():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="world", speaker="SPEAKER_01"),
        ],
    )
    assert result.has_speakers
    assert result.to_txt() == "[Speaker 1] hello\n[Speaker 2] world"
    # blank_lines puts an empty line between turns.
    assert result.to_txt(blank_lines=True) == "[Speaker 1] hello\n\n[Speaker 2] world"


def test_to_txt_blank_lines_without_speakers():
    result = TranscriptionResult(
        text="one\ntwo",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="one"),
            Segment(start=1.0, end=2.0, text="two"),
        ],
    )
    assert result.to_txt() == "one\ntwo"
    assert result.to_txt(blank_lines=True) == "one\n\ntwo"


def test_to_srt_with_speakers():
    result = TranscriptionResult(
        text="hi",
        language="en",
        language_probability=1.0,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="hi", speaker="SPEAKER_02")],
    )
    assert "[Speaker 3] hi" in result.to_srt()


def test_speaker_names_remap_txt_and_srt():
    result = TranscriptionResult(
        text="hello\nworld",
        language="en",
        language_probability=1.0,
        duration=2.0,
        segments=[
            Segment(start=0.0, end=1.0, text="hello", speaker="SPEAKER_00"),
            Segment(start=1.0, end=2.0, text="world", speaker="SPEAKER_01"),
        ],
    )
    names = {"SPEAKER_00": "Xin"}
    txt = result.to_txt(names)
    assert "[Xin] hello" in txt
    assert "[Speaker 2] world" in txt  # unnamed: shown by its number
    assert "[Xin] hello" in result.to_srt(names)


def test_transcribe_audio_missing_file(tmp_path):
    # FileNotFoundError is raised before the optional backend is needed.
    with pytest.raises(FileNotFoundError):
        transcribe_audio(tmp_path / "does-not-exist.mp3")


def test_transcribe_audio_without_backend(tmp_path):
    """When faster-whisper is not installed, a clear RuntimeError is raised."""
    try:
        import faster_whisper  # noqa: F401
    except ImportError:
        pass
    else:
        pytest.skip("faster-whisper is installed; backend-missing path not exercised")

    media = tmp_path / "clip.mp3"
    media.write_bytes(b"\x00")  # file must exist so we reach the import guard
    with pytest.raises(RuntimeError, match="faster-whisper is not installed"):
        transcribe_audio(media)


# -- silence skipping -------------------------------------------------------
# Field report: with "Skip silence" on, a pause was followed by speech being
# dropped until somebody spoke up. Silero enters a speech region at `threshold`
# but only leaves below `neg_threshold`, so on muffled audio - where the speech
# probability sits around 0.3-0.5 - a genuine pause ends the region and the
# quieter speech after it never reaches faster-whisper's default 0.5 to start a
# new one. These pin the settings that answer that, and the reporting that
# makes the loss visible when it still happens.


def test_speech_is_entered_below_faster_sotto_default():
    """0.5 is tuned for clean speech and discards muffled speech after a pause."""
    assert transcription.VAD_THRESHOLD < 0.5
    assert transcription.vad_options()["threshold"] == transcription.VAD_THRESHOLD


def test_leaving_speech_is_harder_than_entering_it():
    """Hysteresis has to stay in the right direction, or regions never close."""
    assert transcription.VAD_NEG_THRESHOLD < transcription.VAD_THRESHOLD


def test_a_pause_longer_than_a_conversation_beat_is_needed_to_close_a_region():
    assert transcription.VAD_MIN_SILENCE_MS >= 2000


def test_word_onsets_are_padded_more_than_the_default():
    """The start of a word is its quietest part, and the first thing clipped."""
    assert transcription.VAD_SPEECH_PAD_MS > 400


def test_every_tuned_value_reaches_faster_whisper():
    options = transcription.vad_options()
    assert options == {
        "threshold": transcription.VAD_THRESHOLD,
        "neg_threshold": transcription.VAD_NEG_THRESHOLD,
        "min_silence_duration_ms": transcription.VAD_MIN_SILENCE_MS,
        "speech_pad_ms": transcription.VAD_SPEECH_PAD_MS,
    }


# -- reporting what was skipped --------------------------------------------


def _result(duration, after_vad):
    return transcription.TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=duration,
        duration_after_vad=after_vad,
    )


def test_a_recording_that_was_heard_whole_reports_nothing_skipped():
    result = _result(600.0, 600.0)
    assert result.skipped_seconds == 0.0
    assert result.kept_fraction == pytest.approx(1.0)


def test_what_the_filter_removed_is_measurable():
    result = _result(600.0, 150.0)
    assert result.skipped_seconds == pytest.approx(450.0)
    assert result.kept_fraction == pytest.approx(0.25)


def test_an_unfiltered_run_reports_no_fraction_rather_than_a_full_one():
    """Silence skipping off is not the same as it having kept everything."""
    result = _result(600.0, None)
    assert result.kept_fraction is None
    assert result.skipped_seconds == 0.0


def test_a_zero_length_recording_does_not_divide_by_zero():
    assert _result(0.0, 0.0).kept_fraction is None


# -- the settings have to survive the trip to the model ---------------------


class _FakeInfo:
    language = "en"
    language_probability = 1.0
    duration = 600.0
    duration_after_vad = 120.0


class _FakeModel:
    """Records what transcribe() was handed, and returns one empty segment."""

    def __init__(self):
        self.kwargs = None

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        return iter(()), _FakeInfo()


def _run(monkeypatch, tmp_path, **overrides):
    model = _FakeModel()
    monkeypatch.setattr(transcription, "_load_model", lambda *a, **k: model)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")
    result = transcription.transcribe_audio(audio, **overrides)
    return model, result


def test_the_tuned_settings_are_handed_to_the_model(monkeypatch, tmp_path):
    model, _ = _run(monkeypatch, tmp_path, vad_filter=True)
    assert model.kwargs["vad_filter"] is True
    assert model.kwargs["vad_parameters"] == transcription.vad_options()


def test_no_settings_are_sent_when_silence_skipping_is_off(monkeypatch, tmp_path):
    model, _ = _run(monkeypatch, tmp_path, vad_filter=False)
    assert model.kwargs["vad_filter"] is False
    assert model.kwargs["vad_parameters"] is None


def test_how_much_was_skipped_comes_back_on_the_result(monkeypatch, tmp_path):
    _, result = _run(monkeypatch, tmp_path, vad_filter=True)
    assert result.duration_after_vad == pytest.approx(120.0)
    assert result.kept_fraction == pytest.approx(0.2)


def test_an_unfiltered_run_does_not_claim_a_filtered_duration(monkeypatch, tmp_path):
    """The model still reports duration_after_vad; it means nothing here."""
    _, result = _run(monkeypatch, tmp_path, vad_filter=False)
    assert result.duration_after_vad is None
    assert result.kept_fraction is None


# -- translating needs a model that was trained to ----------------------------


@pytest.mark.parametrize(
    "model",
    [
        "turbo",
        "base.en",
        "medium.en",
        "/models/faster-whisper-large-v3-turbo",
        "distil-large-v3",
    ],
)
def test_models_that_cannot_translate_are_named(model):
    from sotto.transcription import translation_refusal

    reason = translation_refusal(model)
    assert reason and "medium or large-v3" in reason


@pytest.mark.parametrize("model", ["medium", "large-v3", "small", "/models/large-v3"])
def test_multilingual_models_can_translate(model):
    from sotto.transcription import translation_refusal

    assert translation_refusal(model) is None


def test_a_translation_left_in_chinese_is_spotted():
    from sotto.transcription import UNTRANSLATED_SHARE, Segment, untranslated_share

    chinese = [
        Segment(0, 2, "他们就不许摘了之后呢然后杭州的查克索六几年的时候去那个武夷山")
    ]
    english = [Segment(0, 2, "Then in the sixties he went to Wuyi Mountain (武夷山).")]
    assert untranslated_share(chinese) > UNTRANSLATED_SHARE
    assert untranslated_share(english) < UNTRANSLATED_SHARE
    assert untranslated_share([Segment(0, 1, "85 60 ...")]) == 0.0


# -- a lapse into capitals stays where it happened ------------------------------


@pytest.mark.parametrize(
    "model, expected",
    [
        ("medium", True),
        ("small.en", True),
        ("large-v3", False),
        ("turbo", False),
        ("/assets/models/turbo", False),
        ("/models/faster-whisper-large-v3-turbo", False),
        ("distil-large-v3", False),
    ],
)
def test_large_v3_and_turbo_do_not_carry_earlier_text(model, expected):
    assert transcription.carries_context(model) is expected


def test_a_model_that_carries_context_is_primed_once(monkeypatch, tmp_path):
    model, _ = _run(
        monkeypatch, tmp_path, model_size="medium", initial_prompt="Kowalczyk"
    )
    assert model.kwargs["condition_on_previous_text"] is True
    assert model.kwargs["initial_prompt"] == "Kowalczyk"
    assert model.kwargs["hotwords"] is None


def test_turbo_starts_each_stretch_afresh_but_keeps_the_expected_words(
    monkeypatch, tmp_path
):
    # As an initial prompt, the expected words would reach only the first
    # 30 s once earlier text is not carried; as hotwords they reach them all.
    model, _ = _run(
        monkeypatch, tmp_path, model_size="turbo", initial_prompt="Kowalczyk"
    )
    assert model.kwargs["condition_on_previous_text"] is False
    assert model.kwargs["initial_prompt"] is None
    assert model.kwargs["hotwords"] == "Kowalczyk"


def test_the_record_says_whether_earlier_text_was_context():
    from sotto.provenance import AnalysisProvenance, TranscriptionProvenance

    record = TranscriptionProvenance(model_name="turbo", previous_text_context=False)
    assert TranscriptionProvenance.from_dict(record.to_dict()) == record
    lines = AnalysisProvenance(transcription=record).describe()
    assert (
        "Earlier text given to the model as context: no "
        "(each 30-second stretch started afresh)"
    ) in lines
    older = TranscriptionProvenance.from_dict({"model_name": "turbo"})
    assert older.previous_text_context is None
    assert not any(
        "as context" in line
        for line in AnalysisProvenance(transcription=older).describe()
    )


# -- verbatim: keeping the hesitations ----------------------------------------


def test_only_a_line_that_is_nothing_but_the_example_counts_as_leaked():
    leaked = transcription.is_leaked_example
    assert leaked(transcription.VERBATIM_EXAMPLE)
    assert leaked(" Umm, let me think like, hmm.")
    assert leaked("Okay, here's what I'm, like, thinking.")
    # Things people say, even where they overlap the example.
    assert not leaked("Okay.")
    assert not leaked("Let me think like...")
    assert not leaked("Okay, here's what I'm thinking about the car.")
    assert not leaked("I, I, I went - I drove there.")


@pytest.mark.parametrize(
    "language, model, expected",
    [
        ("en", "medium", True),
        ("en-GB", "turbo", True),
        ("ru", "medium", False),
        (None, "medium.en", True),
        (None, "/models/small.en", True),
        (None, "turbo", None),
    ],
)
def test_english_is_known_up_front_only_when_set_or_english_only(
    language, model, expected
):
    assert transcription.known_english(language, model) is expected


class _Seg:
    def __init__(self, start, end, text):
        self.start, self.end, self.text = start, end, text
        self.words = []
        self.avg_logprob = -0.2


class _Info(_FakeInfo):
    def __init__(self, language="en"):
        self.language = language


class _DetectingModel(_FakeModel):
    """Answers language detection, and returns the given lines."""

    def __init__(self, detected="en", lines=()):
        super().__init__()
        self.detected = detected
        self.lines = list(lines)
        self.detected_on = None

    def detect_language(self, audio, vad_filter, vad_parameters):
        self.detected_on = audio
        return self.detected, 0.93, []

    def transcribe(self, path, **kwargs):
        self.kwargs = kwargs
        self.audio = path
        return iter(self.lines), _Info(kwargs.get("language") or self.detected)


@pytest.fixture
def fake_decoder(monkeypatch):
    """faster-whisper's audio decoder and VAD options, without the library."""
    import sys
    import types

    audio_mod = types.ModuleType("faster_whisper.audio")
    audio_mod.decode_audio = lambda path, sampling_rate: f"samples of {path}"
    vad_mod = types.ModuleType("faster_whisper.vad")
    vad_mod.VadOptions = lambda **kwargs: ("vad", tuple(sorted(kwargs)))
    package = types.ModuleType("faster_whisper")
    for name, module in (
        ("faster_whisper", package),
        ("faster_whisper.audio", audio_mod),
        ("faster_whisper.vad", vad_mod),
    ):
        monkeypatch.setitem(sys.modules, name, module)


def _verbatim_run(monkeypatch, tmp_path, model, **overrides):
    monkeypatch.setattr(transcription, "_load_model", lambda *a, **k: model)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")
    return transcription.transcribe_audio(audio, verbatim=True, **overrides)


def test_verbatim_gives_every_stretch_the_example(monkeypatch, tmp_path):
    model = _DetectingModel()
    result = _verbatim_run(
        monkeypatch,
        tmp_path,
        model,
        model_size="medium",
        language="en",
        initial_prompt="Kowalczyk",
    )
    # Expected words as before; the example ahead of every stretch, where a
    # reset of the running context cannot drop it.
    assert model.kwargs["initial_prompt"] == "Kowalczyk"
    assert model.kwargs["hotwords"] == transcription.VERBATIM_EXAMPLE
    assert result.verbatim is True
    assert model.detected_on is None  # English was known; nothing detected


def test_turbo_gets_the_expected_words_and_the_example_together(monkeypatch, tmp_path):
    model = _DetectingModel()
    _verbatim_run(
        monkeypatch,
        tmp_path,
        model,
        model_size="turbo",
        language="en",
        initial_prompt="Kowalczyk",
    )
    assert model.kwargs["initial_prompt"] is None
    assert model.kwargs["hotwords"] == f"Kowalczyk {transcription.VERBATIM_EXAMPLE}"


@pytest.mark.parametrize(
    "overrides",
    [
        {"language": "ru"},
        {"language": "en", "task": "translate"},
    ],
)
def test_no_example_for_other_languages_or_translations(
    monkeypatch, tmp_path, overrides
):
    model = _DetectingModel()
    result = _verbatim_run(
        monkeypatch, tmp_path, model, model_size="medium", **overrides
    )
    assert model.kwargs["hotwords"] is None
    assert result.verbatim is False


def test_with_auto_language_english_is_detected_before_the_example_is_given(
    monkeypatch, tmp_path, fake_decoder
):
    model = _DetectingModel(detected="en")
    result = _verbatim_run(monkeypatch, tmp_path, model, model_size="turbo")
    assert model.detected_on.startswith("samples of")
    # Decoded once: the same samples go on to be transcribed.
    assert model.audio == model.detected_on
    assert model.kwargs["language"] == "en"
    assert model.kwargs["hotwords"] == transcription.VERBATIM_EXAMPLE
    # The probability is the detection's, not the 1.0 of a language given.
    assert result.language_probability == pytest.approx(0.93)


def test_auto_language_that_is_not_english_gets_no_example(
    monkeypatch, tmp_path, fake_decoder
):
    model = _DetectingModel(detected="ru")
    result = _verbatim_run(monkeypatch, tmp_path, model, model_size="medium")
    assert model.kwargs["language"] == "ru"
    assert model.kwargs["hotwords"] is None
    assert result.verbatim is False and result.language == "ru"


def test_a_line_that_only_repeats_the_example_is_left_out_and_counted(
    monkeypatch, tmp_path
):
    model = _DetectingModel(
        lines=[
            _Seg(0.0, 2.0, " I, I went - I drove there."),
            _Seg(
                2.0,
                9.0,
                " Umm, let me think like, hmm... Okay, here's what I'm, like, thinking.",
            ),
            _Seg(9.0, 10.0, " Okay."),
        ]
    )
    said = []
    result = _verbatim_run(
        monkeypatch,
        tmp_path,
        model,
        model_size="medium",
        language="en",
        progress=said.append,
    )
    assert [s.text for s in result.segments] == ["I, I went - I drove there.", "Okay."]
    assert result.example_lines_removed == 1
    assert "repeated the verbatim example" in " ".join(said)
    assert "Umm, let me think" not in result.text


def test_without_verbatim_a_matching_line_is_left_alone(monkeypatch, tmp_path):
    # The guard is for text the example put there; with no example given, a
    # line is speech however it reads.
    model = _DetectingModel(lines=[_Seg(0.0, 3.0, " Umm, let me think like, hmm.")])
    monkeypatch.setattr(transcription, "_load_model", lambda *a, **k: model)
    audio = tmp_path / "clip.wav"
    audio.write_bytes(b"\x00")
    result = transcription.transcribe_audio(audio, language="en")
    assert len(result.segments) == 1 and result.example_lines_removed == 0


def test_the_record_says_whether_hesitations_were_kept():
    from sotto.provenance import AnalysisProvenance, TranscriptionProvenance

    record = TranscriptionProvenance(
        model_name="turbo", verbatim=True, example_lines_removed=2
    )
    assert TranscriptionProvenance.from_dict(record.to_dict()) == record
    lines = AnalysisProvenance(transcription=record).describe()
    assert (
        "Hesitations and repeats kept (verbatim example given): yes; 2 line(s) "
        "that only repeated the example were left out"
    ) in lines
    older = TranscriptionProvenance.from_dict({"model_name": "turbo"})
    assert older.verbatim is None
    assert not any(
        "Hesitations" in line
        for line in AnalysisProvenance(transcription=older).describe()
    )


def test_a_recording_with_nothing_to_detect_is_transcribed_without_the_example(
    monkeypatch, tmp_path, fake_decoder
):
    class Silent(_DetectingModel):
        def detect_language(self, audio, vad_filter, vad_parameters):
            raise ValueError("max() arg is an empty sequence")

    model = Silent()
    result = _verbatim_run(monkeypatch, tmp_path, model, model_size="medium")
    assert model.kwargs["language"] is None  # left to faster-whisper, as before
    assert model.kwargs["hotwords"] is None
    assert result.verbatim is False


# -- Times on each line, and the spreadsheet ---------------------------------


def _two_speakers():
    from sotto.transcription import Segment, TranscriptionResult, Word

    return TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=4000.0,
        segments=[
            Segment(5.25, 7.0, "Hello there.", speaker="SPEAKER_00"),
            Segment(
                3725.5,
                3727.0,
                "=SUM(A1) he said",
                speaker="SPEAKER_01",
                words=[Word(3725.5, 3727.0, " =SUM(A1)", probability=0.2)],
            ),
        ],
    )


def test_clock_time_is_hours_minutes_seconds():
    from sotto.transcription import clock_time

    assert clock_time(754.5) == "00:12:34"
    assert clock_time(754.5, millis=True) == "00:12:34.500"
    assert clock_time(7205.0004, millis=True) == "02:00:05.000"
    assert clock_time(-3) == "00:00:00"


def test_times_start_each_line_only_when_asked():
    result = _two_speakers()
    names = {"SPEAKER_00": "Ann"}
    plain = result.to_txt(names)
    timed = result.to_txt(names, times=True)
    assert plain.splitlines()[0] == "[Ann] Hello there."
    assert timed.splitlines()[0] == "[00:00:05] [Ann] Hello there."
    assert timed.splitlines()[1].startswith("[01:02:05] [")


def test_times_without_speakers():
    from sotto.transcription import Segment, TranscriptionResult

    result = TranscriptionResult(
        "",
        "en",
        1.0,
        10.0,
        segments=[Segment(1.0, 2.0, "one"), Segment(65.0, 66.0, "two")],
    )
    assert result.to_txt(times=True) == "[00:00:01] one\n[00:01:05] two"


def test_the_spreadsheet_has_a_row_per_line_and_never_a_formula():
    import csv
    import io

    rows = list(csv.reader(io.StringIO(_two_speakers().to_csv({"SPEAKER_00": "Ann"}))))
    assert rows[0] == [
        "Start",
        "End",
        "Start (s)",
        "End (s)",
        "Speaker",
        "Text",
        "Low confidence",
        "Analyst note",
    ]
    assert rows[1] == [
        "00:00:05.250",
        "00:00:07.000",
        "5.250",
        "7.000",
        "Ann",
        "Hello there.",
        "",
        "",
    ]
    # Someone else's words can't become a formula on the analyst's machine.
    assert rows[2][5] == "'=SUM(A1) he said"
    assert rows[2][6] == "yes"  # a low-confidence word marks the line
    assert len(rows) == 3


# -- Analyst notes -------------------------------------------------------------


def test_a_note_belongs_to_the_line_holding_its_moment():
    from sotto.transcription import Note, Segment, note_lines

    lines = [Segment(0, 4, "a"), Segment(5, 9, "b"), Segment(12, 15, "c")]
    notes = [
        Note(5.0, "on b"),
        Note(10.5, "in the gap"),
        Note(-1, "early"),
        Note(14, "c"),
    ]
    assert note_lines(lines, notes) == {1: [0, 1], 0: [2], 2: [3]}
    assert note_lines([], notes) == {}


def test_notes_are_written_under_their_lines_and_labelled():
    from sotto.transcription import Note

    result = _two_speakers()
    result.notes = [Note(5.25, "Sounds nervous.\nCheck the 05:10 call too.")]
    text = result.to_txt({"SPEAKER_00": "Ann"})
    assert text.splitlines()[:3] == [
        "[Ann] Hello there.",
        "    Analyst note: Sounds nervous.",
        "    Check the 05:10 call too.",
    ]
    assert "Analyst note" not in result.to_txt({"SPEAKER_00": "Ann"}, notes=False)


def test_the_spreadsheet_carries_notes_in_their_own_column():
    import csv
    import io

    from sotto.transcription import Note

    result = _two_speakers()
    result.notes = [Note(3725.5, "=not a formula"), Note(3726.0, "second")]
    rows = list(csv.reader(io.StringIO(result.to_csv())))
    assert rows[0][-1] == "Analyst note"
    assert rows[2][-1] == "'=not a formula | second"
    assert rows[1][-1] == ""
    assert all(
        row[-1] == ""
        for row in csv.reader(io.StringIO(result.to_csv(notes=False)))
        if row[0] != "Start"
    )
