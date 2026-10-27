from whispers.project import PROJECT_SUFFIX, load_project, save_project
from whispers.transcription import Segment, TranscriptionResult, Word


def _result():
    return TranscriptionResult(
        text="hello world",
        language="en",
        language_probability=0.99,
        duration=2.0,
        segments=[
            Segment(
                start=0.0,
                end=1.0,
                text="hello",
                speaker="SPEAKER_00",
                avg_logprob=-0.2,
                words=[Word(start=0.0, end=1.0, word=" hello", probability=0.9)],
            ),
            Segment(
                start=1.0,
                end=2.0,
                text="world",
                speaker="SPEAKER_01",
                words=[Word(start=1.0, end=2.0, word=" world", probability=0.4)],
            ),
        ],
    )


def test_save_then_load_round_trips(tmp_path):
    result = _result()
    names = {"SPEAKER_00": "Alice"}
    path = tmp_path / ("clip" + PROJECT_SUFFIX)
    save_project(path, result, names, source="/audio/clip.mp3")
    assert path.exists()

    loaded, loaded_names, source = load_project(path)
    assert source == "/audio/clip.mp3"
    assert loaded_names == names
    assert loaded.language == "en"
    assert loaded.duration == 2.0
    assert [s.speaker for s in loaded.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in loaded.segments] == ["hello", "world"]
    assert loaded.segments[0].words[0].probability == 0.9
    assert loaded.segments[0].avg_logprob == -0.2
    assert loaded.has_speakers


def test_load_tolerates_missing_optional_fields(tmp_path):
    path = tmp_path / ("min" + PROJECT_SUFFIX)
    path.write_text(
        '{"version": 1, "result": {"segments": [{"text": "hi"}]}}', encoding="utf-8"
    )
    loaded, names, source = load_project(path)
    assert names == {}
    assert source is None
    assert loaded.segments[0].text == "hi"
    assert loaded.segments[0].speaker is None


def test_round_trip_preserves_non_diarized(tmp_path):
    result = TranscriptionResult(
        text="one two",
        language="en",
        language_probability=1.0,
        duration=1.0,
        segments=[Segment(start=0.0, end=1.0, text="one two")],
    )
    path = tmp_path / ("nd" + PROJECT_SUFFIX)
    save_project(path, result)
    loaded, names, source = load_project(path)
    assert not loaded.has_speakers
    assert loaded.segments[0].text == "one two"
    assert names == {}


# -- A recognised name reopens as what it is --------------------------------
#
# Before schema 3 a name put on a speaker by voice similarity was stored bare,
# so an older project reopened with "[Viktor Marchenko]" - an unqualified
# identity - and nothing saying it had only ever been a suggestion.

import json  # noqa: E402

from whispers.project import load_project_record  # noqa: E402
from whispers.speaker_names import as_lead, is_lead  # noqa: E402


def _older_project(tmp_path, names, version=2):
    path = tmp_path / f"old{PROJECT_SUFFIX}"
    path.write_text(
        json.dumps(
            {
                "schema_version": version,
                "result": {"text": "", "segments": []},
                "speaker_names": names,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_an_older_project_reopens_a_recognised_name_as_a_lead(tmp_path):
    path = _older_project(tmp_path, {"voice::Viktor Marchenko": "Viktor Marchenko"})
    names = load_project_record(path).speaker_names
    assert names["voice::Viktor Marchenko"] == as_lead("Viktor Marchenko")


def test_a_name_the_operator_typed_over_is_left_alone(tmp_path):
    """Differing from what the recogniser produced, it was their decision."""
    path = _older_project(
        tmp_path, {"voice::Viktor Marchenko": "V. Marchenko", "SPEAKER_01": "Anna"}
    )
    names = load_project_record(path).speaker_names
    assert names == {"voice::Viktor Marchenko": "V. Marchenko", "SPEAKER_01": "Anna"}


def test_a_current_project_keeps_confirmation(tmp_path):
    """From schema 3 a bare name on a voice:: speaker is a confirmed one."""
    path = tmp_path / f"new{PROJECT_SUFFIX}"
    result = TranscriptionResult(
        text="", language="en", language_probability=1.0, duration=0.0, segments=[]
    )
    save_project(
        path,
        result,
        {"voice::Viktor": "Viktor", "voice::Anna": as_lead("Anna")},
        "call.wav",
    )
    names = load_project_record(path).speaker_names
    assert names["voice::Viktor"] == "Viktor", "a confirmed name stays confirmed"
    assert is_lead(names["voice::Anna"]), "an unconfirmed lead stays a lead"
