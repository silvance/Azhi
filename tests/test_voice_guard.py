"""Voices are never measured from processed audio.

QA found that with noise reduction on, the cleaned copy became the audio kept
for the session, so later enrolment, learning from corrections and recognition
all measured cleaned audio and recorded it under the original recording's
hash. The run now pins what voices are measured from to the original; these
check the other half - that a transcript *of* a derived file cannot reach a
voiceprint by any of the Transcribe page's routes.
"""

import pytest

from azhi import enhance, listening

tab = pytest.importorskip("azhi.gui.transcribe_tab")


@pytest.mark.parametrize(
    "name",
    [
        "call.wav" + enhance.SUFFIX,
        "call.wav" + listening.SUFFIX,
        # Windows file names are case-insensitive, so these are the same files.
        "CALL.WAV.CLEANED.WAV",
        "Call.wav.Listening.Wav",
    ],
)
def test_a_transcript_of_a_derived_copy_is_refused_for_voice_work(tmp_path, name):
    message = tab._derived_voice_refusal(tmp_path / name)
    assert message is not None
    assert "not the recording" in message
    assert "Transcribe the original recording" in message


def test_an_original_recording_is_not_refused(tmp_path):
    assert tab._derived_voice_refusal(tmp_path / "call.wav") is None


def test_no_source_at_all_is_not_refused():
    """Nothing to judge yet; the callers have their own "transcribe first"."""
    assert tab._derived_voice_refusal(None) is None
