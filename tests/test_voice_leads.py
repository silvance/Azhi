"""A name suggested by voice similarity is never presented as an identity.

QA found a recognised turn exported as "[Viktor Marchenko] I'll be there at
nine." - indistinguishable from a name the operator typed, in the .txt, .srt,
Word file and analysis report. That is an identity claim made from voice
similarity, which this application must never make. The name now carries a
lead marker everywhere until the operator confirms it.
"""

import pytest

from azhi import speaker_names as sn
from azhi.reports import _transcript_lines
from azhi.transcription import Segment, TranscriptionResult


def _result():
    return TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=10.0,
        segments=[
            Segment(
                start=0.0,
                end=4.0,
                text="I'll be there at nine.",
                speaker="voice::Viktor Marchenko",
            ),
            Segment(start=4.0, end=8.0, text="Fine.", speaker="SPEAKER_01"),
        ],
    )


def test_a_recognised_name_is_shown_as_a_lead():
    names = sn.preset_names(
        ["voice::Viktor Marchenko", "SPEAKER_01"],
        {"voice::Viktor Marchenko": "Viktor Marchenko"},
        ["", "Anna"],
    )
    assert names["voice::Viktor Marchenko"] == "Viktor Marchenko" + sn.LEAD_SUFFIX
    assert sn.is_lead(names["voice::Viktor Marchenko"])


def test_a_typed_name_is_not_a_lead():
    names = sn.preset_names(["SPEAKER_00"], {}, ["Anna"])
    assert names == {"SPEAKER_00": "Anna"}
    assert not sn.is_lead(names["SPEAKER_00"])


def test_marking_is_idempotent_and_reversible():
    lead = sn.as_lead("Viktor")
    assert sn.as_lead(lead) == lead, "a name is never marked twice"
    assert sn.bare_name(lead) == "Viktor"
    assert sn.bare_name("Anna") == "Anna"
    assert not sn.is_lead(None) and not sn.is_lead("")


def _names():
    return {
        "voice::Viktor Marchenko": sn.as_lead("Viktor Marchenko"),
        "SPEAKER_01": "Anna",
    }


@pytest.mark.parametrize("blank_lines", [False, True])
def test_a_text_export_marks_the_lead_and_explains_it(blank_lines):
    text = _result().to_txt(_names(), blank_lines=blank_lines)
    assert "[Viktor Marchenko]" not in text, "never an unqualified identity"
    assert "[Viktor Marchenko (unconfirmed voice lead)] I'll be there at nine." in text
    assert text.startswith(sn.LEAD_LEGEND)
    assert "not an identification" in text


def test_subtitles_mark_the_lead_on_the_line():
    srt = _result().to_srt(_names())
    assert "[Viktor Marchenko]" not in srt
    assert "Viktor Marchenko (unconfirmed voice lead)" in srt


def test_the_analysis_report_marks_and_explains_the_lead():
    lines = _transcript_lines(_result(), _names())
    assert lines[0] == sn.LEAD_LEGEND
    assert any("Viktor Marchenko (unconfirmed voice lead)" in line for line in lines)
    assert not any("[Viktor Marchenko]" in line for line in lines)


def test_no_legend_when_every_name_was_typed_or_confirmed():
    names = {"voice::Viktor Marchenko": "Viktor Marchenko", "SPEAKER_01": "Anna"}
    text = _result().to_txt(names)
    assert sn.LEAD_LEGEND not in text
    assert _transcript_lines(_result(), names)[0] != sn.LEAD_LEGEND


def test_a_word_export_marks_and_explains_the_lead(tmp_path):
    docx = pytest.importorskip("docx")
    from azhi.export import transcript_to_docx

    out = transcript_to_docx(_result(), tmp_path / "t.docx", _names())
    paragraphs = [p.text for p in docx.Document(str(out)).paragraphs]
    assert paragraphs[0] == sn.LEAD_LEGEND
    assert any("Viktor Marchenko (unconfirmed voice lead)" in p for p in paragraphs)
    assert not any("[Viktor Marchenko]" in p for p in paragraphs)
