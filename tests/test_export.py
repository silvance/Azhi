import importlib.util

import pytest

from azhi.export import text_to_docx, transcript_to_docx
from azhi.transcription import Segment, TranscriptionResult

_HAS_DOCX = importlib.util.find_spec("docx") is not None


@pytest.mark.skipif(_HAS_DOCX, reason="python-docx installed; error path not exercised")
def test_text_to_docx_requires_python_docx(tmp_path):
    with pytest.raises(RuntimeError, match="python-docx is not installed"):
        text_to_docx("hello", tmp_path / "out.docx")


@pytest.mark.skipif(_HAS_DOCX, reason="python-docx installed; error path not exercised")
def test_transcript_to_docx_requires_python_docx(tmp_path):
    result = TranscriptionResult(
        text="hi", language="en", language_probability=1.0, duration=1.0
    )
    with pytest.raises(RuntimeError, match="python-docx is not installed"):
        transcript_to_docx(result, tmp_path / "out.docx")


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_text_to_docx_writes_paragraphs(tmp_path):
    from docx import Document

    out = text_to_docx("first block\nsecond line\n\nthird block", tmp_path / "t.docx")
    assert out.exists()
    doc = Document(str(out))
    texts = [p.text for p in doc.paragraphs]
    assert "first block" in texts[0]
    assert "third block" in "\n".join(texts)


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_transcript_to_docx_with_speakers(tmp_path):
    from docx import Document

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
    out = transcript_to_docx(result, tmp_path / "t.docx", {"SPEAKER_00": "Alice"})
    doc = Document(str(out))
    joined = "\n".join(p.text for p in doc.paragraphs)
    assert "[Alice] hello" in joined
    assert "[Speaker 2] world" in joined


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_word_transcript_can_carry_the_time_of_each_line(tmp_path):
    from docx import Document

    result = TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=80.0,
        segments=[
            Segment(start=75.0, end=76.0, text="hello", speaker="SPEAKER_00"),
        ],
    )
    out = transcript_to_docx(
        result, tmp_path / "t.docx", {"SPEAKER_00": "Ann"}, times=True
    )
    joined = "\n".join(p.text for p in Document(str(out)).paragraphs)
    assert "[00:01:15] [Ann] hello" in joined
    untimed = transcript_to_docx(result, tmp_path / "u.docx", {"SPEAKER_00": "Ann"})
    assert "[00:01:15]" not in "\n".join(
        p.text for p in Document(str(untimed)).paragraphs
    )


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_word_notes_follow_their_line_in_italics(tmp_path):
    from docx import Document

    from azhi.transcription import Note

    result = TranscriptionResult(
        text="",
        language="en",
        language_probability=1.0,
        duration=5.0,
        segments=[Segment(start=1.0, end=2.0, text="hello", speaker="SPEAKER_00")],
    )
    result.notes = [Note(1.0, "Who is this?")]
    paragraphs = Document(
        str(transcript_to_docx(result, tmp_path / "n.docx"))
    ).paragraphs
    note = next(p for p in paragraphs if p.text == "Analyst note: Who is this?")
    assert all(run.italic for run in note.runs)
    plain = Document(str(transcript_to_docx(result, tmp_path / "p.docx", notes=False)))
    assert "Analyst note" not in "\n".join(p.text for p in plain.paragraphs)
