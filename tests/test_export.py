import importlib.util

import pytest

from whispers.export import text_to_docx, transcript_to_docx
from whispers.transcription import Segment, TranscriptionResult

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

    from whispers.transcription import Note

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


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_text_from_a_pdf_with_control_characters_saves(tmp_path):
    """A form feed, a stray control code or half a surrogate pair used to make
    Word export fail with a Unicode error."""
    import docx

    text = "Page one\x0cPage two\x07 ends\r\nhere \ud83d."
    out = text_to_docx(text, tmp_path / "out.docx", title="Title\x00")
    paragraphs = [p.text for p in docx.Document(str(out)).paragraphs]
    assert paragraphs == ["Title", "Page one", "Page two ends\nhere �."]


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_arabic_paragraphs_are_laid_out_right_to_left(tmp_path):
    import docx
    from docx.oxml.ns import qn

    out = text_to_docx("مرحبا بالعالم\n\nHello", tmp_path / "out.docx")
    arabic, english = docx.Document(str(out)).paragraphs
    assert arabic._p.pPr.find(qn("w:bidi")) is not None
    assert arabic.runs[0]._r.rPr.find(qn("w:rtl")) is not None
    assert english._p.pPr is None or english._p.pPr.find(qn("w:bidi")) is None


@pytest.mark.skipif(not _HAS_DOCX, reason="python-docx not installed")
def test_a_translated_email_export_reads_as_emails(tmp_path):
    import docx

    from whispers.export import translation_to_docx
    from whispers.translation import (
        Piece,
        TranslatedDocument,
        TranslatedSection,
    )

    email = TranslatedSection(
        fields=[
            ("From", Piece("علي <a@example.com>", "علي <a@example.com>")),
            ("Subject", Piece("جدول الاجتماع", "Meeting agenda", language="ar")),
        ],
        paragraphs=[
            Piece("مرحبا", "Hello", language="ar"),
            Piece("안녕", "안녕", missing="ko"),
        ],
    )
    document = TranslatedDocument(
        sections=[email],
        names={"ar": "Arabic", "ko": "Korean"},
        removed=["CUI // Banner", "Page 1 of 1"],
    )
    out = translation_to_docx(document, tmp_path / "out.docx", originals=True)
    paragraphs = docx.Document(str(out)).paragraphs
    texts = [p.text for p in paragraphs]
    assert texts[0] == "Translation"
    assert "2 repeated page header" in texts[1]
    assert texts[2:] == [
        "Email 1 of 1: Meeting agenda",
        "From: علي <a@example.com>",
        "Subject: Meeting agenda",
        "Arabic original: جدول الاجتماع",
        "Hello",
        "Arabic original: مرحبا",
        "[Korean, not translated: no language pack] 안녕",
        "Left out as page headers and footers",
        "CUI // Banner",
    ]
    assert paragraphs[2].style.name == "Heading 2"
    assert paragraphs[3].runs[0].bold
    assert paragraphs[5].runs[1].italic
