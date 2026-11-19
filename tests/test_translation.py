import importlib.util

import pytest

from whispers import translation
from whispers.documents import Section
from whispers.translation import detect_language, translate_document


def test_chunk_text_rejoins_and_bounds():
    text = "line one\nline two\n\nparagraph two with more words\n"
    chunks = translation._chunk_text(text, max_chars=10)
    # Concatenation is lossless (structure preserved).
    assert "".join(chunks) == text
    # Every chunk respects the cap.
    assert all(len(c) <= 10 for c in chunks)


def test_chunk_text_hard_splits_long_line():
    text = "x" * 25
    chunks = translation._chunk_text(text, max_chars=10)
    assert "".join(chunks) == text
    assert all(len(c) <= 10 for c in chunks)
    assert len(chunks) == 3


def test_chunk_text_empty():
    assert translation._chunk_text("") == [""]


def test_translate_text_without_argos(monkeypatch):
    # argostranslate is not installed in this environment -> clear RuntimeError.
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_text("hola", from_code="es", to_code="en")


def test_available_source_languages_without_argos():
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.available_source_languages()


def test_translate_files_without_argos(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hola", encoding="utf-8")
    with pytest.raises(RuntimeError, match="Argos Translate is not installed"):
        translation.translate_files([f], from_code="es", to_code="en")


def test_detect_language_empty_is_none():
    assert detect_language("") is None
    assert detect_language("   ") is None


def test_detect_language_without_langdetect():
    if importlib.util.find_spec("langdetect") is not None:
        pytest.skip("langdetect installed; missing-dependency path not exercised")
    assert detect_language("this is plainly english text") is None


def test_detect_language_english():
    pytest.importorskip("langdetect")
    assert detect_language("This is a clearly English sentence about the weather.") == (
        "en"
    )


# -- Documents ---------------------------------------------------------------

NAMES = {"zh": "Chinese", "ar": "Arabic", "ko": "Korean"}


class _Fake:
    """Stands in for an Argos translator and counts what it was given."""

    calls: list = []

    def __init__(self, code):
        self.code = code

    def translate(self, text):
        _Fake.calls.append((self.code, text))
        return f"<{self.code}: {text}>"


def _translate(sections, **kwargs):
    _Fake.calls = []
    kwargs.setdefault("forced", None)
    kwargs.setdefault("installed", {"zh", "ar"})
    return translate_document(
        sections,
        names=NAMES,
        detect=lambda _: [("en", 0.99)],
        translator_for=_Fake,
        **kwargs,
    )


def test_only_foreign_paragraphs_go_to_the_translator():
    section = Section(
        paragraphs=[
            "Meeting notes, prepared by the liaison officer for the section.",
            "会议定于下周三上午九点举行。",
            "Smith, John",
        ]
    )
    document = _translate([section])
    assert _Fake.calls == [("zh", "会议定于下周三上午九点举行。")]
    english, chinese, name = document.sections[0].paragraphs
    assert english.text == english.original and not english.translated
    assert chinese.text == "<zh: 会议定于下周三上午九点举行。>"
    assert chinese.language == "zh"
    assert name.text == "Smith, John"


def test_of_an_email_header_only_the_subject_is_translated():
    email = Section(
        fields=[
            ("From", "علي <a@example.com>"),
            ("Sent", "Tuesday, March 4, 2025 9:12 AM"),
            ("Subject", "جدول الاجتماع"),
        ],
        paragraphs=["مرحبا"],
    )
    document = _translate([email])
    assert [text for _, text in _Fake.calls] == ["جدول الاجتماع", "مرحبا"]
    fields = dict(document.sections[0].fields)
    assert fields["From"].text == "علي <a@example.com>"
    assert fields["Subject"].text == "<ar: جدول الاجتماع>"


def test_a_language_with_no_pack_is_marked_not_sent():
    document = _translate([Section(paragraphs=["안녕하세요 세계"])])
    assert _Fake.calls == []
    (piece,) = document.sections[0].paragraphs
    assert piece.missing == "ko" and not piece.translated
    assert "[Korean, not translated: no language pack] 안녕하세요 세계" in (
        document.render_text()
    )
    assert "No language pack in this copy for Korean" in document.summary()


def test_progress_counts_only_what_is_translated():
    seen = []
    _translate(
        [Section(paragraphs=["English text here.", "中文一", "中文二"])],
        on_progress=lambda done, total, left: seen.append((done, total, left)),
    )
    # The English paragraph costs nothing; no estimate before 5 seconds.
    assert seen == [(1, 2, None), (2, 2, None)]


def test_cancelling_stops_before_the_next_paragraph():
    with pytest.raises(translation.CancelledError):
        _translate([Section(paragraphs=["中文"])], cancelled=lambda: True)


def test_rendered_text_keeps_emails_apart_and_lists_what_was_left_out():
    email = Section(
        fields=[("From", "Ali"), ("To", "Sara"), ("Subject", "جدول")],
        paragraphs=["مرحبا"],
    )
    document = _translate(
        [Section(paragraphs=["Cover page"]), email, email],
        removed=["CUI // Banner", "Page 1 of 2", "CUI // Banner", "Page 2 of 2"],
    )
    text = document.render_text(originals=True)
    assert "──────── Email 1 of 2 ────────" in text
    assert "──────── Email 2 of 2 ────────" in text
    assert "Subject: <ar: جدول>\n    Arabic original: جدول" in text
    assert "<ar: مرحبا>\n    Arabic original: مرحبا" in text
    assert "4 repeated page header, footer and page-number line(s)" in text
    # Each left-out line once, and page numbers not listed at all.
    assert text.endswith(
        "──────── Left out as page headers and footers ────────\nCUI // Banner\n"
    )
    assert document.summary() == (
        "Translated 4 passage(s) from Arabic; left 5 as they were (English, "
        "names, numbers)."
    )
    assert "<ar: مرحبا>\n    Arabic" not in document.render_text()
