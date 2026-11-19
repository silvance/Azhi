"""Side-by-side translation of a PDF: what is marked, and what is written."""

import importlib.util

import pytest
from conftest import FRENCH, SPANISH, detect_test_languages

from whispers.passages import read_pdf
from whispers.side_by_side import (
    _latin,
    page_list,
    render_text,
    sha256_of,
    translate_pages,
    write_pdf,
)

NAMES = {"es": "Spanish", "fr": "French"}
SOFTWARE = [("WHISPERS", "test build"), ("Language pack es to English", "1.0")]


class Fake:
    def __init__(self, code):
        self.code = code

    def translate(self, text):
        return f"[{self.code}] the convoy left at six"


def _triage(path):
    pages = read_pdf(path)
    return translate_pages(
        path,
        pages,
        forced=None,
        installed={"es"},
        names=NAMES,
        detect=detect_test_languages,
        translator_for=Fake,
        software=SOFTWARE,
    )


def test_page_numbers_are_listed_in_runs():
    assert page_list([0, 2, 3, 4, 7]) == "1, 3-5, 8"
    assert page_list([]) == ""


def test_text_the_standard_font_cannot_show_is_marked():
    assert _latin("Café “OK” — 5€") == "Café “OK” — 5€"
    assert _latin("Zhang Wei 张伟 said") == "Zhang Wei [?] said"
    assert _latin("Gdańsk") == "Gdansk"


def test_only_foreign_passages_are_marked_and_numbered_per_page(report_pdf):
    triage = _triage(report_pdf)
    assert sorted(triage.marked) == [0, 2]
    (spanish,) = triage.marked[0]
    assert spanish.number == 1
    assert spanish.passage.text == SPANISH
    assert spanish.piece.text == "[es] the convoy left at six"
    (french,) = triage.marked[2]
    assert french.piece.missing == "fr" and not french.piece.translated
    assert triage.sha256 == sha256_of(report_pdf)
    assert triage.summary() == (
        "Foreign-language text on 2 of 3 page(s). Translated 1 passage(s) of "
        "Spanish. No language pack for French: marked, not translated."
    )


def test_the_listing_gives_each_passage_with_where_it_is(report_pdf):
    text = render_text(_triage(report_pdf))
    assert "Unreviewed machine translation" in text
    assert f"SHA-256: {sha256_of(report_pdf)}" in text
    assert "Spanish: 1 passage(s) translated, on page(s) 1" in text
    assert "French: 1 passage(s) NOT translated (no language pack in this copy)" in text
    assert "WHISPERS: test build" in text
    assert f"──────── Page 1 ────────\n\n[1] Spanish\n    Original: {SPANISH}" in text
    assert "    English:  [es] the convoy left at six" in text
    assert f"[1] French: not translated (no language pack)\n    Original: {FRENCH}" in (
        text
    )


def test_the_pdf_has_a_summary_then_each_foreign_page_beside_its_english(
    report_pdf, tmp_path
):
    import pypdfium2 as pdfium

    out = write_pdf(_triage(report_pdf), tmp_path / "report.en.pdf")
    pdf = pdfium.PdfDocument(str(out))
    try:
        assert len(pdf) == 3  # summary, page 1, page 3; page 2 had no foreign text
        texts = [pdf[i].get_textpage().get_text_range() for i in range(len(pdf))]
        width, height = pdf.get_page_size(1)
    finally:
        pdf.close()
    summary, first, third = texts
    assert "Machine translation for language triage" in summary
    assert sha256_of(report_pdf) in summary.replace("\r\n", "")
    assert "Page 1 of 3" in first and "Page 3 of 3" in third
    # The original page is there as it was, its text still searchable.
    assert SPANISH in first and "OK, see you there." in first
    assert "[es] the convoy left at six" in first
    assert "no language pack in this copy, not translated" in third
    assert (width, height) == pytest.approx((1224, 792))


@pytest.mark.skipif(
    importlib.util.find_spec("docx") is None, reason="python-docx not installed"
)
def test_the_word_copy_is_a_table_of_passages(report_pdf, tmp_path):
    import docx

    from whispers.export import side_by_side_docx

    out = side_by_side_docx(_triage(report_pdf), tmp_path / "report.en.docx")
    table = docx.Document(str(out)).tables[0]
    rows = [[cell.text for cell in row.cells] for row in table.rows]
    assert rows[0] == [
        "Page",
        "#",
        "Language",
        "Original",
        "English (machine translation)",
    ]
    assert rows[1] == ["1", "1", "Spanish", SPANISH, "[es] the convoy left at six"]
    assert rows[2] == [
        "3",
        "1",
        "French (no language pack)",
        FRENCH,
        "Not translated: no language pack in this copy.",
    ]
