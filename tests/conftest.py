import random as rand

import numpy
import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "requires_cuda")


@pytest.fixture
def random():
    rand.seed(42)
    numpy.random.seed(42)


SPANISH = "El convoy salió de la base a las seis y llegó al río al mediodía."
FRENCH = "Le convoi a quitté la base à six heures et il est arrivé à midi."


def detect_test_languages(text):
    """A stand-in for langdetect that knows the test sentences."""
    if "convoy" in text:
        return [("es", 0.99)]
    if "convoi" in text:
        return [("fr", 0.99)]
    return [("en", 0.99)]


@pytest.fixture
def report_pdf(tmp_path):
    """A three-page PDF: Spanish beside English, English only, then French.

    Drawn with pdfium's standard font, so only Latin-script languages can be
    in it; detect_test_languages tells them apart.
    """
    pytest.importorskip("pypdfium2")
    from whispers.side_by_side import _Pdf

    pdf = _Pdf()

    def page(lines):
        p = pdf.doc.new_page(612, 792)
        for text, x, y in lines:
            pdf.text(p, text, x, y, pdf.regular, 10)
        p.gen_content()
        p.close()

    page(
        [
            ("Extraction Report - Chats", 72, 740),
            ("1", 60, 700),
            ("3/12/2023 10:15", 90, 700),
            (SPANISH, 200, 700),
            ("2", 60, 670),
            ("3/12/2023 10:16", 90, 670),
            ("OK, see you there.", 200, 670),
        ]
    )
    page([("Examiner notes: nothing further on this page of the report.", 72, 700)])
    page([(FRENCH, 72, 700)])
    path = tmp_path / "report.pdf"
    pdf.save(path)
    return path
