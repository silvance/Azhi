"""The Translate page: mixed-language text, extracted PDFs, Word export.

Builds the real page, so it needs a display; skipped without one. Argos and
Tesseract are stand-ins: no language pack or OCR data is needed.
"""

import functools
import os
import time

import pytest

tk = pytest.importorskip("tkinter")

pytestmark = pytest.mark.skipif(
    os.name != "nt" and not os.environ.get("DISPLAY"),
    reason="needs a display",
)

BANNER = "CUI // Exported from A365 eDiscovery - Case 2025-0117"
PAGES = [
    f"{BANNER}\nFrom: Ali <a@example.com>\nSent: Tuesday\nSubject: جدول الاجتماع\n\n"
    "مرحبا\nPage 1 of 3",
    f"{BANNER}\nFrom: Sara <s@example.com>\nSent: Wednesday\nSubject: تأجيل\n\n"
    "Please confirm receipt.\nPage 2 of 3",
    f"{BANNER}\nFrom: Ali <a@example.com>\nSent: Thursday\nSubject: شكرا\n\n"
    "تم\nPage 3 of 3",
]


class Fake:
    def __init__(self, code):
        self.code = code

    def translate(self, text):
        return f"<{self.code}: {text}>"


class InlineThread:
    def __init__(self, target, daemon=None, args=()):
        self._target, self._args = target, args

    def start(self):
        self._target(*self._args)


@pytest.fixture
def page(tmp_path, monkeypatch):
    from whispers.gui import translate_tab
    from whispers.gui.theme import init_theme
    from whispers.translation import translate_document

    monkeypatch.setattr(
        translate_tab,
        "translate_document",
        functools.partial(
            translate_document, translator_for=Fake, detect=lambda _: [("en", 0.99)]
        ),
    )
    monkeypatch.setattr(translate_tab, "extract_pages", lambda *a, **k: list(PAGES))
    # Workers hand back through root.after, which Tk allows from another
    # thread only inside a running main loop; run them in line here.
    monkeypatch.setattr(translate_tab.threading, "Thread", InlineThread)
    root = tk.Tk()
    init_theme(root)
    try:
        frame = tk.Frame(root)
        frame.pack(fill="both", expand=True)
        tab = translate_tab.TranslateTab(
            frame, root, ocr_available=True, detect_available=True, dnd_ok=False
        )
        _settle(root)
        tab._translate_lang_codes = {
            translate_tab.AUTO_DETECT_LABEL: translate_tab.AUTO_DETECT_CODE,
            "Arabic": "ar",
            "Chinese": "zh",
        }
        tab.translate_from_var.set(translate_tab.AUTO_DETECT_LABEL)
        yield tab, root, tmp_path, translate_tab
    finally:
        root.destroy()


def _settle(root, seconds=0.2):
    deadline = time.time() + seconds
    while time.time() < deadline:
        root.update()
        time.sleep(0.02)


def _output(tab):
    return tab.translate_output.get("1.0", "end-1c")


def test_only_the_foreign_text_of_a_mixed_paste_is_translated(page):
    tab, root, _, _ = page
    tab.translate_input.insert(
        "end",
        "Meeting notes, prepared by the liaison officer.\n\n会议定于下周三举行。",
    )
    tab._translate_paste_in_thread()
    _settle(root)
    out = _output(tab)
    assert "Meeting notes, prepared by the liaison officer." in out
    assert "<zh: 会议定于下周三举行。>" in out
    assert tab.translate_status_var.get().startswith(
        "Translated 1 passage(s) from Chinese; left 1"
    )


def test_an_email_export_comes_out_as_emails_without_its_page_furniture(
    page, monkeypatch
):
    tab, root, tmp_path, module = page
    tab._extract_to_paste_worker(tmp_path / "export.pdf", None)
    _settle(root)
    pasted = tab.translate_input.get("1.0", "end-1c")
    assert BANNER not in pasted and "Page 2 of 3" not in pasted
    assert "leaving out 6 repeated header" in tab.translate_status_var.get()

    tab._translate_paste_in_thread()
    _settle(root)
    out = _output(tab)
    assert "──────── Email 3 of 3 ────────" in out
    assert "From: Sara <s@example.com>" in out
    assert "Subject: <ar: جدول الاجتماع>" in out
    assert "Please confirm receipt." in out
    assert out.rstrip().endswith(BANNER)

    saved = tmp_path / "translation.docx"
    monkeypatch.setattr(module.filedialog, "asksaveasfilename", lambda **_: str(saved))
    tab._save_translation_docx()
    import docx

    headings = [
        p.text
        for p in docx.Document(str(saved)).paragraphs
        if p.style.name.startswith("Heading")
    ]
    assert headings[1:4] == [
        "Email 1 of 3: <ar: جدول الاجتماع>",
        "Email 2 of 3: <ar: تأجيل>",
        "Email 3 of 3: <ar: شكرا>",
    ]


def test_left_out_lines_are_not_reported_for_a_different_paste(page):
    tab, root, tmp_path, _ = page
    tab._extract_to_paste_worker(tmp_path / "export.pdf", None)
    _settle(root)
    tab.translate_input.delete("1.0", "end")
    tab.translate_input.insert("end", "مرحبا بكم")
    tab._translate_paste_in_thread()
    _settle(root)
    assert "Left out" not in _output(tab)


def test_a_batch_pdf_gets_a_text_and_a_word_reading_copy(page):
    tab, root, tmp_path, module = page
    tab.pdf_output_var.set(module.READING_COPY)
    src = tmp_path / "export.pdf"
    src.write_bytes(b"%PDF-1.4 stand-in")
    tab._add_translate_paths([src])
    tab._translate_files_in_thread()
    _settle(root)
    assert (tmp_path / "export.ocr.txt").read_text(encoding="utf-8").startswith(BANNER)
    assert "Email 1 of 3" in (tmp_path / "export.en.txt").read_text(encoding="utf-8")
    assert (tmp_path / "export.en.docx").exists()
    assert tab.translate_status_var.get() == "Done: translated 1, extracted 1"


def test_a_batch_pdf_is_written_side_by_side(page, report_pdf, monkeypatch):
    from conftest import detect_test_languages

    from whispers.side_by_side import translate_pages

    tab, root, _, module = page
    monkeypatch.setattr(
        module,
        "translate_pages",
        functools.partial(
            translate_pages,
            detect=detect_test_languages,
            translator_for=Fake,
            software=[("WHISPERS", "test")],
        ),
    )
    tab._translate_lang_codes["Spanish"] = "es"
    assert tab.pdf_output_var.get() == module.SIDE_BY_SIDE
    tab._add_translate_paths([report_pdf])
    tab._translate_files_in_thread()
    _settle(root)
    folder = report_pdf.parent
    for name in ("report.ocr.txt", "report.en.txt", "report.en.docx", "report.en.pdf"):
        assert (folder / name).exists(), name
    assert "[1] Spanish" in (folder / "report.en.txt").read_text(encoding="utf-8")
    assert tab.translate_status_var.get() == (
        "Done: translated 1, extracted 1 (see Result)"
    )
    assert _output(tab) == (
        "Found:\n\nreport.pdf: Foreign-language text on 2 of 3 page(s). "
        "Translated 1 passage(s) of Spanish. No language pack for French: "
        "marked, not translated. Written to report.en.pdf."
    )


def test_settings_are_remembered(page):
    tab, _, _, module = page
    tab.apply_settings({"keep_original": True, "pdf_output": module.READING_COPY})
    assert tab.get_settings() == {
        "keep_original": True,
        "pdf_output": module.READING_COPY,
    }


def test_time_left_reads_in_words(page):
    _, _, _, module = page
    assert module._duration(40) == "about 40 sec"
    assert module._duration(720) == "about 12 min"
    assert module._duration(7500) == "about 2 h 5 min"
