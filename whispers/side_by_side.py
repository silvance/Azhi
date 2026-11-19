"""Side-by-side machine translation of a PDF, for language triage.

Each page of the original that has foreign-language text is reproduced
unchanged on the left, its foreign passages highlighted and numbered; the
English for each numbered passage is on the right, level with it where there
is room. A first page says what the file was (name and SHA-256), what was
found in which languages, what was not translated, what produced the
translation, and that it is an unreviewed machine translation for deciding
what a qualified linguist needs to see.

The original page is placed as it is (its text stays selectable and
searchable); only the English is drawn, in the PDF standard font Helvetica,
so nothing needs embedding and every reader shows it.
"""

from __future__ import annotations

import ctypes
import datetime
import hashlib
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

from .documents import Section
from .passages import Box, PageText, Passage, read_pdf
from .transcription import CancelCallback, ProgressCallback
from .translation import (
    DocumentProgress,
    Piece,
    detect_language_scores,
    translate_document,
)

PathLike = Union[str, Path]

NOTICE = (
    "Unreviewed machine translation, made offline to find which material "
    "needs a qualified linguist. It is not a certified translation and should "
    "not be quoted as evidence. Have a court-certified linguist translate any "
    "passage that matters."
)


@dataclass
class Marked:
    """A foreign passage, its number on its page, and its translation."""

    number: int
    passage: Passage
    piece: Piece


@dataclass
class Triage:
    """A PDF after its foreign passages were found and translated."""

    source: Path
    sha256: str
    pages: List[PageText]
    marked: Dict[int, List[Marked]] = field(default_factory=dict)
    names: Dict[str, str] = field(default_factory=dict)
    produced: datetime.datetime = field(
        default_factory=lambda: datetime.datetime.now().astimezone()
    )
    # (what, version) as recorded by the software itself; never guessed.
    software: List[Tuple[str, str]] = field(default_factory=list)

    def name(self, code: Optional[str]) -> str:
        return self.names.get(code or "", code or "")

    def all_marked(self) -> List[Marked]:
        return [m for page in sorted(self.marked) for m in self.marked[page]]

    def _tally(self, missing: bool) -> Dict[str, Tuple[int, List[int]]]:
        out: Dict[str, Tuple[int, List[int]]] = {}
        for page in sorted(self.marked):
            for m in self.marked[page]:
                code = m.piece.missing if missing else m.piece.language
                if code is None:
                    continue
                count, pages = out.get(self.name(code), (0, []))
                if not pages or pages[-1] != page:
                    pages = pages + [page]
                out[self.name(code)] = (count + 1, pages)
        return out

    def translated(self) -> Dict[str, Tuple[int, List[int]]]:
        """Language name -> (passages translated, page indexes)."""
        return self._tally(missing=False)

    def untranslated(self) -> Dict[str, Tuple[int, List[int]]]:
        """Language name -> (passages with no language pack, page indexes)."""
        return self._tally(missing=True)

    def summary(self) -> str:
        """One or two sentences for a status line."""
        if not self.marked:
            return "Found no foreign-language text to translate."
        text = (
            f"Foreign-language text on {len(self.marked)} of {len(self.pages)} page(s)."
        )
        done = self.translated()
        if done:
            parts = [f"{n} passage(s) of {lang}" for lang, (n, _) in done.items()]
            text += f" Translated {', '.join(parts)}."
        missing = self.untranslated()
        if missing:
            text += (
                f" No language pack for {', '.join(missing)}: marked, not translated."
            )
        return text


def page_list(indexes: Sequence[int]) -> str:
    """Page indexes as page numbers, runs shortened: "3, 5-9, 12"."""
    numbers = sorted({i + 1 for i in indexes})
    runs: List[str] = []
    start = prev = None
    for n in numbers + [None]:  # type: ignore[list-item]
        if start is not None and n == prev + 1:  # type: ignore[operator]
            prev = n
            continue
        if start is not None:
            runs.append(str(start) if start == prev else f"{start}-{prev}")
        start = prev = n
    return ", ".join(runs)


def sha256_of(path: PathLike) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def software_used(codes: Sequence[str], ocr: bool) -> List[Tuple[str, str]]:
    """What made a translation, as each component reports its own version."""
    from .buildinfo import build_info

    info = build_info()
    whispers = info.application_version
    if info.from_manifest:
        whispers += f" (build {info.build_id}, commit {info.git_commit})"
    else:
        whispers += " (build not identified: no build manifest)"
    out = [("WHISPERS", whispers)]
    out.append(("Argos Translate", _package_version("argostranslate", info)))
    for code, version in sorted(_pack_versions(codes).items()):
        out.append((f"Language pack {code} to English", version or "not recorded"))
    if ocr:
        out.append(("Tesseract OCR", _tesseract_version()))
    return out


def _package_version(name: str, info) -> str:
    recorded = info.dependencies.get(name)
    if recorded:
        return str(recorded)
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001 - a bundle may not carry the metadata
        return "not recorded"


def _pack_versions(codes: Sequence[str]) -> Dict[str, str]:
    try:
        from argostranslate import package

        installed = package.get_installed_packages()
    except Exception:  # noqa: BLE001 - reported as not recorded
        return {}
    return {
        str(pkg.from_code): str(getattr(pkg, "package_version", "") or "")
        for pkg in installed
        if pkg.from_code in codes and pkg.to_code == "en"
    }


def _tesseract_version() -> str:
    try:
        from .ocr import _configure_pytesseract

        return str(_configure_pytesseract().get_tesseract_version())
    except Exception:  # noqa: BLE001 - reported as not recorded
        return "not recorded"


def translate_pages(
    source: PathLike,
    pages: List[PageText],
    *,
    forced: Optional[str],
    installed: set,
    names: Optional[Dict[str, str]] = None,
    detect: Optional[Callable[[str], List[Tuple[str, float]]]] = detect_language_scores,
    on_progress: Optional[DocumentProgress] = None,
    cancelled: Optional[CancelCallback] = None,
    translator_for: Optional[Callable[[str], object]] = None,
    software: Optional[List[Tuple[str, str]]] = None,
) -> Triage:
    """Translate the foreign passages of ``pages`` and number them per page."""
    passages = [p for page in pages for p in page.passages]
    document = translate_document(
        [Section(paragraphs=[p.text for p in passages])],
        forced=forced,
        installed=installed,
        names=names,
        detect=detect,
        on_progress=on_progress,
        cancelled=cancelled,
        translator_for=translator_for,
    )
    pieces = document.sections[0].paragraphs if document.sections else []
    triage = Triage(
        source=Path(source),
        sha256=sha256_of(source),
        pages=pages,
        names=dict(document.names),
    )
    for passage, piece in zip(passages, pieces):
        if piece.translated or piece.missing:
            marks = triage.marked.setdefault(passage.page, [])
            marks.append(Marked(len(marks) + 1, passage, piece))
    if software is None:
        codes = sorted({p.language for p in pieces if p.language})
        software = software_used(codes, any(page.ocr for page in pages))
    triage.software = software
    return triage


def translate_pdf(
    source: PathLike,
    *,
    forced: Optional[str],
    installed: set,
    names: Optional[Dict[str, str]] = None,
    ocr_lang: Optional[str] = None,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[DocumentProgress] = None,
    cancelled: Optional[CancelCallback] = None,
    translator_for: Optional[Callable[[str], object]] = None,
) -> Triage:
    """Find and translate the foreign passages of the PDF at ``source``."""
    pages = read_pdf(source, lang=ocr_lang, progress=progress, cancelled=cancelled)
    return translate_pages(
        source,
        pages,
        forced=forced,
        installed=installed,
        names=names,
        on_progress=on_progress,
        cancelled=cancelled,
        translator_for=translator_for,
    )


# -- Plain text ----------------------------------------------------------------


def header_lines(triage: Triage) -> List[str]:
    """What the file was, what was found, and what made the translation."""
    lines = [
        f"File: {triage.source.name}",
        f"SHA-256: {triage.sha256}",
        f"Pages: {len(triage.pages)}",
    ]
    if triage.marked:
        lines.append(
            f"Pages with foreign-language text: {page_list(list(triage.marked))}"
        )
    for lang, (count, pages) in triage.translated().items():
        lines.append(
            f"{lang}: {count} passage(s) translated, on page(s) {page_list(pages)}"
        )
    for lang, (count, pages) in triage.untranslated().items():
        lines.append(
            f"{lang}: {count} passage(s) NOT translated (no language pack in this "
            f"copy), on page(s) {page_list(pages)}"
        )
    scanned = [page for page in triage.pages if page.ocr]
    if scanned:
        lines.append(
            f"Read from images with OCR: page(s) {page_list([p.index for p in scanned])}"
        )
    lines.append(f"Produced: {triage.produced.strftime('%Y-%m-%d %H:%M %Z').strip()}")
    lines += [f"{what}: {version}" for what, version in triage.software]
    return lines


def render_text(triage: Triage) -> str:
    """Every marked passage with its translation, page by page."""
    lines = ["Machine translation for language triage", "", NOTICE, ""]
    lines += header_lines(triage)
    for page in sorted(triage.marked):
        lines += ["", f"──────── Page {page + 1} ────────"]
        for m in triage.marked[page]:
            lines.append("")
            if m.piece.missing:
                lines.append(
                    f"[{m.number}] {triage.name(m.piece.missing)}: not translated "
                    "(no language pack)"
                )
                lines.append(f"    Original: {m.passage.text}")
                continue
            lines.append(f"[{m.number}] {triage.name(m.piece.language)}")
            lines.append(f"    Original: {m.passage.text}")
            lines.append(f"    English:  {m.piece.text}")
    return "\n".join(lines).rstrip() + "\n"


# -- PDF -----------------------------------------------------------------------

_HIGHLIGHT = (255, 214, 0, 90)
_MISSING_HIGHLIGHT = (230, 60, 60, 70)
_TAG = (176, 82, 0)
_MISSING_TAG = (176, 0, 32)
_GREY = (95, 95, 95)
_BLACK = (0, 0, 0)
_MARGIN = 28.0
_GUTTER = 16.0
_BODY = 9.0
_LEADING = 11.0


def _latin(text: str) -> str:
    """``text`` in characters Helvetica can show; anything else becomes [?].

    A translation can keep words it did not know (a name) in the original
    script; the original is beside it, and the Word copy shows them.
    """
    out: List[str] = []
    unknown = False
    for ch in text:
        if ch in "\r\n\t":
            ch = " "
        try:
            ch.encode("cp1252")
            ok = ch.isprintable()
        except UnicodeEncodeError:
            base = unicodedata.normalize("NFKD", ch)[:1]
            ok = bool(base) and base.isascii() and base.isprintable()
            ch = base
        if ok:
            out.append(ch)
            unknown = False
        elif not unknown:
            out.append("[?]")
            unknown = True
    return "".join(out)


class _Pdf:
    """Drawing on new pages with pdfium: text, boxes, the original page."""

    def __init__(self) -> None:
        import pypdfium2 as pdfium
        import pypdfium2.raw as raw

        self.pdfium = pdfium
        self.raw = raw
        self.doc = pdfium.PdfDocument.new()
        self.regular = raw.FPDFText_LoadStandardFont(self.doc.raw, b"Helvetica")
        self.bold = raw.FPDFText_LoadStandardFont(self.doc.raw, b"Helvetica-Bold")
        self._widths: Dict[Tuple[int, str], float] = {}
        self._buffers: List[ctypes.Array] = []

    def width(self, text: str, font, size: float) -> float:
        total = 0.0
        key_font = id(font)
        for ch in text:
            key = (key_font, ch)
            if key not in self._widths:
                value = ctypes.c_float()
                self.raw.FPDFFont_GetGlyphWidth(
                    font, ord(ch), ctypes.c_float(1000.0), ctypes.byref(value)
                )
                self._widths[key] = value.value / 1000.0
            total += self._widths[key]
        return total * size

    def wrap(self, text: str, font, size: float, width: float) -> List[str]:
        lines: List[str] = []
        for paragraph in _latin(text).split("\n"):
            line = ""
            for word in paragraph.split():
                candidate = f"{line} {word}" if line else word
                if self.width(candidate, font, size) <= width:
                    line = candidate
                    continue
                if line:
                    lines.append(line)
                while self.width(word, font, size) > width and len(word) > 1:
                    cut = len(word) - 1
                    while cut > 1 and self.width(word[:cut], font, size) > width:
                        cut -= 1
                    lines.append(word[:cut])
                    word = word[cut:]
                line = word
            lines.append(line)
        return lines

    def text(self, page, text: str, x: float, y: float, font, size: float, rgb=_BLACK):
        raw = self.raw
        obj = raw.FPDFPageObj_CreateTextObj(self.doc.raw, font, ctypes.c_float(size))
        data = (_latin(text) + "\0").encode("utf-16-le")
        buffer = ctypes.create_string_buffer(data, len(data))
        self._buffers.append(buffer)
        raw.FPDFText_SetText(obj, ctypes.cast(buffer, raw.FPDF_WIDESTRING))
        raw.FPDFPageObj_SetFillColor(obj, *rgb, 255)
        raw.FPDFPageObj_Transform(obj, 1, 0, 0, 1, x, y)
        raw.FPDFPage_InsertObject(page.raw, obj)

    def rect(self, page, box: Box, rgba, stroke: Optional[Tuple[int, int, int]] = None):
        raw = self.raw
        left, bottom, right, top = box
        obj = raw.FPDFPageObj_CreateNewRect(left, bottom, right - left, top - bottom)
        raw.FPDFPageObj_SetFillColor(obj, *rgba)
        if stroke is not None:
            raw.FPDFPageObj_SetStrokeColor(obj, *stroke, 255)
            raw.FPDFPageObj_SetStrokeWidth(obj, ctypes.c_float(0.5))
        raw.FPDFPath_SetDrawMode(obj, raw.FPDF_FILLMODE_WINDING, stroke is not None)
        raw.FPDFPage_InsertObject(page.raw, obj)

    def tag(self, page, number: int, x: float, y_top: float, rgb) -> float:
        """A numbered tag with its top left at (x, y_top); returns its width."""
        label = str(number)
        width = self.width(label, self.bold, 6.5) + 4
        self.rect(page, (x, y_top - 9, x + width, y_top), (*rgb, 255))
        self.text(page, label, x + 2, y_top - 7, self.bold, 6.5, (255, 255, 255))
        return width

    def save(self, path: PathLike) -> None:
        self.doc.save(str(path))
        self.doc.close()


def write_pdf(triage: Triage, path: PathLike) -> Path:
    """Write the side-by-side translation of ``triage`` to ``path``."""
    import pypdfium2 as pdfium

    out = _Pdf()
    source = pdfium.PdfDocument(str(triage.source))
    try:
        first = triage.pages[0] if triage.pages else None
        width = 2 * (first.width if first else 612.0)
        height = first.height if first else 792.0
        _summary_page(out, triage, width, height)
        for index in sorted(triage.marked):
            _translated_page(out, triage, source, index)
    finally:
        source.close()
    target = Path(path)
    out.save(target)
    return target


def _footer(out: _Pdf, page, triage: Triage) -> None:
    out.text(
        page,
        f"{triage.source.name}  ·  SHA-256 {triage.sha256[:16]}…  ·  Unreviewed "
        "machine translation for triage, not a certified translation",
        _MARGIN,
        14,
        out.regular,
        6.5,
        _GREY,
    )


def _summary_page(out: _Pdf, triage: Triage, width: float, height: float) -> None:
    page = out.doc.new_page(width, height)
    y = height - _MARGIN - 16
    out.text(
        page,
        "Machine translation for language triage",
        _MARGIN,
        y,
        out.bold,
        16,
    )
    y -= 26
    text_width = min(width - 2 * _MARGIN, 640)
    notice = out.wrap(NOTICE, out.bold, 10, text_width - 16)
    box_height = 12 * len(notice) + 12
    out.rect(
        page,
        (_MARGIN, y - box_height + 10, _MARGIN + text_width, y + 10),
        (255, 243, 205, 255),
        stroke=(176, 82, 0),
    )
    for line in notice:
        out.text(page, line, _MARGIN + 8, y - 4, out.bold, 10)
        y -= 12
    y -= 18
    for line in header_lines(triage):
        for wrapped in out.wrap(line, out.regular, 9.5, text_width):
            if y < _MARGIN + 30:
                break
            out.text(page, wrapped, _MARGIN, y, out.regular, 9.5)
            y -= 12.5
    y -= 10
    how = (
        "How to read it: each original page with foreign-language text follows, "
        "unchanged, on the left. Its foreign passages are highlighted and "
        "numbered; the English for each number is on the right, level with it "
        "where there is room. Passages in red have no language pack in this copy "
        "and were not translated. Text in English, and pages with no "
        "foreign-language text, were left as they are and are not repeated here."
    )
    for wrapped in out.wrap(how, out.regular, 9.5, text_width):
        out.text(page, wrapped, _MARGIN, y, out.regular, 9.5, _GREY)
        y -= 12.5
    _footer(out, page, triage)
    page.gen_content()
    page.close()


def _translated_page(out: _Pdf, triage: Triage, source, index: int) -> None:
    info = triage.pages[index]
    marks = triage.marked[index]
    panel_left = info.width + _GUTTER
    panel_width = max(160.0, info.width - _GUTTER - _MARGIN)
    total = len(triage.pages)
    # One copy of the original page, shown again on any continuation page.
    original = source.page_as_xobject(index, out.doc)

    def new_page(continued: bool):
        page = out.doc.new_page(info.width * 2, info.height)
        page.insert_obj(original.as_pageobject())
        out.rect(
            page, (info.width, 0, info.width + 0.6, info.height), (180, 180, 180, 255)
        )
        for m in marks:
            missing = bool(m.piece.missing)
            left, bottom, right, top = m.passage.box
            out.rect(
                page,
                (left - 1, bottom - 1, right + 1, top + 1),
                _MISSING_HIGHLIGHT if missing else _HIGHLIGHT,
            )
            tag_width = out.width(str(m.number), out.bold, 6.5) + 4
            x = left - tag_width - 2 if left - tag_width - 2 > 2 else left
            out.tag(page, m.number, x, top + 1, _MISSING_TAG if missing else _TAG)
        title = f"Page {index + 1} of {total}" + (" (continued)" if continued else "")
        out.text(page, title, panel_left, info.height - _MARGIN, out.bold, 9)
        out.text(
            page,
            "English machine translation of the numbered passages",
            panel_left,
            info.height - _MARGIN - 11,
            out.regular,
            7,
            _GREY,
        )
        _footer(out, page, triage)
        return page

    page = new_page(False)
    cursor = info.height - _MARGIN - 28
    aligned = True
    for m in marks:
        missing = bool(m.piece.missing)
        language = triage.name(m.piece.missing or m.piece.language)
        label = (
            f"{language}: no language pack in this copy, not translated"
            if missing
            else language
        )
        body = (
            "Needs a linguist: this copy cannot translate it."
            if missing
            else m.piece.text
        )
        lines = out.wrap(body, out.regular, _BODY, panel_width - 4)
        needed = 11 + _LEADING * len(lines)
        top = min(cursor, m.passage.box[3]) if aligned else cursor
        room = top - _MARGIN - 10
        fits_a_page = needed <= info.height - 2 * _MARGIN - 40
        # A passage that would not fit goes to the next page, unless it would
        # not fit on one anyway: then it starts here and runs on.
        if room < needed and (fits_a_page or room < 11 + 3 * _LEADING):
            page.gen_content()
            page.close()
            page = new_page(True)
            aligned = False
            top = info.height - _MARGIN - 28
        tag_width = out.tag(
            page, m.number, panel_left, top, _MISSING_TAG if missing else _TAG
        )
        out.text(
            page,
            label,
            panel_left + tag_width + 4,
            top - 7,
            out.bold,
            7.5,
            _MISSING_TAG if missing else _GREY,
        )
        y = top - 11 - _BODY
        for line in lines:
            if y < _MARGIN + 10:
                page.gen_content()
                page.close()
                page = new_page(True)
                aligned = False
                y = info.height - _MARGIN - 28 - _BODY
            out.text(page, line, panel_left, y, out.regular, _BODY)
            y -= _LEADING
        cursor = y - 4
    page.gen_content()
    page.close()
    original.close()
