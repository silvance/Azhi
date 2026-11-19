"""Export transcripts and translations to Microsoft Word (``.docx``).

A small wrapper over ``python-docx`` (imported lazily, like the rest of the
package's optional backends) so an operator can hand off a clean Word document
instead of pasting plain text. Diarized transcripts keep their speaker labels in
bold so turns are easy to scan.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

from .documents import RIGHT_TO_LEFT, clean, dominant_script
from .speaker_names import LEAD_LEGEND, any_leads
from .transcription import Segment, TranscriptionResult, clock_time, note_lines

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .side_by_side import Triage
    from .translation import Piece, TranslatedDocument

PathLike = Union[str, Path]


def _document():
    """Return a new python-docx Document, with a clear error if it's missing."""
    try:
        from docx import Document
    except ImportError as exc:
        raise RuntimeError(
            "python-docx is not installed. Install the GUI extra with:  "
            "pip install 'whispers[gui]'"
        ) from exc
    return Document()


def text_to_docx(text: str, path: PathLike, *, title: Optional[str] = None) -> Path:
    """Write ``text`` to a ``.docx`` at ``path``, one paragraph per blank-line block.

    Used for translation output. Blank lines separate paragraphs; single newlines
    inside a block become line breaks within a paragraph.
    """
    document = _document()
    if title:
        document.add_heading(clean(title), level=1)
    # Split on blank lines into paragraphs; keep intra-paragraph line breaks.
    # Cleaned first: a page break or other control character from a PDF made
    # python-docx refuse the whole document.
    blocks = clean(text).split("\n\n")
    for block in blocks:
        paragraph = document.add_paragraph()
        lines = block.split("\n")
        for line_index, line in enumerate(lines):
            if line_index:
                paragraph.add_run().add_break()
            paragraph.add_run(line)
        _direction(paragraph, block)
    out = Path(path)
    document.save(str(out))
    return out


# Where w:bidi goes among a paragraph's properties: Word rejects a file whose
# properties are out of schema order.
_AFTER_BIDI = (
    "w:adjustRightInd",
    "w:snapToGrid",
    "w:spacing",
    "w:ind",
    "w:contextualSpacing",
    "w:mirrorIndents",
    "w:suppressOverlap",
    "w:jc",
    "w:textDirection",
    "w:textAlignment",
    "w:textboxTightWrap",
    "w:outlineLvl",
    "w:divId",
    "w:cnfStyle",
    "w:rPr",
    "w:sectPr",
    "w:pPrChange",
)


def _direction(paragraph: Any, text: str) -> None:
    """Lay ``paragraph`` out right to left when ``text`` is mostly Arabic or Hebrew."""
    if dominant_script(text) not in RIGHT_TO_LEFT:
        return
    from docx.oxml import OxmlElement

    properties = paragraph._p.get_or_add_pPr()
    if properties.find(_qn("w:bidi")) is None:
        properties.insert_element_before(OxmlElement("w:bidi"), *_AFTER_BIDI)
    for run in paragraph.runs:
        run._r.get_or_add_rPr().get_or_add_rtl()


def _qn(tag: str) -> str:
    from docx.oxml.ns import qn

    return qn(tag)


def translation_to_docx(
    translation: "TranslatedDocument",
    path: PathLike,
    *,
    title: Optional[str] = None,
    originals: bool = False,
) -> Path:
    """Write a translated document to ``.docx``, laid out to be read.

    Each email gets a heading (its subject, translated) and its header fields
    as they were written; English is as it was; with ``originals``, each
    translated paragraph is followed by its original, in grey italics. Arabic
    and Hebrew are laid out right to left.
    """
    from docx.shared import RGBColor

    grey = RGBColor(0x59, 0x59, 0x59)
    document = _document()
    document.add_heading(clean(title or "Translation"), level=1)
    note = document.add_paragraph()
    note.add_run(
        "Machine translation into English (offline, Argos Translate). English "
        "text was left as it was."
        + (
            f" {len(translation.removed)} repeated page header, footer and "
            "page-number line(s) were left out; they are listed at the end."
            if translation.removed
            else ""
        )
    ).italic = True

    def original(piece: "Piece") -> None:
        paragraph = document.add_paragraph()
        label = paragraph.add_run(f"{translation.name(piece.language)} original: ")
        label.italic = True
        label.font.color.rgb = grey
        run = paragraph.add_run(clean(piece.original))
        run.italic = True
        run.font.color.rgb = grey
        _direction(paragraph, piece.original)

    emails = sum(1 for section in translation.sections if section.is_email)
    number = 0
    for section in translation.sections:
        if section.is_email:
            number += 1
            subject = next(
                (piece.text for label, piece in section.fields if label == "Subject"),
                "",
            )
            heading = f"Email {number} of {emails}" + (
                f": {clean(subject)}" if subject else ""
            )
            document.add_heading(heading, level=2)
            for label, piece in section.fields:
                paragraph = document.add_paragraph()
                paragraph.add_run(f"{label}: ").bold = True
                paragraph.add_run(clean(piece.text))
                if not piece.translated:
                    _direction(paragraph, piece.text)
                if piece.translated:
                    original(piece)
        for piece in section.paragraphs:
            paragraph = document.add_paragraph()
            if piece.missing:
                flag = paragraph.add_run(
                    f"[{translation.name(piece.missing)}, not translated: no "
                    "language pack] "
                )
                flag.bold = True
            paragraph.add_run(clean(piece.text))
            if not piece.translated:
                _direction(paragraph, piece.text)
            if originals and piece.translated:
                original(piece)
    if translation.left_out():
        document.add_heading("Left out as page headers and footers", level=2)
        for line in translation.left_out():
            paragraph = document.add_paragraph(clean(line))
            _direction(paragraph, line)
    out = Path(path)
    document.save(str(out))
    return out


def side_by_side_docx(triage: "Triage", path: PathLike) -> Path:
    """Write the foreign passages of a PDF beside their translations, as a table.

    One row per passage: page, number (as marked in the side-by-side PDF),
    language, the original and the English. Word lays out Arabic and Chinese
    itself, so this copy shows every character of both, and a linguist can
    edit it.
    """
    from docx.shared import Pt

    from .side_by_side import NOTICE, header_lines

    document = _document()
    document.add_heading("Machine translation for language triage", level=1)
    document.add_paragraph().add_run(NOTICE).bold = True
    for line in header_lines(triage):
        document.add_paragraph(clean(line))
    marks = triage.all_marked()
    if not marks:
        document.add_paragraph("No foreign-language text was found.")
    else:
        table = document.add_table(rows=1, cols=5)
        table.style = "Table Grid"
        for cell, text in zip(
            table.rows[0].cells,
            ("Page", "#", "Language", "Original", "English (machine translation)"),
        ):
            cell.text = ""
            cell.paragraphs[0].add_run(text).bold = True
        for m in marks:
            row = table.add_row().cells
            row[0].text = str(m.passage.page + 1)
            row[1].text = str(m.number)
            code = m.piece.missing or m.piece.language
            row[2].text = triage.name(code) + (
                " (no language pack)" if m.piece.missing else ""
            )
            row[3].text = clean(m.passage.text)
            _direction(row[3].paragraphs[0], m.passage.text)
            row[4].text = (
                "Not translated: no language pack in this copy."
                if m.piece.missing
                else clean(m.piece.text)
            )
        for row in table.rows:
            for cell in row.cells:
                for paragraph in cell.paragraphs:
                    for run in paragraph.runs:
                        run.font.size = Pt(9)
    out = Path(path)
    document.save(str(out))
    return out


def transcript_to_docx(
    result: TranscriptionResult,
    path: PathLike,
    speaker_names: Optional[Dict[str, str]] = None,
    *,
    blank_lines: bool = True,
    title: Optional[str] = None,
    times: bool = False,
    notes: bool = True,
) -> Path:
    """Write a transcript to a ``.docx``.

    When diarized, each paragraph starts with the speaker label in bold
    (optionally remapped via ``speaker_names``); otherwise one paragraph per
    segment. ``times`` starts each paragraph with when it starts in the
    recording, ``[00:12:34]``. ``blank_lines`` is accepted for signature parity
    with the text exporters - Word paragraphs already render with spacing
    between them.
    """
    document = _document()
    if title:
        document.add_heading(title, level=1)

    def at(segment: Segment) -> str:
        return f"[{clock_time(segment.start)}] " if times else ""

    by_line = note_lines(result.segments, result.notes) if notes else {}

    def add_notes(index: int) -> None:
        # Indented and in italics, after the line: a note is the analyst's,
        # and must never read as something that was said.
        from docx.shared import Inches

        for n in by_line.get(index, []):
            paragraph = document.add_paragraph()
            paragraph.paragraph_format.left_indent = Inches(0.5)
            paragraph.add_run(f"Analyst note: {result.notes[n].text}").italic = True

    if not result.has_speakers:
        segments = result.segments
        if segments:
            for index, segment in enumerate(segments):
                document.add_paragraph(at(segment) + segment.text)
                add_notes(index)
        else:
            for line in result.text.split("\n"):
                document.add_paragraph(line)
    else:
        labels = [
            result._speaker_label(segment.speaker, speaker_names)
            for segment in result.segments
        ]
        if any_leads(labels):
            note = document.add_paragraph()
            note.add_run(LEAD_LEGEND).italic = True
        for index, segment in enumerate(result.segments):
            label = result._speaker_label(segment.speaker, speaker_names)
            paragraph = document.add_paragraph()
            if times:
                paragraph.add_run(at(segment))
            run = paragraph.add_run(f"[{label}] ")
            run.bold = True
            paragraph.add_run(segment.text)
            add_notes(index)

    out = Path(path)
    document.save(str(out))
    return out
