"""Offline OCR for images and PDFs, feeding the translation workflow.

This lets an operator drop a photo of a document or a scanned PDF into the
Translate tab and get the foreign text out (which can then be reviewed and
translated to English). It mirrors the rest of the package: the heavy
dependencies (``pytesseract``/Tesseract, ``pypdfium2``, ``Pillow``) are imported
lazily so importing this module never requires them, and long jobs are
cancellable and report progress.

Text extraction has two layers:

* **Images** (`.png`, `.jpg`, ...) are read with Pillow and OCR'd with Tesseract.
* **PDFs** are opened with pypdfium2; each page's embedded text is used directly
  when present (digital PDFs), and pages with little/no text (scans) are
  rendered to an image and OCR'd. So both digital and scanned PDFs work.

Tesseract is used because of its strong coverage of the team's priority
languages, including right-to-left scripts (Arabic, Farsi, Hebrew) and
CJK/Cyrillic/Korean. The language data files (``tessdata``) and the Tesseract
binary are bundled into the offline build (see ``whispers.resources``); from
source it uses a system Tesseract on PATH.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Set, Tuple, Union

from .documents import clean, join_wrapped
from .resources import bundled_tessdata_dir, find_tesseract
from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

# Image containers we offer for OCR (anything Pillow + Tesseract can read).
IMAGE_EXTENSIONS = (
    ".png",
    ".jpg",
    ".jpeg",
    ".tif",
    ".tiff",
    ".bmp",
    ".gif",
    ".webp",
)
PDF_EXTENSIONS = (".pdf",)
OCR_EXTENSIONS = IMAGE_EXTENSIONS + PDF_EXTENSIONS

# Argos / ISO 639-1 source codes -> Tesseract language codes (639-2/3). Covers the
# bundled intel-leaning set plus a few common extras; unknown codes pass through
# (so a raw Tesseract code like "chi_tra" also works if given directly).
_TESSERACT_LANG = {
    "ar": "ara",
    "ru": "rus",
    "zh": "chi_sim",
    "fa": "fas",
    "uk": "ukr",
    "he": "heb",
    "ko": "kor",
    "en": "eng",
    "es": "spa",
    "fr": "fra",
    "de": "deu",
    "it": "ita",
    "pt": "por",
    "nl": "nld",
    "pl": "pol",
    "tr": "tur",
    "ja": "jpn",
    "hi": "hin",
    "ur": "urd",
    "ps": "pus",
}

# Tesseract's script detection (OSD) names -> the language data that reads
# that script. Used when the language is left to be detected: a scanned page
# has no text to detect a language from until it has been read.
_OSD_SCRIPTS = {
    "Han": "chi_sim",
    "HanS": "chi_sim",
    "HanT": "chi_tra",
    "Japanese": "jpn",
    "Katakana": "jpn",
    "Hiragana": "jpn",
    "Hangul": "kor",
    "Korean": "kor",
    "Arabic": "ara",
    "Hebrew": "heb",
    "Cyrillic": "rus",
    "Devanagari": "hin",
    "Greek": "ell",
    "Latin": "eng",
}
# Other language data that reads the same script, tried in turn when the
# usual one is not in this copy.
_SAME_SCRIPT = {
    "ara": ("fas", "urd", "pus"),
    "rus": ("ukr",),
    "chi_sim": ("chi_tra",),
}

# Rendering DPI for scanned PDF pages handed to Tesseract. 300 is the usual sweet
# spot for OCR accuracy without ballooning memory.
_PDF_RENDER_DPI = 300

# A page yielding fewer than this many characters of embedded text is treated as a
# scan and rendered + OCR'd instead.
_MIN_EMBEDDED_CHARS = 16


def tesseract_lang(code: str) -> str:
    """Map an ISO 639-1 / Argos code to a Tesseract language code."""
    return _TESSERACT_LANG.get(code, code)


def is_image(path: PathLike) -> bool:
    return Path(path).suffix.lower() in IMAGE_EXTENSIONS


def is_pdf(path: PathLike) -> bool:
    return Path(path).suffix.lower() in PDF_EXTENSIONS


def is_ocr_file(path: PathLike) -> bool:
    """True if ``path`` is an image or PDF we can extract text from."""
    return Path(path).suffix.lower() in OCR_EXTENSIONS


def ocr_available() -> bool:
    """True if OCR can run: a Tesseract binary plus the Python deps are present.

    Lets the GUI hide/disable OCR controls (and a lean build present cleanly) when
    the engine isn't bundled, without importing the heavy libraries just to check.
    """
    if find_tesseract() is None:
        return False
    try:
        import importlib.util

        return (
            importlib.util.find_spec("pytesseract") is not None
            and importlib.util.find_spec("PIL") is not None
        )
    except ModuleNotFoundError:
        return False


def _configure_pytesseract():
    """Return the pytesseract module wired up to the bundled/Tesseract binary."""
    try:
        import pytesseract
    except ImportError as exc:
        raise RuntimeError(
            "pytesseract is not installed. Install the OCR extra with:  "
            "pip install 'whispers[ocr]'"
        ) from exc
    binary = find_tesseract()
    if binary is None:
        raise RuntimeError(
            "Tesseract OCR was not found (neither bundled in whispers_assets nor on "
            "PATH). Use a build that bundles it, or install Tesseract."
        )
    pytesseract.pytesseract.tesseract_cmd = str(binary)
    return pytesseract


def _ensure_tessdata_prefix() -> None:
    """Point Tesseract at the bundled tessdata via ``TESSDATA_PREFIX``.

    Passing ``--tessdata-dir`` through pytesseract's ``config`` string is
    unreliable: the value is not shell-parsed, so quotes/spaces end up embedded in
    the path (Tesseract then fails with ``Error opening data file "...tessdata"/
    eng.traineddata``). The env var is the robust mechanism and is exactly what
    Tesseract 4/5 expect - set to the ``tessdata`` directory itself.
    """
    data = bundled_tessdata_dir()
    if data is not None:
        os.environ["TESSDATA_PREFIX"] = str(data)


_installed_tessdata: Optional[Set[str]] = None


def installed_ocr_languages() -> Set[str]:
    """The Tesseract language data this copy has (cached; empty if unknown)."""
    global _installed_tessdata
    if _installed_tessdata is None:
        found: Set[str] = set()
        data = bundled_tessdata_dir()
        if data is not None:
            found = {p.stem for p in Path(data).glob("*.traineddata")}
        else:
            try:
                found = set(_configure_pytesseract().get_languages(config=""))
            except Exception:  # noqa: BLE001 - unknown: let Tesseract decide
                found = set()
        _installed_tessdata = found
    return _installed_tessdata


def _with_english(lang: str) -> str:
    """``lang`` plus English, for pages that mix the two (and many do)."""
    installed = installed_ocr_languages()
    if "eng" in lang.split("+") or (installed and "eng" not in installed):
        return lang
    return f"{lang}+eng"


def _detected_lang(image) -> str:
    """The language data for the script Tesseract sees on ``image``.

    English when the script cannot be told (too little text) or this copy has
    nothing that reads it.
    """
    pytesseract = _configure_pytesseract()
    _ensure_tessdata_prefix()
    try:
        osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
    except Exception:  # noqa: BLE001 - too little text to judge, or no osd data
        return "eng"
    wanted = _OSD_SCRIPTS.get(str(osd.get("script", "")), "eng")
    installed = installed_ocr_languages()
    for lang in (wanted, *_SAME_SCRIPT.get(wanted, ())):
        if not installed or lang in installed:
            return lang
    return "eng"


# A line Tesseract read with less average confidence than this (0-100) was
# probably in a script the language data in use cannot read: English OCR
# turns two lines of Chinese into "SWEET PRS LFARER SUES" at about 14.
_UNREAD_CONFIDENCE = 50
# Another language's data has to read those lines at least this well to be
# taken for their language.
_READ_CONFIDENCE = 60


def _lines(data: dict) -> List[Tuple[float, Tuple[int, int, int, int]]]:
    """Each line Tesseract found: its mean word confidence and its box."""
    lines: dict = {}
    for i, word in enumerate(data["text"]):
        if not str(word).strip():
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        box = (
            data["left"][i],
            data["top"][i],
            data["left"][i] + data["width"][i],
            data["top"][i] + data["height"][i],
        )
        lines.setdefault(key, []).append((float(data["conf"][i]), box))
    out = []
    for words in lines.values():
        confidence = sum(c for c, _ in words) / len(words)
        boxes = [b for _, b in words]
        out.append(
            (
                confidence,
                (
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ),
            )
        )
    return out


def _mean_confidence(pytesseract, image, lang: str) -> float:
    data = pytesseract.image_to_data(
        image, lang=lang, output_type=pytesseract.Output.DICT
    )
    confidences = [c for c, _ in _lines(data)]
    return sum(confidences) / len(confidences) if confidences else 0.0


# Language data for scripts other than Latin: the only ones worth trying on
# lines English could not read (a poor scan of English is not read better
# as French).
_OTHER_SCRIPTS = (set(_OSD_SCRIPTS.values()) - {"eng"}) | {
    lang for langs in _SAME_SCRIPT.values() for lang in langs
}


def _second_script(pytesseract, image, lang: str, found: List[str]) -> Optional[str]:
    """The language of lines ``lang`` could not read on a page that mixes scripts.

    Script detection names the script most of a page is in, so English
    commentary around two lines of Chinese is read as all English and the
    Chinese comes out as nonsense. Those lines are the ones read with low
    confidence: the languages in ``found`` (earlier pages of the document)
    are tried on just them first, then this copy's other non-Latin
    languages, and the one that reads them well is returned.
    """
    candidates = sorted(
        (installed_ocr_languages() & _OTHER_SCRIPTS) - set(lang.split("+"))
    )
    if not candidates:
        return None
    data = pytesseract.image_to_data(
        image, lang=lang, output_type=pytesseract.Output.DICT
    )
    unread = [
        box for confidence, box in _lines(data) if confidence < _UNREAD_CONFIDENCE
    ]
    if not unread:
        return None
    margin = 10
    crop = image.crop(
        (
            max(0, min(b[0] for b in unread) - margin),
            max(0, min(b[1] for b in unread) - margin),
            max(b[2] for b in unread) + margin,
            max(b[3] for b in unread) + margin,
        )
    )
    for candidate in found:
        if candidate in candidates:
            if _mean_confidence(pytesseract, crop, candidate) > _READ_CONFIDENCE:
                return candidate
    best, best_confidence = None, float(_READ_CONFIDENCE)
    for candidate in candidates:
        if candidate in found:
            continue
        confidence = _mean_confidence(pytesseract, crop, candidate)
        if confidence > best_confidence:
            best, best_confidence = candidate, confidence
    if best is not None:
        found.append(best)
    return best


def _reading_lang(pytesseract, image, lang: Optional[str], found: List[str]) -> str:
    """The Tesseract language string to read ``image`` with.

    ``lang`` None means the language was not chosen: the script is detected
    first, so a scan in Chinese or Arabic is not read as English, and lines
    in a second script are looked into (see :func:`_second_script`, which
    keeps what it finds in ``found`` for the next page).
    """
    if lang is not None:
        return _with_english(lang)
    chosen = _with_english(_detected_lang(image))
    other = _second_script(pytesseract, image, chosen, found)
    return f"{other}+{chosen}" if other is not None else chosen


def _ocr_image(image, lang: Optional[str], found: Optional[List[str]] = None) -> str:
    """OCR a Pillow image with Tesseract and return stripped text."""
    pytesseract = _configure_pytesseract()
    _ensure_tessdata_prefix()
    chosen = _reading_lang(pytesseract, image, lang, [] if found is None else found)
    return pytesseract.image_to_string(image, lang=chosen).strip()


# A paragraph Tesseract found: its text, and its box in pixels (left, top,
# right, bottom).
OcrParagraph = Tuple[str, Tuple[int, int, int, int]]


def ocr_paragraphs(
    image, lang: Optional[str], found: Optional[List[str]] = None
) -> Tuple[str, List[OcrParagraph]]:
    """The paragraphs on a scanned page and where they are, for marking them.

    Returns the Tesseract language string used and the paragraphs in the
    order Tesseract read them. Words of Chinese, Japanese and Korean are
    joined without spaces.
    """

    pytesseract = _configure_pytesseract()
    _ensure_tessdata_prefix()
    chosen = _reading_lang(pytesseract, image, lang, [] if found is None else found)
    data = pytesseract.image_to_data(
        image, lang=chosen, output_type=pytesseract.Output.DICT
    )
    paragraphs: dict = {}
    for i, word in enumerate(data["text"]):
        word = clean(str(word)).strip()
        if not word:
            continue
        key = (data["block_num"][i], data["par_num"][i])
        line = data["line_num"][i]
        box = (
            data["left"][i],
            data["top"][i],
            data["left"][i] + data["width"][i],
            data["top"][i] + data["height"][i],
        )
        paragraphs.setdefault(key, []).append((line, word, box))
    out: List[OcrParagraph] = []
    for words in paragraphs.values():
        text = ""
        for _, word, _ in words:
            text = join_wrapped(text, word) if text else word
        boxes = [box for _, _, box in words]
        out.append(
            (
                text,
                (
                    min(b[0] for b in boxes),
                    min(b[1] for b in boxes),
                    max(b[2] for b in boxes),
                    max(b[3] for b in boxes),
                ),
            )
        )
    return chosen, out


def _extract_image_file(path: Path, lang: Optional[str]) -> str:
    try:
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError(
            "Pillow is not installed. Install the OCR extra with:  "
            "pip install 'whispers[ocr]'"
        ) from exc
    with Image.open(path) as image:
        return _ocr_image(image, lang)


def _extract_pdf_pages(
    path: Path,
    lang: Optional[str],
    progress: Optional[ProgressCallback],
    cancelled: Optional[CancelCallback],
) -> List[str]:
    try:
        import pypdfium2 as pdfium
    except ImportError as exc:
        raise RuntimeError(
            "pypdfium2 is not installed. Install the OCR extra with:  "
            "pip install 'whispers[ocr]'"
        ) from exc

    pdf = pdfium.PdfDocument(str(path))
    found: List[str] = []
    try:
        pages: list[str] = []
        total = len(pdf)
        for index in range(total):
            if cancelled is not None and cancelled():
                raise CancelledError("OCR cancelled.")
            page = pdf[index]
            # Use embedded text when the page has it (digital PDF); otherwise the
            # page is a scan, so render it and OCR the image.
            textpage = page.get_textpage()
            text = textpage.get_text_range().strip()
            if len(text) < _MIN_EMBEDDED_CHARS:
                if progress is not None:
                    progress(f"OCR page {index + 1}/{total} (scanned)…")
                bitmap = page.render(scale=_PDF_RENDER_DPI / 72)
                image = bitmap.to_pil()
                try:
                    text = _ocr_image(image, lang, found)
                finally:
                    image.close()
            elif progress is not None:
                progress(f"Read page {index + 1}/{total} (text)…")
            pages.append(text)
        return pages
    finally:
        pdf.close()


def extract_pages(
    path: PathLike,
    *,
    lang: Optional[str] = "eng",
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[str]:
    """Extract the text of each page of a PDF (an image is one page).

    Parameters
    ----------
    path
        An image or PDF file (see ``OCR_EXTENSIONS``).
    lang
        A Tesseract language code (e.g. ``"ara"``) for pages that have to be
        read as images; English is read alongside it. None detects the script
        of each scanned page first. Use :func:`tesseract_lang` to map an ISO
        639-1 / Argos code (``"ar"``) to a Tesseract one.
    progress, cancelled
        Optional status callback and cooperative-cancel predicate.

    Characters that could not be saved in a Word document are taken out.
    """
    src = Path(path)
    if not src.exists():
        raise FileNotFoundError(f"Input file does not exist: {src}")
    if is_pdf(src):
        pages = _extract_pdf_pages(src, lang, progress, cancelled)
    elif is_image(src):
        if progress is not None:
            progress(f"OCR {src.name}…")
        pages = [_extract_image_file(src, lang)]
    else:
        raise ValueError(f"Not an OCR-able image or PDF: {src.name}")
    return [clean(page).strip() for page in pages]


def extract_text(
    path: PathLike,
    *,
    lang: Optional[str] = "eng",
    progress: Optional[ProgressCallback] = None,
    cancelled: Optional[CancelCallback] = None,
) -> str:
    """Extract text from an image or PDF, pages separated by a blank line.

    See :func:`extract_pages`.
    """
    pages = extract_pages(path, lang=lang, progress=progress, cancelled=cancelled)
    return "\n\n".join(pages).strip()
