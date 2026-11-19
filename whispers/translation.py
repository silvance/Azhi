"""Offline text translation backend using Argos Translate (CTranslate2 / CPU).

This mirrors :mod:`whispers.transcription`: ``argostranslate`` is imported lazily so
importing this module does not require the optional dependency, and long jobs are
processed in bounded chunks and are cancellable, so a large batch cannot exhaust
memory (the failure mode of the old tool).

Translation runs fully offline from bundled Argos language packs (see
``whispers.resources.configure_offline_translation``); from source it uses whatever
packs Argos has installed.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple, Union

from .documents import LANGUAGE_NAMES, Section, choose_language
from .transcription import CancelCallback, CancelledError, ProgressCallback

PathLike = Union[str, Path]

# Default target language for the foreign -> English workflow.
DEFAULT_TARGET = "en"

# Cap the text handed to a single translate() call so peak memory stays bounded
# no matter how large the input file is.
MAX_CHUNK_CHARS = 2000


def _argos_translate():
    """Import argostranslate.translate, with a clear error if it's missing."""
    try:
        import argostranslate.translate as translate_mod
    except ImportError as exc:
        raise RuntimeError(
            "Argos Translate is not installed. Install the translation extra "
            "with:  pip install 'whispers[translate]'"
        ) from exc
    return translate_mod


# langdetect returns a few codes that differ from the ISO 639-1 codes Argos uses
# for its packs; normalise them so a detected language maps to an installed pack.
_DETECT_ALIASES = {
    "zh-cn": "zh",
    "zh-tw": "zh",
}


def detect_language(text: str) -> Optional[str]:
    """Best-effort detect the language of ``text`` (ISO 639-1), or ``None``.

    Uses langdetect offline (it ships its own profiles - no model download). The
    detector is seeded for reproducibility. Returns ``None`` when the library is
    missing or detection fails (e.g. too little text), so callers can fall back to
    a manually chosen language.
    """
    if not text or not text.strip():
        return None
    try:
        from langdetect import DetectorFactory, detect
    except ImportError:
        return None
    DetectorFactory.seed = 0
    try:
        code = detect(text)
    except Exception:  # noqa: BLE001 - langdetect raises LangDetectException; treat as unknown
        return None
    return _DETECT_ALIASES.get(code, code)


def detect_language_scores(text: str) -> List[Tuple[str, float]]:
    """Likely languages of ``text`` with their probabilities, most likely first.

    Empty when langdetect is missing or cannot tell (too little text).
    """
    if not text or not text.strip():
        return []
    try:
        from langdetect import DetectorFactory, detect_langs
    except ImportError:
        return []
    DetectorFactory.seed = 0
    try:
        found = detect_langs(text)
    except Exception:  # noqa: BLE001 - langdetect raises LangDetectException
        return []
    return [(_DETECT_ALIASES.get(g.lang, g.lang), float(g.prob)) for g in found]


def available_source_languages(
    target: str = DEFAULT_TARGET,
) -> List[Tuple[str, str]]:
    """Return ``(code, name)`` for installed languages translatable to ``target``.

    Used to populate the GUI's "From" dropdown. Empty when no packs are installed.
    """
    translate_mod = _argos_translate()
    languages = translate_mod.get_installed_languages()
    target_lang = next((lang for lang in languages if lang.code == target), None)
    if target_lang is None:
        return []
    out: List[Tuple[str, str]] = []
    for lang in languages:
        if lang.code == target:
            continue
        if lang.get_translation(target_lang) is not None:
            out.append((lang.code, lang.name))
    return sorted(out, key=lambda pair: pair[1])


def _make_translator(from_code: str, to_code: str):
    """Build a reusable translation object (loads the model once)."""
    translate_mod = _argos_translate()
    translation = translate_mod.get_translation_from_codes(from_code, to_code)
    if translation is None:
        raise RuntimeError(
            f"No installed translation for {from_code!r} -> {to_code!r}. "
            "Bundle or install the matching Argos language pack."
        )
    return translation


def _chunk_text(text: str, max_chars: int = MAX_CHUNK_CHARS) -> List[str]:
    """Split ``text`` into <= ``max_chars`` pieces on line boundaries.

    Newline separators are preserved inside the chunks, so concatenating the
    translated chunks keeps the original line/paragraph structure. Pathologically
    long single lines are hard-split as a last resort.
    """
    chunks: List[str] = []
    buf = ""
    # Keep the "\n" separators (odd indices) so structure is preserved.
    for part in re.split(r"(\n)", text):
        if buf and len(buf) + len(part) > max_chars:
            chunks.append(buf)
            buf = ""
        buf += part
        while len(buf) > max_chars:
            chunks.append(buf[:max_chars])
            buf = buf[max_chars:]
    if buf:
        chunks.append(buf)
    return chunks or [""]


def translate_text(
    text: str,
    *,
    from_code: str,
    to_code: str = DEFAULT_TARGET,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> str:
    """Translate ``text`` from ``from_code`` to ``to_code``, preserving structure.

    The text is translated chunk-by-chunk (bounded memory) and the job can be
    cancelled between chunks.
    """
    translation = _make_translator(from_code, to_code)
    chunks = _chunk_text(text)
    out: List[str] = []
    for index, chunk in enumerate(chunks):
        if cancelled is not None and cancelled():
            raise CancelledError("Translation cancelled.")
        out.append(translation.translate(chunk) if chunk.strip() else chunk)
        if on_progress is not None:
            on_progress((index + 1) / len(chunks))
    return "".join(out)


def translate_files(
    paths: Sequence[PathLike],
    *,
    from_code: str,
    to_code: str = DEFAULT_TARGET,
    progress: Optional[ProgressCallback] = None,
    on_progress: Optional[Callable[[float], None]] = None,
    cancelled: Optional[CancelCallback] = None,
) -> List[Path]:
    """Translate each text file, writing ``<name>.<to_code><suffix>`` beside it.

    One translation model is loaded and reused across all files, and each file is
    streamed through ``_chunk_text`` so peak memory does not grow with the batch
    size. Returns the list of output paths.
    """
    translation = _make_translator(from_code, to_code)
    outputs: List[Path] = []
    total = max(1, len(paths))
    for file_index, raw_path in enumerate(paths):
        if cancelled is not None and cancelled():
            raise CancelledError("Translation cancelled.")
        src = Path(raw_path)
        if progress is not None:
            progress(f"Translating {src.name}...")
        text = src.read_text(encoding="utf-8", errors="replace")
        parts: List[str] = []
        for chunk in _chunk_text(text):
            if cancelled is not None and cancelled():
                raise CancelledError("Translation cancelled.")
            parts.append(translation.translate(chunk) if chunk.strip() else chunk)
        dest = src.with_name(f"{src.stem}.{to_code}{src.suffix}")
        dest.write_text("".join(parts), encoding="utf-8")
        outputs.append(dest)
        if progress is not None:
            progress(f"Wrote {dest.name}")
        if on_progress is not None:
            on_progress((file_index + 1) / total)
    return outputs


# -- Documents: translate what is foreign, leave the English alone -------------

# Labels for the header fields whose value is translated; the rest (names,
# addresses, dates) are kept exactly as written.
TRANSLATED_FIELDS = {"Subject"}


@dataclass
class Piece:
    """One paragraph or field value: what it said, and what is shown for it."""

    original: str
    # The translation; the original itself when it was left as it was.
    text: str
    # The language translated from; None when left as it was.
    language: Optional[str] = None
    # A foreign language this copy has no pack for (left untranslated).
    missing: Optional[str] = None

    @property
    def translated(self) -> bool:
        return self.language is not None


@dataclass
class TranslatedSection:
    fields: List[Tuple[str, Piece]] = field(default_factory=list)
    paragraphs: List[Piece] = field(default_factory=list)

    @property
    def is_email(self) -> bool:
        return bool(self.fields)


@dataclass
class TranslatedDocument:
    """A document after translation, with what it takes to say how it was made."""

    sections: List[TranslatedSection] = field(default_factory=list)
    # Language code -> name, for labelling originals.
    names: Dict[str, str] = field(default_factory=dict)
    # Running page headers and footers taken out before translating.
    removed: List[str] = field(default_factory=list)

    def left_out(self) -> List[str]:
        """The page headers and footers taken out, each once, page numbers aside."""
        seen: Dict[str, str] = {}
        for line in self.removed:
            key = re.sub(r"\d+", "#", line)
            if not re.fullmatch(
                r"\W*(?:page|p\.|pg\.?|صفحة)?\W*#(?:\W*(?:of|/|من)\W*#)?\W*",
                key,
                re.IGNORECASE,
            ):
                seen.setdefault(key, line)
        return list(seen.values())

    def pieces(self) -> List[Piece]:
        out: List[Piece] = []
        for section in self.sections:
            out += [piece for _, piece in section.fields]
            out += section.paragraphs
        return out

    def name(self, code: Optional[str]) -> str:
        return self.names.get(code or "", code or "")

    def summary(self) -> str:
        """What was translated and what was not, in one sentence or two."""
        pieces = [p for p in self.pieces() if p.original.strip()]
        translated = [p for p in pieces if p.translated]
        languages = sorted({self.name(p.language) for p in translated})
        kept = len(pieces) - len(translated)
        missing = sorted({self.name(p.missing) for p in pieces if p.missing})
        if translated:
            text = (
                f"Translated {len(translated)} passage(s) from "
                f"{', '.join(languages)}; left {kept} as they were (English, "
                "names, numbers)."
            )
        else:
            text = "Found no foreign-language text to translate."
        if missing:
            text += (
                f" No language pack in this copy for {', '.join(missing)}: "
                "that text was left untranslated and is marked."
            )
        return text

    def render_text(self, *, originals: bool = False) -> str:
        """The translation as plain text, emails separated and headers kept."""
        lines = [
            "Machine translation into English (offline, Argos Translate). "
            "English text was left as it was.",
        ]
        if self.removed:
            lines.append(
                f"{len(self.removed)} repeated page header, footer and page-number "
                "line(s) were left out; they are listed at the end."
            )
        emails = sum(1 for section in self.sections if section.is_email)
        number = 0
        for section in self.sections:
            lines.append("")
            if section.is_email:
                number += 1
                lines.append(f"──────── Email {number} of {emails} ────────")
                for label, piece in section.fields:
                    lines.append(f"{label}: {piece.text}")
                    if piece.translated:
                        lines.append(
                            f"    {self.name(piece.language)} original: {piece.original}"
                        )
                if section.paragraphs:
                    lines.append("")
            for index, piece in enumerate(section.paragraphs):
                if index:
                    lines.append("")
                lines.append(self._shown(piece))
                if originals and piece.translated:
                    lines.append(
                        f"    {self.name(piece.language)} original: {piece.original}"
                    )
        if self.left_out():
            lines += ["", "──────── Left out as page headers and footers ────────"]
            lines += self.left_out()
        return "\n".join(lines).rstrip() + "\n"

    def _shown(self, piece: Piece) -> str:
        if piece.missing:
            return f"[{self.name(piece.missing)}, not translated: no language pack] {piece.text}"
        return piece.text


# Progress: (paragraphs done, paragraphs to translate, seconds left or None).
DocumentProgress = Callable[[int, int, Optional[float]], None]


def translate_document(
    sections: Sequence[Section],
    *,
    forced: Optional[str],
    installed: Set[str],
    names: Optional[Dict[str, str]] = None,
    detect: Optional[Callable[[str], List[Tuple[str, float]]]] = detect_language_scores,
    on_progress: Optional[DocumentProgress] = None,
    cancelled: Optional[CancelCallback] = None,
    removed: Sequence[str] = (),
    translator_for: Optional[Callable[[str], object]] = None,
) -> TranslatedDocument:
    """Translate the foreign paragraphs of ``sections`` to English.

    Each paragraph is routed on its own (see :func:`whispers.documents.choose_language`):
    English, names and numbers are left exactly as written, so they cost
    nothing and come out unchanged. Of an email's header, only the subject is
    translated. ``forced`` is the language chosen under From, or None for
    automatic detection. Progress reports how many paragraphs are done and,
    once there is a rate to go on, roughly how long is left.
    """
    make = translator_for or (lambda code: _make_translator(code, DEFAULT_TARGET))
    translators: Dict[str, object] = {}
    document = TranslatedDocument(
        names={**LANGUAGE_NAMES, **(names or {})}, removed=list(removed)
    )

    plan: List[Tuple[Piece, Optional[str]]] = []

    def route(text: str) -> Piece:
        code, missing = choose_language(
            text, forced=forced, installed=installed, detect=detect
        )
        piece = Piece(original=text, text=text, missing=missing or None)
        plan.append((piece, code))
        return piece

    for section in sections:
        out = TranslatedSection()
        for label, value in section.fields:
            if label in TRANSLATED_FIELDS:
                out.fields.append((label, route(value)))
            else:
                out.fields.append((label, Piece(original=value, text=value)))
        out.paragraphs = [route(paragraph) for paragraph in section.paragraphs]
        document.sections.append(out)

    work = [(piece, code) for piece, code in plan if code is not None]
    total_chars = sum(len(piece.original) for piece, _ in work) or 1
    done_chars = 0
    started = time.monotonic()
    # A chat report repeats itself ("[图片]", system messages, quoted
    # replies): each distinct passage is translated once.
    done: Dict[Tuple[str, str], str] = {}
    for index, (piece, code) in enumerate(work):
        if cancelled is not None and cancelled():
            raise CancelledError("Translation cancelled.")
        assert code is not None
        key = (code, piece.original)
        if key not in done:
            translator = translators.get(code)
            if translator is None:
                translator = translators[code] = make(code)
            parts = [
                translator.translate(chunk) if chunk.strip() else chunk  # type: ignore[attr-defined]
                for chunk in _chunk_text(piece.original)
            ]
            done[key] = "".join(parts).strip() or piece.original
        piece.text = done[key]
        piece.language = code
        done_chars += len(piece.original)
        if on_progress is not None:
            elapsed = time.monotonic() - started
            left: Optional[float] = None
            if elapsed >= 5 and done_chars:
                left = elapsed * (total_chars - done_chars) / done_chars
            on_progress(index + 1, len(work), left)
    return document
