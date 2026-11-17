"""Turn text pulled out of a document into blocks worth translating.

Text extracted from a PDF or a scan is not prose. Each printed line arrives as
a line, so a sentence that wraps arrives in pieces; every page carries its
running header and footer (a classification banner, "Page 3 of 20", the
mailbox an export came from); an exported email starts with a block of From,
Sent, To and Subject lines; and a document can hold two languages, English
commentary around a foreign original. Handed to a translation model as it
comes, that is slow (each line, English included, goes through the model on
its own) and the result is hard to read.

This module puts the structure back, without translating anything:

* :func:`clean` - characters that cannot go into a Word document, such as page
  breaks and stray control codes, out;
* :func:`remove_page_furniture` - lines repeated at the top or bottom of most
  pages, and bare page numbers, out (never an email header line);
* :func:`parse` - lines joined back into paragraphs, and email header blocks
  recognised as such;
* :func:`choose_language` - for one paragraph, the language to translate it
  from, or None to leave it as it is (English, or nothing to translate).

All of it is plain text processing: no model, no network.
"""

from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

# -- Cleaning ------------------------------------------------------------------

# Characters an XML document (so a Word file) cannot hold, beyond the line
# breaks handled first: the C0 controls other than tab and newline, DEL, and
# the two non-characters.
_ILLEGAL = re.compile("[\x00-\x08\x0e-\x1f\x7f￾￿]")
# A word broken at the end of a line: PDF text marks the break with a soft
# hyphen or an STX control; either way the halves belong together.
_LINE_END_HYPHEN = re.compile("[­\x02][ \t]*\n")


def clean(text: str) -> str:
    """``text`` with line endings made ``\\n`` and unrepresentable characters out.

    A page break becomes a paragraph break, a vertical tab a line break, and a
    broken character (half of a surrogate pair) the replacement character, so
    the text can always be saved as a Word document.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _LINE_END_HYPHEN.sub("", text)
    text = text.replace("\x0c", "\n\n").replace("\x0b", "\n")
    text = text.encode("utf-16", "surrogatepass").decode("utf-16", "replace")
    return _ILLEGAL.sub("", text)


# -- Page furniture ------------------------------------------------------------

# How many lines at each end of a page are looked at for repeats: running
# headers and footers are a line or two, and looking deeper starts to catch
# text that only happens to repeat.
EDGE_LINES = 2
# A line repeated at a page edge on at least this share of the pages (and on
# at least _MIN_REPEATS of them) is a running header or footer.
_REPEAT_SHARE = 0.5
_MIN_REPEATS = 3
_PAGE_NUMBER = re.compile(
    r"^\s*[-–—]?\s*(?:page|p\.|pg\.?|صفحة)?\s*\d{1,4}\s*"
    r"(?:(?:of|/|من)\s*\d{1,4})?\s*[-–—]?\s*$",
    re.IGNORECASE,
)


# Numbers are only set aside in lines this short: a running footer such as
# "Case 2025-0117 | Page 3" is, and two lines of text that differ only in a
# number are not the same line.
_FOOTER_WORDS = 8


def _signature(line: str) -> str:
    """A line with case and spacing flattened, and in a short line its numbers
    too, so "Page 3" and "Page 4" count as the same running footer."""
    flat = " ".join(line.lower().split())
    if len(flat.split()) <= _FOOTER_WORDS:
        flat = re.sub(r"\d+", "#", flat)
    return flat


def _edge_indexes(lines: Sequence[str]) -> List[int]:
    filled = [i for i, line in enumerate(lines) if line.strip()]
    return sorted(set(filled[:EDGE_LINES] + filled[-EDGE_LINES:]))


def remove_page_furniture(pages: Sequence[str]) -> Tuple[List[str], List[str]]:
    """``pages`` without their running headers and footers, and the lines taken out.

    A line counts as furniture when it is one of the first or last two lines
    of a page and either is a bare page number or recurs (numbers aside) at the
    edges of at least half the pages, and of three at the least. Email header
    lines are never removed, however often they repeat: a thread from one
    sender repeats its From line. The lines taken out are returned so that
    what was left out can be shown.
    """
    split = [clean(page).split("\n") for page in pages]
    counts: Counter = Counter()
    for lines in split:
        counts.update({_signature(lines[i]) for i in _edge_indexes(lines)})
    needed = max(_MIN_REPEATS, math.ceil(_REPEAT_SHARE * len(split)))
    repeated = {sig for sig, n in counts.items() if sig and n >= needed}
    removed: List[str] = []
    out: List[str] = []
    for lines in split:
        drop = set()
        for i in _edge_indexes(lines):
            line = lines[i]
            if field_label(line) is not None:
                continue
            if _PAGE_NUMBER.match(line) or _signature(line) in repeated:
                drop.add(i)
        removed += [lines[i].strip() for i in sorted(drop)]
        out.append("\n".join(line for i, line in enumerate(lines) if i not in drop))
    return out, removed


# -- Email header fields -------------------------------------------------------

# Header labels as Outlook and other mail clients print them, in English and
# Arabic, mapped to one English name.
_LABELS: Dict[str, str] = {
    "from": "From",
    "sent": "Sent",
    "date": "Date",
    "to": "To",
    "cc": "Cc",
    "bcc": "Bcc",
    "subject": "Subject",
    "attachments": "Attachments",
    "importance": "Importance",
    "reply-to": "Reply-To",
    "من": "From",
    "المرسل": "From",
    "أرسل": "Sent",
    "أُرسل": "Sent",
    "تم الإرسال": "Sent",
    "التاريخ": "Date",
    "إلى": "To",
    "الى": "To",
    "نسخة": "Cc",
    "نسخة إلى": "Cc",
    "الموضوع": "Subject",
    "المرفقات": "Attachments",
}
_FIELD = re.compile(
    r"^\s*("
    + "|".join(re.escape(label) for label in sorted(_LABELS, key=len, reverse=True))
    + r")\s*[:：]\s?(.*)$",
    re.IGNORECASE,
)
# A line that is mostly right to left (an Arabic subject) comes out of a PDF
# with its English label at the end: "<subject> :Subject".
_FIELD_AT_END = re.compile(
    r"^(.*?)\s*[:：]\s*("
    + "|".join(
        re.escape(label)
        for label in sorted(_LABELS, key=len, reverse=True)
        if label.isascii()
    )
    + r")\s*$",
    re.IGNORECASE,
)
# Fields whose value can run onto the next lines (long address lists).
_LISTS = {"To", "Cc", "Bcc", "Attachments"}
# A header block has at least two of these.
_CORE = {"From", "Sent", "Date", "To", "Subject"}


def _field_match(line: str) -> Optional[Tuple[str, str]]:
    match = _FIELD.match(line)
    if match:
        return _LABELS[match.group(1).lower()], match.group(2).strip()
    match = _FIELD_AT_END.match(line)
    if match and dominant_script(match.group(1)) in RIGHT_TO_LEFT:
        return _LABELS[match.group(2).lower()], match.group(1).strip()
    return None


def field_label(line: str) -> Optional[str]:
    """The English name of the header field ``line`` holds, or None."""
    found = _field_match(line)
    return found[0] if found else None


def _field(line: str) -> Tuple[str, str]:
    found = _field_match(line)
    assert found is not None
    return found


# -- Scripts -------------------------------------------------------------------

_RANGES: Tuple[Tuple[str, int, int], ...] = (
    ("han", 0x3400, 0x4DBF),
    ("han", 0x4E00, 0x9FFF),
    ("han", 0xF900, 0xFAFF),
    ("han", 0x20000, 0x2FA1F),
    ("kana", 0x3040, 0x30FF),
    ("kana", 0x31F0, 0x31FF),
    ("hangul", 0x1100, 0x11FF),
    ("hangul", 0x3130, 0x318F),
    ("hangul", 0xAC00, 0xD7AF),
    ("arabic", 0x0600, 0x06FF),
    ("arabic", 0x0750, 0x077F),
    ("arabic", 0x08A0, 0x08FF),
    ("arabic", 0xFB50, 0xFDFF),
    ("arabic", 0xFE70, 0xFEFF),
    ("hebrew", 0x0590, 0x05FF),
    ("cyrillic", 0x0400, 0x052F),
    ("greek", 0x0370, 0x03FF),
    ("devanagari", 0x0900, 0x097F),
    ("thai", 0x0E00, 0x0E7F),
)
# Scripts written without spaces between words: one character carries about
# as much as a short word, so it is weighed as several letters.
_DENSE = {"han", "kana", "hangul", "thai"}
_DENSE_WEIGHT = 3
# The languages each script is used for, most likely first, as Argos codes.
SCRIPT_LANGUAGES: Dict[str, Tuple[str, ...]] = {
    "han": ("zh", "zt"),
    "kana": ("ja",),
    "hangul": ("ko",),
    "arabic": ("ar", "fa", "ur", "ps"),
    "hebrew": ("he",),
    "cyrillic": ("ru", "uk", "bg", "sr"),
    "greek": ("el",),
    "devanagari": ("hi",),
    "thai": ("th",),
}
# Scripts read right to left, for laying out a Word document.
RIGHT_TO_LEFT = {"arabic", "hebrew"}
# Names for languages that may be found without a pack installed to name them.
LANGUAGE_NAMES: Dict[str, str] = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "de": "German",
    "el": "Greek",
    "es": "Spanish",
    "fa": "Persian",
    "fr": "French",
    "he": "Hebrew",
    "hi": "Hindi",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "ps": "Pashto",
    "pt": "Portuguese",
    "ru": "Russian",
    "so": "Somali",
    "sr": "Serbian",
    "th": "Thai",
    "tl": "Tagalog",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
    "zt": "Chinese (Traditional)",
}


def char_script(ch: str) -> Optional[str]:
    """The script of one letter (``"latin"`` for Latin letters), or None."""
    code = ord(ch)
    for name, low, high in _RANGES:
        if low <= code <= high:
            return name
    if ch.isalpha():
        return "latin" if code < 0x0250 or 0x1E00 <= code <= 0x1EFF else "other"
    return None


def script_weights(text: str) -> Dict[str, int]:
    """How much of ``text`` is in each script (dense scripts weighed up)."""
    weights: Counter = Counter()
    for ch in text:
        name = char_script(ch)
        if name is not None:
            weights[name] += _DENSE_WEIGHT if name in _DENSE else 1
    return dict(weights)


def dominant_script(text: str) -> Optional[str]:
    """The script most of ``text`` is written in, or None when it has no letters."""
    weights = script_weights(text)
    if not weights:
        return None
    if "kana" in weights and "han" in weights:
        # Japanese writes with both; kana is what marks it as Japanese.
        weights["kana"] += weights.pop("han")
    return max(weights, key=lambda name: weights[name])


# -- Paragraphs ----------------------------------------------------------------

_TERMINAL = tuple(".!?:;。！？：；…؟»\"')]”’")
_BULLET = re.compile(r"^\s*(?:[•●▪◦\-*–]|\(?\d{1,3}[.)]|\(?[a-zA-Z][.)])\s+")


def _width(text: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in text)


def _tight(last: str, first: str) -> bool:
    """Whether a line break between ``last`` and ``first`` stands for no space.

    Chinese and Japanese, with their full-width punctuation, and Thai are
    written without spaces between words; Korean is written with them.
    """
    scripts = {char_script(last), char_script(first)}
    if scripts == {"thai"}:
        return True
    wide = all(unicodedata.east_asian_width(ch) in "WF" for ch in (last, first))
    return wide and "hangul" not in scripts


def join_wrapped(previous: str, line: str) -> str:
    """Two pieces of one wrapped paragraph put back together.

    Chinese and Japanese join without a space; a word hyphenated at a line
    end joins without its hyphen.
    """
    if previous.endswith("-") and line[:1].islower():
        return previous[:-1] + line
    if previous and line and _tight(previous[-1], line[0]):
        return previous + line
    return previous + " " + line


def _wraps(previous: str, line: str, wrap_width: int) -> bool:
    """Whether ``line`` carries on the paragraph ``previous`` ended with.

    A printed line that fills most of the width and stops mid-sentence ran on;
    a short one, one that ends a sentence, a bullet, or a change of script
    (English commentary, then the foreign original) starts afresh.
    """
    last_line = previous.rsplit("\n", 1)[-1]
    if not last_line or last_line.rstrip().endswith(_TERMINAL):
        return False
    if _BULLET.match(line) or field_label(line) is not None:
        return False
    if field_label(last_line) is not None:
        return False
    if _width(last_line) < 0.7 * wrap_width:
        return False
    return dominant_script(last_line) == dominant_script(line)


def _paragraphs(lines: Sequence[str], wrap_width: int) -> List[str]:
    paragraphs: List[str] = []
    current = ""
    previous_line = ""
    for raw in lines:
        line = raw.strip()
        if not line:
            if current:
                paragraphs.append(current)
            current, previous_line = "", ""
            continue
        if current and _wraps(previous_line, line, wrap_width):
            current = join_wrapped(current, line)
        else:
            if current:
                paragraphs.append(current)
            current = line
        previous_line = line
    if current:
        paragraphs.append(current)
    return paragraphs


def _wrap_width(lines: Iterable[str]) -> int:
    """The width text was wrapped at: the 90th percentile of line widths."""
    widths = sorted(_width(line.strip()) for line in lines if line.strip())
    if not widths:
        return 0
    return widths[min(len(widths) - 1, int(0.9 * len(widths)))]


@dataclass
class Section:
    """A run of a document: an email (header fields, then body) or plain text."""

    # (English field name, value as written), in the order printed. Empty for
    # text that is not an email.
    fields: List[Tuple[str, str]] = field(default_factory=list)
    paragraphs: List[str] = field(default_factory=list)

    @property
    def is_email(self) -> bool:
        return bool(self.fields)

    def value(self, label: str) -> str:
        return next((value for name, value in self.fields if name == label), "")


def _header_end(lines: Sequence[str], start: int) -> int:
    """Where the email header block starting at ``start`` ends, or ``start``.

    A block is consecutive field lines (a long To or Cc list may run onto the
    following lines) holding at least two of From, Sent/Date, To and Subject.
    """
    names: Set[str] = set()
    index = start
    last: Optional[str] = None
    while index < len(lines):
        line = lines[index]
        label = field_label(line)
        if label is not None:
            names.add(label)
            last = label
        elif (
            last in _LISTS
            and line.strip()
            and (";" in line or "@" in line or line[:1].isspace())
        ):
            pass
        else:
            break
        index += 1
    return index if len(names & _CORE) >= 2 else start


def _fields(lines: Sequence[str]) -> List[Tuple[str, str]]:
    fields: List[Tuple[str, str]] = []
    for line in lines:
        if field_label(line) is not None:
            fields.append(_field(line))
        elif fields:
            label, value = fields[-1]
            fields[-1] = (label, f"{value} {line.strip()}".strip())
    return fields


def parse(text: str) -> List[Section]:
    """``text`` as sections: an email for each header block, plain text otherwise.

    Lines are joined back into the paragraphs they were wrapped from. Text
    before the first email (a cover page, the mailbox name) is a section of its
    own.
    """
    lines = clean(text).split("\n")
    wrap_width = _wrap_width(lines)
    sections: List[Section] = []
    body: List[str] = []
    current = Section()
    index = 0
    while index < len(lines):
        end = _header_end(lines, index) if field_label(lines[index]) else index
        if end > index:
            current.paragraphs = _paragraphs(body, wrap_width)
            if current.fields or current.paragraphs:
                sections.append(current)
            current = Section(fields=_fields(lines[index:end]))
            body = []
            index = end
            continue
        body.append(lines[index])
        index += 1
    current.paragraphs = _paragraphs(body, wrap_width)
    if current.fields or current.paragraphs:
        sections.append(current)
    return sections


# -- Which language to translate from -----------------------------------------

Detector = Callable[[str], Sequence[Tuple[str, float]]]
# A Latin-script paragraph is only taken for a foreign language when it is at
# least this long and the detector is at least this sure: short lines (names,
# codes, "OK") are where language detection guesses.
_LATIN_MIN_LETTERS = 40
_LATIN_MIN_CONFIDENCE = 0.9


def choose_language(
    text: str,
    *,
    forced: Optional[str],
    installed: Set[str],
    detect: Optional[Detector] = None,
) -> Tuple[Optional[str], str]:
    """The language to translate ``text`` from, or None to leave it as it is.

    Returns ``(code, note)``. ``code`` is an installed source language; None
    means the paragraph is English (or has nothing to translate), or is in a
    language this copy has no pack for, which ``note`` then names.

    The script decides first, because it is certain: Han characters are
    Chinese, Hangul Korean. Within a script used by several languages (Arabic
    script for Arabic, Farsi, Urdu, Pashto), the language chosen under From
    wins, then the detector. Latin-script text is left alone unless a
    Latin-script language was chosen, or it is long and plainly not English.
    """
    script = dominant_script(text)
    if script is None or script == "other":
        return None, ""
    if script == "latin":
        letters = sum(1 for ch in text if char_script(ch) == "latin")
        if letters < _LATIN_MIN_LETTERS or detect is None:
            return None, ""
        guesses = list(detect(text))
        if not guesses:
            return None, ""
        code, confidence = guesses[0]
        if code == "en" or confidence < _LATIN_MIN_CONFIDENCE:
            return None, ""
        if forced is not None and forced in installed and code == forced:
            return forced, ""
        return (code, "") if code in installed else (None, code)
    candidates = SCRIPT_LANGUAGES.get(script, ())
    if forced is not None and forced in candidates and forced in installed:
        return forced, ""
    available = [code for code in candidates if code in installed]
    if len(available) > 1 and detect is not None:
        guessed = next((code for code, _ in detect(text) if code in available), None)
        if guessed is not None:
            return guessed, ""
    if available:
        return available[0], ""
    return None, candidates[0] if candidates else script
