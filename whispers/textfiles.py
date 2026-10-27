"""Reading a text file for translation without guessing its encoding.

Batch translation read every text file as UTF-8 with undecodable bytes
replaced, so a Windows-1251, Windows-1256 or GBK file - all common where these
recordings and documents come from - was silently turned into replacement
characters and then "translated". Guessing a legacy encoding instead is no
better: almost any bytes decode as *something* in a single-byte code page, so
a wrong guess produces fluent-looking nonsense with no sign that it is wrong.

So a file is read only when its encoding is certain: UTF-8 (with or without a
byte-order mark) or UTF-16 with its byte-order mark. Anything else is refused
with a reason the operator can act on.
"""

from __future__ import annotations

import codecs
from pathlib import Path
from typing import Union

PathLike = Union[str, Path]


class UnreadableText(ValueError):
    """A text file whose encoding cannot be known for certain."""


def read_text(path: PathLike) -> str:
    """The text of ``path`` if it is UTF-8 or BOM-marked UTF-16; else raise."""
    data = Path(path).read_bytes()
    if data.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    else:
        try:
            return data.decode("utf-8-sig")
        except UnicodeDecodeError:
            pass
    raise UnreadableText(
        f"{Path(path).name} is not saved as UTF-8 text, so its characters "
        "cannot be read reliably (it may be in an older Cyrillic, Arabic or "
        "Chinese encoding). Open it in Notepad, choose Save As with the "
        "encoding set to UTF-8, and add it again."
    )
