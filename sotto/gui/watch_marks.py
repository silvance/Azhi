"""Highlight watch-list hits in a Tk text widget (see :mod:`sotto.watchlist`).

Shared by the Transcribe page's transcript and the live transcript, so a name
on the list looks the same wherever it turns up. The highlight sits under the
selection and under Find's, so neither is ever hidden by it.
"""

from __future__ import annotations

import tkinter as tk
from typing import Dict, Optional

from ..watchlist import Matcher

WATCH_TAG = "watch"
WATCH_CURRENT_TAG = "watch_current"
# Analyst notes shown in a transcript carry this tag (see transcript_view).
NOTE_TAG = "note"
# Violet: apart from Find's yellow, low confidence's red and the moved-words
# green, so a hit is never mistaken for one of those.
_WATCH_COLOR = "#4b2a72"
_WATCH_CURRENT_COLOR = "#7a4bb5"


def mark(widget: tk.Text, found: Optional[Matcher]) -> Dict[str, int]:
    """Highlight every hit in ``widget``; returns hits per term, in list order."""
    widget.tag_config(WATCH_TAG, background=_WATCH_COLOR)
    widget.tag_config(WATCH_CURRENT_TAG, background=_WATCH_CURRENT_COLOR)
    widget.tag_remove(WATCH_TAG, "1.0", "end")
    widget.tag_remove(WATCH_CURRENT_TAG, "1.0", "end")
    if not found:
        return {}
    counts: Dict[str, int] = {}
    for start, end, term in found.finditer(widget.get("1.0", "end-1c")):
        first = f"1.0 + {start} chars"
        if NOTE_TAG in widget.tag_names(first):
            continue  # an analyst's note: not something said
        widget.tag_add(WATCH_TAG, first, f"1.0 + {end} chars")
        counts[term] = counts.get(term, 0) + 1
    widget.tag_lower(WATCH_TAG)
    return {term: counts[term] for term in found.terms if term in counts}


def next_hit(widget: tk.Text, after: str) -> Optional[str]:
    """Step to the first hit after index ``after`` (wrapping); its start, or None."""
    ranges = widget.tag_ranges(WATCH_TAG)
    if not ranges:
        return None
    pairs = list(zip(ranges[0::2], ranges[1::2]))
    start, end = next(
        ((s, e) for s, e in pairs if widget.compare(s, ">", after)), pairs[0]
    )
    widget.tag_remove(WATCH_CURRENT_TAG, "1.0", "end")
    widget.tag_add(WATCH_CURRENT_TAG, start, end)
    widget.tag_raise(WATCH_CURRENT_TAG)
    widget.see(start)
    return str(start)
