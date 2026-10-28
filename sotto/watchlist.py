"""A watch list: names, places and other terms to pick out in every transcript.

The analyst keeps one list (saved with the settings, and with an operation
profile, so each operation can have its own). Every transcript is searched for
it as it is shown, and each hit is highlighted.

Matching is whole-word and ignores case: "Ann" finds "ann" and "ANN" but not
"Anna" or "Joanne". A term of several words matches with any spacing between
them. Speaker tags (``[Anna]`` at the start of a line) are not searched: a hit
is something said. A hit only means the transcript has those letters, and a
miss only means it does not: the model may have spelled a name another way,
which is why this is a reading aid and never a finding.
"""

from __future__ import annotations

import re
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

# ``[Speaker 1]``, ``[Anna]``, ``[00:12:34]`` or ``[…]`` at the start of a line.
_LINE_TAGS = re.compile(r"^(?:\[[^\]\n]*\]\s*)+", re.MULTILINE)


def parse(text: str) -> List[str]:
    """Terms from what was typed: one per line, blanks and repeats dropped.

    A repeat differing only in case or spacing is the same term; the first
    spelling is kept.
    """
    terms: List[str] = []
    seen = set()
    for line in text.splitlines():
        term = " ".join(line.split())
        key = term.casefold()
        if term and key not in seen:
            seen.add(key)
            terms.append(term)
    return terms


class Matcher:
    """Finds the terms in text, whole words only, ignoring case."""

    def __init__(self, terms: Sequence[str]) -> None:
        self.terms = list(terms)
        self._by_key: Dict[str, str] = {_key(t): t for t in self.terms}
        alternatives = [
            r"\s+".join(re.escape(part) for part in term.split())
            for term in sorted(self.terms, key=len, reverse=True)
        ]
        self._pattern = (
            re.compile(rf"(?<!\w)(?:{'|'.join(alternatives)})(?!\w)", re.IGNORECASE)
            if alternatives
            else None
        )

    def __bool__(self) -> bool:
        return self._pattern is not None

    def finditer(self, text: str) -> Iterator[Tuple[int, int, str]]:
        """``(start, end, term)`` for each hit, outside speaker tags."""
        if self._pattern is None:
            return
        tags = [m.span() for m in _LINE_TAGS.finditer(text)]
        t = 0
        for match in self._pattern.finditer(text):
            start, end = match.span()
            while t < len(tags) and tags[t][1] <= start:
                t += 1
            if t < len(tags) and tags[t][0] <= start < tags[t][1]:
                continue  # inside a speaker tag
            found = _key(match.group(0))
            yield start, end, self._by_key.get(found, match.group(0))

    def counts(self, text: str) -> Dict[str, int]:
        """Hits per term, in the list's order; terms with none are left out."""
        tally = {term: 0 for term in self.terms}
        for _start, _end, term in self.finditer(text):
            tally[term] = tally.get(term, 0) + 1
        return {term: n for term, n in tally.items() if n}


def matcher(text: str) -> Optional[Matcher]:
    """A matcher for what was typed, or None for an empty list."""
    found = Matcher(parse(text))
    return found if found else None


def describe(counts: Dict[str, int], *, listed: bool) -> str:
    """One line for under the transcript: what was found, or that nothing was."""
    if not listed:
        return ""
    if not counts:
        return "Watch list: nothing in this transcript."
    found = ", ".join(f"{term} ×{n}" for term, n in counts.items())
    return f"Watch list: {found}."


def _key(text: str) -> str:
    return " ".join(text.split()).casefold()
