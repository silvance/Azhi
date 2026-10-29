"""How the questioned and reference audio were recorded, side by side.

A similarity score moves with the recordings as well as with the voices: a
telephone line against a studio microphone, heavy background noise, clipping
or a few seconds of speech all lower it, whoever is speaking. This sets the
measurements behind both sides of a comparison next to each other and says,
in plain words, where they differ enough to matter, so a Low or Intermediate
result on mismatched recordings is not read as much as one on matched ones.

Nothing here says anything about who is speaking. It reads only the quality
measurements already taken for each window and each reference sample (see
:mod:`azhi.quality`), and a measurement that was never taken is shown as not
measured rather than filled in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Any, List, Mapping, Optional, Sequence

from .quality import GOOD_MIN_SECONDS, HIGH_CLIPPING_PCT, describe_bandwidth

# Differences large enough to say out loud. Stated here, like the quality
# rules, so the reason for each note is plain.
BANDWIDTH_APART_HZ = 2000.0
MARGIN_APART_DB = 10.0


@dataclass
class Side:
    """The measurements behind one side of a comparison."""

    speech_seconds: float = 0.0
    # Windows on the questioned side, trusted samples on the reference side.
    pieces: int = 0
    # Medians across the pieces that measured them; None where none did.
    bandwidth_hz: Optional[float] = None
    # How many pieces measured it: samples enrolled by older versions did not.
    bandwidth_pieces: int = 0
    margin_db: Optional[float] = None
    # The worst piece, since a little clipping anywhere distorts that piece.
    clipping_percent: Optional[float] = None


@dataclass
class Conditions:
    """Both sides, and what differs between them."""

    questioned: Side
    reference: Side
    notes: List[str] = field(default_factory=list)

    def rows(self) -> List[tuple]:
        """``(label, questioned, reference)`` in the order they are shown."""
        q, r = self.questioned, self.reference
        return [
            (
                "Speech measured",
                f"{q.speech_seconds:.1f} sec in {q.pieces} window(s)",
                f"{r.speech_seconds:.1f} sec in {r.pieces} sample(s)",
            ),
            ("Frequency range", _bandwidth(q, "windows"), _bandwidth(r, "samples")),
            (
                "Speech above background",
                _decibels(q.margin_db),
                _decibels(r.margin_db),
            ),
            (
                "Clipped samples",
                _percent(q.clipping_percent),
                _percent(r.clipping_percent),
            ),
        ]

    def lines(self) -> List[str]:
        """The block for a text result or report."""
        out = ["Recording conditions (questioned / reference):"]
        for label, questioned, reference in self.rows():
            out.append(f"  {label}: {questioned} / {reference}")
        out += [f"  Note: {note}" for note in self.notes]
        return out


def _bandwidth(measured: Side, pieces: str) -> str:
    text = describe_bandwidth(measured.bandwidth_hz or 0.0)
    if 0 < measured.bandwidth_pieces < measured.pieces:
        text += f" (from {measured.bandwidth_pieces} of {measured.pieces} {pieces})"
    return text


def _decibels(value: Optional[float]) -> str:
    return "not measurable" if value is None else f"about {value:.0f} dB"


def _percent(value: Optional[float]) -> str:
    return "not measured" if value is None else f"{value:.1f}%"


def _number(data: Mapping[str, Any], key: str) -> Optional[float]:
    value = data.get(key)
    return float(value) if isinstance(value, (int, float)) else None


def side(reports: Sequence[Mapping[str, Any]], *, speech_seconds: float) -> Side:
    """One side from its pieces' quality measurements (``QualityReport.to_dict``)."""
    out = Side(speech_seconds=round(speech_seconds, 2), pieces=len(reports))
    bandwidths = [hz for hz in (_number(r, "bandwidth_hz") for r in reports) if hz]
    if bandwidths:
        out.bandwidth_hz = median(bandwidths)
        out.bandwidth_pieces = len(bandwidths)
    # A piece with no pause in it has no background to measure against; its
    # margin is the spread within speech, and would read as noise. One whose
    # pauses are digital silence has nothing to measure either (the margin is
    # left at 0 there), and would read as the noisiest audio there is.
    margins = [
        db
        for db in (
            _number(r, "level_margin_db")
            for r in reports
            if not r.get("level_spread_low")
        )
        if db is not None and db > 0
    ]
    if margins and len(margins) * 2 >= len(reports):
        out.margin_db = median(margins)
    clipping = [
        c for c in (_number(r, "clipping_percent") for r in reports) if c is not None
    ]
    if clipping:
        out.clipping_percent = max(clipping)
    return out


def compare(questioned: Side, reference: Side) -> Conditions:
    """Set the two sides side by side and note what differs enough to matter."""
    result = Conditions(questioned=questioned, reference=reference)
    notes = result.notes
    q_hz, r_hz = questioned.bandwidth_hz, reference.bandwidth_hz
    if q_hz and r_hz and abs(q_hz - r_hz) >= BANDWIDTH_APART_HZ:
        narrower = "questioned" if q_hz < r_hz else "reference"
        notes.append(
            f"The {narrower} audio reaches only {min(q_hz, r_hz) / 1000:.1f} kHz "
            f"against {max(q_hz, r_hz) / 1000:.1f} kHz, as through a telephone "
            "or a low-quality recorder. A different recording channel lowers "
            "similarity whoever is speaking, so a lower score says less here."
        )
    elif q_hz and not r_hz:
        notes.append(
            "The reference samples were enrolled before Azhí measured "
            "frequency range, so the two channels cannot be compared."
        )
    q_db, r_db = questioned.margin_db, reference.margin_db
    if q_db is not None and r_db is not None and abs(q_db - r_db) >= MARGIN_APART_DB:
        noisier = "questioned" if q_db < r_db else "reference"
        notes.append(
            f"The {noisier} speech stands much less clear of its background "
            f"(about {min(q_db, r_db):.0f} dB against {max(q_db, r_db):.0f} dB). "
            "Background noise lowers similarity whoever is speaking."
        )
    for name, value in (
        ("questioned", questioned.clipping_percent),
        ("reference", reference.clipping_percent),
    ):
        if value is not None and value >= HIGH_CLIPPING_PCT:
            notes.append(
                f"Some of the {name} audio is clipped ({value:.1f}% of samples); "
                "the distortion lowers similarity."
            )
    if 0 < questioned.speech_seconds < GOOD_MIN_SECONDS:
        notes.append(
            f"Only {questioned.speech_seconds:.1f} sec of questioned speech: a "
            "score from this little speech varies more from one recording to "
            "the next."
        )
    return result
