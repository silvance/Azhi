"""Two views under a comparison's result: window by window, and the recordings.

* :class:`WindowStrip` - one bar per 8-second window of the questioned speech,
  each compared on its own with the reference, against the band lines. What it
  is for is finding the window that disagrees with the rest (another voice, a
  burst of noise) and listening to it; no single bar is a finding.
* :class:`ConditionsTable` - how the questioned and reference audio were made,
  side by side, with the differences that lower similarity whoever is speaking.

Neither shows anything of the voice measurement itself: the vectors stay in
memory, and these draw only scores and audio measurements.
"""

from __future__ import annotations

import tkinter as tk
from functools import partial
from tkinter import ttk
from typing import Callable, List, Optional

from ..conditions import Conditions
from ..matching import WindowDetail, WindowScore
from ..thresholds import BAND_HIGH, BAND_INTERMEDIATE
from .theme import SPACE_MD, SPACE_SM, SPACE_XS, Style, palette, theme
from .widgets import subtle_button, wrap_to

_HEIGHT = 128
_TOP = 10
_BOTTOM = 18  # room for the times under the bars
_GAP = 3
_MIN_BAR = 8
_MAX_BAR = 44

STRIP_GUIDE = (
    "Each bar is one 8-second window of the questioned speech, compared on "
    "its own. A single window scores lower and varies more than the whole "
    "selection, which averages them, so read the pattern, not a bar: look for "
    "a window that stands apart, then click it to listen."
)
CONDITIONS_GUIDE = (
    "How each recording was made. Differences here lower similarity whoever "
    "is speaking; they say nothing about who it is."
)


class WindowStrip(ttk.Frame):
    """Bars for each questioned window, against the comparison's band lines."""

    def __init__(
        self,
        parent: tk.Misc,
        *,
        on_play: Callable[[WindowScore], None],
        on_stop: Callable[[], None],
    ) -> None:
        super().__init__(parent, style=Style.CARD_INNER)
        self._on_play = on_play
        self._windows: List[WindowScore] = []
        self._apart: List[int] = []
        self._high = 0.0
        self._intermediate = 0.0
        self._selected: Optional[int] = None

        guide = ttk.Label(
            self, text=STRIP_GUIDE, style=Style.MUTED, wraplength=640, justify="left"
        )
        guide.pack(anchor="w")
        wrap_to(self, guide)

        colors = palette()
        holder = ttk.Frame(self, style=Style.CARD_INNER)
        holder.pack(fill="x", pady=(SPACE_SM, 0))
        self.canvas = tk.Canvas(
            holder,
            height=_HEIGHT,
            background=colors.surface_alt,
            highlightthickness=0,
            borderwidth=0,
        )
        self.canvas.pack(fill="x")
        # Only when the bars need more room than there is: an hour of speech
        # is 450 windows.
        self._scroll = ttk.Scrollbar(
            holder, orient="horizontal", command=self.canvas.xview
        )
        self.canvas.configure(xscrollcommand=self._scroll.set)
        self.canvas.bind("<Configure>", lambda _e: self._draw())

        row = ttk.Frame(self, style=Style.CARD_INNER)
        row.pack(fill="x", pady=(SPACE_XS, 0))
        self.detail_var = tk.StringVar(value="")
        ttk.Label(row, textvariable=self.detail_var, style=Style.BODY).pack(
            side="left", anchor="w"
        )
        subtle_button(row, "Stop audio", on_stop).pack(side="right")

        self.notes_var = tk.StringVar(value="")
        self._notes = ttk.Label(
            self,
            textvariable=self.notes_var,
            style=Style.WARNING,
            wraplength=640,
            justify="left",
        )
        wrap_to(self, self._notes)

    def show(self, detail: WindowDetail) -> None:
        self._windows = list(detail.windows)
        self._apart = [self._windows.index(w) for w in detail.windows_apart()]
        self._high = detail.thresholds.comparison_high
        self._intermediate = detail.thresholds.comparison_intermediate
        self._selected = None
        used = sum(1 for w in self._windows if w.used)
        self.detail_var.set(
            f"{used} of {len(self._windows)} window(s) used. Point at a bar for "
            "its time and score."
        )
        notes = detail.window_notes()
        self.notes_var.set("\n".join(notes))
        if notes:
            self._notes.pack(anchor="w", fill="x", pady=(SPACE_SM, 0))
        else:
            self._notes.pack_forget()
        self._draw()

    # -- Drawing ------------------------------------------------------------

    def _bar_width(self, available: int) -> int:
        count = max(1, len(self._windows))
        width = (available - _GAP * count) // count
        return int(max(_MIN_BAR, min(_MAX_BAR, width)))

    def _y(self, score: float) -> float:
        span = _HEIGHT - _TOP - _BOTTOM
        return _TOP + span * (1.0 - min(1.0, max(0.0, score)))

    def _draw(self) -> None:
        canvas = self.canvas
        canvas.delete("all")
        if not self._windows:
            return
        colors = palette()
        small = theme().meta
        lines = [
            (self._high, f"{BAND_HIGH.split()[0]} {self._high:.2f}"),
            (
                self._intermediate,
                f"{BAND_INTERMEDIATE.split()[0]} {self._intermediate:.2f}",
            ),
        ]
        left = max(small.measure(text) for _value, text in lines) + 16
        visible = max(1, canvas.winfo_width())
        bar = self._bar_width(visible - left - SPACE_MD)
        total = left + len(self._windows) * (bar + _GAP) + SPACE_MD
        if total > visible:
            canvas.configure(scrollregion=(0, 0, total, _HEIGHT))
            if not self._scroll.winfo_manager():
                self._scroll.pack(fill="x")
        else:
            canvas.configure(scrollregion=(0, 0, visible, _HEIGHT))
            self._scroll.pack_forget()
        right = max(total, visible)

        # The band lines the whole comparison is read against.
        for value, text in lines:
            y = self._y(value)
            canvas.create_line(left - 4, y, right, y, fill=colors.border, dash=(4, 3))
            canvas.create_text(
                left - 8, y, text=text, anchor="e", fill=colors.text_muted, font=small
            )
        base = self._y(0.0)
        canvas.create_line(left - 4, base, right, base, fill=colors.border)

        # A time under every bar that has room for one.
        label_every = max(1, -(-44 // (bar + _GAP)))
        for index, window in enumerate(self._windows):
            x = left + index * (bar + _GAP)
            tag = f"w{index}"
            if window.score is None:
                # Not used: an empty frame, so the gap reads as no measurement
                # rather than as a low score.
                canvas.create_rectangle(
                    x,
                    self._y(1.0),
                    x + bar,
                    base,
                    outline=colors.border,
                    dash=(2, 2),
                    fill="",
                    tags=(tag, "bar"),
                )
            else:
                canvas.create_rectangle(
                    x,
                    self._y(window.score),
                    x + bar,
                    base,
                    fill=self._colour(window.score),
                    outline=(
                        colors.warning
                        if index in self._apart
                        else self._colour(window.score)
                    ),
                    width=2 if index in self._apart else 1,
                    tags=(tag, "bar"),
                )
            # An invisible full-height target, so a short bar is as easy to
            # click as a tall one.
            canvas.create_rectangle(
                x, _TOP, x + bar, base, outline="", fill="", tags=(tag, "hit")
            )
            if index % label_every == 0:
                canvas.create_text(
                    x + bar / 2,
                    base + 3,
                    text=_short_clock(window.start),
                    anchor="n",
                    fill=colors.text_faint,
                    font=small,
                )
            canvas.tag_bind(tag, "<Enter>", partial(self._describe, index))
            canvas.tag_bind(tag, "<Button-1>", partial(self._play, index))
        if self._selected is not None:
            self._mark_selected()

    def _colour(self, score: float) -> str:
        # Blue at high similarity, as on the result above (green would read as
        # "right"); the bar's height says the rest.
        colors = palette()
        return colors.accent if score >= self._high else colors.text_faint

    def _mark_selected(self) -> None:
        self.canvas.delete("selected")
        if self._selected is None:
            return
        box = self.canvas.bbox(f"w{self._selected}")
        if box:
            self.canvas.create_rectangle(
                box[0] - 2,
                _TOP - 4,
                box[2] + 2,
                self._y(0.0) + 1,
                outline=palette().text,
                tags=("selected",),
            )

    # -- Interaction --------------------------------------------------------

    def _describe(self, index: int, _event: object = None) -> None:
        self.detail_var.set(describe_window(self._windows[index]))

    def _play(self, index: int, _event: object = None) -> None:
        window = self._windows[index]
        self._selected = index
        self._mark_selected()
        self._describe(index)
        if window.spans:
            self._on_play(window)


def describe_window(window: WindowScore) -> str:
    """One window in words, as the line under the strip shows it."""
    if window.score is None:
        return f"{window.where()}: not used ({window.skipped})"
    return (
        f"{window.where()}: {window.score:.2f}, from "
        f"{window.speech_seconds:.1f} sec of speech ({window.quality.lower()} audio)"
    )


def _short_clock(seconds: float) -> str:
    total = int(round(max(0.0, seconds)))
    minutes, secs = divmod(total, 60)
    return f"{minutes}:{secs:02d}"


class ConditionsTable(ttk.Frame):
    """The questioned and reference recordings' measurements, side by side."""

    def __init__(self, parent: tk.Misc) -> None:
        super().__init__(parent, style=Style.CARD_INNER)
        guide = ttk.Label(
            self,
            text=CONDITIONS_GUIDE,
            style=Style.MUTED,
            wraplength=640,
            justify="left",
        )
        guide.pack(anchor="w")
        wrap_to(self, guide)
        self._grid = ttk.Frame(self, style=Style.CARD_INNER)
        self._grid.pack(anchor="w", pady=(SPACE_SM, 0))
        self.notes_var = tk.StringVar(value="")
        self._notes = ttk.Label(
            self,
            textvariable=self.notes_var,
            style=Style.WARNING,
            wraplength=640,
            justify="left",
        )
        wrap_to(self, self._notes)

    def show(self, conditions: Conditions) -> None:
        for child in self._grid.winfo_children():
            child.destroy()
        header = ("", "Questioned", "Reference")
        for column, text in enumerate(header):
            ttk.Label(self._grid, text=text, style=Style.FIELD_LABEL).grid(
                row=0, column=column, sticky="w", padx=(0, SPACE_MD)
            )
        for row, (label, questioned, reference) in enumerate(conditions.rows(), 1):
            ttk.Label(self._grid, text=label, style=Style.FIELD_LABEL).grid(
                row=row, column=0, sticky="w", padx=(0, SPACE_MD), pady=(SPACE_XS, 0)
            )
            for column, value in ((1, questioned), (2, reference)):
                ttk.Label(self._grid, text=value, style=Style.BODY).grid(
                    row=row,
                    column=column,
                    sticky="w",
                    padx=(0, SPACE_MD),
                    pady=(SPACE_XS, 0),
                )
        self.notes_var.set("\n".join(conditions.notes))
        if conditions.notes:
            self._notes.pack(anchor="w", fill="x", pady=(SPACE_SM, 0))
        else:
            self._notes.pack_forget()
