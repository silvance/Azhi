"""Choose which diarized speaker is the person of interest - by ear.

Speaker Profiles (building a reference voice) and Compare Speakers (choosing
the questioned speaker) both separate a recording into speakers and ask which
one is wanted. The question used to be a drop-down of "SPEAKER_01 - 45s of
speech" with the longest talker already selected, and the advice to "listen
first if you are unsure" with no way to listen. On Speaker Profiles the answer
becomes reference material every later comparison is measured against.

Here each speaker has a Play button for their longest turn, nothing is chosen
until the operator chooses, and where this computer cannot play audio the turn
times are shown so it can be found in another player.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Callable, Optional, Sequence, Tuple

from ..enrollment import SpeakerOption
from ..playback import PlaybackError, SegmentPlayer, playback_available
from .dialogs import present
from .errors import friendly_error
from .theme import SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import primary_button, secondary_button

_WRAP = 460

NO_PLAYER = (
    "Audio cannot be played on this computer. The times show each speaker's "
    "longest turn, to find in another player before choosing."
)


def clock(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class SampleListener:
    """Plays spans of one audio file, off the UI thread."""

    def __init__(self, wav: Path) -> None:
        self._wav = wav
        self._player = SegmentPlayer()

    @classmethod
    def for_file(cls, wav: Path) -> "Optional[SampleListener]":
        """A listener for ``wav``, or None where nothing can be played."""
        return cls(wav) if playback_available() else None

    def play(self, span: Tuple[float, float], on_error: Callable[[str], None]) -> None:
        def _worker() -> None:
            try:
                self._player.play_segment(self._wav, span[0], span[1])
            except PlaybackError as exc:
                on_error(friendly_error(exc))

        threading.Thread(target=_worker, daemon=True).start()

    def stop(self) -> None:
        self._player.stop()


def ask_pick_speaker(
    root: tk.Misc,
    *,
    title: str,
    guidance: str,
    options: Sequence[SpeakerOption],
    listener: Optional[SampleListener] = None,
) -> Optional[str]:
    """Ask which speaker is wanted. Returns the speaker id, or None if cancelled."""
    win = tk.Toplevel(root)
    win.title(title)
    win.configure(background=palette().surface)
    win.resizable(False, False)
    chosen = tk.StringVar(value="")
    answer: dict = {"value": None}

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=title, style=Style.SECTION_TITLE).pack(anchor="w")
    ttk.Label(
        frame, text=guidance, style=Style.MUTED, wraplength=_WRAP, justify="left"
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    message = tk.StringVar(value="" if listener is not None else NO_PLAYER)

    def _say(text: str) -> None:
        win.after(0, lambda: message.set(text))

    def _play(option: SpeakerOption) -> None:
        if listener is None:
            return
        message.set(
            f"Playing {option.speaker}, {clock(option.sample[0])}–"
            f"{clock(option.sample[1])}…"
        )
        listener.play(option.sample, _say)

    for option in options:
        row = ttk.Frame(frame, style=Style.CARD_INNER)
        row.pack(fill="x", pady=(0, SPACE_SM))
        text = ttk.Frame(row, style=Style.CARD_INNER)
        text.pack(side="left", fill="x", expand=True)
        turns = "1 turn" if option.turns == 1 else f"{option.turns} turns"
        ttk.Radiobutton(
            text,
            text=f"{option.speaker} — {option.seconds:.0f} s of speech in {turns}",
            variable=chosen,
            value=option.speaker,
            style=Style.RADIO,
            command=lambda: use.configure(state="normal"),
        ).pack(anchor="w")
        ttk.Label(
            text,
            text=(f"Longest turn {clock(option.sample[0])}–{clock(option.sample[1])}"),
            style=Style.META,
        ).pack(anchor="w", padx=(SPACE_XL, 0))
        if listener is not None:
            secondary_button(row, "▶ Play sample", lambda o=option: _play(o)).pack(
                side="right", padx=(SPACE_SM, 0)
            )

    ttk.Label(
        frame,
        textvariable=message,
        style=Style.META,
        wraplength=_WRAP,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, 0))

    def _close() -> None:
        if listener is not None:
            listener.stop()
        win.destroy()

    def _use() -> None:
        if not chosen.get():
            return
        answer["value"] = chosen.get()
        _close()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x", pady=(SPACE_MD, 0))
    secondary_button(buttons, "Cancel", _close).pack(side="right")
    # Off until a speaker is chosen: nothing is picked on the operator's behalf.
    use = primary_button(buttons, "Use this speaker", _use)
    use.configure(state="disabled")
    use.pack(side="right", padx=(0, SPACE_SM))
    win.bind("<Return>", lambda _e: _use())
    present(win, root, modal=True, on_close=_close)
    win.wait_window()
    return answer["value"]


class PickCancelled(Exception):
    """The operator closed the speaker picker without choosing."""


def ask_from_worker(root: tk.Misc, ask: Callable[[], Optional[str]]) -> Optional[str]:
    """Run ``ask`` on the UI thread and wait for its answer, however long.

    The callers used to give up after five minutes and carry on as if the
    question had been cancelled, leaving a dialog on screen that no longer did
    anything.
    """
    answer: dict = {"value": None}
    done = threading.Event()

    def _ask() -> None:
        try:
            answer["value"] = ask()
        finally:
            done.set()

    root.after(0, _ask)
    done.wait()
    return answer["value"]
