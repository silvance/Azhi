"""Confirm or correct a name suggested by voice similarity.

A speaker the recogniser thinks sounds like a learned voice is shown as
"Viktor Marchenko (unconfirmed voice lead)". Turning that into a name has
consequences beyond the label: it teaches the operation profile that voice from
every line attributed to the speaker, and proposes those lines to Viktor
Marchenko's reference profile for review.

Before this dialog the same step was the ordinary rename box, pre-filled with
the suggested name and with OK focused - so one press of Enter accepted the
recogniser's guess and fed its own audio back in as training. The recogniser
would then be more sure of the same guess next time. This dialog makes
confirming a deliberate act: nothing is filled in, "Leave as a suggestion" has
focus so Enter leaves it, there is a button to listen, and it says what
confirming will do.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Callable, Dict, Optional, Tuple

from .dialogs import present
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import secondary_button

CONFIRM = "confirm"
CORRECT = "correct"
_WRAP = 460


def ask_confirm_lead(
    root: tk.Misc,
    *,
    name: str,
    lines: int,
    seconds: float,
    on_play: Optional[Callable[[], None]] = None,
) -> "Optional[Tuple[str, str]]":
    """Ask the operator to confirm, correct or leave a voice-similarity suggestion.

    Returns ``(CONFIRM, name)``, ``(CORRECT, typed_name)``, or ``None`` when
    they leave it as a suggestion.
    """
    win = tk.Toplevel(root)
    win.title("Suggested by voice similarity")
    win.configure(background=palette().surface)
    win.resizable(False, False)

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)

    ttk.Label(frame, text=name, style=Style.SECTION_TITLE).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            f"This speaker sounds like {name}, a voice this profile has learned. "
            "That is a lead from voice similarity — not an identification."
        ),
        style=Style.MUTED,
        wraplength=_WRAP,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    turns = "line" if lines == 1 else "lines"
    ttk.Label(
        frame,
        text=f"{lines} {turns}, {seconds:.0f} seconds of speech, are attributed to this speaker.",
        style=Style.META,
        wraplength=_WRAP,
        justify="left",
    ).pack(anchor="w")
    if on_play is not None:
        secondary_button(frame, "Play a line", on_play).pack(
            anchor="w", pady=(SPACE_SM, 0)
        )

    ttk.Label(
        frame,
        text=(
            "Confirming teaches this profile the voice from all of this "
            f"speaker's lines, and proposes them to {name}'s reference profile "
            "for review. Confirm only if you have listened and are satisfied it "
            "is them."
        ),
        style=Style.WARNING,
        wraplength=_WRAP,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_LG, SPACE_MD))

    ttk.Label(
        frame, text="Not them? Type the right name:", style=Style.FIELD_LABEL
    ).pack(anchor="w")
    typed = tk.StringVar(value="")
    entry = ttk.Entry(frame, textvariable=typed, width=40)
    entry.pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    chosen: Dict[str, Optional[Tuple[str, str]]] = {"value": None}

    def _confirm() -> None:
        chosen["value"] = (CONFIRM, name)
        win.destroy()

    def _correct() -> None:
        value = typed.get().strip()
        if value:
            chosen["value"] = (CORRECT, value)
            win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x", pady=(SPACE_SM, 0))
    cancel = secondary_button(buttons, "Leave as a suggestion", win.destroy)
    cancel.pack(side="right")
    use_typed = secondary_button(buttons, "Use the name I typed", _correct)
    use_typed.pack(side="right", padx=(0, SPACE_SM))
    use_typed.configure(state="disabled")
    secondary_button(buttons, "Confirm – I have listened", _confirm).pack(side="left")

    def _typed_changed(*_args: object) -> None:
        use_typed.configure(state="normal" if typed.get().strip() else "disabled")

    typed.trace_add("write", _typed_changed)
    # Enter in the name box uses that name. Anywhere else, Enter presses the
    # focused button - which starts as "Leave as a suggestion" - so no single
    # keystroke can confirm the suggestion; reaching Confirm takes a click or
    # a deliberate Tab.
    entry.bind("<Return>", lambda _e: _correct())

    present(win, root, modal=True)
    cancel.focus_set()
    win.wait_window()
    return chosen["value"]
