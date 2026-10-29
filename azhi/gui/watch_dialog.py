"""Edit the watch list: the names, places and terms picked out in transcripts.

One term per line. Saved with the settings on this machine, and with an
operation profile, so each operation can carry its own. See
:mod:`azhi.watchlist` for how terms are matched.
"""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional

from .dialogs import present
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import primary_button, secondary_button, style_text_widget


def ask_watch_list(root: tk.Misc, current: str) -> "Optional[str]":
    """Show the list for editing. Returns the new text, or None if cancelled."""
    win = tk.Toplevel(root)
    win.title("Watch list")
    win.configure(background=palette().surface)

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text="Watch list", style=Style.SECTION_TITLE).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            "Names, places, call signs: anything to pick out of every transcript. "
            "One per line. Matched as whole words, in any case, so “Ann” "
            "finds “ANN” but not “Anna”."
        ),
        style=Style.MUTED,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    box = tk.Text(frame, width=48, height=12, wrap="none", undo=True)
    style_text_widget(box)
    box.insert("1.0", current)
    box.pack(fill="both", expand=True)

    ttk.Label(
        frame,
        text=(
            "A hit means the transcript has those letters; a miss means it does "
            "not. The model may spell a name another way, so read around it."
        ),
        style=Style.META,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_SM, SPACE_LG))

    chosen: Dict[str, Optional[str]] = {"value": None}

    def _save() -> None:
        chosen["value"] = box.get("1.0", "end-1c")
        win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x")
    secondary_button(buttons, "Cancel", win.destroy).pack(side="right")
    primary_button(buttons, "Save", _save).pack(side="right", padx=(0, SPACE_SM))

    present(win, root, modal=True)
    box.focus_set()
    win.wait_window()
    return chosen["value"]
