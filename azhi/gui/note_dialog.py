"""Write, change or delete an analyst's note on a transcript line."""

from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, Optional, Tuple

from .dialogs import present
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import primary_button, secondary_button, style_text_widget


def ask_note(
    root: tk.Misc, *, line: str, when: str, text: str = ""
) -> "Optional[Tuple[str, str]]":
    """Ask for a note on ``line`` (said at ``when``).

    Returns ``("save", text)``, ``("delete", "")`` for an existing note the
    analyst removed, or None if they cancelled. ``text`` is the note being
    changed; empty for a new one.
    """
    win = tk.Toplevel(root)
    win.title("Analyst note")
    win.configure(background=palette().surface)

    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text="Change the note" if text else "Add a note",
        style=Style.SECTION_TITLE,
    ).pack(anchor="w")
    shown = line if len(line) <= 160 else line[:157] + "…"
    ttk.Label(
        frame,
        text=f"{when}  {shown}",
        style=Style.MUTED,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_XS, SPACE_MD))

    box = tk.Text(frame, width=52, height=6, wrap="word", undo=True)
    style_text_widget(box)
    box.insert("1.0", text)
    box.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text=(
            "Yours, not the recording's: it is shown and saved labelled as an "
            "analyst note, and stays with this moment if the line is split or "
            "joined."
        ),
        style=Style.META,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_SM, SPACE_LG))

    chosen: Dict[str, Optional[Tuple[str, str]]] = {"value": None}

    def _save() -> None:
        written = box.get("1.0", "end-1c").strip()
        if written:
            chosen["value"] = ("save", written)
        elif text:
            chosen["value"] = ("delete", "")  # emptied: the same as deleting
        win.destroy()

    def _delete() -> None:
        chosen["value"] = ("delete", "")
        win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x")
    secondary_button(buttons, "Cancel", win.destroy).pack(side="right")
    primary_button(buttons, "Save", _save).pack(side="right", padx=(0, SPACE_SM))
    if text:
        secondary_button(buttons, "Delete note", _delete).pack(side="left")

    present(win, root, modal=True)
    box.focus_set()
    win.wait_window()
    return chosen["value"]
