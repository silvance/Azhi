"""The transcript pane: renders a result and supports speaker/word corrections.

Diarization on hard or overlapping audio is never perfect, so the transcript is
interactive. A click plays: a word from there to the end of its line, a
``[speaker]`` tag the whole line. A right-click offers the corrections: on a
tag, rename that speaker everywhere or move the line to another speaker; on a
word, move just that word (or from it onward). A speaker tag, or a selected run
of words, can also be dragged onto another speaker's line. Without a way to
play audio, a click opens the corrections instead. An edit keeps the view
where it was and briefly marks what moved. The underlying split/merge logic
lives in :mod:`sotto.editing`; this module is the Tk view around it.
"""

from __future__ import annotations

import tkinter as tk
from functools import partial
from tkinter import simpledialog
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional, Tuple, cast

from ..editing import (
    coalesce_segments,
    split_segment_on_span,
    split_segment_on_word,
)
from ..speaker_names import bare_name, is_lead, shown_name
from ..transcription import (
    TranscriptionResult,
    is_low_confidence_segment,
    is_low_confidence_word,
)
from .confirm_lead import ask_confirm_lead
from .dialogs import active_window
from .theme import palette

# Foreground colour for low-confidence text when highlighting is enabled.
_LOW_CONFIDENCE_COLOR = "#ff7a7a"
# Background colours for in-transcript search (all matches / the current one).
_SEARCH_COLOR = "#5b5300"
_SEARCH_CURRENT_COLOR = "#b58900"

# How long words just moved stay marked, so the eye can find what changed.
_MOVED_MS = 1600
# How close to the top or bottom edge a drag scrolls the transcript (pixels).
_EDGE = 24
# How far the pointer must travel before a press on a speaker tag is a drag.
_DRAG_START = 5
# A click plays after this long, unless it turns out to be the first half of a
# double-click (which selects a word, and should not set the audio going).
_CLICK_MS = 250

# In place of the correction items while a run is going.
PAUSED_NOTE = "Speaker corrections wait until the run finishes"


class TranscriptView:
    """Owns the transcript text widget plus the current result and speaker names."""

    def __init__(
        self,
        parent: tk.Misc,
        root: tk.Misc,
        blank_lines_var: tk.BooleanVar,
        on_change: Callable[[], None],
        highlight_var: Optional[tk.BooleanVar] = None,
        on_play: Optional[Callable[[float, float], None]] = None,
        on_enroll: Optional[Callable[[str, List[Tuple[float, float]]], None]] = None,
        on_compare: Optional[Callable[[str], None]] = None,
        placeholder: str = "",
    ) -> None:
        self.root = root
        self.blank_lines_var = blank_lines_var
        # Shown (greyed) when there's no result yet - a first-run "do this" hint.
        self._placeholder = placeholder
        # When set/enabled, low-confidence words (or segments) are colored.
        self.highlight_var = highlight_var
        # Called after an edit mutates the result, so the owner can re-save outputs.
        self._on_change = on_change
        # Play a [start, end] span of the source audio; None disables playback.
        self._on_play = on_play
        # Learn a speaker's voice from a correction: (display name, audio spans).
        # None disables voiceprint enrolment (no profile / no audio).
        self._on_enroll = on_enroll
        # Search the speaker profiles for one speaker's lines (speaker id).
        # None where this build has no speaker profiles.
        self._on_compare = on_compare
        self._result: Optional[TranscriptionResult] = None
        self._speaker_names: Dict[str, str] = {}
        # Off while a run or redo is going: an edit made then was lost when the
        # run put up the next transcript. Playing a line stays available.
        self._editable = True
        # Where the next "find" search starts (advances as you step through hits).
        self._find_pos = "1.0"
        self.widget = ScrolledText(
            parent, wrap="word", state="disabled", height=14, font="TkFixedFont"
        )
        # Right-click: the corrections for whatever is under the pointer - a
        # highlighted span, a speaker tag or a word. Button-3 on Windows/Linux,
        # Button-2 on macOS.
        self.widget.bind("<Button-3>", self._context_menu)
        self.widget.bind("<Button-2>", self._context_menu)
        # A double-click selects a word; the click it began with must not play.
        self.widget.bind("<Double-Button-1>", self._cancel_click, add="+")
        self._click_job: Optional[str] = None
        # Drag-to-move: after selecting a run of words, press within the selection
        # and drag it onto another speaker's line to reassign it (no menu needed).
        # In-progress drag state, or None. See _on_drag_press/_motion/_release.
        self._drag: Optional[Dict[str, object]] = None
        # The words an edit just moved, as (first start, last end, speaker), to
        # be marked after the next render; and the pending job that unmarks them.
        self._moved: Optional[Tuple[float, float, str]] = None
        self._moved_job: Optional[str] = None
        self._scroll_job: Optional[str] = None
        self.widget.bind("<Button-1>", self._on_drag_press)
        self.widget.bind("<B1-Motion>", self._on_drag_motion)
        self.widget.bind("<ButtonRelease-1>", self._on_drag_release)

    def set_editable(self, editable: bool) -> None:
        """Allow or pause speaker corrections (playback is unaffected)."""
        self._editable = editable
        if not editable:
            self._drag = None
            self.widget.config(cursor="")

    @staticmethod
    def _popup(menu: tk.Menu, event: object) -> None:
        try:
            menu.tk_popup(event.x_root, event.y_root)  # type: ignore[attr-defined]
        finally:
            menu.grab_release()

    def _add_paused_note(self, menu: tk.Menu) -> None:
        menu.add_command(label=PAUSED_NOTE, state="disabled")

    def set_result(
        self, result: Optional[TranscriptionResult], speaker_names: Dict[str, str]
    ) -> None:
        """Show ``result`` (sharing the ``speaker_names`` dict with the owner)."""
        self._result = result
        self._speaker_names = speaker_names
        self._find_pos = "1.0"
        self.render()

    def get_text(self) -> str:
        return self.widget.get("1.0", "end-1c")

    def find(self, query: str, *, backwards: bool = False) -> int:
        """Highlight all case-insensitive matches of ``query`` and step to one.

        Returns the total number of matches. Stepping wraps around the end. An
        empty query just clears the highlights.
        """
        w = self.widget
        w.tag_config("search", background=_SEARCH_COLOR)
        w.tag_config("search_current", background=_SEARCH_CURRENT_COLOR)
        w.tag_remove("search", "1.0", "end")
        w.tag_remove("search_current", "1.0", "end")
        if not query:
            self._find_pos = "1.0"
            return 0

        total = 0
        idx = "1.0"
        while True:
            pos = w.search(query, idx, nocase=True, stopindex="end")
            if not pos:
                break
            end = f"{pos}+{len(query)}c"
            w.tag_add("search", pos, end)
            idx = end
            total += 1
        if total == 0:
            return 0

        if backwards:
            start = self._find_pos or "end"
            pos = w.search(query, start, nocase=True, backwards=True, stopindex="1.0")
            if not pos:  # wrap to the end
                pos = w.search(
                    query, "end", nocase=True, backwards=True, stopindex="1.0"
                )
        else:
            start = self._find_pos or "1.0"
            pos = w.search(query, start, nocase=True, stopindex="end")
            if not pos:  # wrap to the top
                pos = w.search(query, "1.0", nocase=True, stopindex="end")
        if pos:
            end = f"{pos}+{len(query)}c"
            w.tag_add("search_current", pos, end)
            w.tag_raise("search_current")
            w.see(pos)
            # Advance so the next step moves past (forward) / before (back) this hit.
            self._find_pos = pos if backwards else end
        return total

    def render(self, *, keep_view: bool = False) -> None:
        """Render the current result, with clickable speaker tags / words.

        ``keep_view`` keeps the same part of the transcript on screen, as an
        edit or a display option should; a new result starts at the top.
        """

        highlight = self.highlight_var is not None and self.highlight_var.get()

        def _lowconf_seg(segment) -> tuple:
            return (
                ("lowconf",) if highlight and is_low_confidence_segment(segment) else ()
            )

        def _do() -> None:
            result = self._result
            anchor = self._view_anchor() if keep_view else None
            self.widget.configure(state="normal")
            self.widget.delete("1.0", "end")
            # Underlined as well as coloured: the red alone is the red used
            # for errors, and colour alone is lost on some screens and eyes.
            self.widget.tag_config(
                "lowconf", foreground=_LOW_CONFIDENCE_COLOR, underline=True
            )
            if result is None:
                if self._placeholder:
                    self.widget.tag_config(
                        "placeholder", foreground=palette().text_muted
                    )
                    self.widget.insert("end", self._placeholder, ("placeholder",))
                self.widget.configure(state="disabled")
                return
            # Blank line between segments/turns when enabled (easier to read/paste).
            line_end = "\n\n" if self.blank_lines_var.get() else "\n"
            if not result.has_speakers:
                if result.segments:
                    for index, segment in enumerate(result.segments):
                        if index:
                            self.widget.insert("end", line_end)
                        line_tag = f"line::{index}"
                        tags = (line_tag, *_lowconf_seg(segment))
                        # A click plays this line, as it does in a diarized
                        # transcript; right-click offers the same as a menu.
                        if self._on_play is not None:
                            # ButtonRelease (not press) so a click-drag can select
                            # words instead of setting the audio going.
                            self.widget.tag_bind(
                                line_tag,
                                "<ButtonRelease-1>",
                                self._click_handler(partial(self._play_line, index)),
                            )
                            self.widget.tag_bind(
                                line_tag, "<Enter>", self._cursor_handler("hand2")
                            )
                            self.widget.tag_bind(
                                line_tag, "<Leave>", self._cursor_handler("")
                            )
                        self.widget.insert("end", segment.text, tags)
                    self.widget.insert("end", "\n")
                else:
                    self.widget.insert("end", result.text + "\n")
            else:
                bound: set[str] = set()
                for index, segment in enumerate(result.segments):
                    sid = segment.speaker or "UNKNOWN"
                    name = shown_name(sid, self._speaker_names)
                    spk_tag = f"spk::{sid}"
                    line_tag = f"line::{index}"
                    if sid not in bound:
                        bound.add(sid)
                        self.widget.tag_config(spk_tag, underline=True)
                    # Bind on a per-line tag so a click knows which segment it hit.
                    # A click plays the line (or, with no way to play, opens the
                    # menu that fixes this line or renames the speaker). On
                    # ButtonRelease so a click-drag selects words instead.
                    self.widget.tag_bind(
                        line_tag,
                        "<ButtonRelease-1>",
                        self._click_handler(
                            partial(self._play_line, index),
                            self._speaker_menu_handler(index),
                        ),
                    )
                    self.widget.tag_bind(
                        line_tag, "<Enter>", self._cursor_handler("hand2")
                    )
                    self.widget.tag_bind(line_tag, "<Leave>", self._cursor_handler(""))
                    self.widget.insert("end", f"[{name}]", (spk_tag, line_tag))
                    if segment.words:
                        # Render words individually so a single misattributed word
                        # can be clicked and moved to another speaker.
                        for w_index, word in enumerate(segment.words):
                            text = word.word
                            if w_index == 0 and not text[:1].isspace():
                                text = " " + text
                            wtag = f"word::{index}::{w_index}"
                            self.widget.tag_bind(
                                wtag,
                                "<ButtonRelease-1>",
                                self._click_handler(
                                    partial(self._play_from_word, index, w_index),
                                    self._word_menu_handler(index, w_index),
                                ),
                            )
                            self.widget.tag_bind(
                                wtag, "<Enter>", self._cursor_handler("hand2")
                            )
                            self.widget.tag_bind(
                                wtag, "<Leave>", self._cursor_handler("")
                            )
                            wtags: tuple[str, ...] = (wtag,)
                            if highlight and is_low_confidence_word(word):
                                wtags = (wtag, "lowconf")
                            self.widget.insert("end", text, wtags)
                        self.widget.insert("end", line_end)
                    else:
                        self.widget.insert(
                            "end", f" {segment.text}", _lowconf_seg(segment)
                        )
                        self.widget.insert("end", line_end)
            self.widget.configure(state="disabled")
            if anchor is not None:
                self._restore_view(anchor)
            self._mark_moved()

        self.root.after(0, _do)

    def _changed(self) -> None:
        """Re-render after an edit and notify the owner to persist the change."""
        self.render(keep_view=True)
        self._on_change()

    # -- Keeping the place across a re-render ------------------------------

    def _segment_at(self, index: str) -> Optional[int]:
        for tag in self.widget.tag_names(index):
            if tag.startswith(("line::", "word::")):
                try:
                    return int(tag.split("::")[1])
                except (IndexError, ValueError):
                    continue
        return None

    def _view_anchor(self) -> Tuple[Optional[float], float]:
        """The start time of the first line in view, and the scroll fraction.

        A time rather than a line number: an edit splits and merges segments,
        which renumbers them, but no word changes its time.
        """
        fraction = self.widget.yview()[0]
        result = self._result
        if result is None:
            return None, fraction
        top = self.widget.index("@0,0")
        for offset in range(4):  # past a blank line between turns
            seg = self._segment_at(f"{top} +{offset} lines")
            if seg is not None and seg < len(result.segments):
                return result.segments[seg].start, fraction
        return None, fraction

    def _restore_view(self, anchor: Tuple[Optional[float], float]) -> None:
        start, fraction = anchor
        result = self._result
        if start is not None and result is not None:
            for index, segment in enumerate(result.segments):
                # The first line still running at that time: the same line, or
                # the part of it an edit split off, or the one it merged into.
                if segment.end > start + 1e-6:
                    ranges = self.widget.tag_ranges(f"line::{index}")
                    if ranges:
                        self.widget.yview(f"{ranges[0]} linestart")
                        return
                    break
        self.widget.yview_moveto(fraction)

    def _mark_moved(self) -> None:
        """Briefly mark the words an edit just moved, and keep them in view."""
        moved, self._moved = self._moved, None
        result = self._result
        if moved is None or result is None:
            return
        first, last, speaker = moved
        self.widget.tag_config("moved", background=palette().success_bg)
        self.widget.tag_remove("moved", "1.0", "end")
        marked: List[str] = []
        for s_idx, segment in enumerate(result.segments):
            if (segment.speaker or "UNKNOWN") != speaker:
                continue
            if segment.words:
                for w_idx, word in enumerate(segment.words):
                    if word.start >= first - 1e-6 and word.end <= last + 1e-6:
                        ranges = self.widget.tag_ranges(f"word::{s_idx}::{w_idx}")
                        if ranges:
                            self.widget.tag_add("moved", ranges[0], ranges[1])
                            marked.append(str(ranges[0]))
            elif segment.start >= first - 1e-6 and segment.end <= last + 1e-6:
                ranges = self.widget.tag_ranges(f"line::{s_idx}")
                if ranges:
                    self.widget.tag_add("moved", ranges[0], f"{ranges[0]} lineend")
                    marked.append(str(ranges[0]))
        if not marked:
            return
        self.widget.see(marked[0])
        if self._moved_job is not None:
            self.widget.after_cancel(self._moved_job)
        self._moved_job = self.widget.after(
            _MOVED_MS, lambda: self.widget.tag_remove("moved", "1.0", "end")
        )

    def _cursor_handler(self, cursor: str) -> Callable[[object], None]:
        def handler(_event: object) -> None:
            self.widget.config(cursor=cursor)

        return handler

    def _ordered_speaker_ids(self) -> List[str]:
        """Distinct speaker ids in first-appearance order across the result."""
        ids: List[str] = []
        seen: set[str] = set()
        if self._result is not None:
            for segment in self._result.segments:
                sid = segment.speaker or "UNKNOWN"
                if sid not in seen:
                    seen.add(sid)
                    ids.append(sid)
        return ids

    def _speaker_menu_handler(self, index: int) -> Callable[[object], None]:
        def handler(event: object) -> None:
            if self._has_selection():
                return  # a drag-select; leave it for right-click "Move selection"
            if self._drag is not None and self._drag.get("moved"):
                return  # the end of a drag, handled as a drop
            result = self._result
            if result is None or index >= len(result.segments):
                return
            current = result.segments[index].speaker or "UNKNOWN"
            menu = tk.Menu(self.root, tearoff=0)
            if self._on_play is not None:
                menu.add_command(
                    label="▶ Play this line",
                    command=lambda: self._play_line(index),
                )
                menu.add_separator()
            on_compare = self._on_compare
            if on_compare is not None and result.segments[index].speaker:
                # Reads the transcript, changes nothing: offered mid-run too.
                shown = shown_name(current, self._speaker_names)
                menu.add_command(
                    label=f"Compare \u2018{shown}\u2019 to speaker profiles…",
                    command=lambda: on_compare(current),
                )
                menu.add_separator()
            if not self._editable:
                self._add_paused_note(menu)
                self._popup(menu, event)
                return
            # Reassign just this line to the correct speaker - the fix for the
            # boundary/overlap errors diarization can't get right on its own.
            for sid in self._ordered_speaker_ids():
                name = shown_name(sid, self._speaker_names)
                mark = "  ✓" if sid == current else ""
                menu.add_command(
                    label=f"This line is {name}{mark}",
                    command=self._reassign_command(index, sid),
                )
            menu.add_separator()
            cur_name = shown_name(current, self._speaker_names)
            menu.add_command(
                label=(
                    f"Confirm or correct \u2018{bare_name(cur_name)}\u2019…"
                    if is_lead(cur_name)
                    else f"Rename \u2018{cur_name}\u2019 everywhere…"
                ),
                command=lambda: self._rename_speaker(current),
            )
            self._popup(menu, event)

        return handler

    def _reassign_command(self, index: int, speaker_id: str) -> Callable[[], None]:
        def command() -> None:
            self._reassign_segment(index, speaker_id)

        return command

    def _reassign_segment(self, index: int, speaker_id: str) -> None:
        result = self._result
        if not self._editable or result is None or index >= len(result.segments):
            return
        segment = result.segments[index]
        segment.speaker = speaker_id
        self._moved = (segment.start, segment.end, speaker_id)
        # This line's audio is now confirmed to be that speaker - learn from it.
        self._enroll(speaker_id, [(segment.start, segment.end)])
        self._changed()

    def _enroll(self, speaker_id: str, spans: List[Tuple[float, float]]) -> None:
        """Ask the owner to fold ``spans`` into ``speaker_id``'s voiceprint.

        A no-op unless voiceprint learning is wired up (a profile is active with
        audio available) and the speaker has a display name to learn under - an
        anonymous ``SPEAKER_xx`` id is nothing to enrol against.
        """
        if self._on_enroll is None or not spans:
            return
        name = self._speaker_names.get(speaker_id)
        if name:
            self._on_enroll(name, spans)

    def _confirm_lead(self, speaker_id: str, name: str) -> Optional[str]:
        """Ask about a lead; the name to apply, or None to leave it."""
        result = self._result
        turns = [
            seg
            for seg in (result.segments if result is not None else [])
            if (seg.speaker or "UNKNOWN") == speaker_id
        ]
        seconds = sum(max(0.0, seg.end - seg.start) for seg in turns)
        play = None
        if self._on_play is not None and turns:
            longest = max(turns, key=lambda seg: seg.end - seg.start)
            on_play = self._on_play

            def play() -> None:
                on_play(longest.start, longest.end)

        outcome = ask_confirm_lead(
            self.root, name=name, lines=len(turns), seconds=seconds, on_play=play
        )
        return outcome[1] if outcome is not None else None

    # -- Playback ----------------------------------------------------------

    def _click_handler(
        self,
        play: Callable[[], None],
        menu: Optional[Callable[[object], None]] = None,
    ) -> Callable[[object], None]:
        """A left click: play, once it is clear it was not a double-click.

        Not a click at all after a drag or a drag-select. Where nothing can be
        played, the click opens ``menu`` (the corrections) instead, as it did
        before clicking played.
        """

        def handler(event: object) -> None:
            if self._has_selection():
                return  # a drag-select; leave it for right-click "Move selection"
            if self._drag is not None and self._drag.get("moved"):
                return  # the end of a drag, handled as a drop
            if self._on_play is None:
                if menu is not None:
                    menu(event)
                return
            self._cancel_click()

            def run() -> None:
                self._click_job = None
                play()

            self._click_job = self.widget.after(_CLICK_MS, run)

        return handler

    def _cancel_click(self, _event: object = None) -> None:
        if self._click_job is not None:
            self.widget.after_cancel(self._click_job)
            self._click_job = None

    def _context_menu(self, event: object) -> Optional[str]:
        """Right-click: the corrections for what is under the pointer."""
        if self._has_selection() and self._index_in_selection(event):
            self._selection_menu(event)
            return "break"
        try:
            idx = self.widget.index(f"@{event.x},{event.y}")  # type: ignore[attr-defined]
        except tk.TclError:
            return None
        # A selection elsewhere would make the menus think this is a drag-select.
        self.widget.tag_remove("sel", "1.0", "end")
        names = self.widget.tag_names(idx)
        seg = self._segment_at(idx)
        if seg is None:
            return None
        if any(tag.startswith("spk::") for tag in names):
            self._speaker_menu_handler(seg)(event)
            return "break"
        for tag in names:
            if tag.startswith("word::"):
                try:
                    word = int(tag.split("::")[2])
                except (IndexError, ValueError):
                    continue
                self._word_menu_handler(seg, word)(event)
                return "break"
        if self._on_play is not None:
            self._line_menu_handler(seg)(event)
            return "break"
        return None

    def _line_menu_handler(self, index: int) -> Callable[[object], None]:
        """Right-click menu for a non-diarized line (just Play)."""

        def handler(event: object) -> None:
            if self._on_play is None or self._has_selection():
                return
            menu = tk.Menu(self.root, tearoff=0)
            menu.add_command(
                label="▶ Play this line", command=lambda: self._play_line(index)
            )
            self._popup(menu, event)

        return handler

    def _play_line(self, index: int) -> None:
        result = self._result
        if self._on_play is None or result is None or index >= len(result.segments):
            return
        segment = result.segments[index]
        self._on_play(segment.start, segment.end)

    def _play_from_word(self, seg_index: int, word_index: int) -> None:
        result = self._result
        if self._on_play is None or result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        if word_index >= len(segment.words):
            return
        self._on_play(segment.words[word_index].start, segment.end)

    def _word_menu_handler(
        self, seg_index: int, word_index: int
    ) -> Callable[[object], None]:
        def handler(event: object) -> None:
            if self._has_selection():
                return  # a drag-select; leave it for right-click "Move selection"
            if self._drag is not None and self._drag.get("moved"):
                return  # the end of a drag, handled as a drop
            result = self._result
            if result is None or seg_index >= len(result.segments):
                return
            segment = result.segments[seg_index]
            if word_index >= len(segment.words):
                return
            current = segment.speaker or "UNKNOWN"
            word_text = segment.words[word_index].word.strip()
            menu = tk.Menu(self.root, tearoff=0)
            if self._on_play is not None:
                menu.add_command(
                    label="▶ Play from here",
                    command=lambda: self._play_from_word(seg_index, word_index),
                )
                menu.add_separator()
            if not self._editable:
                self._add_paused_note(menu)
                self._popup(menu, event)
                return
            # Move just this word (splits the segment around it).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = shown_name(sid, self._speaker_names)
                menu.add_command(
                    label=f"Move “{word_text}” → {name}",
                    command=self._word_command(seg_index, word_index, sid, False),
                )
            menu.add_separator()
            # Reassign from this word to the end of the line (the common case: a
            # new speaker's turn actually starts mid-line).
            for sid in self._ordered_speaker_ids():
                if sid == current:
                    continue
                name = shown_name(sid, self._speaker_names)
                menu.add_command(
                    label=f"From “{word_text}” onward → {name}",
                    command=self._word_command(seg_index, word_index, sid, True),
                )
            self._popup(menu, event)

        return handler

    def _word_command(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> Callable[[], None]:
        def command() -> None:
            self._reassign_word_span(seg_index, word_index, speaker_id, to_end)

        return command

    def _reassign_word_span(
        self, seg_index: int, word_index: int, speaker_id: str, to_end: bool
    ) -> None:
        result = self._result
        if not self._editable or result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        if word_index >= len(segment.words):
            return
        last = len(segment.words) - 1 if to_end else word_index
        moved = segment.words[word_index : last + 1]
        parts = split_segment_on_word(segment, word_index, speaker_id, to_end=to_end)
        result.segments[seg_index : seg_index + 1] = parts
        result.segments = coalesce_segments(result.segments)
        if moved:
            self._moved = (moved[0].start, moved[-1].end, speaker_id)
            self._enroll(speaker_id, [(moved[0].start, moved[-1].end)])
        self._changed()

    # -- Move a highlighted span of words ----------------------------------

    def _has_selection(self) -> bool:
        """True if a non-empty text selection exists (i.e. the user drag-selected).

        A plain click clears the selection on press, so by ButtonRelease this is
        only true after a drag - letting us suppress the per-word/line menu and
        leave the selection for a right-click "Move selection".
        """
        try:
            return self.widget.index("sel.first") != self.widget.index("sel.last")
        except tk.TclError:
            return False

    def _selected_word_span(self) -> "Optional[tuple[int, int, int]]":
        """Map the current text selection to ``(seg_index, first_word, last_word)``.

        Returns ``None`` unless the selection covers a contiguous run of words
        within a single segment (the supported case: a sentence mid-line).
        """
        result = self._result
        if result is None:
            return None
        try:
            sel_first = self.widget.index("sel.first")
            sel_last = self.widget.index("sel.last")
        except tk.TclError:
            return None  # nothing selected
        hits: List[tuple[int, int]] = []
        for s_idx, segment in enumerate(result.segments):
            for w_idx in range(len(segment.words)):
                ranges = self.widget.tag_ranges(f"word::{s_idx}::{w_idx}")
                if not ranges:
                    continue
                w_start, w_end = str(ranges[0]), str(ranges[1])
                # Overlap test between the word's range and the selection.
                if self.widget.compare(w_start, "<", sel_last) and self.widget.compare(
                    w_end, ">", sel_first
                ):
                    hits.append((s_idx, w_idx))
        if not hits:
            return None
        if len({s for s, _ in hits}) != 1:
            return None  # selection spans multiple segments - not supported
        s_idx = hits[0][0]
        w_idxs = sorted(w for _, w in hits)
        if w_idxs != list(range(w_idxs[0], w_idxs[-1] + 1)):
            return None  # non-contiguous (shouldn't happen for a normal drag)
        return (s_idx, w_idxs[0], w_idxs[-1])

    def _selection_menu(self, event: object) -> None:
        if not self._editable:
            return
        result = self._result
        span = self._selected_word_span()
        if result is None or span is None:
            return
        seg_index, first_word, last_word = span
        current = result.segments[seg_index].speaker or "UNKNOWN"
        targets = [sid for sid in self._ordered_speaker_ids() if sid != current]
        if not targets:
            return
        menu = tk.Menu(self.root, tearoff=0)
        for sid in targets:
            name = shown_name(sid, self._speaker_names)
            menu.add_command(
                label=f"Move selection → {name}",
                command=self._span_command(seg_index, first_word, last_word, sid),
            )
        self._popup(menu, event)

    def _span_command(
        self, seg_index: int, first_word: int, last_word: int, speaker_id: str
    ) -> Callable[[], None]:
        def command() -> None:
            self._reassign_span(seg_index, first_word, last_word, speaker_id)

        return command

    def _reassign_span(
        self, seg_index: int, first_word: int, last_word: int, speaker_id: str
    ) -> None:
        result = self._result
        if not self._editable or result is None or seg_index >= len(result.segments):
            return
        segment = result.segments[seg_index]
        moved = segment.words[first_word : last_word + 1]
        parts = split_segment_on_span(segment, first_word, last_word, speaker_id)
        result.segments[seg_index : seg_index + 1] = parts
        result.segments = coalesce_segments(result.segments)
        if moved:
            self._moved = (moved[0].start, moved[-1].end, speaker_id)
            self._enroll(speaker_id, [(moved[0].start, moved[-1].end)])
        self._changed()

    # -- Drag a highlighted span onto another speaker ----------------------

    def _index_in_selection(self, event: object) -> bool:
        """True if the pointer is over the current text selection."""
        try:
            idx = self.widget.index(f"@{event.x},{event.y}")  # type: ignore[attr-defined]
            return self.widget.compare(idx, ">=", "sel.first") and self.widget.compare(
                idx, "<", "sel.last"
            )
        except tk.TclError:
            return False

    def _speaker_at_event(self, event: object) -> Optional[str]:
        """The speaker id of the line under the pointer, or ``None``."""
        result = self._result
        if result is None:
            return None
        try:
            idx = self.widget.index(f"@{event.x},{event.y}")  # type: ignore[attr-defined]
        except tk.TclError:
            return None
        for tag in self.widget.tag_names(idx):
            if tag.startswith("spk::"):
                return tag[len("spk::") :]
            if tag.startswith("line::") or tag.startswith("word::"):
                try:
                    seg = int(tag.split("::")[1])
                except (IndexError, ValueError):
                    continue
                if seg < len(result.segments):
                    return result.segments[seg].speaker or "UNKNOWN"
        return None

    def _on_drag_press(self, event: object) -> Optional[str]:
        """Begin a drag: of the selection if the press lands inside it, or of a
        whole line if it lands on that line's speaker tag."""
        self._drag = None
        if not self._editable:
            return None
        if self._has_selection() and self._index_in_selection(event):
            span = self._selected_word_span()
            if span is None:
                return None
            # Keep the selection (suppress the default click that would clear
            # it) so the following motion drags the span, not a new selection.
            self._drag = {"span": span, "moved": False, "from": span[0]}
            return "break"
        line = self._tag_line_at_event(event)
        if line is None:
            return None
        x, y = event.x, event.y  # type: ignore[attr-defined]
        # Not yet a drag: a press that does not travel is a click, and opens
        # the speaker menu as it always has.
        self._drag = {"line": line, "moved": False, "from": line, "x": x, "y": y}
        return "break"

    def _tag_line_at_event(self, event: object) -> Optional[int]:
        """The line whose ``[speaker]`` tag is under the pointer, if any."""
        try:
            idx = self.widget.index(f"@{event.x},{event.y}")  # type: ignore[attr-defined]
        except tk.TclError:
            return None
        names = self.widget.tag_names(idx)
        if not any(tag.startswith("spk::") for tag in names):
            return None
        return self._segment_at(idx)

    def _on_drag_motion(self, event: object) -> Optional[str]:
        drag = self._drag
        if drag is None:
            return None
        x, y = event.x, event.y  # type: ignore[attr-defined]
        if "line" in drag and not drag["moved"]:
            if max(abs(x - cast(int, drag["x"])), abs(y - cast(int, drag["y"]))) < (
                _DRAG_START
            ):
                return "break"
        drag["moved"] = True
        drag["y"] = y
        if _EDGE <= y <= self.widget.winfo_height() - _EDGE:
            # Clear of both edges once: from now on, nearing one scrolls. A
            # drag that starts on the top line must not scroll off at once.
            drag["armed"] = True
        self.widget.config(cursor="fleur")  # the four-arrow "move" cursor
        self._show_drop_target(event)
        self._autoscroll()
        return "break"  # dragging, not extending a selection

    def _show_drop_target(self, event: object) -> None:
        """Shade the line a drop would land on, when it would change something."""
        self.widget.tag_config("droptarget", background=palette().info_bg)
        self.widget.tag_remove("droptarget", "1.0", "end")
        drag, result = self._drag, self._result
        if drag is None or result is None:
            return
        try:
            idx = self.widget.index(f"@{event.x},{event.y}")  # type: ignore[attr-defined]
        except tk.TclError:
            return
        target = self._segment_at(idx)
        source = cast(int, drag["from"])
        if (
            target is None
            or target >= len(result.segments)
            or source >= len(result.segments)
        ):
            return
        if result.segments[target].speaker == result.segments[source].speaker:
            return
        self.widget.tag_add("droptarget", f"{idx} linestart", f"{idx} lineend")

    def _autoscroll(self) -> None:
        """Scroll while a drag is held near the top or bottom edge."""
        drag = self._drag
        if drag is None or self._scroll_job is not None:
            return
        y = cast(int, drag.get("y", 0))
        height = self.widget.winfo_height()
        if not drag.get("armed") and 0 <= y <= height:
            return
        step = -1 if y < _EDGE else 1 if y > height - _EDGE else 0
        if not step:
            return
        self.widget.yview_scroll(step, "units")

        def again() -> None:
            self._scroll_job = None
            self._autoscroll()

        self._scroll_job = self.widget.after(60, again)

    def _end_drag(self) -> None:
        self._drag = None
        self.widget.config(cursor="")
        self.widget.tag_remove("droptarget", "1.0", "end")
        if self._scroll_job is not None:
            self.widget.after_cancel(self._scroll_job)
            self._scroll_job = None

    def _on_drag_release(self, event: object) -> Optional[str]:
        """Drop: give the dragged words or line to the speaker under the pointer."""
        drag = self._drag
        self._end_drag()
        if drag is None or not drag.get("moved"):
            return None
        result = self._result
        target = self._speaker_at_event(event)
        if result is None or target is None:
            return None
        if "line" in drag:
            line = cast(int, drag["line"])
            if line >= len(result.segments):
                return None
            if (result.segments[line].speaker or "UNKNOWN") != target:
                self._reassign_segment(line, target)
            return "break"
        seg_index, first_word, last_word = cast("Tuple[int, int, int]", drag["span"])
        if seg_index >= len(result.segments):
            return None
        if (result.segments[seg_index].speaker or "UNKNOWN") == target:
            return None  # dropped back on the same speaker - nothing to do
        self._reassign_span(seg_index, first_word, last_word, target)
        return "break"

    def _rename_speaker(self, speaker_id: str) -> None:
        if not self._editable:
            return
        current = shown_name(speaker_id, self._speaker_names)
        if is_lead(current):
            # A voice-similarity suggestion is confirmed in its own dialog, not
            # this rename box: confirming trains the profile, so it has to be
            # a deliberate act - see sotto.gui.confirm_lead.
            new_name = self._confirm_lead(speaker_id, bare_name(current))
        else:
            new_name = simpledialog.askstring(
                "Rename speaker",
                f"New name for {current}:",
                initialvalue=current,
                parent=active_window(self.root),
            )
        if not new_name or not new_name.strip() or not self._editable:
            return
        self._speaker_names[speaker_id] = new_name.strip()
        # Naming a whole speaker confirms every one of their turns - a strong
        # enrolment signal, so learn the voice from all of this speaker's audio.
        result = self._result
        if result is not None:
            spans = [
                (seg.start, seg.end)
                for seg in result.segments
                if (seg.speaker or "UNKNOWN") == speaker_id
            ]
            self._enroll(speaker_id, spans)
        self._changed()
