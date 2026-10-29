"""The Azhi application shell: window, header, tabs, and entry point.

Assembles the tabs this build can actually support: Transcribe always, then Speaker profiles and Speaker Compare when a
speaker-embedding model is bundled, Live when ffmpeg is present, and Translate
when the translation packs are. ``AZHI_MODE=transcribe`` forces the lean
single-purpose window. ``azhi.app`` re-exports :func:`main` so the console
script, ``python -m azhi`` and the PyInstaller bundle entry keep working
unchanged.
"""

from __future__ import annotations

import importlib.util
import tkinter as tk
from functools import partial
from pathlib import Path
from tkinter import messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Dict, List, Tuple

from ..diagnostics import format_report
from ..ocr import ocr_available
from ..resources import (
    bundled_argos_data_dir,
    bundled_embedding_model,
    configure_offline_hf_cache,
    configure_offline_ocr,
    configure_offline_translation,
    env_setting,
    find_ffmpeg,
)
from ..settings import SettingsError, load_settings, update_settings
from .comparison_log_tab import ComparisonLogTab
from .dialogs import active_window, present
from .live_tab import LiveTab
from .speaker_compare_tab import SpeakerCompareTab
from .speaker_profiles_tab import SpeakerProfilesTab
from .theme import SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, init_theme
from .transcribe_tab import TranscribeTab
from .translate_tab import TranslateTab
from .widgets import style_text_widget, subtle_button


def _translation_available() -> bool:
    """True if the translation engine is usable (bundled packs or installed lib).

    Lets a lean transcriber-only build (no argostranslate / no packs) present as a
    single-purpose app, without importing the heavy library just to check.
    """
    if bundled_argos_data_dir() is not None:
        return True
    try:
        return importlib.util.find_spec("argostranslate") is not None
    except ModuleNotFoundError:
        return False


class AzhiApp:
    """The main application window."""

    def __init__(
        self,
        root: tk.Tk,
        *,
        drag_and_drop: bool = False,
        bootstrap: bool = False,
    ) -> None:
        self.root = root
        self.root.title("Azhi")
        # Whether a ttkbootstrap theme is underneath; the styles adapt but never
        # depend on it (see the stock-Tk fallback in main()).
        self._bootstrap = bootstrap
        # Whether tkdnd was loaded (see _enable_drag_and_drop) so tabs can register
        # file-drop targets.
        self._dnd_ok = drag_and_drop
        self._tabs: list = []
        # Remembered preferences from the last session (best-effort).
        self._settings = load_settings()
        self._build_ui()
        # Save preferences on close.
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_ui(self) -> None:
        # 1366x768 is the small end of what these machines have, so the shell has
        # to be usable there: a compact rail, not a wide sidebar.
        self.root.minsize(940, 600)
        self._size_window(1180, 760)
        self.theme = init_theme(self.root, bootstrap=self._bootstrap)

        # Translation and live capture appear only when this build supports them.
        # AZHI_MODE=transcribe forces the lean single-purpose window.
        full_mode = (env_setting("AZHI_MODE") or "").lower() != "transcribe"
        show_translate = _translation_available() and full_mode
        # Live (stream) transcription needs ffmpeg to read the stream.
        show_live = find_ffmpeg() is not None and full_mode
        # Speaker enrolment/comparison need the bundled speaker-embedding model.
        show_speakers = bundled_embedding_model() is not None and full_mode

        self._build_header()

        body = ttk.Frame(self.root, style=Style.PAGE)
        body.pack(fill="both", expand=True)
        self._nav_rail = ttk.Frame(
            body, style=Style.SIDEBAR, padding=(SPACE_SM, SPACE_MD)
        )
        self._pages_host = ttk.Frame(body, style=Style.PAGE)
        self._pages_host.pack(side="right", fill="both", expand=True)
        self._pages_host.rowconfigure(0, weight=1)
        self._pages_host.columnconfigure(0, weight=1)

        # Pages are stacked in one cell and raised, rather than carried by a
        # Notebook: the same behaviour without the tab strip, and each screen
        # still receives a plain frame exactly as before.
        self._pages: Dict[str, ttk.Frame] = {}
        self._nav_buttons: Dict[str, ttk.Button] = {}
        self._current_page = ""

        transcribe_root = self._add_page("transcribe", "Transcribe")
        self.transcribe = TranscribeTab(
            transcribe_root,
            self.root,
            dnd_ok=self._dnd_ok,
        )
        self.transcribe.apply_settings(self._settings.get("transcribe", {}))
        self._tabs.append(self.transcribe)

        # Known-subject reference voices, and comparing a questioned speaker
        # against them. Both need the speaker-embedding model, so they appear
        # only when this build actually bundles one.
        if show_speakers:
            profiles_root = self._add_page("speakers", "Speaker Profiles")
            self.speaker_profiles = SpeakerProfilesTab(profiles_root, self.root)
            self._tabs.append(self.speaker_profiles)

            compare_root = self._add_page("compare", "Compare Speakers")
            self.speaker_compare = SpeakerCompareTab(
                compare_root,
                self.root,
                # Lets a comparison report carry the current transcript and the
                # provenance of the run that produced it.
                get_analysis=self.transcribe.current_analysis,
                dnd_ok=self._dnd_ok,
            )
            self._tabs.append(self.speaker_compare)

            # The record of what was actually run. It sits after the pages that
            # produce it, and refreshes when one of them writes an entry.
            history_root = self._add_page("history", "Comparison History")
            self.comparison_log = ComparisonLogTab(history_root, self.root)
            self._tabs.append(self.comparison_log)
            self.speaker_compare.on_history_changed = self.comparison_log.refresh
            # Any page that changes a subject tells all of them. Only
            # Transcribe -> Speaker Profiles was wired, so Compare went on
            # listing a subject deleted elsewhere and comparing against it from
            # memory, and Speaker Profiles could save an old copy over samples
            # a transcript correction had just proposed.
            self.transcribe.on_profiles_changed = self._profiles_changed
            self.speaker_profiles.on_profiles_changed = self._profiles_changed
            self.transcribe.on_compare_speaker = self._compare_transcript_speaker

        if show_live:
            live_root = self._add_page("live", "Live")
            self.live = LiveTab(
                live_root, self.root, watch=lambda: self.transcribe.watch_matcher
            )
            self.transcribe.on_watch_list_changed = self.live.refresh_watch
            self._tabs.append(self.live)

        if show_translate:
            translate_root = self._add_page("translate", "Translate")
            self.translate = TranslateTab(
                translate_root,
                self.root,
                ocr_available=ocr_available(),
                detect_available=importlib.util.find_spec("langdetect") is not None,
                dnd_ok=self._dnd_ok,
            )
            self._tabs.append(self.translate)

        # A single-purpose build has nowhere to navigate to, so the rail would be
        # chrome for its own sake.
        if len(self._pages) > 1:
            self._nav_rail.pack(side="left", fill="y")
        self.show_page("transcribe")

    def _size_window(self, width: int, height: int) -> None:
        """Open at a comfortable size, but never larger than the screen.

        A fixed 780 is taller than a 1366x768 laptop once the title bar and
        taskbar are taken out, which would put the primary action off the
        bottom on the smallest machine this has to run on. The page scrolls
        either way, so there is nothing to gain by asking for height that is
        not there.
        """
        try:
            available_width = self.root.winfo_screenwidth()
            available_height = self.root.winfo_screenheight()
        except tk.TclError:  # pragma: no cover - no display metrics
            available_width, available_height = width, height
        # Leave room for the window chrome and a taskbar.
        width = max(900, min(width, available_width - 80))
        height = max(560, min(height, available_height - 120))
        try:
            self.root.geometry(f"{width}x{height}")
        except tk.TclError:  # pragma: no cover - a toolkit that refuses it
            pass

    def _build_header(self) -> None:
        """Identity, posture, and the two things that are not part of a workflow."""
        colors = self.theme.palette
        header = ttk.Frame(self.root, style=Style.HEADER, padding=(SPACE_XL, SPACE_MD))
        header.pack(fill="x")

        titles = ttk.Frame(header, style=Style.HEADER)
        titles.pack(side="left")
        ttk.Label(titles, text="Azhi", style=Style.APP_TITLE).pack(anchor="w")
        ttk.Label(titles, text="Offline audio analysis", style=Style.APP_SUBTITLE).pack(
            anchor="w"
        )

        subtle_button(header, "System status", self._show_diagnostics).pack(
            side="right", padx=(SPACE_SM, 0)
        )
        subtle_button(header, "Help", self._show_help).pack(side="right")

        # What is actually guaranteed: the analysis happens here, on this
        # machine, with no cloud service behind it. Not "offline" - the Live
        # page connects to a stream an operator supplies, and a badge that
        # contradicts a feature of the same application is worse than no badge.
        posture = ttk.Frame(header, style=Style.HEADER)
        posture.pack(side="right", padx=(0, SPACE_XL))
        ttk.Label(posture, text="●", style=Style.SUCCESS).pack(side="left")
        ttk.Label(posture, text="Local processing", style=Style.MUTED).pack(
            side="left", padx=(SPACE_XS, 0)
        )
        ttk.Frame(self.root, height=1, style=Style.PAGE).pack(fill="x")
        ttk.Separator(self.root, orient="horizontal").pack(fill="x")
        self._header_colors = colors

    def _add_page(self, key: str, label: str) -> ttk.Frame:
        """Register a navigable page and return the frame its screen builds into."""
        frame = ttk.Frame(self._pages_host, style=Style.PAGE)
        frame.grid(row=0, column=0, sticky="nsew")
        self._pages[key] = frame
        # functools.partial rather than a default-argument lambda: the closure
        # must capture this key, not whatever the loop variable ends up as.
        button = ttk.Button(
            self._nav_rail,
            text=label,
            style=Style.NAV,
            width=16,
            command=partial(self.show_page, key),
        )
        button.pack(fill="x", pady=(0, SPACE_XS))
        self._nav_buttons[key] = button
        return frame

    def _compare_transcript_speaker(
        self, source: Path, label: str, spans: List[Tuple[float, float]]
    ) -> None:
        """A speaker clicked in the transcript, searched against every profile."""
        self.show_page("compare")
        self.speaker_compare.search_transcript_speaker(source, label, spans)

    def show_page(self, key: str) -> None:
        """Raise one page and mark its rail entry as the current one."""
        if key not in self._pages:
            return
        self._pages[key].tkraise()
        self._current_page = key
        for name, button in self._nav_buttons.items():
            button.configure(style=Style.NAV_ACTIVE if name == key else Style.NAV)

    def _show_help(self) -> None:
        """Open a short, plain-language guide for first-time / non-technical users."""
        guide = (
            "Azhi — quick start\n"
            "==================\n"
            "\n"
            "Azhi is the Navajo word for voice. The name honours the Navajo\n"
            "Code Talkers.\n"
            "\n"
            "Transcribe a recording\n"
            "----------------------\n"
            "1. On the Transcribe page, drop an audio or video file onto the\n"
            "   Recording box, or select Choose file.\n"
            "2. Set anything you need under Options: the language, how thorough\n"
            "   to be, whether to identify who is speaking, and whether to save\n"
            "   a copy to a folder.\n"
            "3. Select Transcribe recording. The text appears below when it is\n"
            "   finished.\n"
            "\n"
            "Tips\n"
            "----\n"
            "• A more thorough setting is more accurate but slower. Start with\n"
            "   the default and move up if the words are not right.\n"
            "• “Expected words” (under Advanced options): type names, places or\n"
            "   callsigns you expect, so they are spelled correctly.\n"
            "• Copy transcript and Save as… (Word, spreadsheet or text) are\n"
            "   below the transcript.\n"
            "• Right-click a line › Add a note… to write down what you make\n"
            "   of it; notes are saved with the project, labelled as notes.\n"
            "• Watch › Edit list…: names or places to pick out of every\n"
            "   transcript, highlighted wherever they are said.\n"
            "\n"
            "Who said what\n"
            "-------------\n"
            "• Tick “Identify who is speaking” before you start.\n"
            "• In the transcript you can rename a speaker, or drag a run of\n"
            "   words onto another speaker's line to move it.\n"
            "\n"
            "Known people\n"
            "------------\n"
            "• Speaker Profiles builds a reference voice for someone you know,\n"
            "   from recordings you already have.\n"
            "• Once you have corrected the speaker tags on a transcript, Save\n"
            "   speaker to profile… (under the transcript) adds that speaker's\n"
            "   audio to a subject. It is held pending review until you approve\n"
            "   it on Speaker Profiles.\n"
            "• Comparison History is the record of every comparison this copy\n"
            "   has run: the recording, the speech measured, the score, and what\n"
            "   the reference profile held at the time.\n"
            "• Compare Speakers measures how similar a speaker in a questioned\n"
            "   recording is to one of those references. The result is an\n"
            "   investigative lead for review, never an identification.\n"
            "\n"
            "If something goes wrong\n"
            "-----------------------\n"
            "• The message on the page says what happened in plain language;\n"
            "   the Status tab holds the technical detail.\n"
            "• System status (top right) lists exactly what this copy can do,\n"
            "   and says READY or NOT READY before an operation.\n"
            "• Azhi analyses everything on this machine. It needs no\n"
            "   internet connection and uses no cloud service, and nothing you\n"
            "   load (recordings, transcripts or voices) leaves the computer.\n"
            "• The one exception is the Live page, which connects to a stream\n"
            "   address you enter yourself. It does nothing unless you use it.\n"
        )
        self._text_window("Azhi — Help", guide, wrap="word", width=74)

    def _show_diagnostics(self) -> None:
        """Open a window listing which engines/models this build actually has."""
        report = format_report()
        self._text_window(
            "Azhi — system status",
            report,
            # Wrapped: with "none" and no sideways scrollbar, the "Missing:"
            # line and the network statement - the longest lines - were cut.
            wrap="word",
            width=78,
            monospace=True,
            copyable=True,
        )

    def _text_window(
        self,
        title: str,
        body: str,
        *,
        wrap: str = "word",
        width: int = 74,
        monospace: bool = False,
        copyable: bool = False,
    ) -> None:
        """A themed read-only window. Escape closes it; Close takes focus."""
        window = tk.Toplevel(self.root)
        window.title(title)
        window.configure(background=self.theme.palette.background)
        frame = ttk.Frame(window, style=Style.PAGE, padding=SPACE_XL)
        frame.pack(fill="both", expand=True)
        text = ScrolledText(
            frame,
            wrap=wrap,
            width=width,
            height=24,
            font=self.theme.mono if monospace else self.theme.body,
        )
        style_text_widget(text)
        text.pack(fill="both", expand=True)
        text.insert("end", body)
        text.configure(state="disabled")

        buttons = ttk.Frame(frame, style=Style.PAGE)
        buttons.pack(fill="x", pady=(SPACE_MD, 0))
        close = ttk.Button(
            buttons, text="Close", command=window.destroy, style=Style.SECONDARY
        )
        close.pack(side="right")
        if copyable:

            def _copy() -> None:
                self.root.clipboard_clear()
                self.root.clipboard_append(body)

            ttk.Button(buttons, text="Copy", command=_copy, style=Style.SECONDARY).pack(
                side="right", padx=(0, SPACE_SM)
            )
        present(window, self.root)
        close.focus_set()

    def _profiles_changed(self) -> None:
        """Reload every page that shows subject profiles."""
        for name in ("speaker_profiles", "speaker_compare", "comparison_log"):
            page = getattr(self, name, None)
            if page is not None:
                page.refresh()

    def _on_close(self) -> None:
        """Persist preferences, then close the window."""
        # Closing used to end a ninety-minute run, or discard corrections never
        # saved as a project, without a word.
        losses: List[str] = []
        transcribing = self.transcribe.busy_description()
        if transcribing:
            losses.append(transcribing)
        for name, loss in (
            ("translate", "A translation is still running and will be lost."),
            ("speaker_compare", "A speaker comparison is still running."),
            ("speaker_profiles", "A recording is still being added to a profile."),
        ):
            page = getattr(self, name, None)
            if page is not None and page.is_busy():
                losses.append(loss)
        if losses and not messagebox.askyesno(
            "Close Azhi?",
            "\n\n".join(losses) + "\n\nClose anyway?",
            icon="warning",
            default="no",
            parent=active_window(self.root),
        ):
            return
        try:
            saved = update_settings("transcribe", self.transcribe.get_settings())
            if saved.set_aside is not None:
                # Said once: the file is gone from the way, so the next close
                # is quiet.
                messagebox.showinfo(
                    "Settings file replaced",
                    f"The settings file could not be read, so it was kept as "
                    f"{saved.set_aside.name} beside the new one. Anything an "
                    "administrator had set in it - comparison thresholds, for "
                    "example - is in that copy, and is not in use until it is "
                    "restored.",
                    parent=active_window(self.root),
                )
        except SettingsError as exc:
            # Said, not swallowed - but closing still goes ahead: losing the
            # remembered options is not worth keeping someone in the app.
            messagebox.showwarning(
                "Settings not saved", str(exc), parent=active_window(self.root)
            )
        except Exception:  # noqa: BLE001 - never block closing on anything else
            pass
        # Release any tab resources (e.g. a running live ffmpeg session).
        for tab in self._tabs:
            close = getattr(tab, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:  # noqa: BLE001 - never block closing on cleanup
                    pass
        self.root.destroy()


def _enable_drag_and_drop(root: tk.Misc) -> bool:
    """Initialise tkinterdnd2's tkdnd on ``root``; return True on success.

    We load tkdnd onto the existing (ttkbootstrap-themed) root rather than using
    ``TkinterDnD.Tk()`` so the theme is preserved. Individual widgets opt in via
    :func:`azhi.gui.widgets.register_drop`.
    """
    try:
        from tkinterdnd2 import TkinterDnD

        TkinterDnD._require(root)
        return True
    except Exception:  # noqa: BLE001 - DnD is a convenience; never block startup
        return False


def main() -> None:
    """Launch the Azhi GUI."""
    # Must happen before faster-whisper / pyannote / argostranslate are imported
    # (which only occurs once a job runs), so configuring at startup is early
    # enough. Points HF, Argos and Tesseract at the bundled offline caches when
    # present; all are no-ops otherwise.
    configure_offline_hf_cache()
    configure_offline_translation()
    configure_offline_ocr()

    bootstrap = True
    try:
        # ttkbootstrap gives a modern theme; fall back to stock Tk if absent.
        import ttkbootstrap as tb

        root = tb.Window(themename="darkly")
    except ImportError:
        root = tk.Tk()
        bootstrap = False

    # Best-effort: load the tkdnd extension so files can be dragged onto the
    # window. No-op (and the app works normally) when tkinterdnd2 isn't bundled.
    dnd_ok = _enable_drag_and_drop(root)

    AzhiApp(root, drag_and_drop=dnd_ok, bootstrap=bootstrap)
    root.mainloop()


if __name__ == "__main__":
    main()
