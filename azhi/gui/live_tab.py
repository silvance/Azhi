"""The Live tab: near-real-time transcription of a stream or a microphone.

Point it at a stream ffmpeg can read - e.g. a GoPro pushing RTMP to a local RTMP
server - or at a microphone on this PC, and it transcribes short segments as
they arrive (see :mod:`azhi.live`). Kept deliberately simple: a source, a
model, Start/Stop, and a live-updating transcript. Optionally, who is speaking
is worked out behind the words (:mod:`azhi.live_speakers`) and the
transcript relabelled as it goes.

A microphone's audio exists nowhere else, so all of it is kept in a folder the
analyst chose, with a record of the capture beside it
(:mod:`azhi.microphone`), and the page says plainly while it is recording.
"""

from __future__ import annotations

import threading
import tkinter as tk
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import IO, Callable, List, Optional

from .. import speaker_count
from ..atomicio import write_text_atomic
from ..diarization import chosen_backend
from ..diarization import is_available as diarization_available
from ..hashing import short
from ..live import DEFAULT_SEGMENT_SECONDS, LiveTranscriber, test_connection
from ..live_speakers import SpeakerReviewer
from ..microphone import (
    Microphone,
    MicrophoneError,
    explain_capture_error,
    list_microphones,
    read_capture_record,
    recording_path,
    write_capture_record,
)
from ..resources import bundled_models, find_ffmpeg, is_bundle
from ..speaker_names import shown_name
from ..transcription import (
    MODEL_SIZES,
    ModelUnavailable,
    Segment,
    resolve_model_choice,
)
from ..watchlist import Matcher
from .dialogs import active_window
from .errors import friendly_error
from .popout import PanelWindow
from .theme import SPACE_LG, SPACE_SM, SPACE_XS, Style, theme
from .transcribe_tab import ENGINE_CHOICES, ENGINE_LABELS, engine_labels
from .watch_marks import mark
from .widgets import (
    Card,
    PageHeader,
    append_line,
    bind_wheel,
    primary_button,
    scrollable_body,
    secondary_button,
    style_text_widget,
    subtle_button,
)

_LANGUAGES = ["Auto", "en", "es", "fr", "de", "it", "pt", "ru", "ar", "zh", "ja", "ko"]


def default_recording_folder() -> Path:
    """Documents/Azhi recordings, or under the home folder without one."""
    home = Path.home()
    documents = home / "Documents"
    return (documents if documents.is_dir() else home) / "Azhi recordings"


@dataclass
class Capture:
    """A microphone session: what is being recorded, from what, since when."""

    device: str
    started: datetime
    path: Path
    model: str


def finish_capture(capture: Capture) -> str:
    """Write the record beside a finished recording; say where it is."""
    if not capture.path.is_file():
        return "Nothing was recorded."
    try:
        write_capture_record(
            capture.path,
            device=capture.device,
            started=capture.started,
            stopped=datetime.now().astimezone(),
            transcription_model=capture.model,
        )
        digest = read_capture_record(capture.path).get("sha256")
    except (OSError, ValueError) as exc:
        return (
            f"The recording is in {capture.path}, but its record could not be "
            f"written: {friendly_error(exc)}"
        )
    return (
        f"Recording kept: {capture.path.name} (SHA-256 {short(str(digest))}\u2026) "
        f"in {capture.path.parent}."
    )


class LiveTab:
    """Builds and drives the Live (stream) tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        watch: "Optional[Callable[[], Optional[Matcher]]]" = None,
    ) -> None:
        self.parent = parent
        self.root = root
        # The watch list as the Transcribe page has it, read each time the
        # transcript changes, so an edit there shows here at once.
        self._watch = watch

        self._bundled_models = bundled_models()
        default_model = "base.en"
        if self._bundled_models:
            for preferred in ("base.en", "small.en", "small"):
                if preferred in self._bundled_models:
                    default_model = preferred
                    break
            else:
                default_model = next(iter(self._bundled_models))

        self.source_kind_var = tk.StringVar(value="stream")
        self.source_var = tk.StringVar(value="rtmp://localhost/live/stream")
        self.listen_var = tk.BooleanVar(value=False)
        self.mic_var = tk.StringVar(value="")
        self.record_dir_var = tk.StringVar(value=str(default_recording_folder()))
        self.recording_var = tk.StringVar(value="")
        self.model_var = tk.StringVar(value=default_model)
        self.language_var = tk.StringVar(value="Auto")
        self.vad_var = tk.BooleanVar(value=True)
        self.segment_var = tk.StringVar(value=str(DEFAULT_SEGMENT_SECONDS))
        self.save_var = tk.BooleanVar(value=False)
        self.save_path_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Ready")
        # Who is speaking, worked out behind the words. Off by default: it is a
        # second model running beside the first, on the same processor.
        self.speakers_var = tk.BooleanVar(value=False)
        self.people_var = tk.StringVar(value="")
        self.engine_var = tk.StringVar(value=ENGINE_LABELS[0])

        self._live: Optional[LiveTranscriber] = None
        self._save_fh: Optional[IO[str]] = None
        self._reviewer: Optional[SpeakerReviewer] = None
        # With speakers on, the file is rewritten from the transcript each time
        # it changes: what it held before this session, then this session.
        self._save_path: Optional[Path] = None
        self._save_prefix = ""
        self._save_failed = False
        # Lines that ended before Clear was pressed are not shown again.
        self._hide_before = 0.0
        self._mics: List[Microphone] = []
        self._mics_listed = False
        self._capture: Optional[Capture] = None
        self._tick_job: Optional[str] = None
        # What Stop said about the recording, kept for the speakers' last word.
        self._stopped_note = ""

        self._build()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Live",
            "Transcribe a stream or a microphone as it is heard, a few seconds "
            "at a time.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        source_card = Card(container, "Source")
        source_card.pack(fill="x")
        settings = source_card.body
        settings.columnconfigure(1, weight=1)

        ttk.Label(settings, text="Listen to", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        kinds = ttk.Frame(settings, style=Style.CARD_INNER)
        kinds.grid(row=0, column=1, columnspan=2, sticky="w", pady=4)
        self._kind_buttons = []
        for value, label in (
            ("stream", "A stream"),
            ("microphone", "A microphone on this PC"),
        ):
            button = ttk.Radiobutton(
                kinds,
                text=label,
                value=value,
                variable=self.source_kind_var,
                command=self._show_source_kind,
                style=Style.RADIO,
            )
            button.pack(side="left", padx=(0, SPACE_LG))
            self._kind_buttons.append(button)

        # Rows 1-3 hold either the stream's settings or the microphone's.
        url_label = ttk.Label(settings, text="Stream URL", style=Style.FIELD_LABEL)
        url_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        url_entry = ttk.Entry(settings, textvariable=self.source_var)
        url_entry.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        listen = ttk.Checkbutton(
            settings,
            text="Wait for an incoming connection (this PC hosts the stream)",
            variable=self.listen_var,
            style=Style.CHECK,
        )
        listen.grid(row=2, column=1, columnspan=2, sticky="w")
        stream_hint = ttk.Label(
            settings,
            text=(
                "For example, a GoPro streaming to a local RTMP server. Tick "
                "the box only if the source sends straight to this PC."
            ),
            wraplength=560,
            justify="left",
            style=Style.META,
        )
        stream_hint.grid(row=3, column=1, columnspan=2, sticky="w", pady=(SPACE_XS, 0))
        self._stream_rows = [url_label, url_entry, listen, stream_hint]

        mic_label = ttk.Label(settings, text="Microphone", style=Style.FIELD_LABEL)
        mic_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=4)
        mic_row = ttk.Frame(settings, style=Style.CARD_INNER)
        mic_row.grid(row=1, column=1, columnspan=2, sticky="ew", pady=4)
        mic_row.columnconfigure(0, weight=1)
        self.mic_combo = ttk.Combobox(
            mic_row, textvariable=self.mic_var, values=[], state="readonly"
        )
        self.mic_combo.grid(row=0, column=0, sticky="ew")
        self.refresh_button = secondary_button(
            mic_row, "Refresh", self._refresh_microphones
        )
        self.refresh_button.grid(row=0, column=1, padx=(8, 0))
        folder_label = ttk.Label(
            settings, text="Keep recordings in", style=Style.FIELD_LABEL
        )
        folder_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=4)
        folder_row = ttk.Frame(settings, style=Style.CARD_INNER)
        folder_row.grid(row=2, column=1, columnspan=2, sticky="ew", pady=4)
        folder_row.columnconfigure(0, weight=1)
        self.record_dir_entry = ttk.Entry(folder_row, textvariable=self.record_dir_var)
        self.record_dir_entry.grid(row=0, column=0, sticky="ew")
        self.record_dir_button = secondary_button(
            folder_row, "Choose…", self._choose_record_dir
        )
        self.record_dir_button.grid(row=0, column=1, padx=(8, 0))
        mic_hint = ttk.Label(
            settings,
            text=(
                "Everything the microphone hears is kept in this folder as a WAV "
                "file, with a record beside it of the device, the times and the "
                "file's SHA-256. Nothing is sent anywhere."
            ),
            wraplength=560,
            justify="left",
            style=Style.META,
        )
        mic_hint.grid(row=3, column=1, columnspan=2, sticky="w", pady=(SPACE_XS, 0))
        self._mic_rows = [mic_label, mic_row, folder_label, folder_row, mic_hint]
        for widget in self._mic_rows:
            widget.grid_remove()

        model_values = (
            list(self._bundled_models) if self._bundled_models else list(MODEL_SIZES)
        )
        ttk.Label(settings, text="Model", style=Style.FIELD_LABEL).grid(
            row=4, column=0, sticky="w", padx=(0, 8), pady=4
        )
        # Pick-only wherever models are bundled, and always in a shipped copy:
        # a typed name the bundle does not hold could only be downloaded.
        locked = bool(self._bundled_models) or is_bundle()
        ttk.Combobox(
            settings,
            textvariable=self.model_var,
            values=model_values,
            width=18,
            state="readonly" if locked else "normal",
        ).grid(row=4, column=1, sticky="w", pady=4)
        ttk.Label(
            settings,
            text="Use a small model (base.en / small.en) so it keeps up with live.",
            style=Style.META,
        ).grid(row=5, column=1, columnspan=2, sticky="w")

        ttk.Label(settings, text="Language", style=Style.FIELD_LABEL).grid(
            row=6, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            settings,
            textvariable=self.language_var,
            values=_LANGUAGES,
            width=10,
        ).grid(row=6, column=1, sticky="w", pady=4)

        ttk.Label(
            settings, text="Chunk length (seconds)", style=Style.FIELD_LABEL
        ).grid(row=7, column=0, sticky="w", padx=(0, 8), pady=4)
        ttk.Entry(settings, textvariable=self.segment_var, width=8).grid(
            row=7, column=1, sticky="w", pady=4
        )
        ttk.Label(
            settings,
            text="Shorter = more live but choppier; longer = smoother but more lag.",
            style=Style.META,
        ).grid(row=8, column=1, columnspan=2, sticky="w")

        ttk.Checkbutton(
            settings,
            text="Skip silence",
            variable=self.vad_var,
            style=Style.CHECK,
        ).grid(row=9, column=1, columnspan=2, sticky="w", pady=(4, 0))

        if diarization_available():
            self._build_speaker_rows(settings, row=10)

        # Optional: append the live transcript to a file as it comes in.
        save_row = ttk.Frame(settings, style=Style.CARD_INNER)
        save_row.grid(row=14, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        save_row.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            save_row,
            text="Save to file",
            variable=self.save_var,
            command=self._update_save_state,
            style=Style.CHECK,
        ).grid(row=0, column=0, sticky="w")
        self.save_entry = ttk.Entry(save_row, textvariable=self.save_path_var)
        self.save_entry.grid(row=0, column=1, sticky="ew", padx=(8, 0))
        self.save_button = secondary_button(save_row, "Choose…", self._choose_save_path)
        self.save_button.grid(row=0, column=2, padx=(8, 0))

        # --- Controls ------------------------------------------------------
        controls = ttk.Frame(container, style=Style.PAGE)
        controls.pack(fill="x", pady=(SPACE_LG, 0))
        self.start_button = primary_button(
            controls, "Start live transcription", self._start
        )
        self.start_button.pack(side="left")
        self.stop_button = secondary_button(controls, "Stop", self._stop)
        self.stop_button.configure(state="disabled")
        self.stop_button.pack(side="left", padx=(SPACE_SM, 0))
        self.test_button = subtle_button(
            controls, "Test connection", self._test_connection
        )
        self.test_button.pack(side="left", padx=(SPACE_SM, 0))
        # Shown only while a microphone is being recorded.
        self.recording_label = ttk.Label(
            container, textvariable=self.recording_var, style=Style.RECORDING
        )
        self.status_label = ttk.Label(
            container, textvariable=self.status_var, style=Style.PAGE_SUBTITLE
        )
        self.status_label.pack(anchor="w", pady=(SPACE_SM, 0))

        # --- Live transcript ----------------------------------------------
        # Watching a feed come in is the case for a second screen even more
        # than reading a finished transcript is, so this panel pops out too.
        self._panel = PanelWindow(
            container,
            title="Live transcript",
            window_title="Azhi — Live transcript",
            note=("The live transcript is open in its own window. It keeps running."),
            pack_options={"fill": "both", "expand": True, "pady": (SPACE_LG, 0)},
            on_error=self._popout_failed,
        )
        transcript_card = self._panel.card
        self.transcript = ScrolledText(
            transcript_card.body, wrap="word", state="disabled", height=16
        )
        style_text_widget(self.transcript)
        self.transcript.configure(font=theme().mono)
        self.transcript.pack(fill="both", expand=True)

        actions = ttk.Frame(transcript_card.body, style=Style.CARD_INNER)
        actions.pack(fill="x", pady=(SPACE_SM, 0))
        secondary_button(actions, "Copy", self._copy).pack(side="left")
        subtle_button(actions, "Clear", self._clear).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        self._panel.toggle_button(actions).pack(side="right")

        self._update_save_state()
        bind_wheel(canvas, container)

    def _build_speaker_rows(self, settings: ttk.Frame, *, row: int) -> None:
        ttk.Checkbutton(
            settings,
            text="Show who is speaking",
            variable=self.speakers_var,
            style=Style.CHECK,
        ).grid(row=row, column=1, columnspan=2, sticky="w", pady=(SPACE_SM, 0))
        ttk.Label(settings, text="People speaking", style=Style.FIELD_LABEL).grid(
            row=row + 1, column=0, sticky="w", padx=(0, 8), pady=4
        )
        people = ttk.Frame(settings, style=Style.CARD_INNER)
        people.grid(row=row + 1, column=1, columnspan=2, sticky="w")
        ttk.Entry(people, textvariable=self.people_var, width=6).pack(side="left")
        ttk.Label(
            people, text="leave empty to let it work it out", style=Style.META
        ).pack(side="left", padx=(SPACE_SM, 0))
        ttk.Label(settings, text="Method", style=Style.FIELD_LABEL).grid(
            row=row + 2, column=0, sticky="w", padx=(0, 8), pady=4
        )
        ttk.Combobox(
            settings,
            textvariable=self.engine_var,
            values=engine_labels(),
            state="readonly",
            width=34,
        ).grid(row=row + 2, column=1, sticky="w", pady=4)
        ttk.Label(
            settings,
            text=(
                "Speaker labels follow a little behind the words, and an "
                "earlier line can change speaker as more of each voice is "
                "heard. Lines marked [\u2026] have no speaker yet. On a slow "
                "machine, both fall further behind."
            ),
            wraplength=560,
            justify="left",
            style=Style.META,
        ).grid(row=row + 3, column=1, columnspan=2, sticky="w", pady=(SPACE_XS, 0))

    # -- Source: a stream or a microphone ------------------------------------

    @property
    def microphone_chosen(self) -> bool:
        return self.source_kind_var.get() == "microphone"

    def _show_source_kind(self) -> None:
        microphone = self.microphone_chosen
        for widget in self._stream_rows:
            if microphone:
                widget.grid_remove()
            else:
                widget.grid()
        for widget in self._mic_rows:
            if microphone:
                widget.grid()
            else:
                widget.grid_remove()
        if microphone:
            self.test_button.pack_forget()
            if not self._mics_listed:
                self._refresh_microphones()
        else:
            self.test_button.pack(
                side="left", padx=(SPACE_SM, 0), after=self.stop_button
            )

    def _refresh_microphones(self) -> None:
        """Ask ffmpeg what is plugged in, off the UI thread."""
        ffmpeg = find_ffmpeg()
        if ffmpeg is None:
            self.status_var.set("ffmpeg was not found, so microphones can't be listed.")
            return
        self.refresh_button.configure(state="disabled")
        self.status_var.set("Looking for microphones…")

        def _worker() -> None:
            problem: Optional[str] = None
            try:
                found = list_microphones(ffmpeg)
            except MicrophoneError as exc:
                found, problem = [], str(exc)
            self.root.after(0, lambda: self._show_microphones(found, problem))

        threading.Thread(target=_worker, daemon=True).start()

    def _show_microphones(
        self, found: List[Microphone], problem: Optional[str] = None
    ) -> None:
        self._mics = found
        self._mics_listed = True
        if self._live is None:
            self.refresh_button.configure(state="normal")
        names = [mic.name for mic in found]
        self.mic_combo.configure(values=names)
        if self.mic_var.get() not in names:
            self.mic_var.set(names[0] if names else "")
        if problem:
            self.status_var.set(problem)
        elif not names:
            self.status_var.set("No microphone found. Plug one in, then press Refresh.")
        else:
            self.status_var.set(
                "Found 1 microphone."
                if len(names) == 1
                else f"Found {len(names)} microphones."
            )

    def _choose_record_dir(self) -> None:
        folder = filedialog.askdirectory(
            title="Keep microphone recordings in",
            initialdir=self.record_dir_var.get() or str(Path.home()),
            parent=active_window(self.root),
        )
        if folder:
            self.record_dir_var.set(folder)

    def _set_source_locked(self, locked: bool) -> None:
        """While a session runs its source can't change under it."""
        state = "disabled" if locked else "normal"
        for button in self._kind_buttons:
            button.configure(state=state)
        self.mic_combo.configure(state="disabled" if locked else "readonly")
        for widget in (
            self.refresh_button,
            self.record_dir_entry,
            self.record_dir_button,
        ):
            widget.configure(state=state)

    def _tick(self) -> None:
        """Keep the recording line true: device and time, or that it ended."""
        if self._tick_job is not None:
            self.root.after_cancel(self._tick_job)
            self._tick_job = None
        capture, live = self._capture, self._live
        if capture is None:
            self.recording_var.set("")
            self.recording_label.pack_forget()
            return
        if live is not None and not live.running:
            self.recording_var.set(
                "\u25a0 Recording ended: the microphone closed. Press Stop to finish."
            )
            return
        seconds = int((datetime.now().astimezone() - capture.started).total_seconds())
        minutes, secs = divmod(seconds, 60)
        hours, minutes = divmod(minutes, 60)
        self.recording_var.set(
            f"\u25cf Recording from {capture.device}   {hours}:{minutes:02d}:{secs:02d}"
        )
        self._tick_job = self.root.after(1000, self._tick)

    def _popout_failed(self, detail: str) -> None:
        """Tk refused to give the panel a window. Say so where the eye is."""
        self.status_var.set(f"Could not move the live transcript window: {detail}")

    # -- Save-to-file state ------------------------------------------------

    def _update_save_state(self) -> None:
        state = "normal" if self.save_var.get() else "disabled"
        self.save_entry.configure(state=state)
        self.save_button.configure(state=state)

    def _choose_save_path(self) -> None:
        path = filedialog.asksaveasfilename(
            title="Save live transcript",
            defaultextension=".txt",
            initialfile="live-transcript.txt",
            filetypes=[("Text file", "*.txt"), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if path:
            self.save_path_var.set(path)

    # -- Run/stop ----------------------------------------------------------

    def _parse_segment_seconds(self) -> int:
        try:
            return max(2, int(self.segment_var.get().strip()))
        except ValueError:
            return DEFAULT_SEGMENT_SECONDS

    def _test_connection(self) -> None:
        """Probe the stream for a few seconds and report if it's reachable."""
        source = self.source_var.get().strip()
        if not source:
            self.status_var.set("Enter a stream URL first.")
            return
        listen = self.listen_var.get()
        self.test_button.configure(state="disabled")
        self.status_var.set("Testing connection…")

        def _worker() -> None:
            ok, message = test_connection(source, listen=listen)

            def _apply() -> None:
                self.status_var.set(("✓ " if ok else "✗ ") + message)
                # Re-enable only if no session started in the meantime.
                if self._live is None:
                    self.test_button.configure(state="normal")

            self.root.after(0, _apply)

        threading.Thread(target=_worker, daemon=True).start()

    def _start(self) -> None:
        source = self.source_var.get().strip()
        mic: Optional[Microphone] = None
        if self.microphone_chosen:
            mic = next((m for m in self._mics if m.name == self.mic_var.get()), None)
            if mic is None:
                self.status_var.set(
                    "Pick a microphone first. Press Refresh to look again."
                )
                return
        elif not source:
            self.status_var.set("Enter a stream URL first.")
            return
        language = self.language_var.get().strip()
        language_arg = None if language in ("", "Auto") else language
        try:
            # The same policy as the Transcribe page, so the two cannot differ
            # about what this copy may load.
            model = resolve_model_choice(self.model_var.get(), self._bundled_models)
        except ModelUnavailable as exc:
            self.status_var.set(str(exc))
            return

        speakers = self.speakers_var.get() and diarization_available()
        people: Optional[int] = None
        if speakers and self.people_var.get().strip():
            people = speaker_count.parse(self.people_var.get().strip())
            if people is None:
                self.status_var.set(
                    "People speaking should be a number from 1 to "
                    f"{speaker_count.MAX_SPEAKERS}, or empty to work it out."
                )
                return

        # A microphone is always recorded: make room for it before anything
        # else is opened, so a folder that can't be used stops nothing midway.
        started = datetime.now().astimezone()
        record_to: Optional[Path] = None
        if mic is not None:
            folder = self.record_dir_var.get().strip()
            if not folder:
                self.status_var.set("Choose a folder to keep the recording in.")
                return
            try:
                Path(folder).mkdir(parents=True, exist_ok=True)
                record_to = recording_path(folder, started)
            except OSError as exc:
                self.status_var.set(
                    f"The recording folder can't be used: {friendly_error(exc)}"
                )
                return

        # Open the save file (append) if requested.
        self._save_fh = None
        self._save_path = None
        self._save_failed = False
        if self.save_var.get() and self.save_path_var.get().strip():
            path = Path(self.save_path_var.get().strip())
            try:
                if speakers:
                    # Rewritten as labels change; what was there is kept above.
                    self._save_prefix = (
                        path.read_text(encoding="utf-8") if path.is_file() else ""
                    )
                    self._save_path = path
                else:
                    self._save_fh = open(path, "a", encoding="utf-8")
            except (OSError, UnicodeDecodeError) as exc:
                self.status_var.set(friendly_error(exc))
                return

        if self._reviewer is not None:
            # The last session's final check is still going: this one's lines
            # must not be drawn over by it.
            self._reviewer.cancel()
            self._reviewer = None
        reviewer: Optional[SpeakerReviewer] = None
        if speakers:
            self._hide_before = 0.0
            reviewer = SpeakerReviewer(
                on_change=self._on_speakers_change,
                backend=chosen_backend(
                    ENGINE_CHOICES.get(self.engine_var.get(), "auto"), people
                ),
                num_speakers=people,
                on_status=self._on_status,
                on_error=self._on_speaker_error,
                on_finished=self._on_speakers_finished,
            )
            self._reviewer = reviewer

        self._live = LiveTranscriber(
            source,
            model_size=model,
            language=language_arg,
            segment_seconds=self._parse_segment_seconds(),
            vad_filter=self.vad_var.get(),
            listen=self.listen_var.get(),
            speakers=reviewer,
            input_args=mic.input_args if mic is not None else None,
            record_to=record_to,
            watch_silence=mic is not None,
        )
        if mic is not None and record_to is not None:
            # Before start: ffmpeg's first complaint is about this microphone.
            self._capture = Capture(mic.name, started, record_to, self.model_var.get())
        try:
            self._live.start(
                on_text=self._on_text,
                on_status=self._on_status,
                on_error=self._on_error,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            self.status_var.set(friendly_error(exc))
            self._close_save_fh()
            self._live = None
            self._capture = None
            if reviewer is not None:
                reviewer.cancel()
                self._reviewer = None
            return
        self.start_button.configure(state="disabled")
        self.test_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._set_source_locked(True)
        self._stopped_note = ""
        self.status_var.set("Listening…")
        if self._capture is not None:
            self.recording_label.pack(
                anchor="w", pady=(SPACE_SM, 0), before=self.status_label
            )
            self._tick()

    def _stop(self) -> None:
        # stop() blocks (it joins the ffmpeg process + threads), so run it off the
        # UI thread and re-enable the controls when it returns.
        live = self._live
        if live is None:
            return
        self.stop_button.configure(state="disabled")
        self.status_var.set("Stopping…")
        capture = self._capture

        def _worker() -> None:
            note = ""
            try:
                live.stop()
            finally:
                # Hashing a long recording takes a moment: not on the UI thread.
                if capture is not None:
                    note = finish_capture(capture)
                self.root.after(0, lambda: self._on_stopped(note))

        threading.Thread(target=_worker, daemon=True).start()

    def _on_stopped(self, note: str = "") -> None:
        self._live = None
        self._capture = None
        self._stopped_note = note
        self._tick()
        self._close_save_fh()
        self._set_source_locked(False)
        self.start_button.configure(state="normal")
        self.test_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        speakers = " Checking the speakers once more…" if self._reviewer else ""
        self.status_var.set(" ".join(filter(None, ["Stopped.", note])) + speakers)

    # -- Callbacks (from background threads) -------------------------------

    def _on_text(self, text: str) -> None:
        if self._reviewer is not None:
            return  # drawn, and saved, from the speaker reviewer's lines
        append_line(self.transcript, text)
        # After the line is in: append_line draws on the UI thread, in order.
        self.root.after(0, self.refresh_watch)
        fh = self._save_fh
        if fh is not None:
            try:
                fh.write(text + "\n")
                fh.flush()
            except OSError:
                pass

    def _on_status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    def _on_error(self, exc: Exception) -> None:
        explained = explain_capture_error(str(exc)) if self._capture else None
        message = explained or friendly_error(exc)
        self.root.after(0, lambda: self.status_var.set(message))

    def _on_speaker_error(self, exc: Exception) -> None:
        message = (
            f"Could not work out who is speaking: {friendly_error(exc)} "
            "The words are unaffected."
        )
        self.root.after(0, lambda: self.status_var.set(message))

    def _on_speakers_finished(self) -> None:
        reviewer = self._reviewer

        def _done() -> None:
            if reviewer is not None and reviewer is self._reviewer:
                self._reviewer = None
                if self._live is None:
                    said = [
                        "Stopped.",
                        self._stopped_note,
                        "Speakers checked to the end.",
                    ]
                    self.status_var.set(" ".join(filter(None, said)))

        self.root.after(0, _done)

    def _on_speakers_change(
        self, labelled: List[Segment], until: float, pending: List[Segment]
    ) -> None:
        # Called with the reviewer's lock held: hand on, draw on the UI thread.
        reviewer = self._reviewer
        self.root.after(0, lambda: self._show_speakers(reviewer, labelled, pending))

    def _show_speakers(
        self,
        reviewer: Optional[SpeakerReviewer],
        labelled: List[Segment],
        pending: List[Segment],
    ) -> None:
        if reviewer is None or reviewer is not self._reviewer:
            return  # from a session that has since been replaced
        text = speaker_text(labelled, pending, hide_before=self._hide_before)
        widget = self.transcript
        at_end = widget.yview()[1] >= 0.999
        top = widget.yview()[0]
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")
        if at_end:
            widget.see("end")
        else:
            widget.yview_moveto(top)
        self.refresh_watch()
        self._save_speaker_text(text)

    def _save_speaker_text(self, text: str) -> None:
        if self._save_path is None or self._save_failed:
            return
        try:
            write_text_atomic(self._save_path, self._save_prefix + text)
        except OSError as exc:
            # Said once; the transcript on screen carries on regardless.
            self._save_failed = True
            self.status_var.set(
                f"The live transcript is not being saved: {friendly_error(exc)}"
            )

    def _close_save_fh(self) -> None:
        if self._save_fh is not None:
            try:
                self._save_fh.close()
            except OSError:
                pass
            self._save_fh = None

    def refresh_watch(self) -> None:
        """Highlight the watch list in the live transcript as it now stands."""
        mark(self.transcript, self._watch() if self._watch is not None else None)

    # -- Transcript actions ------------------------------------------------

    def _copy(self) -> None:
        text = self.transcript.get("1.0", "end-1c")
        if not text.strip():
            self.status_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.status_var.set("Transcript copied to clipboard.")

    def _clear(self) -> None:
        if self._reviewer is not None:
            self._hide_before = self._reviewer.seconds
        self.transcript.configure(state="normal")
        self.transcript.delete("1.0", "end")
        self.transcript.configure(state="disabled")

    # -- Shutdown hook -----------------------------------------------------

    def close(self) -> None:
        """Release resources when the app closes."""
        live = self._live
        if live is not None:
            try:
                live.stop()
            except Exception:  # noqa: BLE001 - never block closing
                pass
        if self._capture is not None:
            try:
                finish_capture(self._capture)  # the record, even on the way out
            except Exception:  # noqa: BLE001 - never block closing
                pass
            self._capture = None
        if self._reviewer is not None:
            self._reviewer.cancel()
        self._close_save_fh()

    # Kept for API parity with the other tabs (no persisted settings here).
    def get_settings(self) -> "dict[str, object]":
        return {}

    def apply_settings(self, data: "dict[str, object]") -> None:
        return None


def speaker_text(
    labelled: List[Segment], pending: List[Segment], *, hide_before: float = 0.0
) -> str:
    """The live transcript: lines given to speakers, then lines not yet."""
    lines = []
    for segment in labelled:
        if segment.end > hide_before and segment.text.strip():
            lines.append(f"[{_speaker_label(segment.speaker)}] {segment.text.strip()}")
    for segment in pending:
        if segment.end > hide_before and segment.text.strip():
            lines.append(f"[\u2026] {segment.text.strip()}")
    return "\n".join(lines) + ("\n" if lines else "")


def _speaker_label(speaker: Optional[str]) -> str:
    return shown_name(speaker)
