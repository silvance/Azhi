"""The Speaker Compare tab: one questioned speaker against a known subject.

A first-class workflow rather than a small dialog over two JSON files:

* **Reference** - pick a known subject; the tab shows the evidence behind that
  profile (sample count, total reference speech, embedding model and hash,
  quality warnings) so an operator can see what they are comparing against.
* **Questioned** - pick a new recording and say which speech to compare: the
  whole file, a diarized speaker, or explicit time ranges.
* **Result** - a similarity score out of 1.00, an assessment band, the
  operational threshold, both durations, both quality bands, model
  compatibility, and the investigative-only disclaimer.

Nothing here reports a percentage or a probability of identity, and an
insufficient or mismatched input produces an explicit refusal or
"Insufficient data" rather than a number that looks conclusive.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Any, Callable, List, Optional, Tuple

from .. import listening
from ..comparison_log import (
    ProfileError as _LogError,
)
from ..comparison_log import (
    record_from_comparison,
    record_from_gallery,
    save_comparison,
)
from ..enrollment import (
    parse_time_ranges,
    prepare_source,
    spans_for_speaker,
    speaker_options,
)
from ..matching import (
    ComparisonResult,
    compare_questioned_to_profile,
    search_gallery_for_questioned,
)
from ..questioned import (
    SELECTION_DIARIZED,
    SELECTION_RANGES,
    SELECTION_WHOLE,
    QuestionedSpeaker,
    measure_from_wav,
)
from ..reports import GallerySearch, ReportEntry, write_analysis_report
from ..speaker_profiles import (
    SpeakerProfile,
    bundled_model_identity,
    display_labels,
    find_speaker_profile_by_id,
    list_speaker_profiles,
)
from ..thresholds import (
    BAND_HIGH,
    BAND_INSUFFICIENT,
    DISCLAIMER,
    NO_LEAD_EXPLAINED,
    OUTCOME_LEAD,
    OUTCOME_NO_LEAD,
    active,
    describe_active,
)
from ..transcription import AUDIO_EXTENSIONS
from ..voiceprints import SpeakerEmbedder
from .dialogs import active_window, present
from .errors import friendly_error
from .pick_speaker import (
    PickCancelled,
    SampleListener,
    ask_from_worker,
    ask_pick_speaker,
)
from .theme import (
    SPACE_LG,
    SPACE_MD,
    SPACE_SM,
    SPACE_XL,
    SPACE_XS,
    Style,
    palette,
    theme,
)
from .widgets import (
    Card,
    Disclosure,
    EmptyState,
    FileDropZone,
    KeyValueRow,
    PageHeader,
    StatusBanner,
    bind_wheel,
    primary_button,
    register_drop,
    scrollable_body,
    secondary_button,
    set_readonly_text,
    style_text_widget,
    subtle_button,
    wrap_to,
)

MODE_WHOLE = "whole"
MODE_DIARIZE = "diarize"
MODE_RANGES = "ranges"


class SpeakerCompareTab:
    """Builds and drives the Speaker Compare tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        *,
        get_analysis: Optional[Callable[[], Any]] = None,
        dnd_ok: bool = False,
    ) -> None:
        self._dnd_ok = dnd_ok
        self.parent = parent
        self.root = root
        # Supplies (result, speaker_names, provenance) from the Transcribe tab so
        # a report can carry the transcript and its traceability. Optional.
        self._get_analysis = get_analysis
        self._profiles: List[SpeakerProfile] = []
        # subject_id -> the unique label shown in the picker.
        self._labels: dict = {}
        # The diarized cluster the operator chose, for the selection record.
        self._picked_speaker: Optional[str] = None
        # Every comparison and search made on this page since it opened (or
        # since "Start a new session report"), for the session report.
        self._session: List[ReportEntry] = []
        # The result the card is showing, which is what Export writes. It used
        # to write the session's one-to-one comparisons instead, so with a
        # search on screen the report held a different recording's result.
        self._on_screen: Optional[ReportEntry] = None
        # Set by the app so a new comparison shows up in the history page.
        self.on_history_changed: Optional[Callable[[], None]] = None
        self._measuring = False
        self._embedder: Optional[SpeakerEmbedder] = None

        self.reference_var = tk.StringVar(value="")
        self.reference_detail_var = tk.StringVar(value="No reference selected.")
        self.questioned_var = tk.StringVar(value="")
        self.mode_var = tk.StringVar(value=MODE_WHOLE)
        self.ranges_var = tk.StringVar(value="")
        self.status_var = tk.StringVar(value="Idle")
        self.allow_unverified_var = tk.BooleanVar(value=False)

        self._build()
        self.refresh()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Compare Speakers",
            "Measure how similar a speaker in one recording is to a known "
            "person's reference voice.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        self._build_reference_step(container)
        self._build_questioned_step(container)
        self._build_action_step(container)
        self._build_result_step(container)
        bind_wheel(canvas, container)

    # -- Step 1: the known person -----------------------------------------

    def _build_reference_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "1.  Reference speaker")
        card.pack(fill="x")
        row = ttk.Frame(card.body, style=Style.CARD_INNER)
        row.pack(fill="x")
        self.reference_combo = ttk.Combobox(
            row, textvariable=self.reference_var, state="readonly", width=34
        )
        self.reference_combo.pack(side="left")
        self.reference_combo.bind(
            "<<ComboboxSelected>>", lambda _e: self._render_reference()
        )
        subtle_button(row, "Refresh", self.refresh).pack(
            side="left", padx=(SPACE_SM, 0)
        )

        self._reference_facts = ttk.Frame(card.body, style=Style.CARD_INNER)
        self._reference_facts.pack(fill="x", pady=(SPACE_MD, 0))
        self._reference_rows = {
            "speech": KeyValueRow(self._reference_facts, "Reference speech"),
            "samples": KeyValueRow(self._reference_facts, "Trusted samples"),
            "model": KeyValueRow(self._reference_facts, "Voice model"),
        }
        for row_widget in self._reference_rows.values():
            row_widget.pack(fill="x", pady=(0, SPACE_XS))
        self._reference_empty = EmptyState(
            card.body,
            "No reference profile selected",
            "Create one under Speaker Profiles from a recording you know contains "
            "that person.",
        )

    # -- Step 2: the recording in question ---------------------------------

    def _build_questioned_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "2.  Questioned recording")
        card.pack(fill="x", pady=(SPACE_MD, 0))
        self._questioned_zone = FileDropZone(
            card.body,
            self.questioned_var,
            self._choose_questioned,
            # Only invite a drop where one works: without drag-and-drop the
            # invitation did nothing.
            prompt=(
                "Drop the recording to check here"
                if self._dnd_ok
                else "Choose the recording to check"
            ),
            button_text="Choose recording",
            change_text="Change recording",
        )
        self._questioned_zone.pack(fill="x")
        for target in (self._questioned_zone, self._questioned_zone.empty):
            register_drop(self.root, self._dnd_ok, target, self._on_drop_questioned)

        whose = ttk.Label(
            card.body, text="Whose voice should be measured?", style=Style.FIELD_LABEL
        )
        whose.pack(anchor="w", pady=(SPACE_MD, SPACE_XS))
        # Under the file it is about, not in the status line at the foot of the
        # page, which at 1366x768 is below the fold.
        self._questioned_banner = StatusBanner(card.body, wraplength=640)
        self._questioned_banner.insert_before(whose)
        self._questioned_banner.wrap_to(card.body)
        # However the file gets chosen - the button, a drop, a project - the
        # same check runs.
        self.questioned_var.trace_add("write", lambda *_: self._on_questioned_changed())
        for text, value in (
            ("Only one person is speaking — use the whole recording", MODE_WHOLE),
            ("Several people — separate them and let me pick", MODE_DIARIZE),
            ("Only these parts of the recording", MODE_RANGES),
        ):
            ttk.Radiobutton(
                card.body,
                text=text,
                variable=self.mode_var,
                value=value,
                style=Style.RADIO,
            ).pack(anchor="w")
        ranges = ttk.Frame(card.body, style=Style.CARD_INNER)
        ranges.pack(anchor="w", padx=(SPACE_XL, 0))
        ttk.Entry(ranges, textvariable=self.ranges_var, width=40).pack(side="left")
        ttk.Label(ranges, text="e.g. 0:10-0:45, 1:20-2:00", style=Style.META).pack(
            side="left", padx=(SPACE_SM, 0)
        )

    # -- The comparison ----------------------------------------------------

    def _build_action_step(self, parent: tk.Misc) -> None:
        actions = ttk.Frame(parent, style=Style.PAGE)
        actions.pack(fill="x", pady=(SPACE_LG, 0))
        self._compare_button = primary_button(
            actions, "Compare speakers", self._compare
        )
        self._compare_button.pack(side="left")
        self._gallery_button = secondary_button(
            actions, "Search all profiles", self._search_gallery
        )
        self._gallery_button.pack(side="left", padx=(SPACE_SM, 0))
        subtle_button(actions, "Thresholds…", self._show_thresholds).pack(side="right")

        # Under the buttons that cause them. At the foot of the page, "Select a
        # reference subject first." and save failures were below the fold.
        self._status_label = ttk.Label(
            parent, textvariable=self.status_var, style=Style.PAGE_SUBTITLE
        )
        self._status_label.pack(anchor="w", pady=(SPACE_SM, 0))
        # A failure has to be visible on the page, not only in the status line.
        self.banner = StatusBanner(parent, wraplength=680)
        self.banner.insert_before(self._status_label)
        self.banner.wrap_to(parent)

        advanced = Disclosure(parent, "Advanced")
        advanced.pack(fill="x", pady=(SPACE_SM, 0))
        ttk.Checkbutton(
            advanced.body,
            text="Allow a profile whose voice model cannot be verified",
            variable=self.allow_unverified_var,
        ).pack(anchor="w")
        ttk.Label(
            advanced.body,
            text=(
                "Only for an older profile that does not record which model made "
                "it. A profile from a different model is always refused."
            ),
            style=Style.META,
            wraplength=560,
            justify="left",
        ).pack(anchor="w", padx=(SPACE_XL, 0))

    # -- Step 3: what was found --------------------------------------------

    def _build_result_step(self, parent: tk.Misc) -> None:
        card = Card(parent, "3.  Result")
        card.pack(fill="both", expand=True, pady=(SPACE_LG, 0))
        self._result_card = card

        self._result_empty = EmptyState(
            card.body,
            "No comparison yet",
            "Choose a reference profile and a questioned recording, then select "
            "Compare speakers.",
        )
        self._result_empty.pack(fill="x")

        # Shown while a comparison is running, *instead of* the previous result.
        # Leaving the last result on screen with only a small status line
        # underneath is how a finding about one recording gets read as a finding
        # about the one being measured now.
        self._result_busy = ttk.Frame(
            card.body, style=Style.CARD_INNER, padding=(0, SPACE_MD)
        )
        self.busy_var = tk.StringVar(value="")
        ttk.Label(self._result_busy, textvariable=self.busy_var, style=Style.BODY).pack(
            anchor="w"
        )
        self._busy_bar = ttk.Progressbar(self._result_busy, mode="indeterminate")
        self._busy_bar.pack(fill="x", pady=(SPACE_SM, 0))

        self._result_body = ttk.Frame(card.body, style=Style.CARD_INNER)

        # The band, not the number, is the finding. It is stated in words, in
        # the largest type on the card, and never as a percentage.
        self.band_var = tk.StringVar(value="")
        self.band_label = ttk.Label(
            self._result_body, textvariable=self.band_var, style=Style.SECTION_TITLE
        )
        # Set on the widget, not the style: _set_band swaps the style for the
        # colour, and each colour style carries body-size type.
        self.band_label.configure(font=theme().app_title)
        self.band_label.pack(anchor="w")
        self.lead_var = tk.StringVar(value="")
        lead = ttk.Label(
            self._result_body,
            textvariable=self.lead_var,
            style=Style.BODY,
            wraplength=680,
            justify="left",
        )
        lead.pack(anchor="w", pady=(SPACE_XS, 0))
        wrap_to(card.body, lead)

        facts = ttk.Frame(self._result_body, style=Style.CARD_INNER)
        facts.pack(fill="x", pady=(SPACE_MD, 0))
        self._result_rows = {
            "score": KeyValueRow(facts, "Similarity score"),
            "threshold": KeyValueRow(facts, "Operational threshold"),
            "questioned": KeyValueRow(facts, "Questioned speech"),
            "reference": KeyValueRow(facts, "Reference speech"),
            "quality": KeyValueRow(facts, "Audio quality"),
            "model": KeyValueRow(facts, "Voice model"),
            "source": KeyValueRow(facts, "Recording"),
        }
        for row_widget in self._result_rows.values():
            row_widget.pack(fill="x", pady=(0, SPACE_XS))

        self.result_warnings = StatusBanner(self._result_body, wraplength=680)
        self.result_warnings.wrap_to(card.body)

        # The full text block stays: it is what an analyst copies into a case
        # file, and it holds the detail the summary above deliberately omits.
        self._detail_disclosure = Disclosure(self._result_body, "Full result text")
        self.result_warnings.insert_before(self._detail_disclosure)
        self._detail_disclosure.pack(fill="x", pady=(SPACE_MD, 0))
        self.result_text = ScrolledText(
            self._detail_disclosure.body, wrap="word", height=12, state="disabled"
        )
        style_text_widget(self.result_text)
        self.result_text.configure(font=theme().mono)
        self.result_text.pack(fill="both", expand=True)

        # Kept as the last thing built but remembered, so the result can be
        # inserted above it rather than after it.
        self._disclaimer_label = ttk.Label(
            card.body,
            text=DISCLAIMER,
            style=Style.META,
            wraplength=680,
            justify="left",
        )
        self._disclaimer_label.pack(anchor="w", pady=(SPACE_MD, 0))
        wrap_to(card.body, self._disclaimer_label)

        result_actions = ttk.Frame(card.body, style=Style.CARD_INNER)
        result_actions.pack(fill="x", pady=(SPACE_MD, 0))
        export = secondary_button(
            result_actions, "Export this result…", self._export_report
        )
        export.pack(side="left")
        copy = secondary_button(result_actions, "Copy result", self._copy_result)
        copy.pack(side="left", padx=(SPACE_SM, 0))
        session = secondary_button(
            result_actions, "Export session report…", self._export_session_report
        )
        session.pack(side="left", padx=(SPACE_SM, 0))
        # Not "Clear recorded comparisons", which reads as deleting History.
        restart = subtle_button(
            result_actions, "Start a new session report", self._clear_comparisons
        )
        restart.pack(side="left", padx=(SPACE_SM, 0))
        # Everything that acts on "the result" is held while a new one is being
        # measured: until it finishes, these still refer to the previous
        # recording, and a copied or exported result about the wrong recording
        # is worse than one you had to wait for.
        self._result_actions = (export, copy, session, restart)

    # -- Reference ----------------------------------------------------------

    def refresh(self) -> None:
        # The choice is held by subject id across the reload: a label changes
        # when a subject is renamed, and may then belong to nobody.
        previous = self._selected_profile()
        previous_id = previous.subject_id if previous is not None else None
        self._profiles = list_speaker_profiles()
        # Two subjects can share a display name; picking by name alone would let
        # an operator compare against one while believing they chose the other.
        self._labels = display_labels(self._profiles)
        labels = [self._labels[p.subject_id] for p in self._profiles]
        self.reference_combo.configure(values=labels)
        if previous_id is not None:
            label = self._labels.get(previous_id)
            if label is not None:
                self.reference_var.set(label)
            else:
                # Deleted on Speaker Profiles. This used to fall through to the
                # first subject in the list - so the next Compare measured
                # against someone the operator never chose.
                self.reference_var.set("")
                self._status(
                    f"{previous.display_name} is no longer stored, so it is no "
                    "longer selected. Choose a reference subject."
                )
        elif labels and self.reference_var.get() not in labels:
            self.reference_var.set(labels[0])
        if not labels:
            self.reference_var.set("")
        self._render_reference()

    def _selected_profile(self) -> Optional[SpeakerProfile]:
        """The subject behind the chosen label - by subject id, never by name."""
        label = self.reference_var.get()
        for profile in self._profiles:
            if self._labels.get(profile.subject_id) == label:
                return profile
        return None

    def _render_reference(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._reference_facts.pack_forget()
            self._reference_empty.pack(fill="x")
            self.reference_detail_var.set(
                "No reference selected. Create one under Speaker Profiles."
            )
            return
        self._reference_empty.pack_forget()
        self._reference_facts.pack(fill="x", pady=(SPACE_MD, 0))

        summary = profile.summary()
        minimum = active().min_reference_seconds
        thin = profile.total_reference_seconds < minimum
        self._reference_rows["speech"].set(
            f"{profile.total_reference_seconds:.1f} sec"
            + (f"  —  less than the recommended {minimum:.0f} sec" if thin else ""),
            style_name=Style.WARNING if thin else Style.BODY,
        )
        self._reference_rows["samples"].set(str(summary["trusted_sample_count"]))
        if profile.embedding_model:
            self._reference_rows["model"].set(
                profile.embedding_model.describe(), style_name=Style.BODY
            )
        else:
            self._reference_rows["model"].set(
                "unknown (older profile) — a comparison needs explicit "
                "confirmation under Advanced",
                style_name=Style.WARNING,
            )
        # Still maintained for anything that reads the plain string.
        self.reference_detail_var.set(
            f"{profile.display_name} — {summary['trusted_sample_count']} trusted "
            f"samples, {profile.total_reference_seconds:.1f}s reference speech"
        )

    # -- Questioned ---------------------------------------------------------

    def _choose_questioned(self) -> None:
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Questioned recording",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if path:
            self.questioned_var.set(path)

    def _on_drop_questioned(self, paths: List[Path]) -> None:
        """A dropped file is chosen like one picked with the button."""
        if not paths or self._measuring:
            return
        # The variable's trace runs the same processed-audio check as the button.
        self.questioned_var.set(str(paths[0]))

    def _questioned_refusal(self, path: str) -> Optional[str]:
        """Why ``path`` must not be compared, or None if it may be.

        Both kinds of processing alter exactly the detail a voiceprint measures
        - noise reduction by removing what it takes for noise, a listening copy
        by deliberately changing the balance of the voice - so a score from one
        is not comparable with a profile built from original audio.

        This used to be a warning in the status line at the foot of the page,
        with Compare and Search left enabled: the comparison ran and was logged.
        The evidence is only a file name, and someone who renames the file can
        still get it in; but that is a deliberate act, and Speaker Profiles
        already refused on the same evidence.
        """
        kind = listening.derived_kind(path) if path else None
        if kind is None:
            return None
        return (
            f"That is a {kind}, not the recording, so it cannot be compared. "
            "Processing changes the detail a voice comparison measures, and a "
            "score from it would not be comparable with a profile built from "
            "original audio. Choose the original recording."
        )

    def _on_questioned_changed(self) -> None:
        refusal = self._questioned_refusal(self.questioned_var.get().strip())
        if refusal is None:
            self._questioned_banner.hide()
        else:
            self._questioned_banner.show("error", refusal)
        self._update_action_state()

    def _update_action_state(self) -> None:
        """Compare and Search stay off while a derived copy is chosen."""
        refused = self._questioned_refusal(self.questioned_var.get().strip())
        # Not while a comparison is running. Choosing another recording
        # mid-run used to switch both back on, so a second comparison could
        # start on top of the first.
        state = "disabled" if refused or self._measuring else "normal"
        # The trace is attached in step 2 and the buttons are built in step 3,
        # so a write during construction arrives before they exist.
        for name in ("_compare_button", "_gallery_button"):
            button = getattr(self, name, None)
            if button is not None:
                button.configure(state=state)

    def _get_embedder(self) -> SpeakerEmbedder:
        """Load the speaker-embedding model once (sherpa loads on first use)."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder()
        return self._embedder

    def _prepare_spans(self, wav: Path) -> Optional[List[Tuple[float, float]]]:
        """Resolve the questioned speech according to the selected mode."""
        mode = self.mode_var.get()
        if mode == MODE_RANGES:
            return parse_time_ranges(self.ranges_var.get())
        if mode == MODE_DIARIZE:
            from ..diarization import diarize

            self._status("Separating speakers…")
            segments = diarize(wav, progress=self._status)
            options = speaker_options(segments)
            if not options:
                return None
            picked = ask_from_worker(
                self.root,
                lambda: ask_pick_speaker(
                    self.root,
                    title="Which speaker is the questioned speaker?",
                    guidance=(
                        "Choose the speaker to compare against the reference "
                        "profile. Listen to each one first: the speaker numbers "
                        "come from the order they were found, not from who they "
                        "are."
                    ),
                    options=options,
                    listener=SampleListener.for_file(wav),
                ),
            )
            if picked is None:
                raise PickCancelled()
            self._picked_speaker = picked
            return spans_for_speaker(segments, picked)
        return [(0.0, _duration(wav))]

    def _compare(self) -> None:
        profile = self._selected_profile()
        if profile is None:
            self._status("Select a reference subject first.")
            return
        # The list on this page is a snapshot. Compare against the subject as
        # it is stored now: one deleted on Speaker Profiles was still compared
        # against, from memory, and logged; samples removed there still counted.
        fresh = find_speaker_profile_by_id(profile.subject_id)
        if fresh is None:
            self.refresh()
            return
        profile = fresh
        source = self.questioned_var.get().strip()
        if not source:
            self._status("Choose a questioned recording first.")
            return
        refusal = self._questioned_refusal(source)
        if refusal is not None:
            self._questioned_banner.show("error", refusal)
            return
        self._start_measuring(Path(source), "Comparing")
        threading.Thread(
            target=self._compare_worker, args=(profile, Path(source)), daemon=True
        ).start()

    def _measure_questioned(self, source: Path) -> Optional[QuestionedSpeaker]:
        """Measure the questioned speaker, keeping the audio and the source hash.

        The selection is resolved first (which may need diarization), then every
        usable window of it is embedded and averaged - so the duration and
        quality reported are the ones behind the embedding, and the recording
        identifies itself in the result and any report.
        """
        embedder = self._get_embedder()
        wav, digest, temporary = prepare_source(source, progress=self._status)
        try:
            spans = self._prepare_spans(wav)
            if not spans:
                return None
            measured = measure_from_wav(
                wav,
                spans,
                embedder,
                selection_mode=self._selection_description(spans),
                progress=self._status,
            )
        finally:
            if temporary:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass
        measured.source_filename = source.name
        measured.source_sha256 = digest
        try:
            measured.source_size = source.stat().st_size
        except OSError:
            measured.source_size = None
        return measured

    def _selection_description(self, spans) -> str:
        """How the questioned speech was chosen, in words, for the record."""
        mode = self.mode_var.get()
        if mode == MODE_RANGES:
            return f"{SELECTION_RANGES} ({self.ranges_var.get().strip()})"
        if mode == MODE_DIARIZE:
            speaker = self._picked_speaker or "selected speaker"
            return f"{SELECTION_DIARIZED} ({speaker}, {len(spans)} turn(s))"
        return SELECTION_WHOLE

    def _compare_worker(self, profile: SpeakerProfile, source: Path) -> None:
        try:
            measured = self._measure_questioned(source)
            if measured is None or not measured.usable:
                self._report_unusable(measured)
                return
            result = compare_questioned_to_profile(
                measured,
                profile,
                questioned_model=bundled_model_identity(),
                allow_unverified_model=self.allow_unverified_var.get(),
            )
            self._session.append(result)
            self._log(
                record_from_comparison(
                    result,
                    profile=profile,
                    questioned_path=str(source),
                    questioned_spans=measured.embedded_spans,
                )
            )
            self.root.after(0, lambda: self._show_comparison(result))
            self._status("Comparison complete.")
        except PickCancelled:
            self._pick_cancelled("compared")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._fail(exc)
        finally:
            self.root.after(0, self._finish_measuring)

    def _search_gallery(self) -> None:
        source = self.questioned_var.get().strip()
        if not source:
            self._status("Choose a questioned recording first.")
            return
        refusal = self._questioned_refusal(source)
        if refusal is not None:
            self._questioned_banner.show("error", refusal)
            return
        self._start_measuring(Path(source), "Searching all profiles against")
        threading.Thread(
            target=self._gallery_worker, args=(Path(source),), daemon=True
        ).start()

    def _gallery_worker(self, source: Path) -> None:
        try:
            measured = self._measure_questioned(source)
            if measured is None or not measured.usable:
                self._report_unusable(measured)
                return
            result = search_gallery_for_questioned(
                measured,
                self._profiles,
                questioned_model=bundled_model_identity(),
            )
            self._log(
                record_from_gallery(
                    result,
                    questioned_seconds=measured.speech_seconds,
                    questioned_quality=measured.quality,
                    questioned_window_count=measured.window_count,
                    questioned_path=str(source),
                    questioned_spans=measured.embedded_spans,
                    embedding_model=self._model_description(),
                )
            )
            search = GallerySearch(
                result,
                questioned_seconds=measured.speech_seconds,
                questioned_quality=str(measured.quality),
                questioned_window_count=measured.window_count,
            )
            self._session.append(search)
            self.root.after(0, lambda: self._show_gallery(measured, search))
            self._status("Search complete.")
        except PickCancelled:
            self._pick_cancelled("searched")
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._fail(exc)
        finally:
            self.root.after(0, self._finish_measuring)

    @staticmethod
    def _model_description() -> str:
        """The embedding model behind a gallery search, for the record."""
        identity = bundled_model_identity()
        return identity.describe() if identity is not None else ""

    def _log(self, record: object) -> None:
        """Write the record of a comparison, without ever failing the comparison.

        The finding is on screen either way; a history that could not be written
        is worth saying out loud but is not a reason to lose the result.
        """
        try:
            save_comparison(record)  # type: ignore[arg-type]
        except _LogError as exc:
            message = (
                "The comparison ran, but it could not be added to the history: "
                f"{friendly_error(exc)}"
            )
            self.root.after(0, lambda: self.banner.show("warning", message))
            return
        refresh = self.on_history_changed
        if refresh is not None:
            self.root.after(0, refresh)

    # -- Output -------------------------------------------------------------

    def _show(self, lines: List[str]) -> None:
        """The full text block - what an analyst copies into a case file."""
        set_readonly_text(self.result_text, "\n".join(lines) + "\n\n" + DISCLAIMER)
        self.root.after(0, self._reveal_result)

    def _busy_disabled(self) -> tuple:
        """Controls that must not act while a comparison is in flight."""
        return (self._compare_button, self._gallery_button, *self._result_actions)

    def _start_measuring(self, source: Path, what: str) -> None:
        """Clear the previous finding and show that a new one is being made.

        The old result comes off the screen the moment a new comparison starts.
        It described a different recording, and leaving it up - even with a
        status line running underneath - invites reading it as the answer for
        this one.
        """
        self._measuring = True
        self._on_screen = None
        # The file being measured stays the file on screen until it is done;
        # the busy line names it, and step 2 must not say something else.
        self._questioned_zone.set_enabled(False)
        self.banner.hide()
        self._result_empty.pack_forget()
        self._result_body.pack_forget()
        self.busy_var.set(f"{what} {source.name}…")
        self._result_busy.pack(fill="x", before=self._disclaimer_label)
        self._busy_bar.start(12)
        for button in self._busy_disabled():
            button.configure(state="disabled")

    def _finish_measuring(self) -> None:
        self._measuring = False
        self._questioned_zone.set_enabled(True)
        self._busy_bar.stop()
        self._result_busy.pack_forget()
        for button in self._busy_disabled():
            button.configure(state="normal")
        self._update_action_state()
        # Nothing was produced (an error, or a cancelled speaker picker): show
        # the empty state rather than restoring a result about another
        # recording.
        if not self._result_body.winfo_manager():
            self._result_empty.pack(fill="x", before=self._disclaimer_label)

    def _reveal_result(self) -> None:
        self._result_empty.pack_forget()
        self._result_busy.pack_forget()
        self._result_body.pack(fill="both", expand=True, before=self._disclaimer_label)

    def _show_comparison(self, result: ComparisonResult) -> None:
        """Present a comparison as a finding, with the numbers supporting it.

        The band leads, in words. The score is shown out of 1.00 beside it -
        never as a percentage, which reads as a probability that these are the
        same person, which is not what a cosine similarity is. A result the
        audio cannot support outranks whatever the number happens to be.
        """
        self._on_screen = result
        self._show(result.format_lines())

        if result.refused:
            self._set_band("Comparison refused", Style.DANGER)
            self.lead_var.set(result.refusal_reason)
        elif result.band == BAND_INSUFFICIENT:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                "There is not enough usable speech to support any assessment. "
                "The score below is not meaningful."
            )
        elif result.band == BAND_HIGH:
            # Blue, not green. Green is this application's "success / ready",
            # and a green speaker result reads as "we got the right person" -
            # the exact conclusion every word on this card is written to avoid.
            self._set_band(result.band, Style.ACCENT)
            self.lead_var.set(
                f"The questioned speaker produced high similarity to the "
                f"{result.reference_name} reference profile. Further review is "
                "warranted."
            )
        else:
            self._set_band(result.band, Style.BODY)
            self.lead_var.set(
                f"The questioned speech is not strongly similar to the "
                f"{result.reference_name} reference profile."
            )

        self._result_rows["score"].set(
            "—" if result.refused else f"{result.score:.2f} / 1.00"
        )
        self._result_rows["threshold"].set(
            f"{result.operational_threshold:.2f} for high similarity"
        )
        measured = (
            f" across {result.questioned_window_count} window(s)"
            if result.questioned_window_count
            else ""
        )
        self._result_rows["questioned"].set(
            f"{result.questioned_seconds:.1f} sec{measured}"
        )
        self._result_rows["reference"].set(f"{result.reference_seconds:.1f} sec")
        self._result_rows["quality"].set(
            f"questioned {result.questioned_quality.lower()}, "
            f"reference {result.reference_quality.lower()}"
        )
        self._result_rows["model"].set(result.embedding_model)
        self._result_rows["source"].set(
            result.questioned_source_filename or "not recorded"
        )

        if result.warnings:
            self.result_warnings.show("warning", "  ".join(result.warnings))
        else:
            self.result_warnings.hide()

    def _pick_cancelled(self, what: str) -> None:
        """Closing the picker is a cancellation, not a finding about the audio."""
        # It was reported as "No usable speech was found in the questioned
        # selection", which says something about the recording that nobody
        # measured.
        message = f"Cancelled. No speaker was chosen, so nothing was {what}."
        self._status(message)
        self.root.after(0, lambda: self.banner.show("info", message))

    def _fail(self, exc: Exception) -> None:
        """A failed comparison must not leave the previous one standing."""
        message = friendly_error(exc)
        self._status(message)
        self.root.after(0, lambda: self.banner.show("error", message))

    def _report_unusable(self, measured: object) -> None:
        """Not enough speech to measure is a result, and it has to look like one."""
        message = "No usable speech was found in the questioned selection."
        self._status(message)
        if measured is None:
            return
        describe = getattr(measured, "describe", None)
        warnings = list(getattr(measured, "warnings", []))
        self._show((list(describe()) if callable(describe) else []) + [""] + warnings)

        def _paint() -> None:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                message + " Select more of the speaker's audio, or a recording "
                "with more of them speaking."
            )
            for row in self._result_rows.values():
                row.set("—")
            self._result_rows["source"].set(
                str(getattr(measured, "source_filename", "") or "not recorded")
            )
            if warnings:
                self.result_warnings.show("warning", "  ".join(warnings))

        self.root.after(0, _paint)

    def _set_band(self, text: str, style_name: str) -> None:
        self.band_var.set(text)
        self.band_label.configure(style=style_name)

    def _show_gallery(self, measured: object, search: GallerySearch) -> None:
        """A ranked search is context, not an identification."""
        self._on_screen = search
        result = search.result
        describe = getattr(measured, "describe", None)
        lines = list(describe()) if callable(describe) else []
        self._show(lines + [""] + result.summary_lines())
        if result.inadequate_reason:
            self._set_band(BAND_INSUFFICIENT, Style.WARNING)
            self.lead_var.set(
                f"{result.inadequate_reason} The ranking below supports no "
                "conclusion about any subject."
            )
        elif result.accepted_name:
            # The same words History uses for the same search.
            self._set_band(f"{OUTCOME_LEAD}: {result.accepted_name}", Style.ACCENT)
            self.lead_var.set(
                f"The questioned speaker produced high similarity to the "
                f"{result.accepted_name} reference profile. Further review is "
                "warranted."
            )
        else:
            self._set_band(OUTCOME_NO_LEAD, Style.BODY)
            self.lead_var.set(NO_LEAD_EXPLAINED)
        top = result.matches[0] if result.matches else None
        if top is None:
            score_text = "—"
        elif result.inadequate_reason:
            # A name beside a score is a finding; on audio too poor to assess
            # there is none to state.
            score_text = f"{top.score:.2f} / 1.00 (not assessed)"
        else:
            score_text = f"{top.score:.2f} / 1.00 ({top.display_name})"
        self._result_rows["score"].set(score_text)
        self._result_rows["threshold"].set(
            f"{result.thresholds.recognition_acceptance:.2f} to accept, "
            f"{result.thresholds.recognition_margin:.2f} clear of the next"
        )
        self._result_rows["questioned"].set(
            f"{getattr(measured, 'speech_seconds', 0.0):.1f} sec"
        )
        self._result_rows["reference"].set(f"{result.searched} profile(s) searched")
        self._result_rows["quality"].set(str(getattr(measured, "quality", "—")))
        self._result_rows["model"].set("—")
        self._result_rows["source"].set(
            str(getattr(measured, "source_filename", "") or "not recorded")
        )
        if result.skipped:
            self.result_warnings.show(
                "warning",
                f"{len(result.skipped)} profile(s) skipped: "
                + "; ".join(result.skipped[:2]),
            )
        else:
            self.result_warnings.hide()

    def _show_thresholds(self) -> None:
        """Show the decision thresholds in force (read-only)."""
        window = tk.Toplevel(self.root)
        window.title("Whispers - active thresholds")
        window.configure(background=palette().background)
        frame = ttk.Frame(window, style=Style.PAGE, padding=SPACE_XL)
        frame.pack(fill="both", expand=True)
        text = ScrolledText(frame, wrap="word", width=76, height=22, font=theme().mono)
        style_text_widget(text)
        text.pack(fill="both", expand=True)
        text.insert("end", "\n".join(describe_active()))
        text.configure(state="disabled")
        secondary_button(frame, "Close", window.destroy).pack(
            side="right", pady=(SPACE_MD, 0)
        )
        present(window, self.root)

    def _copy_result(self) -> None:
        text = self.result_text.get("1.0", "end-1c")
        if not text.strip():
            self._status("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self._status("Result copied to clipboard.")

    @staticmethod
    def _analysis_matches(provenance, entries: List[ReportEntry]) -> bool:
        """True when the open analysis is of a recording ``entries`` measured.

        Compared by source SHA-256: a matching name proves nothing, and an
        analysis with no recorded hash cannot be shown to be the right one.
        """
        digest = None
        if provenance is not None and getattr(provenance, "source", None):
            digest = provenance.source.sha256
        if not digest:
            return False
        return any(entry.questioned_source_sha256 == digest for entry in entries)

    def _export_report(self) -> None:
        """Write the result on screen, with the transcript if it is of that recording."""
        entry = self._on_screen
        if entry is None:
            self._status("There is no comparison or search result on screen to export.")
            return
        self._write_report([entry], "speaker-analysis-report.docx")

    def _export_session_report(self) -> None:
        """Write every comparison and search made since the session report began."""
        if not self._session:
            self._status(
                "No comparison or search has been made since the report began."
            )
            return
        self._write_report(list(self._session), "speaker-analysis-session-report.docx")

    def _write_report(self, entries: List[ReportEntry], initialfile: str) -> None:
        result = names = provenance = None
        if self._get_analysis is not None:
            try:
                result, names, provenance = self._get_analysis()
            except Exception:  # noqa: BLE001 - a report without the transcript is fine
                result = names = provenance = None
        # A transcript from a *different* recording must not be presented as the
        # source of these results. Attach it only when its source hash is one
        # of the recordings actually measured for them.
        left_out = result is not None and not self._analysis_matches(
            provenance, entries
        )
        if left_out:
            result = names = provenance = None
        path = filedialog.asksaveasfilename(
            title="Export analysis report",
            defaultextension=".docx",
            initialfile=initialfile,
            filetypes=[("Word document", "*.docx"), ("Text file", "*.txt")],
            parent=active_window(self.root),
        )
        if not path:
            return
        try:
            write_analysis_report(
                path,
                result=result,
                speaker_names=names,
                provenance=provenance,
                comparisons=entries,
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._fail(exc)
            return
        count = len(entries)
        what = "1 result" if count == 1 else f"{count} results"
        note = (
            " The open transcript is of a different recording, so it was left out."
            if left_out
            else ""
        )
        self._status(f"Saved {what} to {path}.{note}")

    def _clear_comparisons(self) -> None:
        count = len(self._session)
        self._session = []
        self._status(
            "The session report starts again from the next result"
            + (f" ({count} earlier result(s) left out)" if count else "")
            + ". Comparison History keeps every record."
        )

    def _status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    # -- Shared hooks -------------------------------------------------------

    def get_settings(self) -> dict:
        return {}

    def apply_settings(self, data: dict) -> None:
        return None


def _duration(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)
