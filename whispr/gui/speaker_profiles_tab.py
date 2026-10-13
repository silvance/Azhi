"""The Speaker profiles tab: build known-subject reference voices from audio.

The deliberate enrolment workflow: create a subject, point at historical
recordings that contain them, choose which speech is theirs (the whole file, a
diarized speaker, or explicit time ranges), and store trusted reference samples.
No transcription pass is required to get from *historical recordings* to a
*known-actor profile*.

Automatically learned samples (from transcript corrections elsewhere in the app)
appear here too, but as pending items an operator reviews and promotes - they are
never silently folded into a trusted reference.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, simpledialog, ttk
from typing import Callable, List, Optional, Sequence, Tuple

from .. import listening
from ..enrollment import (
    enroll_from_wav,
    parse_time_ranges,
    prepare_source,
    spans_for_speaker,
    speaker_options,
)
from ..hashing import short
from ..speaker_profiles import (
    ProfileError,
    SpeakerProfile,
    delete_speaker_profile,
    find_speaker_profile_by_id,
    list_speaker_profiles,
    load_profile_file,
    new_subject_id,
    profile_path,
    save_speaker_profile,
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
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XL, SPACE_XS, Style, palette
from .widgets import (
    Card,
    EmptyState,
    KeyValueRow,
    PageHeader,
    StatusBanner,
    bind_wheel,
    danger_button,
    primary_button,
    scrollable_body,
    secondary_button,
    subtle_button,
)

# Enrolment modes offered when a recording is added.
MODE_WHOLE = "whole"
MODE_DIARIZE = "diarize"
MODE_RANGES = "ranges"


# Narrower than this, the subject list goes above the detail instead of beside
# it. The detail needs about 600 pixels for its sample table and buttons.
_STACK_BELOW = 900


class SpeakerProfilesTab:
    """Builds and drives the Speaker profiles tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
    ) -> None:
        self.parent = parent
        self.root = root
        self._profiles: List[SpeakerProfile] = []
        self._selected: Optional[SpeakerProfile] = None
        self._embedder: Optional[SpeakerEmbedder] = None
        # Set by the application: called when this page changes a subject, so
        # Compare and History reload rather than working from an old copy.
        self.on_profiles_changed: Optional[Callable[[], None]] = None
        self.status_var = tk.StringVar(value="Idle")
        self._build()
        self.refresh()

    # -- UI ----------------------------------------------------------------

    def _build(self) -> None:
        canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Speaker Profiles",
            "Build a reference voice for someone you know, from recordings you "
            "already have.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        # At the top, not the foot of the page: below a long sample list the
        # result of an enrolment, or why it failed, was out of sight.
        self._status_label = ttk.Label(
            container, textvariable=self.status_var, style=Style.PAGE_SUBTITLE
        )
        self._status_label.pack(anchor="w", pady=(0, SPACE_MD))
        self.banner = StatusBanner(container)
        self.banner.insert_before(self._status_label)

        # Master and detail: who exists on the left, everything about the
        # selected person on the right. At 1366 wide both fit; the list is kept
        # narrow so the detail - which is where the work happens - gets the room.
        columns = ttk.Frame(container, style=Style.PAGE)
        columns.pack(fill="both", expand=True)
        self._columns = columns
        self._stacked: Optional[bool] = None

        self._build_subject_list(columns)
        self._build_detail(columns)
        self._arrange(stacked=False)
        columns.bind("<Configure>", lambda e: self._arrange(e.width < _STACK_BELOW))
        bind_wheel(canvas, container)

    def _arrange(self, stacked: bool) -> None:
        """Side by side when there is room; the list above the detail when not.

        In a 940-pixel window the detail beside the list was cut off: Remove
        sample off the edge, Approve half shown, the sample table truncated,
        and no way to scroll sideways to them.
        """
        if stacked == self._stacked:
            return
        self._stacked = stacked
        columns = self._columns
        if stacked:
            columns.columnconfigure(0, weight=1, minsize=0)
            columns.columnconfigure(1, weight=0, minsize=0)
            columns.rowconfigure(0, weight=0)
            columns.rowconfigure(1, weight=1)
            self._list_card.grid(
                row=0, column=0, columnspan=2, sticky="nsew", padx=0, pady=(0, SPACE_MD)
            )
            self._detail_card.grid(row=1, column=0, columnspan=2, sticky="nsew")
            self.subject_tree.configure(height=4)
        else:
            columns.columnconfigure(0, weight=0, minsize=300)
            columns.columnconfigure(1, weight=1)
            columns.rowconfigure(0, weight=1)
            columns.rowconfigure(1, weight=0)
            self._list_card.grid(
                row=0, column=0, columnspan=1, sticky="nsew", padx=(0, SPACE_MD), pady=0
            )
            self._detail_card.grid(row=0, column=1, columnspan=1, sticky="nsew")
            self.subject_tree.configure(height=12)

    def _build_subject_list(self, parent: tk.Misc) -> None:
        card = Card(parent, "Subjects")
        self._list_card = card

        self.subject_tree = ttk.Treeview(
            card.body,
            columns=("samples",),
            show="tree",
            height=12,
            selectmode="browse",
        )
        self.subject_tree.column("#0", width=180, stretch=True)
        self.subject_tree.column("samples", width=90, anchor="e", stretch=False)
        self.subject_tree.pack(fill="both", expand=True)
        self.subject_tree.bind("<<TreeviewSelect>>", lambda _e: self._on_select())

        # Shown instead of the list until there is a first subject: a blank box
        # gives a first-time operator nothing to act on.
        self.subjects_empty = EmptyState(
            card.body,
            "No speaker profiles yet",
            "Create one from a recording you know contains a particular person, "
            "so later recordings can be compared against it.",
        )

        buttons = ttk.Frame(card.body, style=Style.CARD_INNER)
        buttons.pack(fill="x", pady=(SPACE_MD, 0))
        primary_button(buttons, "New profile", self._new_subject).pack(side="left")
        subtle_button(buttons, "Import…", self._import_profile).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        subtle_button(buttons, "Export…", self._export_profile).pack(side="left")
        subtle_button(buttons, "Refresh", self.refresh).pack(side="left")

    def _build_detail(self, parent: tk.Misc) -> None:
        card = Card(parent, "")
        self._detail_card = card

        # Nothing selected: say what to do rather than showing an empty table.
        self.detail_empty = EmptyState(
            card.body,
            "No subject selected",
            "Choose someone on the left, or create a profile to begin.",
        )
        self.detail_empty.pack(fill="both", expand=True)

        self.detail_body = ttk.Frame(card.body, style=Style.CARD_INNER)

        heading = ttk.Frame(self.detail_body, style=Style.CARD_INNER)
        heading.pack(fill="x")
        self.subject_name_var = tk.StringVar(value="")
        ttk.Label(
            heading, textvariable=self.subject_name_var, style=Style.SECTION_TITLE
        ).pack(side="left")
        danger_button(heading, "Delete profile", self._delete_profile).pack(
            side="right"
        )

        facts = ttk.Frame(self.detail_body, style=Style.CARD_INNER)
        facts.pack(fill="x", pady=(SPACE_MD, 0))
        self._fact_rows = {
            "reference": KeyValueRow(facts, "Reference speech"),
            "trusted": KeyValueRow(facts, "Trusted samples"),
            "pending": KeyValueRow(facts, "Awaiting review"),
            "sources": KeyValueRow(facts, "From recordings"),
            "model": KeyValueRow(facts, "Voice model"),
        }
        for row in self._fact_rows.values():
            row.pack(fill="x", pady=(0, SPACE_XS))

        actions = ttk.Frame(self.detail_body, style=Style.CARD_INNER)
        actions.pack(fill="x", pady=(SPACE_MD, 0))
        primary_button(actions, "Add a recording", self._add_recording).pack(
            side="left"
        )
        self._approve_button = secondary_button(
            actions, "Approve sample", self._approve
        )
        self._approve_button.pack(side="left", padx=(SPACE_SM, 0))
        danger_button(actions, "Remove sample", self._remove).pack(side="left")

        ttk.Label(
            self.detail_body,
            text=(
                "Add a recording asks which stretches of it are this person. For "
                "a conversation, it is usually easier to transcribe it first with "
                "\u201cIdentify who is speaking\u201d on, correct the speaker "
                "tags there, then use Save speaker to profile\u2026 on the "
                "Transcribe page."
            ),
            style=Style.META,
            wraplength=520,
            justify="left",
        ).pack(anchor="w", pady=(SPACE_SM, 0))

        ttk.Label(
            self.detail_body, text="Reference samples", style=Style.FIELD_LABEL
        ).pack(anchor="w", pady=(SPACE_LG, SPACE_XS))
        self.sample_tree = ttk.Treeview(
            self.detail_body,
            columns=("state", "source", "span", "speech", "quality"),
            show="headings",
            height=8,
        )
        sample_columns: List[Tuple[str, str, int, bool]] = [
            ("state", "State", 210, False),
            ("source", "Recording", 150, True),
            ("span", "Where", 90, False),
            ("speech", "Speech", 70, False),
            ("quality", "Quality", 80, False),
        ]
        for column, heading_text, width, stretch in sample_columns:
            self.sample_tree.heading(column, text=heading_text)
            self.sample_tree.column(column, width=width, stretch=stretch, anchor="w")
        # Durations line up when they are right-aligned.
        self.sample_tree.column("speech", anchor="e")
        self.sample_tree.pack(fill="both", expand=True)
        # A sample awaiting review is not a row of ordinary text: it is the one
        # thing on this screen that asks the operator for a decision.
        self.sample_tree.tag_configure("pending", foreground=palette().warning)
        self.sample_tree.tag_configure("outlier", foreground=palette().danger_active)

        self.samples_empty = EmptyState(
            self.detail_body,
            "No reference samples yet",
            "Add a recording that contains this person, and say which speech is "
            "theirs \u2014 or correct the speaker tags on a transcript and use "
            "Save speaker to profile\u2026 there.",
        )

    # -- Data --------------------------------------------------------------

    def _announce_change(self) -> None:
        """Tell the other pages a subject changed, so none works from a stale copy."""
        notify = self.on_profiles_changed
        if notify is not None:
            self.root.after(0, notify)

    def refresh(self) -> None:
        """Reload the stored subjects and repaint the list."""
        self._profiles = list_speaker_profiles()
        self.subject_tree.delete(*self.subject_tree.get_children())
        for profile in self._profiles:
            summary = profile.summary()
            pending = int(summary["pending_sample_count"])
            # The list answers "who exists and does anyone need me": the detail
            # panel carries the rest.
            suffix = f"  ·  {pending} to review" if pending else ""
            self.subject_tree.insert(
                "",
                "end",
                iid=profile.subject_id,
                text=f"{profile.display_name}{suffix}",
                tags=("pending",) if pending else (),
            )
        self.subject_tree.tag_configure("pending", foreground=palette().warning)
        if self._profiles:
            self.subjects_empty.pack_forget()
            self.subject_tree.pack(fill="both", expand=True)
        else:
            self.subject_tree.pack_forget()
            self.subjects_empty.pack(fill="both", expand=True)
        self._on_select()

    def _on_select(self) -> None:
        selection = self.subject_tree.selection()
        self._selected = None
        if selection:
            for profile in self._profiles:
                if profile.subject_id == selection[0]:
                    self._selected = profile
                    break
        self._render_detail()

    def _render_detail(self) -> None:
        self.sample_tree.delete(*self.sample_tree.get_children())
        profile = self._selected
        if profile is None:
            self.detail_body.pack_forget()
            self.detail_empty.pack(fill="both", expand=True)
            return
        self.detail_empty.pack_forget()
        self.detail_body.pack(fill="both", expand=True)

        summary = profile.summary()
        pending = int(summary["pending_sample_count"])
        self.subject_name_var.set(profile.display_name)
        self._fact_rows["reference"].set(f"{profile.total_reference_seconds:.1f} sec")
        self._fact_rows["trusted"].set(str(summary["trusted_sample_count"]))
        self._fact_rows["pending"].set(
            f"{pending} — review before they are used" if pending else "None",
            style_name=Style.WARNING if pending else Style.BODY,
        )
        self._fact_rows["sources"].set(
            ", ".join(profile.source_files()) or "none recorded"
        )
        if profile.embedding_model:
            self._fact_rows["model"].set(
                f"{profile.embedding_model.name}  ·  "
                f"{short(profile.embedding_model.sha256)}",
                style_name=Style.BODY,
            )
        else:
            # A profile that cannot prove which model made it is a real
            # limitation on any later comparison, so it is said plainly.
            self._fact_rows["model"].set(
                "unknown — imported from an older format, so it cannot be "
                "proven to match this build",
                style_name=Style.WARNING,
            )

        for sample in profile.samples:
            span = (
                f"{sample.source_start:.0f}-{sample.source_end:.0f}s"
                if sample.source_start is not None and sample.source_end is not None
                else "—"
            )
            outlier = "Outlier" in (sample.notes or "")
            state: str
            tags: Tuple[str, ...]
            if sample.is_trusted:
                state, tags = "Trusted", ()
            elif outlier:
                # Short enough for the column at 1366 wide; it was cut to
                # "Needs review — unli…", the one state that most needs reading.
                state, tags = "Unlike reference — review", ("outlier",)
            else:
                state, tags = "Needs review", ("pending",)
            self.sample_tree.insert(
                "",
                "end",
                iid=sample.sample_id,
                values=(
                    state,
                    sample.source_filename or "—",
                    span,
                    f"{sample.speech_duration:.1f}s",
                    str(sample.quality.get("assessment", "—")),
                ),
                tags=tags,
            )
        if profile.samples:
            self.samples_empty.pack_forget()
            self.sample_tree.pack(fill="both", expand=True)
        else:
            self.sample_tree.pack_forget()
            self.samples_empty.pack(fill="both", expand=True)
        self._approve_button.configure(state="normal" if pending else "disabled")

    # -- Subject management -------------------------------------------------

    def _new_subject(self) -> None:
        name = simpledialog.askstring(
            "New subject", "Subject / display name:", parent=active_window(self.root)
        )
        if not name or not name.strip():
            return
        profile = SpeakerProfile(display_name=name.strip())
        try:
            save_speaker_profile(profile)
        except ProfileError as exc:
            self._status(friendly_error(exc))
            return
        self.refresh()
        self.subject_tree.selection_set(profile.subject_id)
        self._status(f"Created subject '{profile.display_name}'.")

    def _delete_profile(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select a subject first.")
            return
        if not messagebox.askyesno(
            "Delete subject",
            f"Delete '{profile.display_name}' and its enrolled voice samples?",
            parent=active_window(self.root),
        ):
            return
        try:
            delete_speaker_profile(profile)
        except ProfileError as exc:
            self._status(friendly_error(exc))
            return
        self.refresh()
        self._announce_change()
        self._status("Subject deleted.")

    def _export_profile(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select a subject first.")
            return
        from ..speaker_profiles import SPEAKER_PROFILE_SUFFIX

        path = filedialog.asksaveasfilename(
            title="Export speaker profile",
            defaultextension=SPEAKER_PROFILE_SUFFIX,
            initialfile=f"{profile.display_name}{SPEAKER_PROFILE_SUFFIX}",
            filetypes=[("Whispers speaker profile", f"*{SPEAKER_PROFILE_SUFFIX}")],
            parent=active_window(self.root),
        )
        if not path:
            return
        try:
            save_speaker_profile(profile, path)
            self._status(f"Exported to {Path(path).name}")
        except ProfileError as exc:
            self._status(friendly_error(exc))

    def _import_profile(self) -> None:
        path = filedialog.askopenfilename(
            title="Import speaker profile",
            filetypes=[("Whispers profile", "*.json"), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if not path:
            return
        try:
            imported = load_profile_file(path)
            # import_warnings describe *this* read and are not persisted, so
            # collect them before saving - after the reload they are gone, and
            # an operator would never learn that samples were dropped.
            issues = [
                (profile.display_name, warning)
                for profile in imported
                for warning in profile.import_warnings
            ]
            kept: List[SpeakerProfile] = []
            for profile in imported:
                existing = find_speaker_profile_by_id(profile.subject_id)
                if existing is None:
                    save_speaker_profile(profile)
                    kept.append(profile)
                    continue
                # The same subject is already here. Saving over it used to
                # happen without a word: QA imported an export holding one
                # trusted sample over a profile that had grown to three, and
                # two were gone.
                choice = _ask_import_conflict(self.root, existing, profile)
                if choice == _SKIP:
                    continue
                if choice == _KEEP_BOTH:
                    profile.subject_id = new_subject_id()
                    profile.display_name = f"{profile.display_name} (imported)"
                    save_speaker_profile(profile)
                else:
                    old_path = profile_path(existing)
                    new_path = save_speaker_profile(profile)
                    if old_path != new_path:
                        # Renamed since the export: the old file has the same
                        # id, and two files claiming one subject is its own
                        # kind of corruption.
                        delete_speaker_profile(existing)
                kept.append(profile)
            imported = kept
        except ProfileError as exc:
            self._status(friendly_error(exc))
            return
        self.refresh()
        legacy = sum(1 for p in imported if p.is_legacy)
        note = (
            f" {legacy} imported from an older format (embedding model unknown)."
            if legacy
            else ""
        )
        if not imported:
            self._status("Nothing was imported.")
            return
        self._announce_change()
        self._status(f"Imported {len(imported)} subject(s).{note}")
        if issues:
            self._report_import_issues(Path(path).name, issues)

    def _report_import_issues(
        self, filename: str, issues: Sequence[Tuple[str, str]]
    ) -> None:
        """Tell the operator exactly what the import dropped or demoted.

        Data being safely quarantined is not the same as the operator knowing
        it happened: a profile that arrives two samples lighter, or with
        reference material demoted to pending, changes what any later
        comparison means.
        """
        lines = [
            f"{filename} was imported, but not everything in it could be "
            "trusted as it stood:",
            "",
        ]
        for subject, warning in issues:
            lines.append(f"  {subject}: {warning}")
        lines += [
            "",
            "Dropped samples held vectors that could not be compared. Demoted "
            "samples are kept as pending: select the sample in the subject's "
            "Reference samples list and use Approve sample once reviewed.",
        ]
        messagebox.showwarning(
            "Imported with changes", "\n".join(lines), parent=active_window(self.root)
        )
        self.banner.show(
            "warning",
            f"{len(issues)} sample(s) in {filename} were dropped or set aside for "
            "review. See the profile's sample list.",
        )

    # -- Sample review ------------------------------------------------------

    def _selected_sample_id(self) -> Optional[str]:
        selection = self.sample_tree.selection()
        return selection[0] if selection else None

    def _approve(self) -> None:
        sample_id = self._selected_sample_id()
        if self._selected is None or sample_id is None:
            self._status("Select a sample to approve.")
            return
        profile = self._current(self._selected)
        if profile is None:
            return
        sample = next((x for x in profile.samples if x.sample_id == sample_id), None)
        if (
            sample is not None
            and "Outlier" in (sample.notes or "")
            and not messagebox.askyesno(
                "Approve a sample unlike the reference?",
                "This sample sounds unlike the rest of "
                f"{profile.display_name}'s reference. It may be someone else, "
                "or the same person on a very different channel. Approving it "
                "makes it part of the reference every later comparison is "
                "measured against. Approve it only if you have listened and "
                "are sure it is them.",
                icon="warning",
                default="no",
                parent=active_window(self.root),
            )
        ):
            return
        if profile.approve_sample(sample_id) and self._save(profile):
            self._selected = profile
            self._render_detail()
            self._status("Sample promoted into the trusted reference.")

    def _remove(self) -> None:
        sample_id = self._selected_sample_id()
        if self._selected is None or sample_id is None:
            self._status("Select a sample to remove.")
            return
        profile = self._current(self._selected)
        if profile is None:
            return
        if not messagebox.askyesno(
            "Remove sample",
            f"Remove this sample from {profile.display_name}'s profile? This "
            "cannot be undone; the recording itself is not affected.",
            icon="warning",
            default="no",
            parent=active_window(self.root),
        ):
            return
        if profile.remove_sample(sample_id) and self._save(profile):
            self._selected = profile
            self.refresh()
            self._status("Sample removed.")

    def _current(self, profile: SpeakerProfile) -> Optional[SpeakerProfile]:
        """``profile`` as it is stored now, or None (said) if it is gone.

        What this page shows was loaded when the subject was selected. Saving
        that copy after another page had added samples - a transcript
        correction proposing them, say - wrote the old copy back and erased
        them. So every change starts from the stored subject.
        """
        current = find_speaker_profile_by_id(profile.subject_id)
        if current is None:
            self._status(f"{profile.display_name} is no longer stored.")
            self.refresh()
        return current

    def _save(self, profile: SpeakerProfile) -> bool:
        try:
            save_speaker_profile(profile)
            self._announce_change()
            return True
        except ProfileError as exc:
            # Losing operational work silently is not acceptable; surface it.
            messagebox.showerror(
                "Could not save", str(exc), parent=active_window(self.root)
            )
            self._status(friendly_error(exc))
            return False

    # -- Enrolment ----------------------------------------------------------

    def _add_recording(self) -> None:
        profile = self._selected
        if profile is None:
            self._status("Select or create a subject first.")
            return
        patterns = " ".join(f"*{ext}" for ext in AUDIO_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Historical recording containing this subject",
            filetypes=[("Audio/Video", patterns), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if not path:
            return
        kind = listening.derived_kind(path)
        if kind is not None:
            # A reference voice is the thing every later comparison is measured
            # against; building one from processed audio quietly biases all of
            # it, whichever kind of processing produced the file.
            self._status(
                f"That is a {kind}, not the recording. A reference voice must "
                "be built from original audio — processing changes the detail "
                "a voiceprint measures."
            )
            return
        choice = self._ask_mode()
        if choice is None:
            return
        mode, ranges_text = choice
        spans: Optional[List[Tuple[float, float]]] = None
        if mode == MODE_RANGES:
            try:
                spans = parse_time_ranges(ranges_text)
            except ValueError as exc:
                self._status(f"Couldn't read the time ranges: {exc}")
                return
        threading.Thread(
            target=self._enroll_worker,
            args=(profile, Path(path), mode, spans),
            daemon=True,
        ).start()

    def _ask_mode(self) -> Optional[Tuple[str, str]]:
        """Ask which speech in the recording belongs to the subject."""
        win = tk.Toplevel(self.root)
        win.title("Which speech is the subject?")
        win.configure(background=palette().surface)
        win.resizable(False, False)
        mode_var = tk.StringVar(value=MODE_WHOLE)
        ranges_var = tk.StringVar(value="")
        chosen: dict = {"value": None}

        frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
        frame.pack(fill="both", expand=True)
        ttk.Label(
            frame, text="Which speech is the subject?", style=Style.SECTION_TITLE
        ).pack(anchor="w", pady=(0, SPACE_SM))
        for text, value in (
            ("The whole recording is this subject", MODE_WHOLE),
            ("Several people — separate speakers and let me pick", MODE_DIARIZE),
            ("I'll give the time ranges", MODE_RANGES),
        ):
            ttk.Radiobutton(
                frame,
                text=text,
                variable=mode_var,
                value=value,
                style=Style.RADIO,
            ).pack(anchor="w")
        ttk.Entry(frame, textvariable=ranges_var, width=44).pack(
            anchor="w", padx=(SPACE_XL, 0), pady=(SPACE_XS, 0)
        )
        ttk.Label(frame, text="e.g. 0:10-0:45, 1:20-2:00", style=Style.META).pack(
            anchor="w", padx=(SPACE_XL, 0)
        )

        def _ok() -> None:
            chosen["value"] = (mode_var.get(), ranges_var.get())
            win.destroy()

        row = ttk.Frame(frame, style=Style.CARD_INNER)
        row.pack(fill="x", pady=(SPACE_MD, 0))
        secondary_button(row, "Cancel", win.destroy).pack(side="right")
        primary_button(row, "Continue", _ok).pack(side="right", padx=(0, SPACE_SM))
        win.bind("<Return>", lambda _e: _ok())
        present(win, self.root, modal=True)
        win.wait_window()
        return chosen["value"]

    def _get_embedder(self) -> SpeakerEmbedder:
        """Load the speaker-embedding model once (sherpa loads on first use)."""
        if self._embedder is None:
            self._embedder = SpeakerEmbedder()
        return self._embedder

    def _enroll_worker(
        self,
        profile: SpeakerProfile,
        source: Path,
        mode: str,
        spans: Optional[List[Tuple[float, float]]],
    ) -> None:
        wav = None
        temporary = False
        try:
            # Enrol into the subject as stored, not the copy this page loaded,
            # so samples another page added since are kept.
            stored = find_speaker_profile_by_id(profile.subject_id)
            if stored is None:
                self._status(f"{profile.display_name} is no longer stored.")
                self.root.after(0, self.refresh)
                return
            profile = stored
            self._status(f"Preparing {source.name}…")
            embedder = self._get_embedder()
            wav, digest, temporary = prepare_source(source, progress=self._status)
            if mode == MODE_DIARIZE:
                spans = self._diarize_and_pick(wav, profile.display_name)
                if spans is None:
                    self._status(
                        "No speakers were found in that recording, so nothing "
                        f"was added to {profile.display_name}."
                    )
                    return
            if spans is None:
                spans = [(0.0, _duration(wav))]
            result = enroll_from_wav(
                profile,
                wav,
                spans,
                embedder,
                source_filename=source.name,
                source_sha256=digest,
                progress=self._status,
            )
            if result.added:
                # Not self._save(): that puts up a message box, and this is a
                # worker thread. A ProfileError lands in the handler below.
                save_speaker_profile(profile)
                self.root.after(0, self._announce_change)
            self.root.after(0, self.refresh)
            quality = result.quality.assessment if result.quality else "n/a"
            message = (
                f"Enrolled {result.added_count} sample(s), "
                f"{result.added_seconds:.1f}s of speech (quality: {quality})."
            )
            if result.skipped:
                message += f" Skipped {len(result.skipped)}: {result.skipped[0]}"
            self._status(message)
        except PickCancelled:
            self._status(
                "Cancelled. No speaker was chosen, so nothing was added to "
                f"{profile.display_name}."
            )
        except Exception as exc:  # noqa: BLE001 - surfaced to the operator
            self._fail(
                f"Nothing was added to {profile.display_name}. {friendly_error(exc)}"
            )
        finally:
            if temporary and wav is not None:
                try:
                    Path(wav).unlink()
                except OSError:
                    pass

    def _diarize_and_pick(
        self, wav: Path, subject: str
    ) -> Optional[List[Tuple[float, float]]]:
        """Diarize the recording and let the operator choose the subject's speech.

        None when no speaker was found; raises :class:`PickCancelled` when the
        operator closes the picker.
        """
        from ..diarization import diarize

        self._status("Separating speakers…")
        segments = diarize(wav, progress=self._status)
        options = speaker_options(segments)
        if not options:
            return None
        speaker = ask_from_worker(
            self.root,
            lambda: ask_pick_speaker(
                self.root,
                title=f"Which speaker is {subject}?",
                guidance=(
                    f"The speaker you choose becomes reference audio for "
                    f"{subject}, which every later comparison is measured "
                    "against. Listen to each one before choosing: the speaker "
                    "numbers come from the order they were found, and the "
                    "longest talker is not always the subject."
                ),
                options=options,
                listener=SampleListener.for_file(wav),
            ),
        )
        if speaker is None:
            raise PickCancelled()
        return spans_for_speaker(segments, speaker)

    # -- Shared hooks -------------------------------------------------------

    def _status(self, message: str) -> None:
        self.root.after(0, lambda: self.status_var.set(message))

    def _fail(self, message: str) -> None:
        """A failure, in the banner as well as the status line (any thread)."""
        self._status(message)
        self.root.after(0, lambda: self.banner.show("error", message))

    def get_settings(self) -> dict:
        return {}

    def apply_settings(self, data: dict) -> None:
        return None


def _duration(wav_path: Path) -> float:
    import wave

    with wave.open(str(wav_path), "rb") as wav:
        return wav.getnframes() / float(wav.getframerate() or 1)


_REPLACE = "replace"
_KEEP_BOTH = "keep_both"
_SKIP = "skip"


def _describe_subject(profile: SpeakerProfile) -> str:
    when = (profile.updated_utc or "")[:10] or "date not recorded"
    return (
        f"{len(profile.trusted_samples())} trusted and "
        f"{len(profile.pending_samples())} pending sample(s), last changed {when}"
    )


def _ask_import_conflict(
    root: tk.Misc, existing: SpeakerProfile, incoming: SpeakerProfile
) -> str:
    """Ask what to do with an imported subject that is already here.

    Keeping both is the safe choice and has the focus: nothing is lost, and
    the operator can delete whichever copy they decide against. Merging the
    samples is deliberately not offered - it means settling conflicting
    trusted and pending states between two histories, and doing that wrongly
    corrupts a reference voice silently.
    """
    win = tk.Toplevel(root)
    win.title("Subject already here")
    win.configure(background=palette().surface)
    win.resizable(False, False)
    frame = ttk.Frame(win, padding=SPACE_XL, style=Style.CARD)
    frame.pack(fill="both", expand=True)
    ttk.Label(
        frame,
        text=f"{existing.display_name} is already here",
        style=Style.SECTION_TITLE,
    ).pack(anchor="w")
    for heading, profile in (
        ("On this machine", existing),
        ("In the file being imported", incoming),
    ):
        ttk.Label(frame, text=heading, style=Style.FIELD_LABEL).pack(
            anchor="w", pady=(SPACE_MD, 0)
        )
        ttk.Label(
            frame,
            text=f"{profile.display_name}: {_describe_subject(profile)}",
            style=Style.META,
            wraplength=460,
            justify="left",
        ).pack(anchor="w")
    ttk.Label(
        frame,
        text=(
            "Replacing discards the copy on this machine, and any samples it "
            "has that the imported file does not."
        ),
        style=Style.WARNING,
        wraplength=460,
        justify="left",
    ).pack(anchor="w", pady=(SPACE_LG, SPACE_MD))

    chosen = {"value": _SKIP}

    def choose(value: str) -> None:
        chosen["value"] = value
        win.destroy()

    buttons = ttk.Frame(frame, style=Style.CARD_INNER)
    buttons.pack(fill="x")
    secondary_button(buttons, "Skip it", lambda: choose(_SKIP)).pack(side="right")
    secondary_button(buttons, "Replace", lambda: choose(_REPLACE)).pack(
        side="right", padx=(0, SPACE_SM)
    )
    keep = primary_button(buttons, "Keep both", lambda: choose(_KEEP_BOTH))
    keep.pack(side="left")
    present(win, root, modal=True)
    keep.focus_set()
    win.wait_window()
    return chosen["value"]
