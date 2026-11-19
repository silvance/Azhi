"""The Translate tab: offline foreign->English text translation, with OCR.

Paste text or batch files (translated to ``<name>.en.<ext>`` beside each
original); images and PDFs are OCR'd first when the OCR engine is bundled, and
a translated PDF or image also gets a Word copy, ``<name>.en.docx``.

A document is translated paragraph by paragraph (see :mod:`whispers.documents`):
English is left as it was, running page headers and footers are dropped, and
exported emails keep their header fields, with the subject translated. The
source language can be chosen explicitly or detected for each paragraph.
"""

from __future__ import annotations

import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from typing import Callable, Dict, List, Optional, Set, Tuple

from ..documents import parse, remove_page_furniture
from ..export import side_by_side_docx, text_to_docx, translation_to_docx
from ..ocr import (
    OCR_EXTENSIONS,
    extract_pages,
    is_ocr_file,
    is_pdf,
    tesseract_lang,
)
from ..passages import read_pdf
from ..side_by_side import render_text as render_side_by_side
from ..side_by_side import translate_pages, write_pdf
from ..textfiles import UnreadableText, read_text
from ..transcription import CancelledError
from ..translation import TranslatedDocument, translate_document
from .dialogs import active_window
from .errors import friendly_error
from .theme import SPACE_LG, SPACE_MD, SPACE_SM, SPACE_XS, Style
from .widgets import (
    Card,
    PageHeader,
    bind_wheel,
    primary_button,
    register_drop,
    scrollable_body,
    secondary_button,
    style_text_widget,
    subtle_button,
)

# Sentinel shown in the "From" dropdown for automatic language detection.
AUTO_DETECT_LABEL = "Auto-detect language"
AUTO_DETECT_CODE = "__auto__"
# Lets users OCR an English document by picking English explicitly. There is no
# English->English translation, so this is extract-only (the OCR'd text is the
# output). Mapped to the target code "en".
ENGLISH_LABEL = "English (OCR only, no translation)"
# What a PDF in the batch is written as.
SIDE_BY_SIDE = "side_by_side"
READING_COPY = "reading"
# When "From" is Auto-detect, a scanned page's script is detected before it is
# read (see whispers.ocr), so a Chinese or Arabic scan is not read as English.


class TranslateTab:
    """Builds and drives the Translate tab inside ``parent``."""

    def __init__(
        self,
        parent: ttk.Frame,
        root: tk.Misc,
        *,
        ocr_available: bool,
        detect_available: bool,
        dnd_ok: bool,
    ) -> None:
        self.parent = parent
        self.root = root
        # This page's own cancel signal. It was shared with every page, so
        # starting a translation cleared a transcription's pending cancel.
        self._cancel_event = threading.Event()
        # One job at a time, claimed on the UI thread before its worker starts.
        # Nothing stopped a second Translate click starting a second job, which
        # then cleared the first one's cancel.
        self._busy = False
        self._start_buttons: List[ttk.Button] = []
        self._ocr_available = ocr_available
        self._detect_available = detect_available
        self._dnd_ok = dnd_ok
        self._translate_lang_codes: dict[str, str] = {}
        # The last translation shown in Result, for Save as Word: the text it
        # was rendered as, and the document behind it.
        self._last: Optional[Tuple[str, TranslatedDocument]] = None
        # The text last extracted into the paste box, and the running headers
        # and footers left out of it, reported with its translation.
        self._extracted: Tuple[str, List[str]] = ("", [])
        self.keep_original_var = tk.BooleanVar(value=False)
        self.pdf_output_var = tk.StringVar(value=SIDE_BY_SIDE)
        self._build()

    # -- UI construction ---------------------------------------------------

    def _build(self) -> None:
        translate_canvas, container = scrollable_body(self.parent)

        PageHeader(
            container,
            "Translate",
            "Turn text, images or documents in another language into English.",
        ).pack(fill="x", pady=(0, SPACE_LG))

        languages = Card(container, "Languages")
        languages.pack(fill="x")
        lang_frame = languages.body
        lang_frame.columnconfigure(1, weight=1)
        ttk.Label(lang_frame, text="From", style=Style.FIELD_LABEL).grid(
            row=0, column=0, sticky="w", padx=(0, 8), pady=4
        )
        self.translate_from_var = tk.StringVar()
        self.translate_from_combo = ttk.Combobox(
            lang_frame, textvariable=self.translate_from_var, state="readonly", width=28
        )
        self.translate_from_combo.grid(row=0, column=1, sticky="w", pady=4)
        subtle_button(lang_frame, "Refresh", self._refresh_languages).grid(
            row=0, column=2, padx=(SPACE_SM, 0), pady=SPACE_XS
        )
        ttk.Label(lang_frame, text="To", style=Style.FIELD_LABEL).grid(
            row=1, column=0, sticky="w", padx=(0, SPACE_SM), pady=SPACE_XS
        )
        ttk.Label(lang_frame, text="English", style=Style.BODY).grid(
            row=1, column=1, sticky="w", pady=SPACE_XS
        )
        ttk.Checkbutton(
            lang_frame,
            text="Keep the original under each translated paragraph",
            variable=self.keep_original_var,
            style=Style.CHECK,
        ).grid(row=2, column=0, columnspan=3, sticky="w", pady=(SPACE_SM, 0))
        ttk.Label(
            lang_frame,
            text=(
                "Each paragraph is translated on its own: English is left as it "
                "was, and with a language chosen under From, other languages are "
                "still detected. Repeated page headers and footers are left out, "
                "and an exported email keeps its header, with the subject "
                "translated."
            ),
            wraplength=620,
            justify="left",
            style=Style.META,
        ).grid(row=3, column=0, columnspan=3, sticky="w", pady=(SPACE_XS, 0))
        self.translate_hint_var = tk.StringVar()
        ttk.Label(
            lang_frame,
            textvariable=self.translate_hint_var,
            wraplength=620,
            justify="left",
            style=Style.MUTED,
        ).grid(row=4, column=0, columnspan=3, sticky="w", pady=(SPACE_SM, 0))

        # --- Paste box -----------------------------------------------------
        paste_card = Card(container, "Text")
        paste_card.pack(fill="both", expand=True, pady=(SPACE_MD, 0))
        paste_frame = paste_card.body
        ttk.Label(
            paste_frame, text="Paste the text to translate", style=Style.FIELD_LABEL
        ).pack(anchor="w")
        self.translate_input = ScrolledText(paste_frame, wrap="word", height=6)
        style_text_widget(self.translate_input)
        self.translate_input.pack(fill="both", expand=True, pady=(SPACE_XS, SPACE_SM))
        paste_buttons = ttk.Frame(paste_frame, style=Style.CARD_INNER)
        paste_buttons.pack(fill="x")
        translate_button = primary_button(
            paste_buttons, "Translate", self._translate_paste_in_thread
        )
        translate_button.pack(side="left")
        self._start_buttons.append(translate_button)
        if self._ocr_available:
            # OCR a single image/PDF into the box so its text can be reviewed and
            # corrected before translating (OCR is rarely perfect).
            extract_button = secondary_button(
                paste_buttons, "Extract from image/PDF…", self._extract_to_paste
            )
            extract_button.pack(side="left", padx=(SPACE_SM, 0))
            self._start_buttons.append(extract_button)
        result_header = ttk.Frame(paste_frame, style=Style.CARD_INNER)
        result_header.pack(fill="x", pady=(SPACE_MD, 0))
        ttk.Label(result_header, text="Result", style=Style.FIELD_LABEL).pack(
            side="left"
        )
        secondary_button(result_header, "Copy", self._copy_translation).pack(
            side="right"
        )
        secondary_button(
            result_header, "Save as Word…", self._save_translation_docx
        ).pack(side="right", padx=(0, SPACE_SM))
        self.translate_output = ScrolledText(
            paste_frame, wrap="word", height=6, state="disabled"
        )
        style_text_widget(self.translate_output)
        self.translate_output.pack(fill="both", expand=True, pady=(SPACE_XS, 0))
        # Drop an image/PDF (or text file) onto the paste box to extract its text.
        register_drop(
            self.root, self._dnd_ok, self.translate_input, self._on_drop_source
        )

        # --- Batch files ---------------------------------------------------
        batch_card = Card(container, "Files")
        batch_card.pack(fill="x", pady=(SPACE_MD, 0))
        batch_frame = batch_card.body
        self._translate_files: List[Path] = []
        self.translate_files_var = tk.StringVar(value="No files selected.")
        row = ttk.Frame(batch_frame, style=Style.CARD_INNER)
        row.pack(fill="x")
        files_button = primary_button(
            row, "Translate files", self._translate_files_in_thread
        )
        files_button.pack(side="left")
        self._start_buttons.append(files_button)
        secondary_button(row, "Add files…", self._add_translate_files).pack(
            side="left", padx=(SPACE_SM, 0)
        )
        subtle_button(row, "Clear", self._clear_translate_files).pack(side="left")
        ttk.Label(
            batch_frame,
            textvariable=self.translate_files_var,
            wraplength=620,
            justify="left",
            style=Style.MUTED,
        ).pack(anchor="w", pady=(SPACE_SM, 0))
        batch_help = "Each file is translated to <name>.en.<ext> next to the original."
        if self._ocr_available:
            batch_help += (
                " Images and scanned pages are OCR'd first (the extracted text is "
                "also saved as <name>.ocr.txt), and their translation is also saved "
                "as a Word document, <name>.en.docx."
            )
        ttk.Label(
            batch_frame,
            text=batch_help,
            wraplength=620,
            justify="left",
            style=Style.META,
        ).pack(anchor="w")
        if self._ocr_available:
            ttk.Label(
                batch_frame, text="Write each PDF as", style=Style.FIELD_LABEL
            ).pack(anchor="w", pady=(SPACE_SM, 0))
            ttk.Radiobutton(
                batch_frame,
                text=(
                    "Side by side: each page with foreign-language text beside its "
                    "English, passages numbered (<name>.en.pdf, and a Word table)"
                ),
                variable=self.pdf_output_var,
                value=SIDE_BY_SIDE,
                style=Style.RADIO,
            ).pack(anchor="w")
            ttk.Radiobutton(
                batch_frame,
                text=(
                    "Reading copy: the translation as continuous text, one email at "
                    "a time"
                ),
                variable=self.pdf_output_var,
                value=READING_COPY,
                style=Style.RADIO,
            ).pack(anchor="w")
            ttk.Label(
                batch_frame,
                text=(
                    "Side by side suits forensic reports (Cellebrite, AXIOM) and "
                    "chats: every passage stays on its page, where it can be found "
                    "again. Use the reading copy to read a run of emails or a "
                    "letter through."
                ),
                wraplength=620,
                justify="left",
                style=Style.META,
            ).pack(anchor="w")
        # Drop files onto the batch box to add them to the queue.
        register_drop(self.root, self._dnd_ok, batch_frame, self._on_drop_files)

        # --- Run controls + progress --------------------------------------
        run_frame = ttk.Frame(container, style=Style.PAGE)
        run_frame.pack(fill="x", pady=(SPACE_LG, 0))
        run_frame.columnconfigure(1, weight=1)
        self.translate_cancel_button = secondary_button(
            run_frame, "Cancel", self._cancel
        )
        self.translate_cancel_button.configure(state="disabled")
        self.translate_cancel_button.grid(row=0, column=0, sticky="w")
        self.translate_progress = ttk.Progressbar(run_frame, mode="determinate")
        self.translate_progress.grid(row=0, column=1, sticky="ew", padx=(SPACE_MD, 0))
        self.translate_status_var = tk.StringVar(value="Ready")
        ttk.Label(
            run_frame,
            textvariable=self.translate_status_var,
            style=Style.PAGE_SUBTITLE,
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(SPACE_XS, 0))

        self._refresh_languages()
        bind_wheel(translate_canvas, container)

    # -- Cancellation ------------------------------------------------------

    def is_busy(self) -> bool:
        return self._busy

    def _cancel(self) -> None:
        """Ask this page's job to stop at its next checkpoint."""
        if not self._busy:
            return
        self._cancel_event.set()
        self.translate_cancel_button.configure(state="disabled")
        self.translate_status_var.set("Cancelling…")

    def _start(self, target: Callable[..., None], *args: object) -> None:
        """Run ``target`` on a worker, unless a job is already running."""
        if self._busy:
            self.translate_status_var.set(
                "Wait for the current job to finish, or cancel it."
            )
            return
        self._busy = True
        for button in self._start_buttons:
            button.configure(state="disabled")
        self._cancel_event.clear()

        def run() -> None:
            try:
                target(*args)
            except Exception as exc:  # noqa: BLE001 - e.g. a failed import
                # Each worker frees the page in its own finally; this covers
                # one that fails before reaching it, which would otherwise
                # leave every start button off for good.
                self._set_translate_busy(False, friendly_error(exc))

        threading.Thread(target=run, daemon=True).start()

    # -- Languages ---------------------------------------------------------

    def _refresh_languages(self) -> None:
        self.translate_hint_var.set("Loading languages…")
        threading.Thread(target=self._load_languages, daemon=True).start()

    def _load_languages(self) -> None:
        try:
            from ..translation import available_source_languages

            langs = available_source_languages()
            err: Optional[Exception] = None
        except Exception as exc:  # noqa: BLE001 - surfaced to the user as a hint
            langs, err = [], exc

        def _apply() -> None:
            self._translate_lang_codes = {name: code for code, name in langs}
            names = list(self._translate_lang_codes)
            # Offer automatic detection (langdetect) as the first choice.
            if names and self._detect_available:
                self._translate_lang_codes[AUTO_DETECT_LABEL] = AUTO_DETECT_CODE
                names = [AUTO_DETECT_LABEL] + names
            # Offer English explicitly so English images/PDFs can be OCR'd
            # (extract-only; there's no English->English translation).
            if names and self._ocr_available:
                self._translate_lang_codes[ENGLISH_LABEL] = "en"
                names = names + [ENGLISH_LABEL]
            self.translate_from_combo.configure(values=names)
            if names:
                if self.translate_from_var.get() not in names:
                    self.translate_from_var.set(names[0])
                self.translate_hint_var.set("")
            elif err is not None:
                self.translate_hint_var.set(friendly_error(err))
            else:
                self.translate_hint_var.set(
                    "No language packs found. Use a build with bundled packs, or "
                    "install Argos packs."
                )

        self.root.after(0, _apply)

    def _selected_from_code(self) -> Optional[str]:
        return self._translate_lang_codes.get(self.translate_from_var.get())

    def _installed_from_codes(self) -> set[str]:
        """Set of source language codes that have an installed pack to English."""
        # Exclude the auto sentinel and the target ("en"): there's no en->en pack.
        return {
            code
            for code in self._translate_lang_codes.values()
            if code not in (AUTO_DETECT_CODE, "en")
        }

    def _forced_code(self) -> Optional[str]:
        """The language chosen under From, or None for automatic detection."""
        selected = self._selected_from_code()
        if not selected or selected in (AUTO_DETECT_CODE, "en"):
            return None
        return selected

    def _language_names(self) -> Dict[str, str]:
        return {
            code: name
            for name, code in self._translate_lang_codes.items()
            if code not in (AUTO_DETECT_CODE, "en")
        }

    def _ocr_lang_code(self) -> Optional[str]:
        """Tesseract language for scanned pages, from the 'From' language.

        None (Auto-detect) has OCR detect each page's script; a chosen
        language is read alongside English.
        """
        selected = self._selected_from_code()
        if not selected or selected == AUTO_DETECT_CODE:
            return None
        return tesseract_lang(selected)

    def _progress(self, done: int, total: int, left: Optional[float]) -> None:
        self._set_translate_progress(done / max(1, total))
        when = f", {_duration(left)} left" if left is not None else ""
        self._set_translate_status(f"Translating passage {done} of {total}{when}…")

    def _translate(self, text: str, removed: List[str]) -> TranslatedDocument:
        """Translate the foreign paragraphs of ``text``, reporting progress."""
        return translate_document(
            parse(text),
            forced=self._forced_code(),
            installed=self._installed_from_codes(),
            names=self._language_names(),
            on_progress=self._progress,
            cancelled=self._cancel_event.is_set,
            removed=removed,
        )

    def _side_by_side(self, src: Path) -> Tuple[bool, str]:
        """Write the side-by-side translation of the PDF ``src``.

        Returns whether anything was translated, and what was found.
        """
        self._set_translate_status(f"Reading {src.name}…")
        pages = read_pdf(
            src,
            lang=self._ocr_lang_code(),
            progress=self._set_translate_status,
            cancelled=self._cancel_event.is_set,
        )
        # The extracted text as it came, even if none of it can be translated.
        src.with_name(f"{src.stem}.ocr.txt").write_text(
            "\n\n".join(page.text for page in pages), encoding="utf-8"
        )
        self._set_translate_status(f"Translating {src.name}…")
        triage = translate_pages(
            src,
            pages,
            forced=self._forced_code(),
            installed=self._installed_from_codes(),
            names=self._language_names(),
            on_progress=self._progress,
            cancelled=self._cancel_event.is_set,
        )
        if not triage.marked:
            return False, triage.summary()
        write_pdf(triage, _side_by_side_copy(src))
        src.with_name(f"{src.stem}.en.txt").write_text(
            render_side_by_side(triage), encoding="utf-8"
        )
        side_by_side_docx(triage, _word_copy(src))
        return True, triage.summary()

    def _removed_from(self, text: str) -> List[str]:
        """The page headers and footers left out of ``text`` when it was extracted.

        None once the paste box holds something else: corrections to the
        extracted text keep most of its lines, a different document does not.
        """
        extracted, removed = self._extracted
        lines = {line.strip() for line in extracted.split("\n") if line.strip()}
        if not removed or not lines:
            return []
        kept = lines & {line.strip() for line in text.split("\n")}
        return removed if len(kept) >= len(lines) / 2 else []

    # -- Batch file list ---------------------------------------------------

    def _add_translate_files(self) -> None:
        if self._ocr_available:
            ocr_patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
            filetypes = [
                ("Text, images & PDFs", f"*.txt {ocr_patterns}"),
                ("All files", "*.*"),
            ]
        else:
            filetypes = [("Text files", "*.txt"), ("All files", "*.*")]
        paths = filedialog.askopenfilenames(
            filetypes=filetypes, parent=active_window(self.root)
        )
        self._add_translate_paths([Path(raw) for raw in paths if raw])

    def _add_translate_paths(self, paths: List[Path]) -> None:
        for path in paths:
            if path not in self._translate_files:
                self._translate_files.append(path)
        self._update_translate_files_label()

    def _clear_translate_files(self) -> None:
        self._translate_files = []
        self._update_translate_files_label()

    def _update_translate_files_label(self) -> None:
        count = len(self._translate_files)
        if not count:
            self.translate_files_var.set("No files selected.")
            return
        names = ", ".join(p.name for p in self._translate_files[:5])
        more = "" if count <= 5 else f" (+{count - 5} more)"
        self.translate_files_var.set(f"{count} file(s): {names}{more}")

    # -- Thread-safe widget updates ---------------------------------------

    def _set_translate_busy(self, busy: bool, message: Optional[str] = None) -> None:
        def _do() -> None:
            self.translate_cancel_button.configure(
                state="normal" if busy else "disabled"
            )
            if not busy:
                self.translate_progress["value"] = 0
                self._busy = False
                for button in self._start_buttons:
                    button.configure(state="normal")
            self.translate_status_var.set(message or ("Working…" if busy else "Ready"))

        self.root.after(0, _do)

    def _set_translate_progress(self, fraction: float) -> None:
        pct = max(0.0, min(1.0, fraction)) * 100.0
        self.root.after(0, lambda: self.translate_progress.configure(value=pct))

    def _set_translate_status(self, message: str) -> None:
        self.root.after(0, lambda: self.translate_status_var.set(message))

    def _set_translate_output(self, text: str) -> None:
        def _do() -> None:
            self.translate_output.configure(state="normal")
            self.translate_output.delete("1.0", "end")
            self.translate_output.insert("end", text)
            self.translate_output.configure(state="disabled")

        self.root.after(0, _do)

    def _set_translate_input(self, text: str) -> None:
        def _do() -> None:
            self.translate_input.delete("1.0", "end")
            self.translate_input.insert("end", text)

        self.root.after(0, _do)

    # -- Translate (paste) -------------------------------------------------

    def get_settings(self) -> Dict[str, object]:
        return {
            "keep_original": bool(self.keep_original_var.get()),
            "pdf_output": self.pdf_output_var.get(),
        }

    def apply_settings(self, data: Dict[str, object]) -> None:
        self.keep_original_var.set(bool(data.get("keep_original", False)))
        output = data.get("pdf_output")
        if output in (SIDE_BY_SIDE, READING_COPY):
            self.pdf_output_var.set(str(output))

    def _writes_side_by_side(self, src: Path) -> bool:
        return (
            self._ocr_available
            and is_pdf(src)
            and self.pdf_output_var.get() == SIDE_BY_SIDE
        )

    def _translate_paste_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        text = self.translate_input.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to translate.")
            return
        self._start(self._translate_paste, text)

    def _translate_paste(self, text: str) -> None:
        self._set_translate_busy(True, "Translating…")
        final = "Done"
        try:
            if self._selected_from_code() == "en":
                self._set_translate_output(
                    "That's English already, so there's nothing to translate. To "
                    "pull text out of an image or PDF, use 'Extract from "
                    "image/PDF…'."
                )
                return
            document = self._translate(text, self._removed_from(text))
            shown = document.render_text(originals=self.keep_original_var.get())
            self._last = (shown, document)
            self._set_translate_output(shown)
            final = document.summary()
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(friendly_error(exc))
            final = "Error"
        finally:
            self._set_translate_busy(False, final)

    # -- Translate (batch files) ------------------------------------------

    def _translate_files_in_thread(self) -> None:
        if not self._selected_from_code():
            self.translate_status_var.set("Pick a 'From' language first.")
            return
        if not self._translate_files:
            self.translate_status_var.set("Add files first.")
            return
        files = list(self._translate_files)
        # Existing translations were overwritten without a word.
        existing = [
            out
            for src in files
            for out in _outputs(src, self._writes_side_by_side(src))
            if out.exists()
        ]
        keep: Set[Path] = set()
        if existing:
            replace = messagebox.askyesnocancel(
                "Replace existing translations?",
                f"{len(existing)} file(s) this would write already exist, for "
                f"example {existing[0].name}.\n\nYes replaces them. No leaves "
                "them and skips those recordings. Cancel stops here.",
                icon="warning",
                default="no",
                parent=active_window(self.root),
            )
            if replace is None:
                return
            if not replace:
                keep = {
                    src
                    for src in files
                    if any(
                        o.exists()
                        for o in _outputs(src, self._writes_side_by_side(src))
                    )
                }
        self._start(self._translate_files_worker, files, keep)

    def _translate_files_worker(self, files: List[Path], keep: Set[Path]) -> None:
        self._set_translate_busy(True, "Translating files…")
        final = "Done"
        translated_count = 0
        extracted_count = 0
        skipped = 0
        reasons: List[str] = []
        found: List[str] = []
        try:
            total = max(1, len(files))
            for index, src in enumerate(files):
                if self._cancel_event.is_set():
                    raise CancelledError("Translation cancelled.")
                if src in keep:
                    skipped += 1
                    reasons.append(f"{src.name}: already translated; left as it was.")
                    self._set_translate_progress((index + 1) / total)
                    continue
                if self._writes_side_by_side(src):
                    wrote, summary = self._side_by_side(src)
                    extracted_count += 1
                    if wrote:
                        translated_count += 1
                        found.append(
                            f"{src.name}: {summary} Written to "
                            f"{_side_by_side_copy(src).name}."
                        )
                    else:
                        found.append(f"{src.name}: {summary}")
                    self._set_translate_progress((index + 1) / total)
                    continue
                # Get the foreign text: read text files directly, OCR images/PDFs.
                is_ocr = is_ocr_file(src)
                removed: List[str] = []
                if is_ocr:
                    self._set_translate_status(f"Reading {src.name} (OCR)…")
                    pages = extract_pages(
                        src,
                        lang=self._ocr_lang_code(),
                        progress=self._set_translate_status,
                        cancelled=self._cancel_event.is_set,
                    )
                    # Always keep the extracted text as it came, page headers
                    # and all, even if none of it can be translated.
                    ocr_dest = src.with_name(f"{src.stem}.ocr.txt")
                    ocr_dest.write_text("\n\n".join(pages), encoding="utf-8")
                    extracted_count += 1
                    pages, removed = remove_page_furniture(pages)
                    text = "\n\n".join(pages)
                    dest = src.with_name(f"{src.stem}.en.txt")
                else:
                    try:
                        text = read_text(src)
                    except UnreadableText as exc:
                        skipped += 1
                        reasons.append(str(exc))
                        self._set_translate_progress((index + 1) / total)
                        continue
                    dest = src.with_name(f"{src.stem}.en{src.suffix}")

                self._set_translate_status(f"Translating {src.name}…")
                document = self._translate(text, removed)
                if not any(piece.translated for piece in document.pieces()):
                    # OCR files already produced .ocr.txt; text files yield nothing.
                    if is_ocr:
                        self._set_translate_status(
                            f"{src.name}: extracted text only (no foreign language "
                            "to translate)."
                        )
                    else:
                        skipped += 1
                        reasons.append(f"{src.name}: {document.summary()}")
                    self._set_translate_progress((index + 1) / total)
                    continue
                originals = self.keep_original_var.get()
                dest.write_text(
                    document.render_text(originals=originals), encoding="utf-8"
                )
                if is_ocr:
                    try:
                        translation_to_docx(
                            document,
                            _word_copy(src),
                            title=f"Translation of {src.name}",
                            originals=originals,
                        )
                    except RuntimeError as exc:  # python-docx missing
                        reasons.append(f"{src.name}: no Word copy ({exc})")
                translated_count += 1
                self._set_translate_status(f"Wrote {dest.name}")
                self._set_translate_progress((index + 1) / total)
            parts = []
            if translated_count:
                parts.append(f"translated {translated_count}")
            if extracted_count:
                parts.append(f"extracted {extracted_count}")
            if skipped:
                parts.append(f"skipped {skipped}")
            final = "Done: " + (", ".join(parts) if parts else "nothing to do")
            # The status line holds one sentence; what each PDF held, and
            # why any file was skipped, go where they can be read.
            report = []
            if found:
                report.append("Found:\n\n" + "\n\n".join(found))
            if reasons:
                report.append("Skipped:\n\n" + "\n\n".join(reasons))
            if report:
                self._set_translate_output("\n\n".join(report))
                final += " (see Result)"
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            # Show the cause in the Result box; the status line is too short and
            # gets overwritten by the final "busy off" message below.
            self._set_translate_output(friendly_error(exc))
            final = "Error: see the Result box"
        finally:
            self._set_translate_busy(False, final)

    # -- OCR extract to the paste box --------------------------------------

    def _extract_to_paste(self) -> None:
        """OCR a single image/PDF into the paste box for review before translating."""
        lang = self._ocr_lang_code()
        patterns = " ".join(f"*{ext}" for ext in OCR_EXTENSIONS)
        path = filedialog.askopenfilename(
            title="Choose an image or PDF",
            filetypes=[("Images & PDFs", patterns), ("All files", "*.*")],
            parent=active_window(self.root),
        )
        if not path:
            return
        self._start(self._extract_to_paste_worker, Path(path), lang)

    def _extract_to_paste_worker(self, path: Path, lang: Optional[str]) -> None:
        self._set_translate_busy(True, f"Reading {path.name} (OCR)…")
        final = "Extracted. Review it, then Translate."
        try:
            pages = extract_pages(
                path,
                lang=lang,
                progress=self._set_translate_status,
                cancelled=self._cancel_event.is_set,
            )
            pages, removed = remove_page_furniture(pages)
            text = "\n\n".join(pages).strip()
            self._extracted = (text, removed)
            self._set_translate_input(text)
            if not text.strip():
                final = "No text found in that file."
            elif removed:
                final = (
                    f"Extracted {len(pages)} page(s), leaving out {len(removed)} "
                    "repeated header, footer and page-number line(s). Review it, "
                    "then Translate."
                )
        except CancelledError:
            final = "Cancelled"
        except Exception as exc:  # noqa: BLE001
            self._set_translate_output(friendly_error(exc))
            final = "Error — see Result box"
        finally:
            self._set_translate_busy(False, final)

    # -- Export / clipboard ------------------------------------------------

    def _copy_translation(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to copy yet.")
            return
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.translate_status_var.set("Translation copied to clipboard.")

    def _save_translation_docx(self) -> None:
        text = self.translate_output.get("1.0", "end-1c")
        if not text.strip():
            self.translate_status_var.set("Nothing to save yet.")
            return
        path = filedialog.asksaveasfilename(
            title="Save translation as Word",
            defaultextension=".docx",
            initialfile="translation.docx",
            filetypes=[("Word document", "*.docx")],
            parent=active_window(self.root),
        )
        if not path:
            return
        try:
            if self._last is not None and self._last[0].rstrip() == text.rstrip():
                translation_to_docx(
                    self._last[1], path, originals=self.keep_original_var.get()
                )
            else:
                text_to_docx(text, path)
            self.translate_status_var.set(f"Saved {Path(path).name}")
        except Exception as exc:  # noqa: BLE001
            self.translate_status_var.set(friendly_error(exc))

    # -- Drag-and-drop -----------------------------------------------------

    def _on_drop_source(self, paths: List[Path]) -> None:
        """A file dropped on the paste box: OCR an image/PDF, or load a text file."""
        path = paths[0]
        if is_ocr_file(path):
            lang = self._ocr_lang_code()
            self._start(self._extract_to_paste_worker, path, lang)
        else:
            try:
                self._extracted = ("", [])
                self._set_translate_input(read_text(path))
            except UnreadableText as exc:
                self._set_translate_output(str(exc))
                self.translate_status_var.set("Not loaded: see Result")
            except OSError as exc:
                self.translate_status_var.set(friendly_error(exc))

    def _on_drop_files(self, paths: List[Path]) -> None:
        """Files dropped on the batch box: add them to the queue."""
        self._add_translate_paths(paths)
        self.translate_status_var.set(f"Added {len(paths)} file(s) to the batch.")


def _outputs(src: Path, side_by_side: bool = False) -> List[Path]:
    """The files translating ``src`` writes beside it."""
    if is_ocr_file(src):
        out = [
            src.with_name(f"{src.stem}.ocr.txt"),
            src.with_name(f"{src.stem}.en.txt"),
            _word_copy(src),
        ]
        return out + [_side_by_side_copy(src)] if side_by_side else out
    return [src.with_name(f"{src.stem}.en{src.suffix}")]


def _side_by_side_copy(src: Path) -> Path:
    """Where the side-by-side translation of a PDF is written."""
    return src.with_name(f"{src.stem}.en.pdf")


def _word_copy(src: Path) -> Path:
    """Where the Word copy of a translated PDF or image is written."""
    return src.with_name(f"{src.stem}.en.docx")


def _duration(seconds: Optional[float]) -> str:
    """Roughly how long, in words: "about 40 sec", "about 12 min", "about 2 h 5 min"."""
    if seconds is None:
        return ""
    if seconds < 60:
        return f"about {max(1, round(seconds))} sec"
    minutes = round(seconds / 60)
    if minutes < 60:
        return f"about {minutes} min"
    return f"about {minutes // 60} h {minutes % 60} min"
