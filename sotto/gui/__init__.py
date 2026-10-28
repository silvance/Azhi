"""Tkinter GUI for Sotto, split into focused modules.

- :mod:`sotto.gui.app` - the application shell and ``main`` entry point
- :mod:`sotto.gui.transcribe_tab` - the Transcribe tab (settings + run)
- :mod:`sotto.gui.translate_tab` - the Translate tab (text/OCR translation)
- :mod:`sotto.gui.transcript_view` - the interactive transcript pane
- :mod:`sotto.gui.widgets` / :mod:`sotto.gui.errors` - shared helpers

``sotto.app`` re-exports :func:`main` for the console script, ``python -m
sotto`` and the PyInstaller bundle entry.
"""

from .app import SottoApp, main

__all__ = ["SottoApp", "main"]
