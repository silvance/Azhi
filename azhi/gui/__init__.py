"""Tkinter GUI for Azhi, split into focused modules.

- :mod:`azhi.gui.app` - the application shell and ``main`` entry point
- :mod:`azhi.gui.transcribe_tab` - the Transcribe tab (settings + run)
- :mod:`azhi.gui.translate_tab` - the Translate tab (text/OCR translation)
- :mod:`azhi.gui.transcript_view` - the interactive transcript pane
- :mod:`azhi.gui.widgets` / :mod:`azhi.gui.errors` - shared helpers

``azhi.app`` re-exports :func:`main` for the console script, ``python -m
azhi`` and the PyInstaller bundle entry.
"""

from .app import AzhiApp, main

__all__ = ["AzhiApp", "main"]
