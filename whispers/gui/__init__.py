"""Tkinter GUI for Whispers, split into focused modules.

- :mod:`whispers.gui.app` - the application shell and ``main`` entry point
- :mod:`whispers.gui.transcribe_tab` - the Transcribe tab (settings + run)
- :mod:`whispers.gui.translate_tab` - the Translate tab (text/OCR translation)
- :mod:`whispers.gui.transcript_view` - the interactive transcript pane
- :mod:`whispers.gui.widgets` / :mod:`whispers.gui.errors` - shared helpers

``whispers.app`` re-exports :func:`main` for the console script, ``python -m
whispers`` and the PyInstaller bundle entry.
"""

from .app import WhispersApp, main

__all__ = ["WhispersApp", "main"]
