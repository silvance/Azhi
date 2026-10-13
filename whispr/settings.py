"""Persist GUI preferences, and administrator settings, between launches.

One small JSON file in the user's config location holds the last-used model,
output folder and options *and* anything an administrator configured there -
the comparison thresholds, which the Thresholds dialog tells them to set in
this file. So the file is shared, and writing it means changing one section and
keeping the rest.

Reading stays forgiving: a missing or unreadable file means nothing is
remembered, never a crash at startup. Writing does not: a save that fails is
reported to the caller, so the operator can be told, and a file that cannot be
read is left alone rather than replaced - it may hold an administrator's
thresholds that a corrupt byte has made unreadable.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

from .atomicio import write_json_atomic


class SettingsError(OSError):
    """Settings could not be saved, in words an operator can act on."""


def settings_path() -> Path:
    """Per-user settings file location (``%APPDATA%`` on Windows, XDG elsewhere)."""
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
        return Path(base) / "Whispers" / "settings.json"
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "whispr" / "settings.json"


def load_settings() -> Dict[str, Any]:
    """Return the saved settings, or an empty dict if none/unreadable."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt file -> no saved settings
        return {}
    return data if isinstance(data, dict) else {}


def update_settings(section: str, value: Any) -> Path:
    """Replace one section of the settings file and keep every other.

    Closing the application used to write ``{"transcribe": ...}`` over the
    whole file, which erased the comparison thresholds an administrator had
    configured in it: the next session quietly went back to the defaults. It
    also wrote in place, and swallowed any error.

    Raises :class:`SettingsError` when the file cannot be written, or when it
    exists but cannot be read - overwriting that would destroy whatever it held.
    """
    path = settings_path()
    data: Dict[str, Any] = {}
    if path.exists():
        try:
            loaded = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise SettingsError(
                f"The settings file {path} could not be read, so it was left as "
                "it is rather than replaced. Your choices from this session were "
                f"not saved. ({exc})"
            ) from exc
        if isinstance(loaded, dict):
            data = loaded
    data[section] = value
    try:
        return write_json_atomic(path, data)
    except OSError as exc:
        raise SettingsError(
            f"Your settings could not be saved to {path}: "
            f"{exc.strerror or exc}. They will not be remembered next time."
        ) from exc
