"""Persist GUI preferences, and administrator settings, between launches.

One small JSON file in the user's config location holds the last-used model,
output folder and options *and* anything an administrator configured there -
the comparison thresholds, which the Thresholds dialog tells them to set in
this file. So the file is shared, and writing it means changing one section and
keeping the rest.

Reading stays forgiving: a missing or unreadable file means nothing is
remembered, never a crash at startup. Writing does not: a save that fails is
reported to the caller, so the operator can be told, and a file that cannot be
parsed is moved aside rather than replaced - it may hold an administrator's
thresholds that a corrupt byte has made unreadable.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from .atomicio import write_json_atomic


class SettingsError(OSError):
    """Settings could not be saved, in words an operator can act on."""


# What earlier versions called the program in the names of its folders and
# files, newest first: Whispers, and before that Whispr. Kept only to find what
# they saved and bring it across.
OLD_NAMES = ("whispers", "whispr")
# On Windows the folder was only ever "Whispers".
OLD_WINDOWS_NAMES = ("Whispers",)


def settings_path() -> Path:
    """Per-user settings file location (``%APPDATA%`` on Windows, XDG elsewhere).

    Beside it are the speaker profiles, operation profiles and comparison
    records, so an earlier version's folder (``Whispers`` on Windows;
    ``whispers`` or ``whispr`` elsewhere) is renamed the first time it is
    looked for (see :func:`_settings_folder`).
    """
    if os.name == "nt":
        base = os.environ.get("APPDATA") or str(Path.home())
        return (
            _settings_folder(Path(base), "Sotto", OLD_WINDOWS_NAMES) / "settings.json"
        )
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return _settings_folder(Path(base), "sotto", OLD_NAMES) / "settings.json"


def _settings_folder(base: Path, name: str, old_names: "Sequence[str]") -> Path:
    """``base/name``, after moving an earlier version's folder there if need be.

    A rename within one folder, so the voice profiles in it are never copied
    and never in two places. If it cannot be renamed, it is used where it is:
    starting an empty folder would make every profile look lost. With more
    than one earlier folder, the newest is the one taken.
    """
    folder = base / name
    if folder.exists():
        return folder
    for old_name in old_names:
        old = base / old_name
        if not old.is_dir():
            continue
        try:
            old.rename(folder)
        except OSError:
            return old
        return folder
    return folder


def adopt_old_files(
    folder: Path, suffix: str, old_suffixes: "Union[str, Sequence[str]]"
) -> "List[Path]":
    """Files in ``folder`` ending ``suffix``, renaming any an earlier version left.

    Earlier versions saved the same files under ``old_suffixes`` (newest
    first). Each is renamed in place, once, so it is listed once; one that
    cannot be renamed, or whose new name is already taken, is returned as it is
    rather than left out. Sorted by name; empty when the folder cannot be read.
    """
    olds = [old_suffixes] if isinstance(old_suffixes, str) else list(old_suffixes)
    try:
        for old_suffix in olds:
            for path in sorted(folder.glob(f"*{old_suffix}")):
                target = path.with_name(path.name[: -len(old_suffix)] + suffix)
                if target.exists():
                    continue
                try:
                    path.rename(target)
                except OSError:
                    pass
        current = set(folder.glob(f"*{suffix}"))
        left = {
            path
            for old_suffix in olds
            for path in folder.glob(f"*{old_suffix}")
            if not path.with_name(path.name[: -len(old_suffix)] + suffix).exists()
        }
    except OSError:
        return []
    return sorted(current | left)


def load_settings() -> Dict[str, Any]:
    """Return the saved settings, or an empty dict if none/unreadable."""
    try:
        data = json.loads(settings_path().read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt file -> no saved settings
        return {}
    return data if isinstance(data, dict) else {}


@dataclass
class SettingsSaved:
    """What saving did: where it wrote, and any unreadable file it moved aside."""

    path: Path
    set_aside: Optional[Path] = None


def update_settings(section: str, value: Any) -> SettingsSaved:
    """Replace one section of the settings file and keep every other.

    Closing the application used to write ``{"transcribe": ...}`` over the
    whole file, which erased the comparison thresholds an administrator had
    configured in it: the next session quietly went back to the defaults. It
    also wrote in place, and swallowed any error.

    A file that exists but cannot be parsed is moved aside under a dated name
    and a new one started, rather than replaced or left in the way. Leaving it
    meant a warning on every close, forever, with nothing ever saved again - and
    a 0-byte file, which is what the old in-place writer left after a crash, was
    enough to cause it. Moving it keeps whatever it held recoverable. An empty
    file simply counts as empty.

    Raises :class:`SettingsError` when the file cannot be read or written.
    """
    path = settings_path()
    data: Dict[str, Any] = {}
    set_aside: Optional[Path] = None
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise SettingsError(
                f"The settings file {path} could not be opened, so your choices "
                f"from this session were not saved: {exc.strerror or exc}."
            ) from exc
        if raw.strip():
            try:
                loaded = json.loads(raw)
            except ValueError:
                loaded = None
            if isinstance(loaded, dict):
                data = loaded
            else:
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
                set_aside = path.with_name(f"{path.name}.unreadable-{stamp}")
                try:
                    os.replace(path, set_aside)
                except OSError as exc:
                    raise SettingsError(
                        f"The settings file {path} could not be read, and could "
                        "not be moved aside either, so your choices from this "
                        f"session were not saved: {exc.strerror or exc}."
                    ) from exc
    data[section] = value
    try:
        written = write_json_atomic(path, data)
    except OSError as exc:
        raise SettingsError(
            f"Your settings could not be saved to {path}: "
            f"{exc.strerror or exc}. They will not be remembered next time."
        ) from exc
    return SettingsSaved(written, set_aside)
