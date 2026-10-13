"""Per-operation profiles: saved settings + learned speaker voiceprints.

A *profile* bundles everything that is specific to one recurring operation or
team so an operator can pick it once and get consistent, improving results:

* the transcription settings to use (model, diarization engine, expected speaker
  count, custom vocabulary, ...);
* the roster of speaker voiceprints learned from that operation's corrections
  (see :mod:`whispr.voiceprints`).

Profiles live as small JSON files under the same per-user config location as the
GUI settings, so they persist between launches and survive app updates. As with
:mod:`whispr.settings`, all disk operations are best-effort: a missing or
read-only config dir just means profiles can't be saved, never a crash.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .atomicio import write_json_atomic
from .settings import settings_path
from .voiceprints import Voiceprint


def profiles_dir() -> Path:
    """Directory holding the per-user profile files (beside ``settings.json``)."""
    return settings_path().parent / "profiles"


def _slug(name: str) -> str:
    """A filesystem-safe stem for ``name`` (keeps letters/digits/-/_)."""
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in name.strip()
    ).strip("_")
    return cleaned.lower() or "profile"


@dataclass
class Profile:
    """One operation's saved settings and learned speaker voiceprints."""

    name: str
    settings: Dict[str, object] = field(default_factory=dict)
    voiceprints: Dict[str, Voiceprint] = field(default_factory=dict)

    def voiceprint_for(self, speaker_name: str) -> Voiceprint:
        """The voiceprint for ``speaker_name``, created (empty) if new."""
        vp = self.voiceprints.get(speaker_name)
        if vp is None:
            vp = Voiceprint(name=speaker_name)
            self.voiceprints[speaker_name] = vp
        return vp

    def to_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "settings": self.settings,
            "voiceprints": [vp.to_dict() for vp in self.voiceprints.values()],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Profile":
        name = str(data.get("name", "")) or "Profile"
        settings = data.get("settings")
        settings = settings if isinstance(settings, dict) else {}
        voiceprints: Dict[str, Voiceprint] = {}
        raw = data.get("voiceprints") or []
        if isinstance(raw, list):
            for item in raw:
                if isinstance(item, dict):
                    vp = Voiceprint.from_dict(item)
                    if vp.name:
                        voiceprints[vp.name] = vp
        return cls(name=name, settings=settings, voiceprints=voiceprints)


def list_profiles() -> List[str]:
    """Names of the saved profiles, sorted, or ``[]`` if none/unreadable."""
    names: List[str] = []
    try:
        entries = sorted(profiles_dir().glob("*.json"))
    except OSError:
        return names
    for path in entries:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001 - skip a corrupt/unreadable profile
            continue
        if isinstance(data, dict) and data.get("name"):
            names.append(str(data["name"]))
    return sorted(names)


class ProfileSaveError(OSError):
    """An operation profile could not be saved or removed, in plain words."""


def profile_path(name: str) -> Path:
    """The file the profile called ``name`` is kept in."""
    return profiles_dir() / f"{_slug(name)}.json"


def load_profile(name: str) -> Optional[Profile]:
    """Load the profile called ``name``, or ``None`` if missing/unreadable."""
    path = profile_path(name)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001 - missing/corrupt -> no profile
        return None
    if not isinstance(data, dict):
        return None
    return Profile.from_dict(data)


def existing_profile_name(name: str) -> Optional[str]:
    """The profile that saving under ``name`` would replace, or None.

    Profiles are stored by a simplified file name, so different names can share
    one file: "Op Alpha" and "op alpha" both become ``op_alpha.json``. Creating
    the second used to replace the first without a word, learned voices and
    all. This answers with the *stored* profile's own name, so the question put
    to the operator names the thing they would lose. A file that exists but
    cannot be read still counts as taken - replacing it would destroy it too.
    """
    path = profile_path(name)
    if not path.exists():
        return None
    existing = load_profile(name)
    return existing.name if existing is not None else path.name


def save_profile(profile: Profile) -> Path:
    """Write ``profile`` atomically. Raises :class:`ProfileSaveError` on failure.

    It used to write in place and swallow every error, while the page went on
    to say "Saved", "Created" or "Learned".
    """
    path = profile_path(profile.name)
    try:
        return write_json_atomic(path, profile.to_dict())
    except OSError as exc:
        raise ProfileSaveError(
            f"The profile \u2018{profile.name}\u2019 could not be saved to "
            f"{path}: {exc.strerror or exc}."
        ) from exc


# One read-change-write at a time, so two learning runs finishing together
# cannot each re-read the same copy and keep only the last one's voices.
_UPDATE_LOCK = threading.Lock()


def _stored_for_update(name: str) -> Profile:
    """The stored profile ``name``, re-read now, for a change to be applied to.

    Raises :class:`ProfileSaveError` when there is nothing safe to change: the
    file has gone, cannot be read (it is left as it is), or now holds a
    different profile because it was replaced under another name.
    """
    path = profile_path(name)
    stored = load_profile(name)
    if stored is None:
        if path.exists():
            raise ProfileSaveError(
                f"The profile \u2018{name}\u2019 could not be read, so it was "
                "left as it is."
            )
        raise ProfileSaveError(
            f"The profile \u2018{name}\u2019 is no longer on this computer."
        )
    if stored.name != name:
        raise ProfileSaveError(
            f"The profile \u2018{name}\u2019 has been replaced by "
            f"\u2018{stored.name}\u2019. Choose that profile to go on."
        )
    return stored


def add_learned_voice(
    name: str, speaker: str, vectors: Sequence[Sequence[float]]
) -> Profile:
    """Add newly learned ``vectors`` for ``speaker`` to the stored profile ``name``.

    The profile is re-read first and only that speaker's voice changes. Learning
    takes a few seconds, and saving the copy held when it started wrote over
    whatever was stored in between - an import, a replacement, another window's
    learning - without a word. Returns the profile as saved.
    """
    with _UPDATE_LOCK:
        stored = _stored_for_update(name)
        voiceprint = stored.voiceprint_for(speaker)
        for vector in vectors:
            voiceprint.add(vector)
        save_profile(stored)
    return stored


def store_profile_settings(name: str, settings: Dict[str, object]) -> Profile:
    """Replace the stored profile's settings, keeping the voices it has learned.

    The voices are the ones on disk now, not the ones in memory, for the same
    reason as :func:`add_learned_voice`. Returns the profile as saved.
    """
    with _UPDATE_LOCK:
        stored = _stored_for_update(name)
        stored.settings = dict(settings)
        save_profile(stored)
    return stored


def delete_profile(name: str) -> None:
    """Remove the profile called ``name``. Raises if it is there and cannot go."""
    path = profile_path(name)
    try:
        path.unlink()
    except FileNotFoundError:
        return
    except OSError as exc:
        raise ProfileSaveError(
            f"The profile \u2018{name}\u2019 could not be deleted: "
            f"{exc.strerror or exc}."
        ) from exc


# Suffix for exported/shared profile files (a single self-contained JSON that
# carries the settings and the learned voiceprints). Used to filter file dialogs.
PROFILE_SUFFIX = ".whispr-profile.json"


def export_profile(profile: Profile, path: "str | Path") -> None:
    """Write ``profile`` to ``path`` as a shareable JSON file.

    Unlike :func:`save_profile` this targets a user-chosen location and raises on
    failure (the caller surfaces it), so a failed export isn't silently lost.
    """
    Path(path).write_text(
        json.dumps(profile.to_dict(), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_profile_file(path: "str | Path") -> Profile:
    """Load a :class:`Profile` from an exported file (for import).

    Raises ``ValueError`` if the file isn't a valid profile so the caller can show
    a clear message. Does not save it - the caller decides where/whether to store
    it (e.g. after resolving a name clash) with :func:`save_profile`.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Couldn't read the profile file: {exc}") from exc
    if not isinstance(data, dict) or not data.get("name"):
        raise ValueError("This file isn't a Whispers profile.")
    return Profile.from_dict(data)


# Suffix for a single exported speaker voiceprint (for cross-profile comparison).
