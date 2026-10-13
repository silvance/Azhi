"""Which name goes on which speaker, and when a name may not travel.

Two kinds of name end up on a diarized transcript, and they are not equally
trustworthy:

* a name *recognised* from a voiceprint, measured against the turns in front of
  it. It is earned, every time, by the audio it is put on.
* a name *typed* by an operator into the Speaker 1..N fields before a run. It is
  positional, and the diarizer's numbering is arbitrary, so even on a first run
  it may need swapping - which the interface says.

The distinction matters when a transcript is split a second time. The new split
numbers its speakers from scratch: "Speaker 1" afterwards need not be the person
"Speaker 1" was before. A recognised name survives that because it is worked out
again. A typed name must not, because carrying it across would put an operator's
word - often a subject's name - onto somebody else, while looking like the tool
had preserved their work rather than guessed.
"""

from __future__ import annotations

import re
from typing import Dict, Iterable, Mapping, Optional, Sequence

# The diarizers label their clusters SPEAKER_00, SPEAKER_01, ... - pyannote by
# convention, sherpa-onnx by construction. That number is what the Speaker 1..N
# fields are numbered against.
_CLUSTER = re.compile(r"^SPEAKER_(\d+)$")


def cluster_index(speaker_id: str) -> "Optional[int]":
    """The diarizer's own number for a cluster, or None if it has no number.

    A recognised voice (``voice::Name``) has no cluster number: it is a person,
    not a position, and nothing typed into a positional field belongs to it.
    """
    match = _CLUSTER.match(speaker_id)
    return int(match.group(1)) if match else None


# How a name suggested by voice similarity is shown, everywhere it is shown.
#
# Recognition compares a turn's voice against voices this profile has learned.
# That is similarity, and the rule for this application is that similarity is
# never presented as identity. A bare "[Viktor Marchenko]" on a line is
# indistinguishable from a name the operator typed or confirmed - on screen,
# and worse, in the .txt, .srt, Word file and report that leave this machine.
# So a recognised name carries this marker until the operator confirms it by
# renaming the speaker, which replaces it with a typed name.
#
# It is applied to the display name rather than at each place that renders one,
# because there are five such places and a sixth will be added one day.
#
# The wording was reviewed: "Viktor Marchenko? (voice lead)" read, out of
# context, as doubt about the spelling and as "the main speaker". A line pasted
# into another document carries no legend, so the marker has to say what it
# means on its own.
LEAD_SUFFIX = " (unconfirmed voice lead)"
LEAD_LEGEND = (
    'A name marked "(unconfirmed voice lead)" was suggested by voice similarity to a '
    "voice this profile has learned. It is an investigative lead, not an "
    "identification. Confirm or correct it before relying on it."
)


def as_lead(name: str) -> str:
    """``name`` as shown when it was suggested by voice similarity."""
    return name if is_lead(name) else f"{name}{LEAD_SUFFIX}"


def is_lead(name: "Optional[str]") -> bool:
    """Whether a display name is an unconfirmed voice-similarity suggestion."""
    return bool(name) and str(name).endswith(LEAD_SUFFIX)


def bare_name(name: str) -> str:
    """The name without the lead marker - what confirming it would produce."""
    return name[: -len(LEAD_SUFFIX)] if is_lead(name) else name


def any_leads(names: "Iterable[str]") -> bool:
    """Whether any of these display names is an unconfirmed lead."""
    return any(is_lead(name) for name in names)


def preset_names(
    speaker_ids: "Iterable[str]",
    recognized: "Mapping[str, str]",
    typed: "Sequence[str]",
    *,
    use_typed: bool = True,
) -> "Dict[str, str]":
    """Map speaker id -> display name for one result.

    Recognised voices are applied first, marked as leads (see LEAD_SUFFIX). Whatever is left takes its name from
    the Speaker 1..N field with *its own* number - SPEAKER_00 from the first
    field, SPEAKER_01 from the second - and not from the next unused field.

    That distinction is the whole of this function. If recognition claims
    SPEAKER_00 and leaves SPEAKER_01 anonymous, taking "the next unused field"
    would put the name typed for the first speaker onto the second one. The
    operator typed "Alice" against a position, so Alice belongs to that
    position or to nobody.

    ``use_typed`` is False when the split is being redone: those fields were
    typed against a split that no longer exists, and the new one numbers its
    speakers from scratch, so only the recognised names apply.
    """
    ids = sorted({sid for sid in speaker_ids if sid})
    names: Dict[str, str] = {
        sid: as_lead(recognized[sid]) for sid in ids if sid in recognized
    }
    if not use_typed:
        return names
    for sid in ids:
        if sid in names:
            continue
        index = cluster_index(sid)
        if index is None or index >= len(typed):
            continue
        cleaned = typed[index].strip()
        if cleaned:
            names[sid] = cleaned
    return names
