"""Write a JSON file so that a crash or a full disk never leaves half of one.

Kept in its own module, depending on nothing else in the application, so that
every module that saves something - settings, profiles, projects, history - can
use it. It used to live in ``speaker_profiles``, which ``settings`` cannot import
without a cycle; that is why settings were the one thing still written in place.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Mapping, Union

PathLike = Union[str, Path]


def write_json_atomic(path: PathLike, payload: Mapping[str, object]) -> Path:
    """Write ``payload`` to ``path`` atomically (temp file + replace).

    A crash mid-write leaves the previous file intact rather than a truncated
    one. Errors propagate: losing an operational profile silently is worse than
    a visible failure.
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    handle, tmp_name = tempfile.mkstemp(
        dir=str(out.parent), prefix=out.name, suffix=".tmp"
    )
    tmp = Path(tmp_name)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp, out)
    except OSError:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise
    return out
