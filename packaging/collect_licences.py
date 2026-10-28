"""Copy the licence files of every installed Python package into a bundle.

Run after PyInstaller, before the inventory is written, so ``--verify`` covers
them::

    python packaging/collect_licences.py dist/Sotto

Each distribution's licence, copying and notice files go to
``<bundle>/licenses/<distribution>/``. The build environment holds exactly what
the bundle was made from - and a few build tools besides, whose licences cost
nothing to include. Models and non-Python programs are covered by
``THIRD_PARTY_NOTICES.md``.
"""

from __future__ import annotations

import shutil
import sys
from importlib import metadata
from pathlib import Path
from typing import List, Optional

MARKERS = ("LICEN", "COPYING", "NOTICE", "AUTHORS")


def collect(bundle: Path) -> int:
    """Copy the files; returns how many distributions had any."""
    out = bundle / "licenses"
    count = 0
    for dist in metadata.distributions():
        name = (dist.metadata.get("Name") or "").strip()
        if not name:
            continue
        found = False
        for entry in dist.files or []:
            if not any(marker in entry.name.upper() for marker in MARKERS):
                continue
            source = Path(str(dist.locate_file(entry)))
            if not source.is_file():
                continue
            # Keep the path inside the package, so two files of one name
            # (a licence in the package and one in its metadata) both survive.
            target = out / name / Path(*entry.parts[-2:])
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            found = True
        count += found
    return count


def main(argv: Optional[List[str]] = None) -> int:
    args = sys.argv[1:] if argv is None else list(argv)
    if len(args) != 1 or not Path(args[0]).is_dir():
        print("usage: collect_licences.py BUNDLE_FOLDER")
        return 2
    count = collect(Path(args[0]))
    print(f"licence files from {count} distribution(s) -> {Path(args[0]) / 'licenses'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
