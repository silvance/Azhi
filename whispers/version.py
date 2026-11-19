"""The WHISPERS version.

Builds before the program was renamed reported the version of the speech
library they were forked from, the same in every release. Each build is still
identified exactly by its build id and commit (see :mod:`whispers.buildinfo`).
"""

__version__ = "1.0.0"

# What the name stands for. Shown in Help; see docs/name.md for why each
# word is there.
NAME = "WHISPERS"
FULL_NAME = (
    "Workstation for Human-language Intelligence: Speech Processing, "
    "Exploitation, Reporting & Screening"
)
