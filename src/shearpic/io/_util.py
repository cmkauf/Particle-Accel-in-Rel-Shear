"""Small helpers shared by the readers."""

from __future__ import annotations

import numpy as np


def keep_last_written(t: np.ndarray) -> np.ndarray:
    """Mask that removes rows re-written after a restart.

    Athena++ appends to history and trajectory files on restart, so time can jump
    backwards. A row is kept only if its time is smaller than that of every later row.
    """
    t = np.asarray(t)
    suffix_min = np.minimum.accumulate(t[::-1])[::-1]
    keep = np.ones(t.size, dtype=bool)
    keep[:-1] = t[:-1] < suffix_min[1:]
    return keep


# macOS flag on online-only cloud-storage placeholders; reading one triggers a download.
SF_DATALESS = 0x40000000


def is_dataless(path) -> bool:
    """True if ``path`` is an online-only cloud placeholder; always False without ``st_flags``."""
    import os

    try:
        flags = getattr(os.stat(path), "st_flags", 0)
    except OSError:
        return False
    return bool(flags & SF_DATALESS)
