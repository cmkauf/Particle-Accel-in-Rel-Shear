"""Reader for tracked-particle trajectory files ``trajectory_initmbid_<M>_pid_<P>.tab``.

The files have no header and whitespace-separated columns:

=====  ====================================================================
col    meaning
=====  ====================================================================
0      time t
1-3    position x, y, z
4-6    reduced momentum u = p/m = gamma v
7-9    B in the particle's cell, 13-column files only
10-12  cE = -U x B in the particle's cell, 13-column files only
=====  ====================================================================

Samples are written every fixed number of cycles, so the time spacing is not uniform.
"""

from __future__ import annotations

import io
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from ._util import keep_last_written

__all__ = ["Trajectory", "read_trajectory", "find_trajectory_files", "load_trajectories", "stack_trajectories"]

_NAME_RE = re.compile(r"trajectory_initmbid_(\d+)_pid_(\d+)\.tab$")


@dataclass
class Trajectory:
    """One particle's time series; vectors have shape ``(n_t, 3)``.

    ``x`` is the position wrapped into the periodic box and ``u`` the reduced momentum.
    ``B`` and ``cE`` are taken in the particle's cell and are None for 7-column files.
    ``n_dropped`` counts rows removed when reading.
    """

    t: np.ndarray
    x: np.ndarray
    u: np.ndarray
    B: np.ndarray | None = None
    cE: np.ndarray | None = None
    pid: int | None = None
    init_mbid: int | None = None
    path: Path | None = None
    n_dropped: int = field(default=0, repr=False)

    def __len__(self) -> int:
        return self.t.size

    @property
    def has_fields(self) -> bool:
        return self.B is not None and self.cE is not None

    def select(self, mask_or_slice) -> "Trajectory":
        """Sub-trajectory, e.g. ``traj.select(traj.t > 100)``."""
        pick = lambda a: None if a is None else a[mask_or_slice]  # noqa: E731
        return Trajectory(pick(self.t), pick(self.x), pick(self.u), pick(self.B), pick(self.cE),
                          self.pid, self.init_mbid, self.path, n_dropped=self.n_dropped)

    def time_window(self, t_start: float, t_end: float) -> "Trajectory":
        return self.select((self.t >= t_start) & (self.t <= t_end))


def read_trajectory(path: str | Path) -> Trajectory:
    """Read one trajectory file.

    An unterminated last line, rows with NaN or inf, and rows repeated by a restart are
    dropped with a warning. Raises ValueError for a file without valid rows or with a
    column count other than 7 or 13.
    """
    path = Path(path)
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        cut = raw.rfind(b"\n")
        raw = raw[:cut + 1] if cut >= 0 else b""
        warnings.warn(f"{path.name}: dropped the incomplete last line (file still being written or truncated)",
                      stacklevel=2)
    if not raw.strip():
        raise ValueError(f"{path.name} is empty")
    try:
        data = pd.read_csv(io.BytesIO(raw), sep=r"\s+", header=None, dtype=np.float64, comment="#").to_numpy()
    except pd.errors.EmptyDataError:  # only comment lines
        raise ValueError(f"{path.name} is empty") from None
    if data.ndim != 2 or data.shape[1] not in (7, 13):
        raise ValueError(f"{path.name}: expected 7 or 13 columns, found {data.shape[-1]}")
    finite = np.isfinite(data).all(axis=1)
    n_bad = int((~finite).sum())
    if n_bad:
        warnings.warn(f"{path.name}: dropped {n_bad} rows containing NaN/inf values", stacklevel=2)
        data = data[finite]
        if data.shape[0] == 0:
            raise ValueError(f"{path.name} is empty (no rows with finite values)")
    keep = keep_last_written(data[:, 0])
    n_restart = int((~keep).sum())
    if n_restart:
        warnings.warn(f"{path.name}: dropped {n_restart} rows with non-increasing time (restart?)", stacklevel=2)
        data = data[keep]
    m = _NAME_RE.search(path.name)
    has_fields = data.shape[1] == 13
    return Trajectory(
        t=data[:, 0].copy(),
        x=data[:, 1:4].copy(),
        u=data[:, 4:7].copy(),
        B=data[:, 7:10].copy() if has_fields else None,
        cE=data[:, 10:13].copy() if has_fields else None,
        pid=int(m.group(2)) if m else None,
        init_mbid=int(m.group(1)) if m else None,
        path=path,
        n_dropped=n_bad + n_restart,
    )


def find_trajectory_files(run_dir: str | Path, pattern: str = "**/trajectory_*.tab") -> list[Path]:
    """Sorted trajectory files matching the glob ``pattern`` below ``run_dir``."""
    return sorted(Path(run_dir).glob(pattern))


def load_trajectories(paths_or_run_dir: str | Path | Sequence[str | Path], *, backend: str = "thread",
                      n_workers: int | str = "auto", progress: bool = False,
                      on_error: str = "raise") -> list:
    """Read a run directory, a single file or a list of trajectory files in parallel.

    ``backend``, ``n_workers`` and ``progress`` are passed to
    :func:`shearpic.parallel.parallel_map`. With ``on_error='collect'`` a failed file gives
    a :class:`shearpic.parallel.TaskError` in its place in the returned list; filter those
    with ``isinstance``, since a Trajectory with zero samples is also falsy.
    """
    from ..parallel import parallel_map

    if isinstance(paths_or_run_dir, (str, Path)) and Path(paths_or_run_dir).is_dir():
        paths = find_trajectory_files(paths_or_run_dir)
    elif isinstance(paths_or_run_dir, (str, Path)):
        paths = [Path(paths_or_run_dir)]
    else:
        paths = [Path(p) for p in paths_or_run_dir]
    if not paths:
        raise FileNotFoundError(f"no trajectory files found in {paths_or_run_dir}")
    return parallel_map(read_trajectory, paths, backend=backend, n_workers=n_workers, progress=progress,
                        on_error=on_error)


def stack_trajectories(trajs: Sequence[Trajectory]) -> dict[str, np.ndarray]:
    """Stack trajectories into arrays of shape ``(n_par, n_t[, 3])``; raise if their times differ."""
    t0 = trajs[0].t
    for tr in trajs[1:]:
        if tr.t.shape != t0.shape or not np.allclose(tr.t, t0, rtol=0, atol=1e-6 * max(1.0, abs(t0[-1]))):
            raise ValueError(f"time grids differ ({trajs[0].path} vs {tr.path}); trim with time_window first")
    out = {"t": t0.copy(), "x": np.stack([tr.x for tr in trajs]), "u": np.stack([tr.u for tr in trajs]),
           "pid": np.array([-1 if tr.pid is None else tr.pid for tr in trajs])}
    if all(tr.has_fields for tr in trajs):
        out["B"] = np.stack([tr.B for tr in trajs])
        out["cE"] = np.stack([tr.cE for tr in trajs])
    return out
