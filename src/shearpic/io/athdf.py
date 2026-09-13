"""Reader for Athena++ HDF5 snapshots (``*.athdf``) on uniform meshes.

:class:`Snapshot` stitches the meshblocks into global numpy arrays in code units,
indexed ``(x, y, z)`` and float32 by default. The file is opened read-only without HDF5
locking and only while reading, so a ``Snapshot`` can be passed to worker processes.

Variables of the MHD-PIC runs:

========  =====================================================================
rho       gas density
vel1..3   gas velocity U
np        CR number density per unit volume; count per cell is np * dV
vp1..3    CR mean reduced momentum <u> = <p/m> = <gamma v> in the cell
Bcc1..3   cell-centred magnetic field; magnetic energy density B^2/2
========  =====================================================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np

from ._util import is_dataless


class DatalessFileError(OSError):
    """The file is an online-only cloud placeholder; reading it would download it."""

__all__ = ["Snapshot", "list_snapshots", "select_snapshots", "default_time_tolerance", "snapshot_number",
           "DatalessFileError"]


def _decode(values) -> list[str]:
    return [v.decode() if isinstance(v, bytes) else str(v) for v in values]


def _open(path: Path) -> h5py.File:
    """Open read-only without HDF5 file locking, falling back to the default flags."""
    try:
        return h5py.File(path, "r", locking=False)
    except (OSError, ValueError, TypeError) as err:
        # OSError if the file is already open with locking in this process;
        # ValueError/TypeError if the HDF5 library predates the locking option.
        if "lock" not in str(err).lower():
            raise
        return h5py.File(path, "r")


@dataclass(frozen=True)
class Snapshot:
    """Lazy handle on one ``.athdf`` file; only metadata is read at construction.

    ``time`` is in a/U0, ``shape`` and ``block_shape`` are cell counts ``(nx1, nx2, nx3)``
    of the root grid and of one meshblock, and ``bounds`` are the float32-rounded domain
    bounds stored in the file. :meth:`read` supports only ``max_level == 0``.
    """

    path: Path
    time: float
    cycle: int
    variables: tuple[str, ...]
    shape: tuple[int, int, int]
    block_shape: tuple[int, int, int]
    bounds: tuple[tuple[float, float], ...]
    n_blocks: int
    max_level: int
    _locations: dict = field(compare=False, repr=False)  # variable -> (dataset name, index)

    @classmethod
    def open(cls, path: str | Path) -> "Snapshot":
        """Read the metadata of an ``.athdf`` file; raise NotImplementedError for non-uniform spacing."""
        path = Path(path)
        if is_dataless(path):
            import warnings

            warnings.warn(f"{path.name} is an online-only placeholder; opening it downloads the whole file",
                          stacklevel=2)
        with _open(path) as f:
            a = f.attrs
            datasets = _decode(a["DatasetNames"])
            nvars = [int(n) for n in a["NumVariables"]]
            names = _decode(a["VariableNames"])
            locations = {}
            start = 0
            for ds, n in zip(datasets, nvars):
                for idx, name in enumerate(names[start:start + n]):
                    locations[name] = (ds, idx)
                start += n
            bounds = tuple(
                (float(a[f"RootGridX{d}"][0]), float(a[f"RootGridX{d}"][1])) for d in (1, 2, 3)
            )
            for d in (1, 2, 3):
                if float(a[f"RootGridX{d}"][2]) != 1.0:
                    raise NotImplementedError(f"{path.name}: non-uniform spacing in x{d} is not supported")
            return cls(
                path=path,
                time=float(a["Time"]),
                cycle=int(a["NumCycles"]),
                variables=tuple(names),
                shape=tuple(int(n) for n in a["RootGridSize"]),
                block_shape=tuple(int(n) for n in a["MeshBlockSize"]),
                bounds=bounds,
                n_blocks=int(a["NumMeshBlocks"]),
                max_level=int(a["MaxLevel"]),
                _locations=locations,
            )

    def edges(self, axis: int, bounds: tuple[float, float] | None = None) -> np.ndarray:
        """Cell-face coordinates along ``axis`` (0=x, 1=y, 2=z) as float64.

        Pass exact ``bounds``, e.g. ``cfg.bounds[axis]``, to avoid the float32 rounding
        of the bounds stored in the file.
        """
        lo, hi = bounds if bounds is not None else self.bounds[axis]
        return np.linspace(lo, hi, self.shape[axis] + 1)

    def centers(self, axis: int, bounds: tuple[float, float] | None = None) -> np.ndarray:
        """Cell-centre coordinates along ``axis`` as float64; ``bounds`` as in :meth:`edges`."""
        e = self.edges(axis, bounds)
        return 0.5 * (e[:-1] + e[1:])

    @property
    def x(self) -> np.ndarray:
        return self.centers(0)

    @property
    def y(self) -> np.ndarray:
        return self.centers(1)

    @property
    def z(self) -> np.ndarray:
        return self.centers(2)

    def read(self, names: str | list[str] | tuple[str, ...] | None = None, dtype=np.float32,
             squeeze: bool = False) -> dict[str, np.ndarray]:
        """Read variables into global ``(nx, ny, nz)`` arrays.

        ``names=None`` reads every variable; ``squeeze=True`` drops length-1 axes, so a 2D
        run gives ``(nx, ny)`` arrays.
        """
        if self.max_level != 0:
            raise NotImplementedError(
                f"{self.path.name} uses mesh refinement (MaxLevel={self.max_level}); "
                "load it with yt (pip install 'shearpic[yt]')."
            )
        if names is None:
            names = list(self.variables)
        elif isinstance(names, str):
            names = [names]
        missing = [n for n in names if n not in self._locations]
        if missing:
            raise KeyError(f"{missing} not in {self.path.name}; available: {list(self.variables)}")
        out: dict[str, np.ndarray] = {}
        with _open(self.path) as f:
            loc = f["LogicalLocations"][:]
            for name in names:
                ds, idx = self._locations[name]
                blocks = f[ds][idx]  # (n_block, nz_b, ny_b, nx_b)
                arr = self._assemble(blocks, loc, dtype)
                out[name] = arr.squeeze() if squeeze else arr
        return out

    def __getitem__(self, name: str) -> np.ndarray:
        return self.read(name)[name]

    def _assemble(self, blocks: np.ndarray, loc: np.ndarray, dtype=None) -> np.ndarray:
        """Global C-contiguous ``(nx, ny, nz)`` array from ``(n_block, nz_b, ny_b, nx_b)`` blocks."""
        bx, by, bz = self.block_shape
        nx, ny, nz = self.shape
        n1, n2, n3 = nx // bx, ny // by, nz // bz
        loc = np.asarray(loc, dtype=np.int64)
        n_tiles = n1 * n2 * n3
        valid = (blocks.shape[0] == loc.shape[0] == n_tiles and np.all(loc >= 0)
                 and np.all(loc < np.array([n1, n2, n3])))
        if not valid or np.unique(loc[:, 0] + n1 * (loc[:, 1] + n2 * loc[:, 2])).size != n_tiles:
            raise ValueError(f"{self.path.name}: the {blocks.shape[0]} meshblocks do not tile the "
                             f"{self.shape} root grid exactly")
        out = np.empty((nx, ny, nz), dtype=blocks.dtype if dtype is None else dtype)
        for b, (l1, l2, l3) in enumerate(loc):
            out[l1 * bx:(l1 + 1) * bx, l2 * by:(l2 + 1) * by, l3 * bz:(l3 + 1) * bz] = blocks[b].transpose(2, 1, 0)
        return out


_NUM_RE = re.compile(r"\.(\d+)\.athdf$")


def snapshot_number(path: str | Path) -> int:
    m = _NUM_RE.search(str(path))
    if not m:
        raise ValueError(f"cannot parse the snapshot number from {path}")
    return int(m.group(1))


def list_snapshots(run_dir: str | Path, output: str | None = None, *, skip_dataless: bool = False) -> list[Path]:
    """Sorted ``*.athdf`` files in ``run_dir``.

    ``output`` selects one stream, e.g. ``'out2'``; by default all streams are returned.
    ``skip_dataless=True`` drops online-only cloud placeholders.
    """
    pattern = f"*.{output}.*.athdf" if output else "*.athdf"
    files = sorted(Path(run_dir).glob(pattern), key=lambda p: (p.name.rsplit(".", 2)[0], snapshot_number(p)))
    if skip_dataless:
        files = [p for p in files if not is_dataless(p)]
    return files


def default_time_tolerance(t: float, dt: float | None = None) -> float:
    """Default largest accepted ``|Snapshot.time - t|`` in :func:`select_snapshots`.

    ``max(1e-3 dt, 1e-6 |t|)`` for an output cadence ``dt``, else ``1e-6 max(1, |t|)``.
    Athena++ writes an output at the first step past ``k dt``, so snapshot times exceed
    ``k dt`` by at most one time step, while a time between two outputs is rejected.
    """
    t = abs(float(t))
    if dt:
        return max(1e-3 * abs(float(dt)), 1e-6 * t)
    return 1e-6 * max(1.0, t)


def select_snapshots(run_dir: str | Path, times, *, output: str = "out2", dt: float | None = None,
                     tol: float | None = None, allow_download: bool = False) -> list[Path]:
    """Path of the snapshot nearest in time to each of ``times`` [a/U0], in the requested order.

    Parameters
    ----------
    dt : output cadence of the stream, e.g. ``cfg.output_dt('hdf5')``; sets the default tolerance
    tol : maximum ``|Snapshot.time - t|``; default :func:`default_time_tolerance`
    allow_download : open online-only placeholders when no local snapshot matches

    Raises FileNotFoundError if no snapshot lies within the tolerance, DatalessFileError if
    only a placeholder could match and downloads are not allowed, and ValueError if two
    times resolve to the same file.
    """
    run_dir = Path(run_dir)
    files = list_snapshots(run_dir, output)
    if not files:
        raise FileNotFoundError(f"no *.{output}.*.athdf files in {run_dir}")
    by_number = {snapshot_number(p): p for p in files}
    placeholders = [p for p in files if is_dataless(p)]
    placeholder_set = set(placeholders)
    local = [p for p in files if p not in placeholder_set]
    local_times = {p: Snapshot.open(p).time for p in local}

    downloaded: dict[Path, float] = {}

    def time_of_placeholder(p: Path) -> float:
        if p not in downloaded:
            downloaded[p] = Snapshot.open(p).time
        return downloaded[p]

    chosen: list[Path] = []
    for t in times:
        t = float(t)
        limit = float(tol) if tol is not None else default_time_tolerance(t, dt)
        best = min(local, key=lambda p: (abs(local_times[p] - t), snapshot_number(p)), default=None)
        if best is not None and abs(local_times[best] - t) <= limit:
            chosen.append(best)
            continue
        nearest = "" if best is None else f" (nearest local snapshot: {best.name} at t = {local_times[best]:g})"
        if dt and abs(t - round(t / dt) * dt) > limit:
            # no file of this stream can match, so downloading would not help
            raise FileNotFoundError(f"no {output} snapshot within {limit:g} of t = {t:g} in {run_dir}{nearest}; "
                                    f"t = {t:g} is not an output time of this stream (dt = {dt:g}) -- request an "
                                    "output time, or widen the tolerance (--tol)")
        order = placeholders
        if dt and placeholders:
            # only the files numbered near t / dt can match
            n = int(round(t / dt))
            order = [by_number[k] for k in (n, n - 1, n + 1) if k in by_number and by_number[k] in placeholder_set]
        if order and not allow_download:
            raise DatalessFileError(f"no local {output} snapshot within {limit:g} of t = {t:g} in {run_dir}{nearest}; "
                                    f"{order[0].name} (and possibly others) is an online-only placeholder -- make it "
                                    "available offline or pass --allow-download (allow_download=True)")
        found = None
        for p in order:
            if abs(time_of_placeholder(p) - t) <= limit:
                found = p
                break
        if found is None:
            raise FileNotFoundError(f"no {output} snapshot within {limit:g} of t = {t:g} in {run_dir}{nearest}; "
                                    "request an output time, or widen the tolerance (--tol)")
        chosen.append(found)

    seen: dict[Path, float] = {}
    for t, p in zip(times, chosen):
        if p in seen:
            got = local_times.get(p, downloaded.get(p))
            raise ValueError(f"requested times t = {seen[p]:g} and t = {float(t):g} both resolve to {p.name} "
                             f"(t = {got:g}) in {run_dir}; request distinct output times")
        seen[p] = float(t)
    return chosen
