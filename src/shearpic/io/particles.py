"""Readers for particle snapshots written per meshblock by the Athena++ fork.

File names are ``<basename>.block<gid>.<file_id>.<NNNNN>.par.<kind>`` with two formats:

``tab``
    A line ``# Athena++ particle data at time = <T>``, the column header
    ``born_meshblock particle_id x y z vx vy vz``, then one particle per line with
    ``vx, vy, vz`` the reduced momentum u = p/m.
``bin``
    A 64-byte header of 12 float32 meshblock and mesh bounds, float32 time and dt, and
    the int64 particle count, followed by the :data:`PARBIN_RECORD` records.

One run can hold several output streams ``(basename, file_id, kind)``; the selection
functions raise when files of more than one stream match. A full snapshot has of order
10^8 particles, so map reductions over the block files with
:func:`shearpic.parallel.parallel_map` and read only the ``fields`` needed.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

__all__ = ["ParticleBlock", "read_partab", "read_parbin", "read_particle_block", "list_particle_files",
           "ParticleFileName", "parse_particle_filename", "particle_output_files", "particle_output_numbers",
           "particle_output_stream", "stream_tag", "read_particle_time", "select_particle_output", "PARBIN_RECORD", "PARBIN_HEADER_BYTES",
           "PARTICLE_FIELDS"]

PARBIN_RECORD = np.dtype(
    [("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("ux", "<f4"), ("uy", "<f4"), ("uz", "<f4"),
     ("dpar", "<f4"), ("property", "<f4"), ("pid", "<i8"), ("init_mbid", "<i4")]
)
assert PARBIN_RECORD.itemsize == 44

PARBIN_HEADER_BYTES = 12 * 4 + 2 * 4 + 8
assert PARBIN_HEADER_BYTES == 64

PARTICLE_FIELDS = ("x", "u", "pid", "init_mbid")

_TIME_RE = re.compile(r"time\s*=\s*([-+0-9.eE]+)")
_TAB_COLUMNS = ["born_meshblock", "particle_id", "x", "y", "z", "vx", "vy", "vz"]
_TAB_SOURCE = {"x": ["x", "y", "z"], "u": ["vx", "vy", "vz"], "pid": ["particle_id"],
               "init_mbid": ["born_meshblock"]}
_BIN_SOURCE = {"x": ["x", "y", "z"], "u": ["ux", "uy", "uz"], "pid": ["pid"], "init_mbid": ["init_mbid"]}


@dataclass
class ParticleBlock:
    """Particles of one meshblock file.

    ``x`` and ``u`` are float64 ``(n, 3)`` arrays, ``pid`` and ``init_mbid``, the meshblock
    where the particle was born, are int64 ``(n,)`` arrays. Fields not requested are None.
    """

    time: float
    x: np.ndarray | None
    u: np.ndarray | None
    pid: np.ndarray | None
    init_mbid: np.ndarray | None
    path: Path | None = None
    dt: float | None = None
    n: int | None = None

    def __len__(self) -> int:
        if self.n is not None:
            return int(self.n)
        for arr in (self.pid, self.init_mbid, self.u, self.x):
            if arr is not None:
                return int(arr.shape[0])
        raise ValueError("ParticleBlock has no arrays and no particle count")


def _check_fields(fields: Iterable[str]) -> tuple[str, ...]:
    if isinstance(fields, str):
        fields = (fields,)
    fields = tuple(fields)
    unknown = [f for f in fields if f not in PARTICLE_FIELDS]
    if unknown:
        raise ValueError(f"unknown particle fields {unknown}; choose from {PARTICLE_FIELDS}")
    return fields


def read_partab(path: str | Path, fields: Iterable[str] = PARTICLE_FIELDS) -> ParticleBlock:
    """Read a ``.par.tab`` file, parsing only the columns of ``fields``."""
    path = Path(path)
    fields = _check_fields(fields)
    with open(path) as fh:
        first = fh.readline()
        header = fh.readline().split()
    m = _TIME_RE.search(first)
    if m is None:
        raise ValueError(f"{path.name}: first line does not contain 'time = ...'")
    if header != _TAB_COLUMNS:
        raise ValueError(f"{path.name}: unexpected header {header}")
    # read at least one column so that the particle count is known
    usecols = [col for f in fields for col in _TAB_SOURCE[f]] or ["born_meshblock"]
    dtypes = {col: (np.int64 if col in ("born_meshblock", "particle_id") else np.float64) for col in usecols}
    df = pd.read_csv(path, sep=r"\s+", skiprows=2, header=None, names=_TAB_COLUMNS, usecols=usecols,
                     dtype=dtypes)
    out = {f: None for f in PARTICLE_FIELDS}
    for f in fields:
        cols = _TAB_SOURCE[f]
        if len(cols) == 3:
            out[f] = df[cols].to_numpy(np.float64)
        else:
            out[f] = df[cols[0]].to_numpy(np.int64)
    return ParticleBlock(time=float(m.group(1)), path=path, n=len(df), **out)


def read_parbin(path: str | Path, fields: Iterable[str] = PARTICLE_FIELDS) -> ParticleBlock:
    """Read a ``.par.bin`` file; the records are memory-mapped and only ``fields`` are copied."""
    path = Path(path)
    fields = _check_fields(fields)
    size = path.stat().st_size
    with open(path, "rb") as fh:
        head = fh.read(PARBIN_HEADER_BYTES)
    if len(head) < PARBIN_HEADER_BYTES:
        raise ValueError(f"{path.name}: {len(head)} bytes is shorter than the {PARBIN_HEADER_BYTES}-byte header")
    time, dt = np.frombuffer(head, dtype="<f4", count=2, offset=12 * 4)
    (n,) = np.frombuffer(head, dtype="<i8", count=1, offset=14 * 4)
    n = int(n)
    available = (size - PARBIN_HEADER_BYTES) // PARBIN_RECORD.itemsize
    if n < 0 or available < n:
        raise ValueError(f"{path.name}: header says {n} particles but only {available} records present")
    out = {f: None for f in PARTICLE_FIELDS}
    if n == 0:
        rec = np.empty(0, dtype=PARBIN_RECORD)
    else:
        rec = np.memmap(path, dtype=PARBIN_RECORD, mode="r", offset=PARBIN_HEADER_BYTES, shape=(n,))
    try:
        for f in fields:
            cols = _BIN_SOURCE[f]
            if len(cols) == 3:
                arr = np.empty((n, 3), dtype=np.float64)  # column by column avoids a float32 (n, 3) copy
                for j, col in enumerate(cols):
                    arr[:, j] = rec[col]
            else:
                arr = np.array(rec[cols[0]], dtype=np.int64)
            out[f] = arr
    finally:
        mm = getattr(rec, "_mmap", None)
        del rec
        if mm is not None:
            try:
                mm.close()
            except BufferError:  # pragma: no cover - a view is still alive; the GC closes it later
                pass
    return ParticleBlock(time=float(time), dt=float(dt), path=path, n=n, **out)


def read_particle_block(path: str | Path, fields: Iterable[str] = PARTICLE_FIELDS) -> ParticleBlock:
    """Dispatch on the file extension (``.par.tab`` or ``.par.bin``); ``fields`` as in the readers."""
    name = str(path)
    if name.endswith(".par.tab"):
        return read_partab(path, fields)
    if name.endswith(".par.bin"):
        return read_parbin(path, fields)
    raise ValueError(f"not a particle file: {path}")


_NUMBER_KIND_RE = re.compile(r"\.(\d{5})\.par\.(tab|bin)$")
_NAME_RE = re.compile(r"^(?P<basename>.+)\.block(?P<gid>\d+)\.(?P<file_id>[^./]+)\.(?P<number>\d{5})\.par\."
                      r"(?P<kind>tab|bin)$")


@dataclass(frozen=True)
class ParticleFileName:
    """Parts of a particle file name ``<basename>.block<gid>.<file_id>.<NNNNN>.par.<kind>``.

    ``basename`` is the ``problem_id`` and may contain dots; ``file_id`` is the output
    block name such as ``out4``.
    """

    basename: str
    gid: int
    file_id: str
    number: int
    kind: str

    @property
    def stream(self) -> tuple[str, str, str]:
        """``(basename, file_id, kind)``, shared by all files of one output stream."""
        return (self.basename, self.file_id, self.kind)

    @property
    def stream_tag(self) -> str:
        """``'<basename>.<file_id>.<kind>'``, e.g. ``'org.stir.feedback.out4.tab'``."""
        return stream_tag(self.stream)


def stream_tag(stream) -> str:
    """String ``'<basename>.<file_id>.<kind>'`` naming a particle output stream.

    ``stream`` is a ``(basename, file_id, kind)`` tuple, a dict with those keys or a
    :class:`ParticleFileName`. Product file names embed it so that products of different
    streams of one run never share a name.
    """
    if isinstance(stream, ParticleFileName):
        stream = stream.stream
    if isinstance(stream, dict):
        stream = (stream.get("basename"), stream.get("file_id"), stream.get("kind"))
    basename, file_id, kind = stream
    if not basename or not file_id or kind not in ("tab", "bin"):
        raise ValueError(f"incomplete particle stream {stream!r}: need basename, file_id and kind 'tab'/'bin'")
    return f"{basename}.{file_id}.{kind}"


def parse_particle_filename(path: str | Path) -> ParticleFileName | None:
    """Split a particle file name into its parts; ``None`` if it is not a particle file."""
    m = _NAME_RE.match(Path(path).name)
    if m is None:
        return None
    return ParticleFileName(m["basename"], int(m["gid"]), m["file_id"], int(m["number"]), m["kind"])


def _check_kind(kind: str) -> str:
    if kind not in ("tab", "bin", "*"):
        raise ValueError(f"kind must be 'tab', 'bin' or '*', not {kind!r}")
    return kind


def list_particle_files(run_dir: str | Path, file_number: int | None = None, kind: str = "*", *,
                        warn_mixed: bool = True, file_id: str | None = None,
                        basename: str | None = None) -> list[Path]:
    """Sorted per-meshblock particle files of a run, optionally for one output number.

    ``kind='*'`` matches both formats and, with ``warn_mixed``, warns when an output number
    exists in both. ``file_id`` and ``basename`` restrict the list to one output block or
    ``problem_id``; otherwise it may mix several streams, so use
    :func:`particle_output_files` for the files of one snapshot.
    """
    _check_kind(kind)
    num = "*" if file_number is None else f"{int(file_number):05d}"
    files = sorted(Path(run_dir).glob(f"*.block*.*.{num}.par.{kind}"))
    if file_id is not None or basename is not None:
        keep = []
        for p in files:
            name = parse_particle_filename(p)
            if name is None:
                continue
            if (file_id is None or name.file_id == file_id) and (basename is None or name.basename == basename):
                keep.append(p)
        files = keep
    if warn_mixed and kind == "*" and files:
        kinds: dict[str, set[str]] = {}
        for p in files:
            m = _NUMBER_KIND_RE.search(p.name)
            if m:
                kinds.setdefault(m.group(1), set()).add(m.group(2))
        both = sorted(number for number, found in kinds.items() if len(found) > 1)
        if both:
            warnings.warn(f"particle output(s) {', '.join(both)} in {run_dir} exist as both .par.tab and .par.bin; "
                          "the list contains every particle twice -- pass kind='tab' or kind='bin'", stacklevel=2)
    return files


def _streams(files: Iterable[Path]) -> dict[tuple[str, str, str], list[Path]]:
    out: dict[tuple[str, str, str], list[Path]] = {}
    for p in files:
        name = parse_particle_filename(p)
        if name is None:
            raise ValueError(f"cannot parse the particle file name {Path(p).name!r} "
                             "(expected <basename>.block<gid>.<file_id>.<NNNNN>.par.<tab|bin>)")
        out.setdefault(name.stream, []).append(Path(p))
    return out


def _describe_streams(streams) -> str:
    return "; ".join(f"basename={b!r} file_id={f!r} kind={k!r} ({len(v)} files)"
                     for (b, f, k), v in sorted(streams.items()))


def particle_output_files(run_dir: str | Path, file_number: int, kind: str = "*", *, file_id: str | None = None,
                          basename: str | None = None) -> list[Path]:
    """Sorted block files of output ``file_number`` of a single particle stream.

    Arguments are as in :func:`list_particle_files`. Raises FileNotFoundError if nothing
    matches and ValueError if the matching files belong to more than one stream.
    """
    files = list_particle_files(run_dir, file_number, kind, warn_mixed=False, file_id=file_id, basename=basename)
    if not files:
        raise FileNotFoundError(f"no particle files for output {int(file_number):05d} (kind={kind!r}, "
                                f"file_id={file_id!r}, basename={basename!r}) in {run_dir}")
    streams = _streams(files)
    if len(streams) > 1:
        kinds = {k for (_, _, k) in streams}
        hint = ("pass kind='tab' or kind='bin'" if len(kinds) > 1 else "pass file_id=... or basename=...")
        raise ValueError(f"output {int(file_number):05d} in {run_dir} matches several particle outputs "
                         f"(both .par.tab and .par.bin / different file_id / basename): {_describe_streams(streams)}; "
                         f"{hint}")
    return files


def particle_output_numbers(run_dir: str | Path, kind: str = "tab", *, file_id: str | None = None,
                            basename: str | None = None) -> list[int]:
    """Sorted output numbers of one particle stream; raise ValueError if several streams match."""
    files = list_particle_files(run_dir, None, kind, warn_mixed=False, file_id=file_id, basename=basename)
    streams = _streams(files)
    if len(streams) > 1:
        raise ValueError(f"{run_dir} holds several particle outputs: {_describe_streams(streams)}; "
                         "pass kind, file_id or basename")
    return sorted({parse_particle_filename(p).number for p in files})


def particle_output_stream(run_dir: str | Path, kind: str = "tab", *, file_id: str | None = None,
                           basename: str | None = None) -> tuple[str, str, str]:
    """The single particle stream ``(basename, file_id, kind)`` matching the filters."""
    files = list_particle_files(run_dir, None, kind, warn_mixed=False, file_id=file_id, basename=basename)
    if not files:
        raise FileNotFoundError(f"no particle files (kind={kind!r}, file_id={file_id!r}, basename={basename!r}) "
                                f"in {run_dir}")
    streams = _streams(files)
    if len(streams) > 1:
        raise ValueError(f"{run_dir} holds several particle outputs: {_describe_streams(streams)}; "
                         "pass kind, file_id or basename")
    return next(iter(streams))


def read_particle_time(path: str | Path) -> float:
    """Simulation time from a particle file header; ``.par.bin`` stores it as float32."""
    path = Path(path)
    name = path.name
    if name.endswith(".par.tab"):
        with open(path) as fh:
            first = fh.readline()
        m = _TIME_RE.search(first)
        if m is None:
            raise ValueError(f"{name}: first line does not contain 'time = ...'")
        return float(m.group(1))
    if name.endswith(".par.bin"):
        with open(path, "rb") as fh:
            head = fh.read(PARBIN_HEADER_BYTES)
        if len(head) < PARBIN_HEADER_BYTES:
            raise ValueError(f"{name}: {len(head)} bytes is shorter than the {PARBIN_HEADER_BYTES}-byte header")
        return float(np.frombuffer(head, dtype="<f4", count=1, offset=12 * 4)[0])
    raise ValueError(f"not a particle file: {path}")


def select_particle_output(run_dir: str | Path, time: float, *, kind: str = "tab", file_id: str | None = None,
                           basename: str | None = None, dt: float | None = None, tol: float | None = None,
                           allow_download: bool = False) -> tuple[int, float]:
    """``(number, file_time)`` of the particle output whose header time is closest to ``time``.

    Parameters
    ----------
    dt : output cadence; outputs near ``round(time / dt)`` are tried first
    tol : maximum ``|file_time - time|``; default ``dt/2``, or ``1e-6 max(1, |time|)`` without ``dt``
    allow_download : open online-only placeholders instead of skipping them

    Raises FileNotFoundError if no output lies within ``tol``.
    """
    from ._util import is_dataless

    numbers = particle_output_numbers(run_dir, kind, file_id=file_id, basename=basename)
    if not numbers:
        raise FileNotFoundError(f"no particle outputs (kind={kind!r}, file_id={file_id!r}) in {run_dir}")
    time = float(time)
    limit = tol if tol is not None else (0.5 * dt if dt else 1e-6 * max(1.0, abs(time)))
    order: list[int] = []
    if dt:
        guess = int(round(time / dt))
        order = [k for k in (guess, guess - 1, guess + 1) if k in numbers]
    order += [k for k in numbers if k not in order]
    best: tuple[float, int, float] | None = None
    blocked = []
    for k in order:
        first = particle_output_files(run_dir, k, kind, file_id=file_id, basename=basename)[0]
        if is_dataless(first) and not allow_download:
            blocked.append(first.name)
            continue
        t_file = read_particle_time(first)
        dist = abs(t_file - time)
        if best is None or dist < best[0]:
            best = (dist, k, t_file)
        if dt and dist <= limit:
            break  # outputs are dt apart, so no other one is closer
    if best is None or best[0] > limit:
        found = "" if best is None else f"; closest is output {best[1]:05d} at t = {best[2]:.10g}"
        extra = f"; skipped online-only placeholders {blocked[:3]} (use allow_download)" if blocked else ""
        raise FileNotFoundError(f"no particle output within {limit:g} of t = {time:g} in {run_dir}{found}{extra}")
    return best[1], best[2]
