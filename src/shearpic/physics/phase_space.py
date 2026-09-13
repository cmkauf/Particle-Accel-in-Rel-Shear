"""Phase-space histograms f(u_i, y) of full particle snapshots.

A :class:`PhaseSpaceHist` stores integer particle counts in (u_i, y) bins for one component of
the reduced momentum, together with the total particle number and, per y bin, the particles
outside the momentum range and exact sums of u_i and u_i^2.  A density per unit u_i/c is c
times the density per unit u_i, with c the speed of light of the run itself::

    hists = particle_snapshot_phase_space(run_dir, 12, cfg, u_edges, cfg.edges(1))
    H, x_edges, y_edges = hists["z"].density("total", unit="mc")
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = ["COMPONENTS", "PhaseSpaceHist", "phase_space_histogram", "particle_snapshot_phase_space",
           "layer_asymmetry"]

COMPONENTS = ("x", "y", "z")
_INDEX = {"x": 0, "y": 1, "z": 2}

SpeedOfLight = Union[float, "RunConfig"]


def _edges(name: str, edges) -> np.ndarray:
    e = np.array(edges, dtype=np.float64)
    if e.ndim != 1 or e.size < 2:
        raise ValueError(f"{name} must be a 1D array with at least 2 entries")
    if not np.all(np.isfinite(e)) or np.any(np.diff(e) <= 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return e


def _as_int64(name: str, arr, shape) -> np.ndarray:
    a = np.asarray(arr)
    if a.shape != shape:
        raise ValueError(f"{name} has shape {a.shape}, expected {shape}")
    if a.dtype.kind == "f":
        if not np.all(np.isfinite(a)) or np.any(a != np.round(a)):
            raise ValueError(f"{name} must hold integer particle numbers")
    elif a.dtype.kind not in "iu":
        raise ValueError(f"{name} must be an integer array")
    a = a.astype(np.int64)
    if np.any(a < 0):
        raise ValueError(f"{name} must be non-negative")
    return a


def _opt_float(x) -> float | None:
    return None if x is None else float(x)


@dataclass(frozen=True, eq=False)
class PhaseSpaceHist:
    """Integer histogram of particles in (u_i, y) for one momentum component.

    ``counts`` is indexed [u_bin, y_bin] as in ``np.histogram2d``, with edges in u_i [U0] and
    y [a], and ``n_total`` counts all particles including those outside the bins.  The optional
    per-y arrays hold the particles with u_i outside ``u_edges`` (``out_of_range_y``), all
    particles (``n_y``) and the sums of u_i and u_i^2 over them (``sum_u_y``, ``sum_u2_y``).
    ``c`` is the speed of light of the run, needed for ``unit='mc'``.
    """

    component: str
    u_edges: np.ndarray
    y_edges: np.ndarray
    counts: np.ndarray
    n_total: int
    out_of_range_y: np.ndarray | None = None
    n_y: np.ndarray | None = None
    sum_u_y: np.ndarray | None = None
    sum_u2_y: np.ndarray | None = None
    time: float | None = None
    run_id: int | str | None = None
    c: float | None = None

    def __post_init__(self):
        set_ = functools.partial(object.__setattr__, self)
        if self.component not in COMPONENTS:
            raise ValueError(f"component must be one of {COMPONENTS}, not {self.component!r}")
        ue, ye = _edges("u_edges", self.u_edges), _edges("y_edges", self.y_edges)
        set_("u_edges", ue)
        set_("y_edges", ye)
        nu, ny = ue.size - 1, ye.size - 1
        counts = _as_int64("counts", self.counts, (nu, ny))
        set_("counts", counts)
        if self.n_total is None or float(self.n_total) != round(float(self.n_total)):
            raise ValueError("n_total must be an integer")
        n_total = int(round(float(self.n_total)))
        if n_total < int(counts.sum()):
            raise ValueError(f"n_total = {n_total} is smaller than the {int(counts.sum())} particles in the bins")
        set_("n_total", n_total)
        for name in ("out_of_range_y", "n_y"):
            arr = getattr(self, name)
            if arr is not None:
                set_(name, _as_int64(name, arr, (ny,)))
        for name in ("sum_u_y", "sum_u2_y"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.array(arr, dtype=np.float64)
                if arr.shape != (ny,):
                    raise ValueError(f"{name} has shape {arr.shape}, expected ({ny},)")
                set_(name, arr)
        if self.n_y is not None:
            if self.out_of_range_y is not None and not np.array_equal(self.n_y, counts.sum(axis=0) + self.out_of_range_y):
                raise ValueError("n_y != counts.sum(axis=0) + out_of_range_y")
            if int(self.n_y.sum()) > n_total:
                raise ValueError("sum(n_y) exceeds n_total")
        set_("time", _opt_float(self.time))
        set_("c", _opt_float(self.c))

    # ---------------------------------------------------------------- geometry
    @property
    def shape(self) -> tuple[int, int]:
        return self.counts.shape

    @property
    def du(self) -> np.ndarray:
        return np.diff(self.u_edges)

    @property
    def dy(self) -> np.ndarray:
        return np.diff(self.y_edges)

    @property
    def u_centers(self) -> np.ndarray:
        return 0.5 * (self.u_edges[:-1] + self.u_edges[1:])

    @property
    def y_centers(self) -> np.ndarray:
        return 0.5 * (self.y_edges[:-1] + self.y_edges[1:])

    @property
    def n_in_range(self) -> int:
        return int(self.counts.sum())

    @property
    def has_sums(self) -> bool:
        """True if exact per-``y`` moments over all particles are stored."""
        return self.n_y is not None and self.sum_u_y is not None and self.sum_u2_y is not None

    # ---------------------------------------------------------------- density
    def density(self, normalize: str | None = "total", unit: str = "u") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Phase-space density ``H`` with its momentum and y edges.

        ``normalize='total'`` divides by ``n_total``, ``'in_range'`` by the particles in the bins,
        and ``None`` gives particles per unit area.  ``unit='mc'`` uses the axis p_i/(mc) = u_i/c,
        dividing the edges by c and multiplying the density by c.
        """
        area = self.du[:, None] * self.dy[None, :]
        H = self.counts / area
        if normalize == "total":
            norm = self.n_total
        elif normalize == "in_range":
            norm = self.n_in_range
        elif normalize is None:
            norm = 1
        else:
            raise ValueError("normalize must be 'total', 'in_range' or None")
        if not norm:
            raise ValueError("cannot normalise an empty histogram")
        H = H / norm
        x_edges = self.u_edges.copy()
        if unit == "mc":
            if not self.c:
                raise ValueError("unit='mc' needs the speed of light c of this run")
            x_edges = x_edges / self.c
            H = H * self.c
        elif unit != "u":
            raise ValueError("unit must be 'u' or 'mc'")
        return H, x_edges, self.y_edges.copy()

    def out_of_range_fraction(self) -> float:
        """Fraction of the ``n_total`` particles that are not in any bin."""
        if not self.n_total:
            raise ValueError("empty histogram")
        return (self.n_total - self.n_in_range) / self.n_total

    def out_of_range_fraction_y(self) -> np.ndarray | None:
        """Fraction of the particles in each y bin with u_i outside the edges, or None if not stored."""
        if self.out_of_range_y is None or self.n_y is None:
            return None
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(self.n_y > 0, self.out_of_range_y / np.maximum(self.n_y, 1), np.nan)

    # ---------------------------------------------------------------- moments
    def moments_source(self) -> str:
        """``'sums'`` if exact moments are stored, else ``'bins'`` (see :meth:`moments`)."""
        return "sums" if self.has_sums else "bins"

    def moments(self, source: str = "auto") -> tuple[np.ndarray, np.ndarray]:
        """Mean [U0] and variance [U0^2] of u_i in every y bin, NaN for empty bins.

        ``source='sums'`` uses the exact sums over all particles.  ``'bins'`` places particles at
        their bin centres, which ignores particles outside the momentum range and so
        underestimates the variance there.  ``'auto'`` uses the sums when they are stored.
        """
        if source == "auto":
            source = self.moments_source()
        with np.errstate(invalid="ignore", divide="ignore"):
            if source == "sums":
                if not self.has_sums:
                    raise ValueError("this histogram stores no exact moments (legacy data); use source='bins'")
                n = self.n_y.astype(np.float64)
                mean = self.sum_u_y / n
                var = self.sum_u2_y / n - mean**2
            elif source == "bins":
                n = self.counts.sum(axis=0).astype(np.float64)
                uc = self.u_centers[:, None]
                mean = (self.counts * uc).sum(axis=0) / n
                var = (self.counts * uc**2).sum(axis=0) / n - mean**2
            else:
                raise ValueError("source must be 'auto', 'sums' or 'bins'")
        empty = n == 0
        mean = np.where(empty, np.nan, mean)
        var = np.where(empty, np.nan, np.maximum(var, 0.0))
        return mean, var

    def mean_in_y_range(self, lo: float, hi: float, source: str = "auto") -> float:
        """Particle-weighted mean of u_i over the y bins with centres in [lo, hi]."""
        if source == "auto":
            source = self.moments_source()
        sel = (self.y_centers >= lo) & (self.y_centers <= hi)
        return self._mean_over(sel, source)

    def _mean_over(self, sel: np.ndarray, source: str) -> float:
        if source == "sums":
            if not self.has_sums:
                raise ValueError("this histogram stores no exact moments (legacy data); use source='bins'")
            n, s = float(self.n_y[sel].sum()), float(self.sum_u_y[sel].sum())
        elif source == "bins":
            n = float(self.counts[:, sel].sum())
            s = float((self.counts[:, sel] * self.u_centers[:, None]).sum())
        else:
            raise ValueError("source must be 'auto', 'sums' or 'bins'")
        return s / n if n else math.nan

    # ------------------------------------------------------------ map-reduce
    def __add__(self, other: "PhaseSpaceHist") -> "PhaseSpaceHist":
        """Sum two histograms of the same snapshot, such as two meshblock files.

        Component, edges and time (when both are set) must agree.  Optional per-y arrays are kept
        only when both sides have them, and ``run_id`` and ``c`` only when equal.
        """
        if not isinstance(other, PhaseSpaceHist):
            return NotImplemented
        if other.component != self.component:
            raise ValueError(f"cannot add histograms of u_{self.component} and u_{other.component}")
        if not (np.array_equal(self.u_edges, other.u_edges) and np.array_equal(self.y_edges, other.y_edges)):
            raise ValueError("cannot add histograms with different bin edges")
        if self.time is not None and other.time is not None:
            if not math.isclose(self.time, other.time, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"cannot add histograms at different times ({self.time} vs {other.time})")

        def both(name):
            a, b = getattr(self, name), getattr(other, name)
            return None if a is None or b is None else a + b

        return PhaseSpaceHist(
            component=self.component, u_edges=self.u_edges, y_edges=self.y_edges,
            counts=self.counts + other.counts, n_total=self.n_total + other.n_total,
            out_of_range_y=both("out_of_range_y"), n_y=both("n_y"), sum_u_y=both("sum_u_y"),
            sum_u2_y=both("sum_u2_y"), time=self.time if self.time is not None else other.time,
            run_id=self.run_id if self.run_id == other.run_id else None,
            c=self.c if self.c == other.c else None,
        )

    def __radd__(self, other):
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __repr__(self) -> str:
        t = f", time={self.time:g}" if self.time is not None else ""
        return (f"PhaseSpaceHist(u_{self.component}, {self.shape[0]}x{self.shape[1]} bins, "
                f"u [{self.u_edges[0]:g}, {self.u_edges[-1]:g}], n_total={self.n_total}, "
                f"out_of_range={self.out_of_range_fraction():.3g}{t})")


# ------------------------------------------------------------------ histogram
def _bin_index(values: np.ndarray, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Bin index of each value (last edge in the last bin) and the in-range mask."""
    nb = edges.size - 1
    idx = np.searchsorted(edges, values, side="right") - 1
    idx[values == edges[-1]] = nb - 1
    inside = (idx >= 0) & (idx < nb)
    return idx, inside


def phase_space_histogram(u_component, y, u_edges, y_edges, *, component: str, time: float | None = None,
                          run_id: int | str | None = None, c: float | None = None) -> PhaseSpaceHist:
    """Histogram particles in (u_i, y), keeping totals, per-y out-of-range counts and exact moments.

    Bins are [e_k, e_k+1) except the last, which includes its upper edge.  Particles with
    non-finite u_i or y are dropped with a warning, and those with y outside ``y_edges`` count
    only in ``n_total``.
    """
    ue, ye = _edges("u_edges", u_edges), _edges("y_edges", y_edges)
    u = np.asarray(u_component, dtype=np.float64).ravel()
    yy = np.asarray(y, dtype=np.float64).ravel()
    if u.shape != yy.shape:
        raise ValueError(f"u_component {u.shape} and y {yy.shape} differ in shape")
    finite = np.isfinite(u) & np.isfinite(yy)
    if not finite.all():
        warnings.warn(f"phase_space_histogram: ignoring {int((~finite).sum())} particles with non-finite values",
                      stacklevel=2)
        u, yy = u[finite], yy[finite]
    nu, ny = ue.size - 1, ye.size - 1
    iu, u_in = _bin_index(u, ue)
    iy, y_in = _bin_index(yy, ye)
    both = u_in & y_in
    counts = np.bincount(iu[both] * ny + iy[both], minlength=nu * ny).reshape(nu, ny)
    iyy = iy[y_in]
    n_y = np.bincount(iyy, minlength=ny)
    out_y = np.bincount(iy[y_in & ~u_in], minlength=ny)
    uu = u[y_in]
    sum_u = np.bincount(iyy, weights=uu, minlength=ny)
    sum_u2 = np.bincount(iyy, weights=uu * uu, minlength=ny)
    return PhaseSpaceHist(component=component, u_edges=ue, y_edges=ye, counts=counts, n_total=int(u.size),
                          out_of_range_y=out_y, n_y=n_y, sum_u_y=sum_u, sum_u2_y=sum_u2, time=time, run_id=run_id,
                          c=c)


# ------------------------------------------------------ snapshot map-reduce
def _block_phase_space(path: str | Path, *, u_edges: np.ndarray, y_edges: np.ndarray,
                       components: tuple[str, ...]) -> dict[str, PhaseSpaceHist]:
    """Phase-space histograms of one meshblock file; module level so that it pickles."""
    from ..io.particles import read_particle_block

    block = read_particle_block(path, fields=("x", "u"))
    y = block.x[:, 1]
    return {comp: phase_space_histogram(block.u[:, _INDEX[comp]], y, u_edges, y_edges, component=comp,
                                        time=block.time)
            for comp in components}


@dataclass(frozen=True)
class _Timed:
    time: float


def particle_snapshot_phase_space(run_dir: str | Path, file_number: int, cfg: SpeedOfLight, u_edges, y_edges,
                                  components: Iterable[str] = COMPONENTS, kind: str = "tab", *,
                                  file_id: str | None = None, basename: str | None = None,
                                  backend: str = "auto", n_workers: int | str = "auto", on_error: str = "raise",
                                  expected_n: int | None = None, progress: bool = False,
                                  mem_per_task: int | str | None = "256MB") -> dict[str, PhaseSpaceHist] | None:
    """Phase-space histograms (u_i, y) of all particles of one output, summed over meshblock files.

    Returns a dict mapping each component to its PhaseSpaceHist, or None on MPI ranks other than 0.

    Parameters
    ----------
    run_dir : path
        Directory with ``<basename>.block<gid>.<file_id>.<NNNNN>.par.{tab,bin}`` files.
    file_number : int
        Output number NNNNN, see :func:`shearpic.io.particles.select_particle_output`.
    cfg : RunConfig or float
        Supplies the speed of light and the run id.
    u_edges, y_edges : arrays
        Edges in u_i [U0] and y [a]; ``y_edges`` should cover the box.
    file_id, basename : str, optional
        Select one output stream.
    backend, n_workers, progress, mem_per_task
        Passed to ``parallel_map``.
    on_error : 'raise' or 'collect'
        ``'collect'`` skips unreadable blocks with a warning.
    expected_n : int, optional
        Required ``n_total``, e.g. ``cfg.n_par``; any other value raises ValueError.
    """
    from ..config import parse_run_id
    from ..io.particles import particle_output_files
    from ..parallel import parallel_map
    from .spectra import _speed_of_light, common_block_time, drop_failed_blocks

    components = tuple(components)
    bad = [c for c in components if c not in COMPONENTS]
    if bad or not components:
        raise ValueError(f"components must be a non-empty subset of {COMPONENTS}, got {components}")
    if on_error not in ("raise", "collect"):
        raise ValueError("on_error must be 'raise' or 'collect'")
    ue, ye = _edges("u_edges", u_edges), _edges("y_edges", y_edges)
    c_val = _speed_of_light(cfg) if cfg is not None else None
    run_id = getattr(cfg, "run_id", None)
    files = particle_output_files(run_dir, file_number, kind, file_id=file_id, basename=basename)
    worker = functools.partial(_block_phase_space, u_edges=ue, y_edges=ye, components=components)
    parts = parallel_map(worker, files, backend=backend, n_workers=n_workers, progress=progress,
                         mem_per_task=mem_per_task, desc=f"phase space {int(file_number):05d}", on_error=on_error)
    if parts is None:
        return None
    parts = drop_failed_blocks(parts, files, f"phase space {int(file_number):05d}")
    time = common_block_time([_Timed(p[components[0]].time) for p in parts], int(file_number), run_dir)
    if run_id is None:
        run_id = parse_run_id(run_dir)
    out = {}
    for comp in components:
        total = functools.reduce(lambda a, b: a + b, (p[comp] for p in parts))
        out[comp] = replace(total, time=time, run_id=run_id, c=c_val)
    n = {h.n_total for h in out.values()}
    if len(n) != 1:
        raise ValueError(f"components disagree on the number of particles: {sorted(n)}")
    (n_total,) = n
    if expected_n is not None and n_total != int(expected_n):
        raise ValueError(f"output {int(file_number):05d} of {run_dir}: read {n_total} particles from "
                         f"{len(parts)} of {len(files)} block files, expected {int(expected_n)}")
    return out


# ------------------------------------------------------------------ statistics
def layer_asymmetry(hist: PhaseSpaceHist, layer: float, *, half_width: float = 10.0, fit_half_width: float = 4.0,
                    source: str = "auto") -> dict[str, float | str]:
    """Mean momentum just below and above a shear layer at y = ``layer``, and its gradient across it.

    ``mean_below`` and ``mean_above`` average u_i over the y bins within ``half_width`` on each
    side, and ``slope`` [U0/a] is a linear fit of the per-bin mean against y within
    ``fit_half_width``.  Distances are periodic in the y extent of the histogram, and ``source``
    is as for :meth:`PhaseSpaceHist.moments`.
    """
    if source == "auto":
        source = hist.moments_source()
    Ly = float(hist.y_edges[-1] - hist.y_edges[0])
    d = (hist.y_centers - layer + 0.5 * Ly) % Ly - 0.5 * Ly
    below = (d < 0) & (d >= -half_width)
    above = (d > 0) & (d <= half_width)
    mean, _ = hist.moments(source)
    near = (np.abs(d) <= fit_half_width) & np.isfinite(mean)
    slope = float(np.polyfit(d[near], mean[near], 1)[0]) if int(near.sum()) >= 2 else math.nan
    return {"layer": float(layer), "mean_below": hist._mean_over(below, source),
            "mean_above": hist._mean_over(above, source), "slope": slope, "source": source}
