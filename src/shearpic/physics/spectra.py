"""Particle energy spectra as histograms that keep their particle counts.

A :class:`Spectrum` histograms one of ``gamma``, ``gamma_minus_1`` (gamma - 1) or ``p_over_mc``
(|u|/c).  These are monotonic functions of each other, so :meth:`Spectrum.to` changes variable
by mapping the bin edges and leaves the counts unchanged.  Particles outside the edges are kept
as underflow and overflow and included in ``n_total``::

    spec = particle_snapshot_spectrum(run_dir, 6, cfg, log_edges(1.0, 100.0, 200))
    x, f = spec.centers("geometric"), spec.dN_dx()
"""

from __future__ import annotations

import functools
import math
import warnings
from dataclasses import dataclass, replace
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np

from .relativity import dot

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = [
    "VARIABLES", "Spectrum", "log_edges", "lin_edges", "parse_edges", "histogram", "energy_values",
    "convert_values", "power_law_index", "cutoff_value", "particle_snapshot_spectrum",
]

VARIABLES = ("gamma", "gamma_minus_1", "p_over_mc")
_LOWER_BOUND = {"gamma": 1.0, "gamma_minus_1": 0.0, "p_over_mc": 0.0}
_DOMAIN_TOL = 1e-9  # round-off below the physical domain that is clipped rather than rejected

SpeedOfLight = Union[float, "RunConfig"]


def _check_variable(variable: str) -> str:
    if variable not in VARIABLES:
        raise ValueError(f"unknown spectrum variable {variable!r}; choose from {VARIABLES}")
    return variable


def _speed_of_light(c: SpeedOfLight) -> float:
    """Speed of light from a number or a RunConfig."""
    if isinstance(c, (int, float, np.number)):
        return float(c)
    value = getattr(c, "c", None)
    if value is None:
        raise ValueError(f"cannot get the speed of light from {type(c).__name__} (RunConfig without <particles>?)")
    return float(value)


# ------------------------------------------------------------ variable changes
def _to_gamma_minus_1(x: np.ndarray, variable: str) -> np.ndarray:
    """Map values of ``variable`` to gamma - 1 without cancellation error near gamma = 1."""
    x = np.asarray(x, dtype=np.float64)
    if variable == "gamma_minus_1":
        return x
    if variable == "gamma":
        return x - 1.0  # exact for gamma in [0.5, 2] by Sterbenz's lemma
    p2 = x * x
    return p2 / (1.0 + np.sqrt(1.0 + p2))


def _from_gamma_minus_1(g1: np.ndarray, variable: str) -> np.ndarray:
    g1 = np.asarray(g1, dtype=np.float64)
    if variable == "gamma_minus_1":
        return g1
    if variable == "gamma":
        return 1.0 + g1
    return np.sqrt(g1 * (g1 + 2.0))  # gamma^2 - 1 = (gamma-1)(gamma+1)


def convert_values(x, source: str, target: str) -> np.ndarray:
    """Convert energy-variable values from ``source`` to ``target``.

    Values below the physical domain by round-off are clipped onto it; values clearly outside
    it raise ValueError.
    """
    _check_variable(source)
    _check_variable(target)
    x = np.asarray(x, dtype=np.float64)
    lo = _LOWER_BOUND[source]
    if np.any(x < lo - _DOMAIN_TOL * max(1.0, lo)):
        raise ValueError(f"values below the domain of {source!r} (min {np.nanmin(x)!r} < {lo})")
    if source == target:
        return x.copy()
    g1 = np.maximum(_to_gamma_minus_1(np.maximum(x, lo), source), 0.0)
    return _from_gamma_minus_1(g1, target)


def energy_values(u, c: SpeedOfLight, variable: str = "gamma") -> np.ndarray:
    """Energy variable of particles with reduced momenta ``u`` of shape (..., 3).

    ``c`` is the speed of light or a RunConfig.  gamma - 1 is evaluated as (u/c)^2 / (1 + gamma),
    which stays accurate at u << c.
    """
    _check_variable(variable)
    c = _speed_of_light(c)
    p2 = dot(u, u) / (c * c)
    if variable == "p_over_mc":
        return np.sqrt(p2)
    gamma = np.sqrt(1.0 + p2)
    if variable == "gamma":
        return gamma
    return p2 / (1.0 + gamma)


def log_edges(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` logarithmically spaced bins between ``lo`` and ``hi`` (``n + 1`` edges, exact ends)."""
    if not (lo > 0 and hi > lo):
        raise ValueError("log_edges needs 0 < lo < hi")
    if int(n) < 1:
        raise ValueError("log_edges needs n >= 1 bins")
    edges = np.logspace(math.log10(lo), math.log10(hi), int(n) + 1)
    edges[0], edges[-1] = lo, hi
    return edges


def lin_edges(lo: float, hi: float, n: int) -> np.ndarray:
    """``n`` uniform bins between ``lo`` and ``hi`` (``n + 1`` edges, exact ends)."""
    if not hi > lo:
        raise ValueError("lin_edges needs lo < hi")
    if int(n) < 1:
        raise ValueError("lin_edges needs n >= 1 bins")
    edges = np.linspace(float(lo), float(hi), int(n) + 1)
    edges[0], edges[-1] = lo, hi
    return edges


def parse_edges(spec: str) -> np.ndarray:
    """Bin edges from ``'log:LO:HI:N'`` (:func:`log_edges`) or ``'lin:LO:HI:N'`` (:func:`lin_edges`)."""
    parts = str(spec).split(":")
    if len(parts) != 4 or parts[0] not in ("log", "lin"):
        raise ValueError(f"bad edges specification {spec!r}; expected 'log:LO:HI:N' or 'lin:LO:HI:N'")
    try:
        lo, hi, n = float(parts[1]), float(parts[2]), int(parts[3])
    except ValueError:
        raise ValueError(f"bad edges specification {spec!r}; expected 'log:LO:HI:N' or 'lin:LO:HI:N'") from None
    return log_edges(lo, hi, n) if parts[0] == "log" else lin_edges(lo, hi, n)


# -------------------------------------------------------------------- Spectrum
def _opt_float(x) -> float | None:
    return None if x is None else float(x)


@dataclass(frozen=True, eq=False)
class Spectrum:
    """Histogram of particle energies in one energy variable.

    ``counts`` holds the (weighted) particles per bin and ``n_total`` all particles, including
    ``underflow`` below ``edges[0]`` and ``overflow`` above ``edges[-1]``.  A spectrum whose
    counts are unknown stores only the normalised density ``pdf`` = (dN/dx) / N instead.
    ``time``, ``run_id`` and ``c`` are metadata.
    """

    variable: str
    edges: np.ndarray
    counts: np.ndarray | None = None
    pdf: np.ndarray | None = None
    n_total: float | None = None
    underflow: float | None = None
    overflow: float | None = None
    time: float | None = None
    run_id: int | str | None = None
    c: float | None = None

    def __post_init__(self):
        set_ = functools.partial(object.__setattr__, self)
        _check_variable(self.variable)
        edges = np.array(self.edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise ValueError("edges must be a 1D array with at least 2 entries")
        if not np.all(np.isfinite(edges)) or np.any(np.diff(edges) <= 0):
            raise ValueError("edges must be finite and strictly increasing")
        lo = _LOWER_BOUND[self.variable]
        if edges[0] < lo - _DOMAIN_TOL * max(1.0, lo):
            raise ValueError(f"first edge {edges[0]!r} is below the domain of {self.variable!r} ({lo})")
        set_("edges", edges)
        nb = edges.size - 1
        if self.counts is None and self.pdf is None:
            raise ValueError("a Spectrum needs counts or pdf")
        for name in ("counts", "pdf"):
            arr = getattr(self, name)
            if arr is not None:
                arr = np.array(arr, dtype=np.float64)
                if arr.shape != (nb,):
                    raise ValueError(f"{name} has shape {arr.shape}, expected ({nb},) for {nb + 1} edges")
                set_(name, arr)
        if self.counts is not None:
            under = 0.0 if self.underflow is None else float(self.underflow)
            over = 0.0 if self.overflow is None else float(self.overflow)
            set_("underflow", under)
            set_("overflow", over)
            if self.n_total is None:
                set_("n_total", float(self.counts.sum() + under + over))
        for name in ("n_total", "underflow", "overflow", "time", "c"):
            set_(name, _opt_float(getattr(self, name)))

    # ------------------------------------------------------------- geometry
    @property
    def n_bins(self) -> int:
        return self.edges.size - 1

    @property
    def widths(self) -> np.ndarray:
        return np.diff(self.edges)

    def centers(self, kind: str = "arithmetic") -> np.ndarray:
        """Bin centres, ``'arithmetic'`` (e0 + e1)/2 or ``'geometric'`` sqrt(e0 e1)."""
        e0, e1 = self.edges[:-1], self.edges[1:]
        if kind == "arithmetic":
            return 0.5 * (e0 + e1)
        if kind == "geometric":
            if self.edges[0] <= 0:
                raise ValueError("geometric centres need positive edges")
            return np.sqrt(e0 * e1)
        raise ValueError("kind must be 'arithmetic' or 'geometric'")

    @property
    def has_counts(self) -> bool:
        return self.counts is not None

    @property
    def n_in_range(self) -> float | None:
        return None if self.counts is None else float(self.counts.sum())

    # ------------------------------------------------------------ densities
    def dN_dx(self, normalize: str | None = "total") -> np.ndarray:
        """Differential spectrum dN/dx per unit of ``variable``.

        ``normalize='total'`` divides by ``n_total``, so the result integrates to one minus the
        out-of-range fraction; ``'in_range'`` integrates to 1 over the bins and ``None`` gives
        particles per unit x.  A pdf-only spectrum returns its stored pdf for ``'total'``.
        """
        w = self.widths
        if self.counts is None:
            if normalize is None:
                raise ValueError("this spectrum only stores a normalised pdf (legacy data): counts are unknown")
            if normalize == "total":
                return self.pdf.copy()
            if normalize == "in_range":
                return self.pdf / float(np.sum(self.pdf * w))
            raise ValueError("normalize must be 'total', 'in_range' or None")
        if normalize is None:
            return self.counts / w
        if normalize == "total":
            norm = self.n_total
        elif normalize == "in_range":
            norm = float(self.counts.sum())
        else:
            raise ValueError("normalize must be 'total', 'in_range' or None")
        if not norm:
            raise ValueError("cannot normalise an empty spectrum")
        return self.counts / w / norm

    # ---------------------------------------------------------------- means
    def _bin_weights(self) -> np.ndarray:
        """Particles (counts) or probability (pdf * width) per bin."""
        if self.counts is not None:
            return self.counts
        return self.pdf * self.widths

    def mean(self, variable: str = "gamma_minus_1", *, in_range: bool = False) -> float:
        """Mean of an energy variable, placing each particle at its bin midpoint in ``variable``.

        The mean over all particles raises ValueError when some lie outside the bins;
        ``in_range=True`` averages over the in-range particles only.  The history column
        ``KE_cr / (N c^2)`` corresponds to ``variable='gamma_minus_1'``.
        """
        _check_variable(variable)
        w = self._bin_weights()
        if self.counts is not None and not in_range and (self.underflow or self.overflow):
            raise ValueError(f"{self.underflow:g} particles below and {self.overflow:g} above the bins: the mean of "
                             "all particles is unknown (use mean_bounds, or mean(..., in_range=True))")
        norm = float(w.sum())
        if not norm:
            raise ValueError("cannot take the mean of an empty spectrum")
        e = convert_values(self.edges, self.variable, variable)
        return float(np.sum(w * 0.5 * (e[:-1] + e[1:])) / norm)

    def mean_bounds(self, variable: str = "gamma_minus_1") -> tuple[float, float]:
        """Lower and upper bounds on the mean of ``variable`` from the bin edges.

        With counts the bounds cover all ``n_total`` particles: underflow lies between the domain
        minimum and the first edge, and overflow above the last edge makes the upper bound inf.
        """
        _check_variable(variable)
        w = self._bin_weights()
        e = convert_values(self.edges, self.variable, variable)
        lo, hi = float(np.sum(w * e[:-1])), float(np.sum(w * e[1:]))
        if self.counts is None:
            norm = float(w.sum())
        else:
            norm = float(self.n_total)
            lo += self.underflow * _LOWER_BOUND[variable] + self.overflow * e[-1]
            hi += self.underflow * e[0] + (math.inf if self.overflow else 0.0)
        if not norm:
            raise ValueError("cannot bound the mean of an empty spectrum")
        return lo / norm, hi / norm

    # ------------------------------------------------------ change variable
    def to(self, variable: str) -> "Spectrum":
        """The same histogram in another energy variable, with the edges mapped and counts unchanged.

        A stored ``pdf`` is rescaled so that pdf * width is preserved.
        """
        _check_variable(variable)
        if variable == self.variable:
            return self
        edges = convert_values(self.edges, self.variable, variable)
        pdf = None if self.pdf is None else self.pdf * self.widths / np.diff(edges)
        return replace(self, variable=variable, edges=edges, pdf=pdf)

    # ------------------------------------------------------------ map-reduce
    def __add__(self, other: "Spectrum") -> "Spectrum":
        """Sum two histograms of the same snapshot, such as two meshblock files.

        Variable, edges and time (when both are set) must agree; ``run_id`` and ``c`` are kept
        only when equal.
        """
        if not isinstance(other, Spectrum):
            return NotImplemented
        if other.variable != self.variable:
            raise ValueError(f"cannot add spectra in {self.variable!r} and {other.variable!r}; convert with .to()")
        if other.edges.shape != self.edges.shape or not np.array_equal(other.edges, self.edges):
            raise ValueError("cannot add spectra with different bin edges")
        if self.counts is None or other.counts is None:
            raise ValueError("cannot add pdf-only (legacy) spectra: counts are unknown")
        if self.time is not None and other.time is not None:
            if not math.isclose(self.time, other.time, rel_tol=1e-6, abs_tol=1e-6):
                raise ValueError(f"cannot add spectra at different times ({self.time} vs {other.time})")
        time = self.time if self.time is not None else other.time
        return Spectrum(
            variable=self.variable, edges=self.edges, counts=self.counts + other.counts,
            n_total=self.n_total + other.n_total,
            underflow=self.underflow + other.underflow, overflow=self.overflow + other.overflow,
            time=time,
            run_id=self.run_id if self.run_id == other.run_id else None,
            c=self.c if self.c == other.c else None,
        )

    def __radd__(self, other):  # lets sum(list_of_spectra) work
        if isinstance(other, (int, float)) and other == 0:
            return self
        return NotImplemented

    def __repr__(self) -> str:
        kind = "counts" if self.counts is not None else "pdf"
        n = f", n_total={self.n_total:g}" if self.n_total is not None else ""
        t = f", time={self.time:g}" if self.time is not None else ""
        return (f"Spectrum({self.variable}, {self.n_bins} bins [{self.edges[0]:g}, {self.edges[-1]:g}], "
                f"{kind}{n}{t})")


# ------------------------------------------------------------------ histogram
def histogram(values, edges, variable: str, weights=None, *, time: float | None = None,
              run_id: int | str | None = None, c: float | None = None) -> Spectrum:
    """Histogram energy values into a :class:`Spectrum`, keeping underflow and overflow.

    Bins are [e_i, e_i+1) except the last, which includes its upper edge as in ``np.histogram``.
    Non-finite values are dropped with a warning and not counted in ``n_total``.
    """
    spec_edges = np.asarray(edges, dtype=np.float64)
    x = np.asarray(values, dtype=np.float64).ravel()
    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None and w.shape != x.shape:
        raise ValueError("weights must have the same shape as values")
    finite = np.isfinite(x)
    if not finite.all():
        warnings.warn(f"histogram: ignoring {int((~finite).sum())} non-finite values", stacklevel=2)
        x = x[finite]
        w = None if w is None else w[finite]
    nb = spec_edges.size - 1
    if nb < 1:
        raise ValueError("need at least 2 edges")
    idx = np.searchsorted(spec_edges, x, side="right") - 1
    idx[x == spec_edges[-1]] = nb - 1  # right edge belongs to the last bin
    under_mask = idx < 0
    over_mask = idx >= nb
    inside = ~(under_mask | over_mask)
    if w is None:
        counts = np.bincount(idx[inside], minlength=nb).astype(np.float64)
        under, over, total = float(under_mask.sum()), float(over_mask.sum()), float(x.size)
    else:
        counts = np.bincount(idx[inside], weights=w[inside], minlength=nb).astype(np.float64)
        under, over, total = float(w[under_mask].sum()), float(w[over_mask].sum()), float(w.sum())
    return Spectrum(variable=variable, edges=spec_edges, counts=counts, n_total=total,
                    underflow=under, overflow=over, time=time, run_id=run_id, c=c)


# ----------------------------------------------------------- shape statistics
def _positive_centers(spec: Spectrum) -> np.ndarray:
    e = spec.edges
    if e[0] > 0:
        return spec.centers("geometric")
    out = spec.centers("arithmetic")
    pos = e[:-1] > 0
    out[pos] = np.sqrt(e[:-1][pos] * e[1:][pos])
    return out


def power_law_index(spec: Spectrum, lo: float, hi: float, *, normalize: str | None = "total") -> float:
    """Local power-law index alpha of dN/dx ~ x^-alpha between ``lo`` and ``hi``.

    Unweighted least-squares fit of log dN/dx against log x over the non-empty bins whose
    geometric centre lies in [lo, hi], with x in the spectrum's own variable.
    """
    x = _positive_centers(spec)
    f = spec.dN_dx(normalize)
    use = (x >= lo) & (x <= hi) & (f > 0)
    if int(use.sum()) < 2:
        raise ValueError(f"fewer than 2 non-empty bins with centres in [{lo:g}, {hi:g}]")
    slope = np.polyfit(np.log(x[use]), np.log(f[use]), 1)[0]
    return float(-slope)


def cutoff_value(spec: Spectrum, level: float, *, normalize: str | None = "total") -> float:
    """Value of x where dN/dx first falls below ``level`` above the peak of the spectrum.

    The crossing is interpolated in (log x, log dN/dx) between geometric bin centres, or taken
    at the lower edge of the first bin below ``level`` if that bin is empty.  Returns NaN if the
    spectrum stays above ``level`` and raises ValueError if the peak is below it.
    """
    x = _positive_centers(spec)
    f = spec.dN_dx(normalize)
    peak = int(np.argmax(f))
    if not f[peak] >= level:
        raise ValueError(f"the spectrum peak {f[peak]:g} is below the level {level:g}")
    below = np.nonzero(f[peak:] < level)[0]
    if below.size == 0:
        return math.nan
    k = peak + int(below[0])
    if f[k] <= 0:
        return float(spec.edges[k])
    t = (math.log(level) - math.log(f[k - 1])) / (math.log(f[k]) - math.log(f[k - 1]))
    return float(math.exp(math.log(x[k - 1]) + t * (math.log(x[k]) - math.log(x[k - 1]))))


# ------------------------------------------------------ snapshot map-reduce
def _block_spectrum(path: str | Path, *, c: float, edges: np.ndarray, variable: str) -> Spectrum:
    """Histogram the momenta of one meshblock particle file; module level so that it pickles."""
    from ..io.particles import read_particle_block

    block = read_particle_block(path, fields=("u",))
    vals = energy_values(block.u, c, variable)
    return histogram(vals, edges, variable, time=block.time, c=c)


def particle_snapshot_spectrum(run_dir: str | Path, file_number: int, c: SpeedOfLight, edges,
                               variable: str = "gamma", kind: str = "*", backend: str = "auto",
                               n_workers: int | str = "auto", progress: bool = False,
                               on_error: str = "raise", *, file_id: str | None = None,
                               basename: str | None = None, expected_n: int | None = None,
                               mem_per_task: int | str | None = "256MB") -> Spectrum | None:
    """Energy spectrum of all particles of one particle output, summed over meshblock files.

    Blocks are histogrammed in parallel workers.  MPI ranks other than 0 return None.

    Parameters
    ----------
    run_dir : path
        Directory with ``<basename>.block<gid>.<file_id>.<NNNNN>.par.{tab,bin}`` files.
    file_number : int
        Output number NNNNN.
    c : float or RunConfig
        Speed of light.
    kind : 'tab', 'bin' or '*'
        File format.
    backend, n_workers, progress, mem_per_task
        Passed to :func:`shearpic.parallel.parallel_map`.
    on_error : 'raise' or 'collect'
        ``'collect'`` skips unreadable blocks with a warning; the result then covers the blocks read.
    file_id, basename : str, optional
        Select one output stream; files from several streams raise ValueError.
    expected_n : int, optional
        Required ``n_total``, e.g. ``cfg.n_par``; any other value raises ValueError.
    """
    from ..config import parse_run_id
    from ..io.particles import particle_output_files
    from ..parallel import parallel_map

    _check_variable(variable)
    if on_error not in ("raise", "collect"):
        raise ValueError("on_error must be 'raise' or 'collect'")
    c_val = _speed_of_light(c)
    run_id = getattr(c, "run_id", None)
    edges = np.asarray(edges, dtype=np.float64)
    files = particle_output_files(run_dir, file_number, kind, file_id=file_id, basename=basename)
    worker = functools.partial(_block_spectrum, c=c_val, edges=edges, variable=variable)
    parts = parallel_map(worker, files, backend=backend, n_workers=n_workers, progress=progress,
                         mem_per_task=mem_per_task, desc=f"spectrum {file_number:05d}", on_error=on_error)
    if parts is None:  # results are gathered on MPI rank 0
        return None
    parts = drop_failed_blocks(parts, files, f"spectrum {file_number:05d}")
    time = common_block_time(parts, file_number, run_dir)
    total = functools.reduce(lambda a, b: a + b, parts)
    if expected_n is not None and total.n_total != expected_n:
        raise ValueError(f"output {file_number:05d} of {run_dir}: read {total.n_total:.0f} particles from "
                         f"{len(parts)} of {len(files)} block files, expected {int(expected_n)}")
    if run_id is None:
        run_id = parse_run_id(run_dir)
    return replace(total, time=time, run_id=run_id, c=c_val)


def drop_failed_blocks(parts: list, files: list, what: str) -> list:
    """Drop TaskError results of a block map with a warning naming the files; RuntimeError if all failed."""
    from ..parallel import TaskError

    failed = [p for p in parts if isinstance(p, TaskError)]
    if not failed:
        return parts
    ok = [p for p in parts if not isinstance(p, TaskError)]
    names = ", ".join(Path(files[e.index]).name for e in failed[:5]) + (" ..." if len(failed) > 5 else "")
    if not ok:
        raise RuntimeError(f"all {len(files)} particle files of {what} failed; "
                           f"first error: {failed[0].error!r}\n{failed[0].traceback}")
    warnings.warn(f"{what}: skipped {len(failed)} of {len(files)} unreadable block file(s) ({names}); first error: "
                  f"{failed[0].error!r}.  The result covers only the blocks that were read.", stacklevel=3)
    return ok


def common_block_time(parts, file_number: int, run_dir) -> float:
    """Time shared by all block results; ValueError if they differ."""
    times = np.array([p.time for p in parts], dtype=np.float64)
    if np.ptp(times) > 1e-6 * max(1.0, float(np.max(np.abs(times)))):
        raise ValueError(f"particle files of output {file_number:05d} report different times "
                         f"({times.min()} .. {times.max()}); mixed outputs in {run_dir}?")
    return float(times[0])
