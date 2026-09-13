"""Energy spectra of turbulent fluctuations on periodic, possibly anisotropic grids.

The runs are homogeneous only along x and z, so fluctuations are taken relative to x (and z)
averages at each y, and their energy is binned in shells of |k|.  Spectra E(k) are energy
densities per unit wavenumber, rho0 U0^2 a in code units, and integrate to the fluctuation
energy density by Parseval's theorem.  Arrays are indexed (x, y[, z]).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = [
    "FieldSpectrum", "fluctuations", "shell_spectrum", "rebin_log", "mhd_energy_spectra", "average_spectra",
    "sum_spectra", "fit_power_law", "local_slopes", "power_law_amplitude", "nyquist_wavenumber", "shell_width",
]

PARSEVAL_RTOL = 1e-10
_EDGE_TOL = 1e-9  # relative tolerance for lattice points on a shell edge (they go to the upper shell)

Lengths = Union[Sequence[float], "RunConfig"]


# ============================================================================ container
@dataclass(frozen=True, eq=False)
class FieldSpectrum:
    """Shell-integrated energy spectrum of a fluctuation field.

    ``E`` is the energy density per unit wavenumber in the bins bounded by ``k_edges`` [1/a] and
    ``n_modes`` the number of lattice modes per bin.  ``total`` is the volume-averaged energy
    density of all modes, of which ``energy_beyond_kmax`` lies above the last edge.  ``time`` is
    None for time averages and ``kind`` is a free label such as ``'kin'`` or ``'mag'``.
    """

    k_edges: np.ndarray
    E: np.ndarray
    n_modes: np.ndarray
    total: float
    energy_beyond_kmax: float = 0.0
    time: float | None = None
    kind: str = ""

    def __post_init__(self):
        edges = np.asarray(self.k_edges, dtype=np.float64)
        E = np.asarray(self.E, dtype=np.float64)
        n_modes = np.asarray(self.n_modes, dtype=np.int64)
        if edges.ndim != 1 or edges.size < 2 or np.any(np.diff(edges) <= 0) or edges[0] < 0:
            raise ValueError("k_edges must be a 1D, strictly increasing, non-negative array with >= 2 entries")
        if E.shape != (edges.size - 1,) or n_modes.shape != E.shape:
            raise ValueError(f"E and n_modes must have shape ({edges.size - 1},), got {E.shape} and {n_modes.shape}")
        object.__setattr__(self, "k_edges", edges)
        object.__setattr__(self, "E", E)
        object.__setattr__(self, "n_modes", n_modes)
        object.__setattr__(self, "total", float(self.total))
        object.__setattr__(self, "energy_beyond_kmax", float(self.energy_beyond_kmax))
        if self.time is not None:
            object.__setattr__(self, "time", float(self.time))

    # ------------------------------------------------------------ derived
    @property
    def dk(self) -> np.ndarray:
        """Bin widths ``diff(k_edges)``."""
        return np.diff(self.k_edges)

    @property
    def k(self) -> np.ndarray:
        """Arithmetic bin centres, m dk for the linear shells m >= 1 of :func:`shell_spectrum`."""
        return 0.5 * (self.k_edges[:-1] + self.k_edges[1:])

    @property
    def k_geometric(self) -> np.ndarray:
        """Geometric bin centres (NaN for a bin starting at k = 0)."""
        with np.errstate(invalid="ignore"):
            out = np.sqrt(self.k_edges[:-1] * self.k_edges[1:])
        out[self.k_edges[:-1] <= 0] = np.nan
        return out

    @property
    def bin_energy(self) -> np.ndarray:
        """Energy density in each bin, ``E * dk``."""
        return self.E * self.dk

    @property
    def energy_in_bins(self) -> float:
        """Energy density up to the last edge, sum(E dk)."""
        return float(self.bin_energy.sum())

    def parseval_error(self) -> float:
        """Relative mismatch ``|sum(E dk) + energy_beyond_kmax - total| / total`` (0 for an empty field)."""
        diff = abs(self.energy_in_bins + self.energy_beyond_kmax - self.total)
        return diff / abs(self.total) if self.total != 0 else diff

    def energy_between(self, k_min: float, k_max: float) -> float:
        """Energy density of the bins whose centres lie in ``[k_min, k_max]``."""
        sel = (self.k >= k_min) & (self.k <= k_max)
        return float(self.bin_energy[sel].sum())


# ======================================================================= fluctuations
def _avg_axes(ndim: int) -> tuple[int, ...]:
    if ndim == 2:
        return (0,)
    if ndim == 3:
        return (0, 2)
    raise ValueError(f"fields must be 2D (x, y) or 3D (x, y, z) arrays, got ndim={ndim}")


def fluctuations(rho: np.ndarray, v: Sequence[np.ndarray], B: Sequence[np.ndarray] | None, *,
                 velocity_mean: str = "favre", weight: str = "sqrt_rho",
                 ) -> tuple[tuple[np.ndarray, ...], tuple[np.ndarray, ...] | None]:
    """Kinetic and magnetic fluctuation fields relative to x (and z) averages.

    Returns ``(w, b)``: the weighted velocity fluctuations w = weight * (v - mean) and
    b = B - <B>_x per component, with b None if ``B`` is None.

    Parameters
    ----------
    rho, v, B
        Density and 2 or 3 velocity and magnetic field components, all of one shape.
    velocity_mean : {'favre', 'reynolds'}
        Density-weighted mean <rho v>_x / <rho>_x or plain mean <v>_x.
    weight : {'sqrt_rho', 'mean_rho', 'none'}
        sqrt(rho), sqrt(<rho>) over the box, or 1.  With sqrt(rho) and the Favre mean,
        0.5 <w^2> is the fluctuation kinetic energy density.
    """
    rho = np.asarray(rho, dtype=np.float64)
    axes = _avg_axes(rho.ndim)
    if velocity_mean not in ("favre", "reynolds"):
        raise ValueError("velocity_mean must be 'favre' or 'reynolds'")
    if weight not in ("sqrt_rho", "mean_rho", "none"):
        raise ValueError("weight must be 'sqrt_rho', 'mean_rho' or 'none'")
    if np.any(rho <= 0):
        raise ValueError("density must be positive")

    rho_mean = rho.mean(axis=axes, keepdims=True) if velocity_mean == "favre" else None
    if weight == "sqrt_rho":
        wfac = np.sqrt(rho)
    elif weight == "mean_rho":
        wfac = math.sqrt(float(rho.mean()))
    else:
        wfac = 1.0

    w = []
    for comp in v:
        u = np.asarray(comp, dtype=np.float64)
        if u.shape != rho.shape:
            raise ValueError(f"velocity component shape {u.shape} != density shape {rho.shape}")
        if velocity_mean == "favre":
            mean = (rho * u).mean(axis=axes, keepdims=True) / rho_mean
        else:
            mean = u.mean(axis=axes, keepdims=True)
        w.append(wfac * (u - mean))

    b = None
    if B is not None:
        b = []
        for comp in B:
            f = np.asarray(comp, dtype=np.float64)
            if f.shape != rho.shape:
                raise ValueError(f"magnetic component shape {f.shape} != density shape {rho.shape}")
            b.append(f - f.mean(axis=axes, keepdims=True))
        b = tuple(b)
    return tuple(w), b


# ======================================================================= shell spectra
def _lengths(L: Lengths) -> tuple[float, ...]:
    if hasattr(L, "L"):
        L = L.L
    return tuple(float(x) for x in L)


def _active_axes(shape: tuple[int, ...], L: tuple[float, ...]) -> list[tuple[int, float]]:
    if len(L) < len(shape):
        raise ValueError(f"need one length per array axis: shape {shape}, L {L}")
    active = [(n, L[i]) for i, n in enumerate(shape) if n > 1]
    if not active:
        raise ValueError("fields have no axis with more than one cell")
    if any(not (l > 0) for _, l in active):
        raise ValueError(f"box lengths must be positive, got {L}")
    return active


def shell_width(shape: tuple[int, ...], L: Lengths) -> float:
    """Default shell width ``2 pi / min(L_i)`` over the axes with more than one cell."""
    L = _lengths(L)
    return 2.0 * math.pi / min(l for _, l in _active_axes(tuple(shape), L))


def nyquist_wavenumber(shape: tuple[int, ...], L: Lengths) -> float:
    """``min_i pi n_i / L_i`` over the axes with more than one cell: the largest isotropic ``|k|``."""
    L = _lengths(L)
    return min(math.pi * n / l for n, l in _active_axes(tuple(shape), L))


@dataclass(frozen=True)
class _Binning:
    index: np.ndarray        # flat shell index per mode (int64), n_shells for modes beyond k_max
    n_shells: int
    edges: np.ndarray
    n_modes: np.ndarray


def _binning(shape: tuple[int, ...], L: tuple[float, ...], dk: float | None, k_max) -> _Binning:
    _active_axes(shape, L)  # validates
    if dk is None:
        dk = shell_width(shape, L)
    dk = float(dk)
    if not dk > 0:
        raise ValueError("dk must be positive")
    k2 = np.zeros(shape, dtype=np.float64)
    for axis, n in enumerate(shape):
        ki = 2.0 * np.pi * np.fft.fftfreq(n, d=L[axis] / n)
        sh = [1] * len(shape)
        sh[axis] = n
        k2 = k2 + ki.reshape(sh) ** 2
    K = np.sqrt(k2, out=k2)
    index = np.floor(K / dk + 0.5 + _EDGE_TOL).astype(np.int64)
    del K
    if isinstance(k_max, str):
        if k_max == "nyquist":
            k_max = nyquist_wavenumber(shape, L)
        elif k_max == "corner":
            k_max = float(index.max()) * dk
        else:
            raise ValueError("k_max must be 'nyquist', 'corner' or a number")
    k_max = float(k_max)
    if not k_max > 0:
        raise ValueError("k_max must be positive")
    n_shells = int(math.floor(k_max / dk + _EDGE_TOL)) + 1   # shells m = 0 .. M with m dk <= k_max
    np.minimum(index, n_shells, out=index)                   # overflow bin n_shells
    index = index.ravel()
    counts = np.bincount(index, minlength=n_shells + 1)
    edges = np.concatenate(([0.0], (np.arange(n_shells) + 0.5) * dk))   # 0, dk/2, 3dk/2, ..., (M + 1/2) dk
    return _Binning(index=index, n_shells=n_shells, edges=edges, n_modes=counts[:n_shells].astype(np.int64))


def _spectrum(components: Sequence[np.ndarray], binning: _Binning, prefactor: float, time, kind: str) -> FieldSpectrum:
    energy = np.zeros(binning.n_shells + 1, dtype=np.float64)
    real_space = 0.0
    for f in components:
        f = np.asarray(f, dtype=np.float64)
        if f.size != binning.index.size:
            raise ValueError("all components must have the same shape")
        F = np.fft.fftn(f, norm="forward")
        power = F.real ** 2 + F.imag ** 2
        del F
        energy += np.bincount(binning.index, weights=power.ravel(), minlength=binning.n_shells + 1)
        real_space += float(np.mean(f * f))
    energy *= prefactor
    total_real = prefactor * real_space
    total_modes = float(energy.sum())
    width = np.diff(binning.edges)
    spec = FieldSpectrum(k_edges=binning.edges, E=energy[:-1] / width, n_modes=binning.n_modes,
                         total=total_modes, energy_beyond_kmax=float(energy[-1]), time=time, kind=kind)
    # Parseval: the binned spectrum integrates to the real-space energy density
    lhs = float((spec.E * width).sum()) + spec.energy_beyond_kmax
    if abs(lhs - total_real) > PARSEVAL_RTOL * abs(total_real):
        raise AssertionError(f"Parseval check failed for {kind or 'spectrum'}: sum(E dk) + beyond = {lhs!r}, "
                             f"prefactor <sum f^2> = {total_real!r}")
    return spec


def shell_spectrum(components: Sequence[np.ndarray], L: Lengths, *, dk: float | None = None,
                   k_max: float | str = "nyquist", prefactor: float = 0.5, time: float | None = None,
                   kind: str = "") -> FieldSpectrum:
    """Linear-shell energy spectrum E(|k|) of a scalar or vector field on a periodic grid.

    With F_k = fftn(f, norm='forward') the mode energy is prefactor * sum |F_k|^2 over components.
    Shell m >= 1 holds (m - 1/2) dk <= |k| < (m + 1/2) dk and shell 0 holds |k| < dk/2; lattice
    points on an edge go to the upper shell.  An AssertionError is raised if the result fails
    the Parseval check.

    Parameters
    ----------
    components : sequence of arrays
        Field components of one 2D (nx, ny) or 3D (nx, ny, nz) shape.
    L : (Lx, Ly[, Lz]) or RunConfig
        Box lengths.
    dk : float, optional
        Shell width, by default 2 pi / min(L_i) over the axes with more than one cell.
    k_max : 'nyquist', 'corner' or float
        Largest shell centre; ``'nyquist'`` is min_i pi n_i / L_i and ``'corner'`` keeps every mode.
    prefactor : float
        0.5 for energy densities.
    """
    if isinstance(components, np.ndarray):
        components = [components]
    components = list(components)
    if not components:
        raise ValueError("need at least one component")
    shape = np.shape(components[0])
    if any(np.shape(c) != shape for c in components):
        raise ValueError("all components must have the same shape")
    binning = _binning(tuple(shape), _lengths(L), dk, k_max)
    return _spectrum(components, binning, float(prefactor), time, kind)


def mhd_energy_spectra(rho: np.ndarray, v: Sequence[np.ndarray], B: Sequence[np.ndarray], cfg: Lengths, *,
                       dk: float | None = None, k_max: float | str = "nyquist", velocity_mean: str = "favre",
                       weight: str = "sqrt_rho", time: float | None = None) -> dict[str, FieldSpectrum]:
    """Kinetic, magnetic and total fluctuation energy spectra of one MHD snapshot.

    Arguments are as for :func:`fluctuations` and :func:`shell_spectrum`, with ``cfg`` supplying
    the box lengths.  Returns a dict of ``'kin'``, ``'mag'`` and ``'tot'`` spectra on the same shells.
    """
    w, b = fluctuations(rho, v, B, velocity_mean=velocity_mean, weight=weight)
    binning = _binning(tuple(np.shape(w[0])), _lengths(cfg), dk, k_max)
    kin = _spectrum(w, binning, 0.5, time, "kin")
    del w
    mag = _spectrum(b, binning, 0.5, time, "mag")
    return {"kin": kin, "mag": mag, "tot": sum_spectra(kin, mag, kind="tot")}


# ======================================================================= operations
def _same_edges(a: FieldSpectrum, b: FieldSpectrum) -> bool:
    return a.k_edges.shape == b.k_edges.shape and np.allclose(a.k_edges, b.k_edges, rtol=1e-12, atol=0)


def sum_spectra(a: FieldSpectrum, b: FieldSpectrum, *, kind: str = "") -> FieldSpectrum:
    """Sum of two spectra on identical bins, such as kinetic plus magnetic; ``time`` is kept if both agree."""
    if not _same_edges(a, b) or not np.array_equal(a.n_modes, b.n_modes):
        raise ValueError("spectra must have identical bins to be added")
    time = a.time if (a.time is not None and b.time is not None and abs(a.time - b.time) <= 1e-9 * max(1.0, abs(a.time))) else None
    return FieldSpectrum(k_edges=a.k_edges, E=a.E + b.E, n_modes=a.n_modes, total=a.total + b.total,
                         energy_beyond_kmax=a.energy_beyond_kmax + b.energy_beyond_kmax, time=time, kind=kind)


def average_spectra(spectra: Sequence[FieldSpectrum]) -> FieldSpectrum:
    """Arithmetic mean of spectra on identical bins, with ``time`` None and ``kind`` kept if common."""
    spectra = list(spectra)
    if not spectra:
        raise ValueError("need at least one spectrum")
    first = spectra[0]
    for s in spectra[1:]:
        if not _same_edges(first, s) or not np.array_equal(first.n_modes, s.n_modes):
            raise ValueError("spectra must have identical bins (k_edges and n_modes) to be averaged")
    kinds = {s.kind for s in spectra}
    return FieldSpectrum(
        k_edges=first.k_edges, E=np.mean([s.E for s in spectra], axis=0), n_modes=first.n_modes,
        total=float(np.mean([s.total for s in spectra])),
        energy_beyond_kmax=float(np.mean([s.energy_beyond_kmax for s in spectra])),
        time=None, kind=first.kind if len(kinds) == 1 else "",
    )


def rebin_log(spec: FieldSpectrum, n_per_decade: int = 8) -> FieldSpectrum:
    """Merge adjacent bins into approximately logarithmic bins, conserving energy exactly.

    Target edges k_1 10^(j / n_per_decade), from the first positive edge k_1, are moved up to the
    nearest existing edge, so each new bin is a union of whole original bins and spectra with the
    same edges get the same bins.  A first bin starting at k = 0 is kept.
    """
    if n_per_decade < 1:
        raise ValueError("n_per_decade must be >= 1")
    edges = spec.k_edges
    start = 1 if edges[0] <= 0 else 0
    pos = edges[start:]
    if pos.size < 2:
        return spec
    n_target = int(math.ceil(n_per_decade * math.log10(pos[-1] / pos[0]) - 1e-9))
    target = pos[0] * 10.0 ** (np.arange(n_target + 1) / n_per_decade)
    up = np.searchsorted(pos, target * (1.0 - 1e-12), side="left")   # first edge >= target (round-off tolerant)
    up = np.clip(up, 0, pos.size - 1)
    keep = np.unique(np.concatenate(([0], up, [pos.size - 1])))
    keep_idx = keep + start                                  # indices into edges
    if start:
        keep_idx = np.concatenate(([0], keep_idx))
    new_edges = edges[keep_idx]
    energy = np.concatenate(([0.0], np.cumsum(spec.bin_energy)))
    modes = np.concatenate(([0], np.cumsum(spec.n_modes)))
    new_energy = np.diff(energy[keep_idx])
    new_modes = np.diff(modes[keep_idx])
    return FieldSpectrum(k_edges=new_edges, E=new_energy / np.diff(new_edges), n_modes=new_modes, total=spec.total,
                         energy_beyond_kmax=spec.energy_beyond_kmax, time=spec.time, kind=spec.kind)


_WEIGHTINGS = ("log", "shell")


def _fit_selection(spec: FieldSpectrum, k_min: float, k_max: float, weighting: str) -> tuple[np.ndarray, np.ndarray]:
    """Mask of usable bins (centre in ``[k_min, k_max]``, ``E > 0``) and their fit weights."""
    if weighting not in _WEIGHTINGS:
        raise ValueError(f"weighting must be one of {_WEIGHTINGS}, got {weighting!r}")
    if not k_max > k_min > 0:
        raise ValueError("need 0 < k_min < k_max")
    k = spec.k
    sel = (k >= k_min) & (k <= k_max) & (spec.E > 0)
    if weighting == "log":
        w = (spec.dk / k)[sel]          # d ln k of each bin: equal weight per logarithmic interval
    else:
        w = np.ones(int(sel.sum()))     # equal weight per bin
    return sel, w


def fit_power_law(spec: FieldSpectrum, k_min: float, k_max: float, *, weighting: str = "log") -> tuple[float, float]:
    """Weighted least-squares slope of log E against log k over bins with centres in [k_min, k_max].

    Bins with E <= 0 are excluded.  ``weighting='log'`` weights each bin by dk/k, so that equal
    intervals of ln k count equally; ``'shell'`` weights bins equally, which on linear shells
    favours the high-k end of the range.  Returns ``(slope, stderr)``, where the standard error
    measures only the scatter about a straight line.
    """
    sel, w = _fit_selection(spec, k_min, k_max, weighting)
    n = int(sel.sum())
    if n < 3:
        raise ValueError(f"only {n} bins with E > 0 in [{k_min}, {k_max}]; need >= 3 for a slope and its error")
    x = np.log(spec.k[sel])
    y = np.log(spec.E[sel])
    xm = x - np.dot(w, x) / w.sum()
    ym = y - np.dot(w, y) / w.sum()
    sxx = float(np.dot(w, xm * xm))
    slope = float(np.dot(w, xm * ym) / sxx)
    resid = ym - slope * xm
    stderr = math.sqrt(float(np.dot(w, resid * resid)) / (n - 2) / sxx)
    return slope, stderr


def local_slopes(spec: FieldSpectrum, windows: float | Sequence[Sequence[float]] = 0.5, *,
                 k_min: float | None = None, k_max: float | None = None, weighting: str = "log",
                 min_bins: int = 3) -> list[dict[str, float]]:
    """Local slopes d ln E / d ln k of a spectrum in windows of k.

    A number ``windows`` is a window width in decades; windows overlapping by half within
    [k_min, k_max], by default the range of bins with E > 0, are kept if they hold at least
    ``min_bins`` bins.  A sequence of (k_lo, k_hi) pairs gives the windows explicitly, with a NaN
    slope where there are too few bins.  Returns one dict per window with ``k_lo``, ``k_hi``,
    geometric ``k_centre``, ``slope`` and ``n_bins``.
    """
    if min_bins < 3:
        raise ValueError("min_bins must be >= 3")
    k = spec.k
    ok = (spec.k_edges[:-1] > 0) & (spec.E > 0)
    explicit = not isinstance(windows, (int, float))
    if explicit:
        pairs = [(float(lo), float(hi)) for lo, hi in windows]
    else:
        width = float(windows)
        if not width > 0:
            raise ValueError("window width (decades) must be positive")
        if not ok.any():
            return []
        lo_k = float(k_min) if k_min is not None else float(k[ok].min())
        hi_k = float(k_max) if k_max is not None else float(k[ok].max())
        if not hi_k > lo_k > 0:
            raise ValueError("need 0 < k_min < k_max")
        step = width / 2.0
        j0 = math.ceil(math.log10(lo_k) / step - 1e-9)
        j1 = math.floor((math.log10(hi_k) - width) / step + 1e-9)
        pairs = [(10.0 ** (j * step), 10.0 ** (j * step + width)) for j in range(j0, j1 + 1)]
    rows = []
    for lo, hi in pairs:
        if not hi > lo > 0:
            raise ValueError(f"invalid window ({lo}, {hi})")
        n = int(((k >= lo) & (k <= hi) & ok).sum())
        if n >= min_bins:
            slope = fit_power_law(spec, lo, hi, weighting=weighting)[0]
        elif explicit:
            slope = float("nan")
        else:
            continue
        rows.append({"k_lo": lo, "k_hi": hi, "k_centre": math.sqrt(lo * hi), "slope": slope, "n_bins": n})
    return rows


def power_law_amplitude(spec: FieldSpectrum, slope: float, k_min: float, k_max: float, *,
                        weighting: str = "log") -> float:
    """Amplitude A of A k^slope that best matches log E over [k_min, k_max] for a fixed slope.

    ``weighting`` is as for :func:`fit_power_law` and should match the one used for the slope.
    """
    sel, w = _fit_selection(spec, k_min, k_max, weighting)
    if not sel.any():
        raise ValueError(f"no bins with E > 0 in [{k_min}, {k_max}]")
    return float(np.exp(np.dot(w, np.log(spec.E[sel]) - slope * np.log(spec.k[sel])) / w.sum()))
