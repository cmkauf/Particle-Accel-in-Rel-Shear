"""Diagnostics along tracked-particle trajectories and interpolation of grid fields to particles.

A shear-layer crossing is registered once a particle leaves the band |y - y_l| <= h on the
side opposite to where it was last seen outside it, so gyration about a layer centre is not
counted repeatedly.  Positions are unwrapped with minimum-image steps, which assumes a
particle moves less than Ly/2 between samples.

Trajectory files sample B and cE at the particle's cell (nearest grid point) every fixed
number of cycles, while the pusher interpolates with TSC.  Per-sample v . cE is therefore
aliased at the gyrofrequency and should not be integrated for single-particle energy budgets;
the sampled gamma is reliable, and :func:`sample_field_tsc` evaluates snapshot fields as the
pusher does.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import TYPE_CHECKING, Sequence

import numpy as np

from . import relativity as rel

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig
    from ..io.trajectory import Trajectory

__all__ = [
    "CrossingEvents", "crossing_events", "crossing_events_for", "cumulative_crossings",
    "trajectory_diagnostics", "TRAJECTORY_DIAGNOSTICS", "sample_field_ngp", "sample_field_tsc",
]


# =================================================================== crossings
@dataclass(frozen=True, eq=False)
class CrossingEvents:
    """Shear-layer crossings of one particle, sorted by time.

    ``times`` interpolates the last passage through the layer centre before the particle left
    the hysteresis band on the new side, and ``index`` is the first sample after that passage.
    ``layer`` indexes ``layer_positions`` and ``direction`` is +1 towards larger y, -1 towards
    smaller y.
    """

    times: np.ndarray
    index: np.ndarray
    layer: np.ndarray
    direction: np.ndarray
    layer_positions: tuple[float, ...] = dc_field(default=())
    hysteresis: float = 0.0

    @property
    def count(self) -> int:
        return int(self.times.size)

    def __len__(self) -> int:
        return self.count

    def select(self, layer: int | None = None, direction: int | None = None) -> "CrossingEvents":
        """Events of one layer and/or direction."""
        mask = np.ones(self.count, dtype=bool)
        if layer is not None:
            mask &= self.layer == layer
        if direction is not None:
            mask &= self.direction == direction
        return CrossingEvents(self.times[mask], self.index[mask], self.layer[mask], self.direction[mask],
                              self.layer_positions, self.hysteresis)


def _unwrap_periodic(y: np.ndarray, L: float) -> np.ndarray:
    """Continuous coordinate from periodic samples, assuming |step| < L/2."""
    step = np.diff(y)
    step -= L * np.round(step / L)
    return np.concatenate(([y[0]], y[0] + np.cumsum(step)))


def _layer_crossings(t: np.ndarray, s: np.ndarray, L: float, h: float):
    """Crossings of the images ``k L`` of one layer by the unwrapped offset ``s = y - y_l``."""
    d = s - L * np.round(s / L)  # minimum-image distance to the nearest image
    outside = np.abs(d) > h
    if not outside.any():
        return [], [], []
    # gap g = region (g L + h, (g+1) L - h) between images g and g+1
    gap = np.floor((s - h) / L).astype(np.int64)
    out_idx = np.flatnonzero(outside)
    out_gap = gap[out_idx]
    change = np.flatnonzero(np.diff(out_gap) != 0)
    times, index, direction = [], [], []
    for c in change:
        i0, j = out_idx[c], out_idx[c + 1]
        g0, g1 = out_gap[c], out_gap[c + 1]
        sgn = 1 if g1 > g0 else -1
        seg = s[i0:j + 1]
        for g in range(g0, g1, sgn):
            image = (g + 1) * L if sgn > 0 else g * L
            if sgn > 0:
                passed = (seg[:-1] < image) & (seg[1:] >= image)
            else:
                passed = (seg[:-1] > image) & (seg[1:] <= image)
            k = int(np.flatnonzero(passed)[-1]) + 1  # first sample beyond the last passage
            a, b = seg[k - 1], seg[k]
            frac = (image - a) / (b - a)
            times.append(t[i0 + k - 1] + frac * (t[i0 + k] - t[i0 + k - 1]))
            index.append(i0 + k)
            direction.append(sgn)
    return times, index, direction


def crossing_events(t, y, layers: Sequence[float], Ly: float, hysteresis: float) -> CrossingEvents:
    """Detect crossings of shear layers, with hysteresis, in a y-periodic box of height ``Ly``.

    ``y`` may be wrapped into the box or already unwrapped.  The side of a layer is not updated
    inside the band |y - y_l| <= hysteresis (minimum image), with 0 <= hysteresis < Ly/2.
    """
    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.ndim != 1 or t.shape != y.shape:
        raise ValueError("t and y must be 1D arrays of the same length")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t and y must be finite")
    if not (Ly > 0 and 0 <= hysteresis < 0.5 * Ly):
        raise ValueError("need Ly > 0 and 0 <= hysteresis < Ly/2")
    layers = tuple(float(v) for v in layers)
    times, index, layer, direction = [], [], [], []
    if y.size >= 2:
        s_all = _unwrap_periodic(y, Ly)
        for li, yl in enumerate(layers):
            tt, ii, dd = _layer_crossings(t, s_all - yl, Ly, hysteresis)
            times += tt
            index += ii
            direction += dd
            layer += [li] * len(tt)
    times = np.asarray(times, dtype=np.float64)
    index = np.asarray(index, dtype=np.int64)
    layer = np.asarray(layer, dtype=np.int64)
    direction = np.asarray(direction, dtype=np.int64)
    order = np.lexsort((layer, times))
    return CrossingEvents(times[order], index[order], layer[order], direction[order], layers, float(hysteresis))


def crossing_events_for(traj: "Trajectory", cfg: "RunConfig", hysteresis: float | None = None) -> CrossingEvents:
    """:func:`crossing_events` for a Trajectory with the run's layers; ``hysteresis`` defaults to ``cfg.profile.a``."""
    layers = cfg.profile.layer_positions
    if not layers:
        raise ValueError(f"profile {cfg.profile.kind!r} has no localized shear layers")
    h = cfg.profile.a if hysteresis is None else hysteresis
    return crossing_events(traj.t, traj.x[:, 1], layers, cfg.Ly, h)


def cumulative_crossings(events: CrossingEvents, n_samples: int, *, layer: int | None = None,
                         direction: int | None = None) -> np.ndarray:
    """Number of crossings with ``index <= i`` for each sample i, optionally for one layer or direction."""
    ev = events.select(layer, direction) if (layer is not None or direction is not None) else events
    if ev.count and ev.index.max() >= n_samples:
        raise ValueError("event index beyond n_samples")
    return np.cumsum(np.bincount(ev.index, minlength=n_samples)).astype(np.int64)


# ======================================================= trajectory diagnostics
TRAJECTORY_DIAGNOSTICS = {
    "t": "sample time [a/U0]",
    "gamma": "Lorentz factor sqrt(1 + u^2/c^2)",
    "v": "velocity u/gamma, shape (n, 3) [U0]",
    "E_kin_per_mass": "kinetic energy per unit particle mass (gamma-1) c^2 = u^2/(1+gamma) [U0^2]",
    "E_kin": "kinetic energy of one simulation particle m_cr (gamma-1) c^2 [rho0 U0^2 a^3]",
    "dgamma_dt_fd": "finite difference of the sampled gamma, d gamma/dt [U0/a] (np.gradient)",
    "u_par": "reduced momentum along the local B (signed) [U0]",
    "u_perp": "reduced momentum perpendicular to B [U0]",
    "r_g": "gyroradius u_perp/(q_mc |B|) with the local (NGP) field [a]",
    "Omega": "gyrofrequency q_mc |B|/gamma [U0/a]",
    "mu_invariant": "magnetic moment per mass u_perp^2/(2|B|) (relativistic adiabatic invariant)",
    "pitch_lab": "cos(pitch angle) between u and B in the simulation frame",
    "pitch_drift": "cos(pitch angle) in the local E x B drift frame (exact boost; NaN where |w| >= c or B = 0)",
    "cos_v_cE": "cos of the angle between v and cE",
    "power_per_mass": "q_mc v . cE per unit particle mass [U0^3/a] (aliased, NGP fields; see module docs)",
    "dgamma_dt": "q_mc v . cE / c^2 [U0/a] (aliased and strobed, NGP fields; see module docs)",
}


def trajectory_diagnostics(traj: "Trajectory", cfg: "RunConfig") -> dict[str, np.ndarray | None]:
    """Physical quantities along one trajectory, keyed as in ``TRAJECTORY_DIAGNOSTICS``.

    ``cfg`` supplies c, q_mc and m_cr.  Entries that need B, or B and cE, are None for
    7-column trajectory files, and divisions by |B| = 0 give NaN.  ``power_per_mass`` and
    ``dgamma_dt`` come from sampled NGP fields, so per-particle energy changes should be
    taken from ``gamma`` or ``dgamma_dt_fd``.
    """
    from ..io.trajectory import Trajectory

    if not isinstance(traj, Trajectory):
        raise TypeError(f"trajectory_diagnostics: pass one Trajectory; loop over lists (got {type(traj).__name__})")
    cfg.require_particles()
    c, q_mc = float(cfg.c), float(cfg.q_mc)
    t = np.asarray(traj.t, dtype=np.float64)
    u = np.asarray(traj.u, dtype=np.float64)
    gamma = rel.lorentz_factor(u, c)
    out: dict[str, np.ndarray | None] = dict.fromkeys(TRAJECTORY_DIAGNOSTICS)
    out["t"] = t
    out["gamma"] = gamma
    out["v"] = u / gamma[:, None]
    out["E_kin_per_mass"] = rel.kinetic_energy_per_mass(u, c)
    out["E_kin"] = cfg.m_cr * out["E_kin_per_mass"]
    out["dgamma_dt_fd"] = np.gradient(gamma, t) if t.size >= 2 else np.full(t.shape, np.nan)

    if traj.B is not None:
        B = np.asarray(traj.B, dtype=np.float64)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["u_par"], out["u_perp"] = rel.parallel_perp(u, B)
            out["r_g"] = rel.gyroradius(u, B, q_mc)
            out["Omega"] = rel.gyrofrequency(u, B, q_mc, c)
            out["mu_invariant"] = rel.magnetic_moment_per_mass(u, B)
            out["pitch_lab"] = rel.pitch_cosine(u, B)
    if traj.B is not None and traj.cE is not None:
        cE = np.asarray(traj.cE, dtype=np.float64)
        out["pitch_drift"] = rel.pitch_cosine(u, B, cE=cE, c=c, frame="drift")
        out["cos_v_cE"] = rel.cos_angle(out["v"], cE)
        out["power_per_mass"] = q_mc * rel.dot(out["v"], cE)
        out["dgamma_dt"] = out["power_per_mass"] / c**2
    return out


# ============================================================ grid -> particle
def _grid_geometry(field_: np.ndarray, x: np.ndarray, cfg: "RunConfig"):
    f = np.asarray(field_)
    nx = tuple(cfg.nx)
    if f.ndim == 2 and nx[2] == 1 and f.shape == nx[:2]:
        f = f[:, :, None]
    if f.shape != nx:
        raise ValueError(f"field has shape {np.shape(field_)}, expected {nx} (or {nx[:2]} for 2D runs)")
    x = np.asarray(x, dtype=np.float64)
    if x.shape[-1] != 3:
        raise ValueError("positions must have shape (..., 3)")
    lead = x.shape[:-1]
    x = x.reshape(-1, 3)
    lo = np.array([b[0] for b in cfg.bounds])
    dx = np.array(cfg.dx)
    xi = (x - lo) / dx  # continuous cell coordinate: cell i spans [i, i+1)
    return f, xi, lead


def sample_field_ngp(field: np.ndarray, x, cfg: "RunConfig") -> np.ndarray:
    """Nearest-grid-point value of a cell-centred field at particle positions ``x`` (..., 3).

    ``field`` is indexed (x, y, z), or (x, y) for 2D runs, and positions wrap periodically.
    This is how the fields in trajectory files are sampled.
    """
    f, xi, lead = _grid_geometry(field, x, cfg)
    idx = [np.mod(np.floor(xi[:, a]).astype(np.int64), f.shape[a]) for a in range(3)]
    return f[idx[0], idx[1], idx[2]].reshape(lead)


def sample_field_tsc(field: np.ndarray, x, cfg: "RunConfig") -> np.ndarray:
    """Triangular-shaped-cloud interpolation of a cell-centred field to particle positions.

    The weights are those of ``ParticleMesh::GetWeightTSC``: with d the particle's offset from
    its cell centre in cell units, the cell gets 3/4 - d^2 and its lower and upper neighbours
    (1/2)(1/2 - d)^2 and (1/2)(1/2 + d)^2.  Axes with one cell use weight 1, boundaries are
    periodic, and the arguments are as for :func:`sample_field_ngp`.
    """
    f, xi, lead = _grid_geometry(field, x, cfg)
    n = xi.shape[0]
    idx, wts = [], []
    for a in range(3):
        if f.shape[a] == 1:
            idx.append(np.zeros((n, 1), dtype=np.int64))
            wts.append(np.ones((n, 1)))
            continue
        cell = np.floor(xi[:, a])
        d = xi[:, a] - cell - 0.5
        offsets = np.array([-1, 0, 1])
        idx.append(np.mod(cell.astype(np.int64)[:, None] + offsets, f.shape[a]))
        wts.append(np.stack([0.5 * (0.5 - d) ** 2, 0.75 - d * d, 0.5 * (0.5 + d) ** 2], axis=1))
    out = np.zeros(n, dtype=np.result_type(f.dtype, np.float64))
    for i in range(idx[0].shape[1]):
        for j in range(idx[1].shape[1]):
            wij = wts[0][:, i] * wts[1][:, j]
            for k in range(idx[2].shape[1]):
                out += wij * wts[2][:, k] * f[idx[0][:, i], idx[1][:, j], idx[2][:, k]]
    return out.reshape(lead)
