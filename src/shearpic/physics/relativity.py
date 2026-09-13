"""Relativistic particle kinematics in MHD-PIC code units.

The particles obey du/dt = q_mc (cE + v x B) with u = p/m = gamma v the reduced momentum,
gamma = sqrt(1 + u^2/c^2), c the numerical speed of light, q_mc the charge-to-mass ratio
over c and cE = -U x B the ideal electric field (Sun & Bai 2023).  Vectors carry a trailing
axis of length 3 and broadcast; functions ending in ``_mag`` take magnitudes.
"""

from __future__ import annotations

import warnings

import numpy as np

__all__ = [
    "dot", "norm", "cos_angle",
    "lorentz_factor", "lorentz_factor_mag", "velocity", "momentum_from_velocity",
    "kinetic_energy_per_mass", "kinetic_energy_per_mass_mag", "momentum_mag_from_gamma",
    "parallel_perp", "gyrofrequency", "gyroradius", "magnetic_moment_per_mass",
    "drift_velocity", "boost", "pitch_cosine", "power_per_mass", "dgamma_dt",
]


# ----------------------------------------------------------------- vector helpers
def dot(a, b) -> np.ndarray:
    return np.einsum("...i,...i->...", np.asarray(a, np.float64), np.asarray(b, np.float64))


def norm(a) -> np.ndarray:
    return np.sqrt(dot(a, a))


def _safe_div(num, den) -> np.ndarray:
    num, den = np.broadcast_arrays(np.asarray(num, np.float64), np.asarray(den, np.float64))
    out = np.full(num.shape, np.nan)
    np.divide(num, den, out=out, where=den != 0)
    return out if out.ndim else float(out)


def cos_angle(a, b) -> np.ndarray:
    """Cosine of the angle between two vectors; NaN where either vanishes."""
    return np.clip(_safe_div(dot(a, b), norm(a) * norm(b)), -1.0, 1.0)


# --------------------------------------------------------------------- kinematics
def lorentz_factor_mag(u_mag, c: float) -> np.ndarray:
    u_mag = np.asarray(u_mag, np.float64)
    return np.sqrt(1.0 + (u_mag / c) ** 2)


def lorentz_factor(u, c: float) -> np.ndarray:
    """gamma = sqrt(1 + |u|^2 / c^2) for reduced-momentum vectors u."""
    return np.sqrt(1.0 + dot(u, u) / c**2)


def velocity(u, c: float) -> np.ndarray:
    """Velocity v = u / gamma."""
    u = np.asarray(u, np.float64)
    return u / lorentz_factor(u, c)[..., None]


def momentum_from_velocity(v, c: float) -> np.ndarray:
    """u = gamma v with gamma = 1/sqrt(1 - v^2/c^2)."""
    v = np.asarray(v, np.float64)
    beta2 = dot(v, v) / c**2
    if np.any(beta2 >= 1.0):
        raise ValueError("|v| >= c")
    return v / np.sqrt(1.0 - beta2)[..., None]


def kinetic_energy_per_mass_mag(u_mag, c: float) -> np.ndarray:
    """(gamma - 1) c^2 from |u|, evaluated as u^2 / (1 + gamma) to avoid cancellation at u << c."""
    u2 = np.asarray(u_mag, np.float64) ** 2
    return u2 / (1.0 + np.sqrt(1.0 + u2 / c**2))


def kinetic_energy_per_mass(u, c: float) -> np.ndarray:
    """Kinetic energy per unit mass, (gamma - 1) c^2; multiply by ``cfg.m_cr`` for energy."""
    u2 = dot(u, u)
    return u2 / (1.0 + np.sqrt(1.0 + u2 / c**2))


def momentum_mag_from_gamma(gamma, c: float) -> np.ndarray:
    """|u| = c sqrt(gamma^2 - 1)."""
    g = np.asarray(gamma, np.float64)
    return c * np.sqrt(np.maximum(g * g - 1.0, 0.0))


# ------------------------------------------------------------------- gyromotion
def parallel_perp(vec, B) -> tuple[np.ndarray, np.ndarray]:
    """Components of ``vec`` along B (signed scalar) and perpendicular to B (magnitude)."""
    vec = np.asarray(vec, np.float64)
    Bmag = norm(B)
    par = _safe_div(dot(vec, B), Bmag)
    perp2 = np.maximum(dot(vec, vec) - np.square(par), 0.0)
    return par, np.sqrt(perp2)


def gyrofrequency(u, B, q_mc: float, c: float) -> np.ndarray:
    """Omega = q_mc |B| / gamma."""
    return q_mc * norm(B) / lorentz_factor(u, c)


def gyroradius(u, B, q_mc: float) -> np.ndarray:
    """Gyroradius u_perp / (q_mc |B|) in the local field, as in the history column ``r_g``."""
    _, u_perp = parallel_perp(u, B)
    return _safe_div(u_perp, q_mc * norm(B))


def magnetic_moment_per_mass(u, B) -> np.ndarray:
    """Relativistic magnetic moment per unit mass, u_perp^2 / (2 |B|).

    It is an adiabatic invariant only while r_g is small compared with the field gradient scale.
    """
    _, u_perp = parallel_perp(u, B)
    return _safe_div(u_perp**2, 2.0 * norm(B))


# -------------------------------------------------------------- frames & energy
def drift_velocity(cE, B) -> np.ndarray:
    """E x B drift velocity w = cE x B / |B|^2, which equals U_perp for cE = -U x B."""
    cE = np.asarray(cE, np.float64)
    B = np.asarray(B, np.float64)
    B2 = dot(B, B)[..., None]
    out = np.full(np.broadcast_shapes(cE.shape, B.shape), np.nan)
    np.divide(np.cross(cE, B), B2, out=out, where=B2 != 0)
    return out


def boost(u, w, c: float) -> np.ndarray:
    """Reduced momentum u' in a frame moving with velocity w, by an exact Lorentz boost.

    u'_par = Gamma (u_par - gamma |w|) and u'_perp = u_perp, with Gamma = 1/sqrt(1 - w^2/c^2).
    Elements with |w| >= c or non-finite w (the E x B drift where B is weak or zero) are NaN,
    with one RuntimeWarning per call; a single finite w with |w| >= c raises ValueError.
    """
    u = np.asarray(u, np.float64)
    w = np.asarray(w, np.float64)
    wmag = norm(w)
    with np.errstate(invalid="ignore"):
        bad = ~(np.isfinite(wmag) & (wmag < c))
    if np.ndim(wmag) == 0 and np.isfinite(wmag) and wmag >= c:
        raise ValueError("boost velocity must satisfy |w| < c")
    if np.any(bad):
        n_bad = int(np.count_nonzero(bad))
        warnings.warn(f"boost: {n_bad} of {bad.size} frame velocities have |w| >= c or are not finite; "
                      "returning NaN for them", RuntimeWarning, stacklevel=2)
        w = np.where(bad[..., None], 0.0, w)
        wmag = np.where(bad, 0.0, wmag)
    n = np.zeros(np.broadcast_shapes(w.shape, u.shape))
    np.divide(w, wmag[..., None], out=n, where=wmag[..., None] > 0)
    Gamma = 1.0 / np.sqrt(1.0 - (wmag / c) ** 2)
    gamma = lorentz_factor(u, c)
    u_par = dot(u, n)
    out = u + ((Gamma - 1.0) * u_par - Gamma * gamma * wmag)[..., None] * n
    if np.any(bad):
        out = np.where(bad[..., None], np.nan, out)
    return out


def pitch_cosine(u, B, cE=None, c: float | None = None, frame: str = "lab") -> np.ndarray:
    """Cosine of the pitch angle between the particle momentum and B.

    ``frame='lab'`` uses the simulation frame.  ``frame='drift'`` boosts u into the local
    E x B frame, where the ideal electric field vanishes, and needs ``cE`` and ``c``; the
    result is NaN where the drift speed |cE x B|/B^2 reaches c or B = 0.
    """
    if frame == "lab":
        return cos_angle(u, B)
    if frame != "drift":
        raise ValueError("frame must be 'lab' or 'drift'")
    if cE is None or c is None:
        raise ValueError("frame='drift' needs cE and c")
    u_drift = boost(u, drift_velocity(cE, B), c)
    return cos_angle(u_drift, B)


def power_per_mass(u, cE, q_mc: float, c: float) -> np.ndarray:
    """Rate of work by the electric field per unit particle mass, q_mc v . cE."""
    return q_mc * dot(velocity(u, c), cE)


def dgamma_dt(u, cE, q_mc: float, c: float) -> np.ndarray:
    """dgamma/dt = q_mc v . cE / c^2."""
    return power_per_mass(u, cE, q_mc, c) / c**2
