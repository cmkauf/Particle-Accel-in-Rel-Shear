"""Derived grid quantities from snapshot arrays indexed (x, y, z) or (x, y).

Derivatives are second-order central differences on the periodic box.  Grid spacing is given
as ``(dx, dy, dz)`` or as an object with ``.dx`` such as a RunConfig, and vector fields as
tuples ``(Fx, Fy[, Fz])`` with a missing Fz taken as zero.

For the cosmic rays the grid stores the number density ``np`` and the cell-mean reduced
momentum <u> (``vp1..3``).  The CR current, work rate and energy density need <v> or <gamma>
and have to come from the history file or the particle outputs instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Sequence, Union

import numpy as np

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = [
    "magnitude", "ddx", "gradient", "divergence", "curl", "vorticity", "vorticity_z", "current", "current_z",
    "motional_cE", "x_average", "fluctuation", "kinetic_energy_density", "magnetic_energy_density", "integrate",
    "reynolds_stress_xy", "maxwell_stress_xy", "layer_mask",
    "cr_mass_density", "cr_momentum_density",
]

Vector = Sequence[np.ndarray]
Spacing = Union[Sequence[float], "RunConfig", Any]


def _f64(a) -> np.ndarray:
    return np.asarray(a, dtype=np.float64)


def _spacing(dx_or_cfg: Spacing) -> tuple[float, float, float]:
    """``(dx, dy, dz)`` from a sequence or an object with ``.dx`` (e.g. RunConfig)."""
    dx = getattr(dx_or_cfg, "dx", dx_or_cfg)
    try:
        values = tuple(float(d) for d in dx)
    except TypeError:
        raise TypeError(f"grid spacing must be a sequence (dx, dy, dz) or have a .dx attribute, got {dx_or_cfg!r}") \
            from None
    if len(values) == 2:  # dz is never used when the z axis has length 1
        values = (*values, 1.0)
    if len(values) != 3:
        raise ValueError(f"grid spacing needs 2 or 3 entries, got {len(values)}")
    return values


def _components(F: Vector, name: str = "vector field") -> tuple:
    """``(Fx, Fy, Fz)`` with ``Fz = None`` for two-component input (treated as zero)."""
    comps = tuple(F)
    if len(comps) == 2:
        return (*comps, None)
    if len(comps) != 3:
        raise ValueError(f"{name} needs 2 or 3 components, got {len(comps)}")
    return comps


def magnitude(*components) -> np.ndarray:
    """sqrt(sum of squares), e.g. ``magnitude(Bx, By, Bz)``."""
    return np.sqrt(sum(_f64(c) ** 2 for c in components))


def ddx(f: np.ndarray, spacing: float | Spacing, axis: int) -> np.ndarray:
    """Periodic central difference df/dx along ``axis`` (0, 1, 2 = x, y, z); zero along length-1 axes.

    ``spacing`` is the cell size along ``axis`` or the spacings of all axes (``cfg.dx`` or ``cfg``).
    """
    f = _f64(f)
    if axis >= f.ndim or f.shape[axis] == 1:
        return np.zeros_like(f)
    if hasattr(spacing, "dx") or np.ndim(spacing) > 0:
        h = _spacing(spacing)[axis]
    else:
        h = float(spacing)
    return (np.roll(f, -1, axis=axis) - np.roll(f, 1, axis=axis)) / (2.0 * h)


def _ddx_or_zero(f, dx, axis):
    return 0.0 if f is None else ddx(f, dx[axis], axis)


def gradient(f: np.ndarray, dx_or_cfg: Spacing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(df/dx, df/dy, df/dz)``; ``dx_or_cfg`` is ``(dx, dy, dz)`` or a RunConfig."""
    dx = _spacing(dx_or_cfg)
    return tuple(ddx(f, dx[i], i) for i in range(3))


def divergence(F: Vector, dx_or_cfg: Spacing) -> np.ndarray:
    """div F for ``F = (Fx, Fy[, Fz])`` (a missing Fz counts as zero)."""
    dx = _spacing(dx_or_cfg)
    comps = _components(F)
    return sum(ddx(comps[i], dx[i], i) for i in range(3) if comps[i] is not None)


def curl(F: Vector, dx_or_cfg: Spacing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All three components of curl F; derivatives along length-1 axes vanish."""
    dx = _spacing(dx_or_cfg)
    Fx, Fy, Fz = _components(F)
    return (
        _ddx_or_zero(Fz, dx, 1) - ddx(Fy, dx[2], 2),
        ddx(Fx, dx[2], 2) - _ddx_or_zero(Fz, dx, 0),
        ddx(Fy, dx[0], 0) - ddx(Fx, dx[1], 1),
    )


def _curl_z(Fx, Fy, dx_or_cfg: Spacing) -> np.ndarray:
    dx = _spacing(dx_or_cfg)
    return ddx(Fy, dx[0], 0) - ddx(Fx, dx[1], 1)


def vorticity(v: Vector, dx_or_cfg: Spacing):
    """Vorticity curl v; for 2D runs :func:`vorticity_z` computes the only nonzero component."""
    return curl(v, dx_or_cfg)


def vorticity_z(vx: np.ndarray, vy: np.ndarray, dx_or_cfg: Spacing) -> np.ndarray:
    """Vorticity z component dvy/dx - dvx/dy [U0/a]."""
    return _curl_z(vx, vy, dx_or_cfg)


def current(B: Vector, dx_or_cfg: Spacing):
    """Current density J = curl B, with 4 pi / c absorbed in Athena units."""
    return curl(B, dx_or_cfg)


def current_z(Bx: np.ndarray, By: np.ndarray, dx_or_cfg: Spacing) -> np.ndarray:
    """Current density z component dBy/dx - dBx/dy."""
    return _curl_z(Bx, By, dx_or_cfg)


def motional_cE(v: Vector, B: Vector) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ideal-MHD motional electric field cE = -v x B."""
    vx, vy, vz = (_f64(c) for c in v)
    Bx, By, Bz = (_f64(c) for c in B)
    return (vz * By - vy * Bz, vx * Bz - vz * Bx, vy * Bx - vx * By)


def x_average(f: np.ndarray) -> np.ndarray:
    """Mean profile along y, averaged over x and z, shape (ny,)."""
    f = _f64(f)
    return f.mean(axis=(0, 2)) if f.ndim == 3 else f.mean(axis=0)


def fluctuation(f: np.ndarray) -> np.ndarray:
    """f - <f>_x, with the same shape as f."""
    f = _f64(f)
    mean = f.mean(axis=0, keepdims=True)
    if f.ndim == 3:
        mean = mean.mean(axis=2, keepdims=True)
    return f - mean


def kinetic_energy_density(rho, v: Vector) -> np.ndarray:
    return 0.5 * _f64(rho) * sum(_f64(c) ** 2 for c in v)


def magnetic_energy_density(B: Vector) -> np.ndarray:
    """B^2 / 2 (Athena units)."""
    return 0.5 * sum(_f64(c) ** 2 for c in B)


def integrate(density: np.ndarray, cfg: "RunConfig") -> float:
    """Volume integral sum(density) * dV."""
    return float(_f64(density).sum() * cfg.dV)


def reynolds_stress_xy(rho, vx, vy) -> np.ndarray:
    """Local Reynolds stress rho dvx dvy with fluctuations about the x-average."""
    return _f64(rho) * fluctuation(vx) * fluctuation(vy)


def maxwell_stress_xy(Bx, By) -> np.ndarray:
    """Local Maxwell stress -dBx dBy with fluctuations about the x-average (Athena units)."""
    return -fluctuation(Bx) * fluctuation(By)


def layer_mask(cfg: "RunConfig", cutoff: float = 0.9, y: np.ndarray | None = None) -> np.ndarray:
    """Boolean mask along y selecting cells inside the shear layers (|y - y_i| < width/2)."""
    y = cfg.centers(1) if y is None else np.asarray(y)
    half = 0.5 * cfg.profile.layer_width(cutoff)
    mask = np.zeros(y.shape, dtype=bool)
    for yl in cfg.profile.layer_positions:
        d = (y - yl + 0.5 * cfg.Ly) % cfg.Ly - 0.5 * cfg.Ly  # periodic distance
        mask |= np.abs(d) < half
    return mask


def cr_mass_density(n_cr: np.ndarray, cfg: "RunConfig") -> np.ndarray:
    """CR mass density m_cr * np from the snapshot number density."""
    return cfg.m_cr * _f64(n_cr)


def cr_momentum_density(n_cr: np.ndarray, u_mean: Vector, cfg: "RunConfig"):
    """CR momentum density m_cr * np * <u>, with <u> from ``vp1..3``."""
    rho = cr_mass_density(n_cr, cfg)
    return tuple(rho * _f64(c) for c in u_mean)
