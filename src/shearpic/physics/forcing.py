"""Background shear profile and the stirring force that maintains it.

This follows ShearProfile, ConstForce and DrivenForce in the ``kh_driven`` problem generator
of the Athena++ fork, and has to be kept in step with it.  For ``tau > 0`` and ``stir = 0``
the force per unit mass is::

    a_stir(y) = (U_ref(y) - <v_x>_x(y)) / tau + nu_iso (-U_ref''(y))

with <v_x>_x the unweighted x (and z) average.  The C++ evaluates U_ref and U_ref'' at the
lower cell faces y_min + j dy for cell row j.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .fields import x_average

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = ["ShearProfile", "x_mean_velocity", "stir_acceleration", "stir_power_density", "stir_power"]

PROFILE_KINDS = ("double_tanh", "single_tanh", "sin")


@dataclass(frozen=True)
class ShearProfile:
    """Reference shear velocity profile U_ref(y), the x component of the gas velocity [U0].

    ``'double_tanh'`` (``iprob = 0``) is -S [1 - tanh((y - y1)/a) + tanh((y - y2)/a)], a jet
    +S between the layers y1 < y2 inside a wind -S.  ``'single_tanh'`` is S tanh(y/a) and
    ``'sin'`` (``iprob = 1``) is S sin(2 pi n y / Ly).
    """

    kind: str
    amplitude: float = 1.0   # S [U0]
    a: float = 1.0           # layer half width [a]
    y1: float = -10.0
    y2: float = 10.0
    Ly: float = 1.0          # box height, used only by 'sin'
    n: int = 1               # mode number, used only by 'sin'

    def __post_init__(self):
        if self.kind not in PROFILE_KINDS:
            raise ValueError(f"unknown profile kind {self.kind!r}; choose from {PROFILE_KINDS}")

    # --------------------------------------------------------------- profile
    def U(self, y) -> np.ndarray:
        """Reference velocity U_ref(y) [U0] at positions y [a]."""
        y = np.asarray(y, dtype=np.float64)
        S, a = self.amplitude, self.a
        if self.kind == "double_tanh":
            return -S * (1.0 - np.tanh((y - self.y1) / a) + np.tanh((y - self.y2) / a))
        if self.kind == "single_tanh":
            return S * np.tanh(y / a)
        k = 2.0 * np.pi * self.n / self.Ly
        return S * np.sin(k * y)

    def d2U(self, y) -> np.ndarray:
        """Analytic second derivative U_ref''(y)."""
        y = np.asarray(y, dtype=np.float64)
        S, a = self.amplitude, self.a

        def tanh_dd(z):  # d^2/dy^2 tanh(z/a)
            t = np.tanh(z / a)
            return -2.0 / a**2 * t * (1.0 - t * t)

        if self.kind == "double_tanh":
            return -S * (-tanh_dd(y - self.y1) + tanh_dd(y - self.y2))
        if self.kind == "single_tanh":
            return S * tanh_dd(y)
        k = 2.0 * np.pi * self.n / self.Ly
        return -S * k * k * np.sin(k * y)

    def viscous_compensation(self, y, nu: float) -> np.ndarray:
        """Constant force per unit mass ``nu * (-U'')`` that balances viscous decay."""
        return nu * (-self.d2U(y))

    def on_cpp_grid(self, y_edges: np.ndarray) -> np.ndarray:
        """U_ref per cell row as the C++ stores it, evaluated at the lower faces ``y_edges[:-1]``."""
        return self.U(np.asarray(y_edges)[:-1])

    # ---------------------------------------------------------------- layers
    @property
    def layer_positions(self) -> tuple[float, ...]:
        if self.kind == "double_tanh":
            return (self.y1, self.y2)
        if self.kind == "single_tanh":
            return (0.0,)
        return ()

    def layer_width(self, cutoff: float = 0.9) -> float:
        """Full width of one layer where ``|U| < cutoff * S`` (tanh profiles): ``2 a artanh(cutoff)``."""
        if not 0.0 < cutoff < 1.0:
            raise ValueError("cutoff must be in (0, 1)")
        return 2.0 * self.a * float(np.arctanh(cutoff))


def x_mean_velocity(vx: np.ndarray) -> np.ndarray:
    """Unweighted x (and z) average of vx, shape (ny,), which is the mean the C++ stirring uses."""
    return x_average(vx)


def stir_acceleration(vx: np.ndarray, cfg: "RunConfig", *, cpp_faces: bool = True) -> np.ndarray:
    """Stirring acceleration a_stir(y), shape (ny,).

    ``cpp_faces=True`` samples U_ref at the lower cell faces like the C++, ``False`` at cell centres.
    """
    if cfg.tau is None or cfg.tau <= 0:
        return np.zeros(cfg.nx[1])
    if cfg.stir not in (0, None):
        raise NotImplementedError("only stir = 0 (relaxation to the reference profile) is implemented")
    prof = cfg.profile
    y_edges = cfg.edges(1)
    y = y_edges[:-1] if cpp_faces else 0.5 * (y_edges[:-1] + y_edges[1:])
    u_ref = prof.U(y)
    f0 = prof.viscous_compensation(y, cfg.nu_iso or 0.0)
    return (u_ref - x_mean_velocity(vx)) / cfg.tau + f0


def stir_power_density(rho: np.ndarray, vx: np.ndarray, cfg: "RunConfig", **kw) -> np.ndarray:
    """Local power density rho v_x a_stir of the stirring force."""
    a = stir_acceleration(vx, cfg, **kw)
    shape = [1] * np.ndim(rho)
    shape[1] = -1
    return np.asarray(rho, np.float64) * np.asarray(vx, np.float64) * a.reshape(shape)


def stir_power(rho: np.ndarray, vx: np.ndarray, cfg: "RunConfig", **kw) -> float:
    """Total stirring power sum(rho v_x a_stir) dV, the same quantity as ``hst['Pstir'] * cfg.dV``."""
    return float(stir_power_density(rho, vx, cfg, **kw).sum() * cfg.dV)
