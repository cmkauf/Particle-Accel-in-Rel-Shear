"""Normalised time series, energy budgets and smoothing for the Athena++ history file.

The history columns aggregate differently: ``i-KE`` and ``i-ME`` are volume integrals, ``KE_cr``
is sum_p u^2/(1 + gamma) per unit particle mass, ``Gamma``, ``r_g`` and ``w_c`` are sums over
particles, ``Pideal`` is the total power into the particles and ``Pstir`` lacks the cell volume.
:func:`history_diagnostics` converts them into the quantities of :data:`DEFINITIONS`.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
import pandas as pd

from ..io._util import keep_last_written

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = ["DEFINITIONS", "history_diagnostics", "energy_budget", "smooth_series", "smoothing_lam",
           "smoothing_cutoff_period"]

DEFINITIONS: dict[str, str] = {
    "time": "simulation time [a/U0]",
    "eps_kin": "gas kinetic energy density (1-KE + 2-KE + 3-KE)/V [rho0 U0^2]",
    "eps_mag": "magnetic energy density (1-ME + 2-ME + 3-ME)/V, B^2/2 Athena units [rho0 U0^2]",
    "eps_p": "CR kinetic energy density m_cr KE_cr / V [rho0 U0^2]",
    "mean_gamma": "mean particle Lorentz factor Gamma/np",
    "mean_r_g": "mean particle gyroradius r_g/np with local |B| [a]",
    "mean_omega_c": "mean particle gyrofrequency w_c/np = <q_mc |B|/gamma> [U0/a]",
    "P_ideal": "total power into particles by the motional field, Pideal = sum_p m_cr q_mc v.cE [rho0 U0^3 a^2]",
    "P_ideal_x": "x part of P_ideal, sum_p m_cr q_mc v_x cE_x [rho0 U0^3 a^2]",
    "P_ideal_y": "y part of P_ideal, sum_p m_cr q_mc v_y cE_y [rho0 U0^3 a^2]",
    "P_ideal_z": "z part of P_ideal, sum_p m_cr q_mc v_z cE_z [rho0 U0^3 a^2]",
    "P_stir": "total power of the stirring force, Pstir * dV [rho0 U0^3 a^2]",
    "dEp_dt": "time derivative of the CR kinetic energy d(m_cr KE_cr)/dt (np.gradient) [rho0 U0^3 a^2]",
}

_trapezoid = getattr(np, "trapezoid", None) or np.trapz  # numpy < 2.0 only has trapz

HistoryLike = Union[pd.DataFrame, str, Path]


def _as_history(hst: HistoryLike) -> pd.DataFrame:
    """A history DataFrame; paths (file or run directory) are read with :func:`read_hst`."""
    if isinstance(hst, pd.DataFrame):
        return hst
    if isinstance(hst, (str, Path)):
        from ..io.history import read_hst

        return read_hst(hst)
    raise TypeError(f"hst must be a DataFrame or a path to a .hst file / run directory, got {type(hst).__name__}")


def _clean_time_axis(hst: pd.DataFrame) -> pd.DataFrame:
    """Rows sorted in time with restart duplicates removed (newest rows win)."""
    if "time" not in hst.columns:
        raise ValueError("history DataFrame has no 'time' column")
    t = hst["time"].to_numpy(np.float64)
    if t.size > 1 and np.any(np.diff(t) <= 0):
        keep = keep_last_written(t)
        warnings.warn(f"history time axis not strictly increasing: dropped {int((~keep).sum())} rows "
                      "re-written after a restart (use shearpic.io.history.read_hst to do this on read)",
                      stacklevel=3)
        hst = hst.loc[keep]
    return hst.reset_index(drop=True)


def _sum_columns(hst: pd.DataFrame, names) -> np.ndarray | None:
    present = [n for n in names if n in hst.columns]
    if not present:
        return None
    return hst[present].to_numpy(np.float64).sum(axis=1)


def _per_particle(hst: pd.DataFrame, col: str) -> np.ndarray | None:
    if col not in hst.columns or "np" not in hst.columns:
        return None
    num = hst[col].to_numpy(np.float64)
    n = hst["np"].to_numpy(np.float64)
    out = np.full(num.shape, np.nan)
    np.divide(num, n, out=out, where=n != 0)
    return out


def history_diagnostics(hst: HistoryLike, cfg: "RunConfig") -> pd.DataFrame:
    """Table of the :data:`DEFINITIONS` columns whose source columns exist in the history.

    ``hst`` is a DataFrame from :func:`shearpic.io.history.read_hst` or a path to the ``.hst``
    file or its run directory; rows re-written after a restart are dropped with a warning.
    ``cfg`` supplies V, dV and m_cr.
    """
    hst = _clean_time_axis(_as_history(hst))
    t = hst["time"].to_numpy(np.float64)
    out: dict[str, np.ndarray] = {"time": t}
    V, dV = float(cfg.V), float(cfg.dV)

    ke = _sum_columns(hst, ("1-KE", "2-KE", "3-KE"))
    if ke is not None:
        out["eps_kin"] = ke / V
    me = _sum_columns(hst, ("1-ME", "2-ME", "3-ME"))
    if me is not None:
        out["eps_mag"] = me / V
    Ep = None
    if "KE_cr" in hst.columns:
        Ep = float(cfg.m_cr) * hst["KE_cr"].to_numpy(np.float64)
        out["eps_p"] = Ep / V
    for name, col in (("mean_gamma", "Gamma"), ("mean_r_g", "r_g"), ("mean_omega_c", "w_c")):
        val = _per_particle(hst, col)
        if val is not None:
            out[name] = val
    for name, col in (("P_ideal", "Pideal"), ("P_ideal_x", "Pideal_x"), ("P_ideal_y", "Pideal_y"),
                      ("P_ideal_z", "Pideal_z")):
        if col in hst.columns:
            out[name] = hst[col].to_numpy(np.float64)
    if "Pstir" in hst.columns:
        out["P_stir"] = hst["Pstir"].to_numpy(np.float64) * dV
    if Ep is not None:
        out["dEp_dt"] = np.gradient(Ep, t) if t.size >= 2 else np.full(t.shape, np.nan)
    df = pd.DataFrame(out)
    df.attrs["definitions"] = {k: DEFINITIONS[k] for k in df.columns}
    return df


def energy_budget(hst: HistoryLike, cfg: "RunConfig", t_start: float, t_end: float) -> dict[str, float]:
    """Energy changes and work integrals over the history samples with t_start <= time <= t_end.

    Integrals use the trapezoid rule.  The dict holds the samples used (``t_start``, ``t_end``,
    ``n_samples``), the CR kinetic energy change ``Delta_E_p``, the work done on the particles
    by cE (``int_P_ideal_dt``) and by the stirring force (``int_P_stir_dt``), the gas energy
    changes ``Delta_E_kin``, ``Delta_E_mag`` and ``Delta_E_gas``, and ``ratio_Ep_over_P_ideal``.
    Entries whose history columns are missing are NaN.
    """
    if not t_end > t_start:
        raise ValueError("need t_end > t_start")
    hst = _clean_time_axis(_as_history(hst))
    t_all = hst["time"].to_numpy(np.float64)
    sel = (t_all >= t_start) & (t_all <= t_end)
    if sel.sum() < 2:
        raise ValueError(f"fewer than 2 history samples in [{t_start}, {t_end}] "
                         f"(file covers {t_all.min() if t_all.size else '-'} .. {t_all.max() if t_all.size else '-'})")
    h = hst.loc[sel]
    t = h["time"].to_numpy(np.float64)
    nan = float("nan")

    def delta(values: np.ndarray | None) -> float:
        return nan if values is None else float(values[-1] - values[0])

    Ep = float(cfg.m_cr) * h["KE_cr"].to_numpy(np.float64) if "KE_cr" in h.columns else None
    ke = _sum_columns(h, ("1-KE", "2-KE", "3-KE"))
    me = _sum_columns(h, ("1-ME", "2-ME", "3-ME"))
    P_ideal = float(_trapezoid(h["Pideal"].to_numpy(np.float64), t)) if "Pideal" in h.columns else nan
    P_stir = float(_trapezoid(h["Pstir"].to_numpy(np.float64) * cfg.dV, t)) if "Pstir" in h.columns else nan
    dEp = delta(Ep)
    return {
        "t_start": float(t[0]),
        "t_end": float(t[-1]),
        "n_samples": int(t.size),
        "Delta_E_p": dEp,
        "int_P_ideal_dt": P_ideal,
        "int_P_stir_dt": P_stir,
        "Delta_E_kin": delta(ke),
        "Delta_E_mag": delta(me),
        "Delta_E_gas": delta(ke) + delta(me),
        "ratio_Ep_over_P_ideal": dEp / P_ideal if np.isfinite(P_ideal) and P_ideal != 0 else nan,
    }


# ============================================================================ smoothing
def _median_spacing(t: np.ndarray) -> float:
    return float(np.median(np.diff(t)))


def smoothing_lam(cutoff_period: float, dt: float) -> float:
    """Smoothing-spline penalty whose filter has half amplitude at ``cutoff_period``.

    On a uniform grid of spacing dt the spline is the low-pass filter
    H(omega) = 1 / (1 + lam dt omega^4), so H(2 pi / P_c) = 1/2 gives lam = (P_c / 2 pi)^4 / dt.
    """
    if not (cutoff_period > 0 and dt > 0):
        raise ValueError("cutoff_period and dt must be positive")
    return (cutoff_period / (2.0 * np.pi)) ** 4 / dt


def smoothing_cutoff_period(lam: float, dt: float) -> float:
    """Half-amplitude period P_c = 2 pi (lam dt)^(1/4) of a smoothing spline with penalty ``lam``."""
    if not (lam > 0 and dt > 0):
        raise ValueError("lam and dt must be positive")
    return float(2.0 * np.pi * (lam * dt) ** 0.25)


def smooth_series(t, y, *, cutoff_period: float | None = None, lam: float | None = None,
                  derivative: int = 0) -> np.ndarray:
    """Smoothing-spline estimate of a time series, or of its ``derivative`` (0, 1 or 2), at ``t``.

    Uses :func:`scipy.interpolate.make_smoothing_spline`.  Give either ``cutoff_period``, the
    period at which sinusoids keep half their amplitude (converted with the median sample
    spacing, so the smoothing does not depend on the output cadence), or the raw penalty
    ``lam``.  ``t`` must be strictly increasing with at least 5 samples.
    """
    from scipy.interpolate import make_smoothing_spline

    t = np.asarray(t, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if t.ndim != 1 or y.shape != t.shape:
        raise ValueError(f"t and y must be 1D arrays of the same length, got {t.shape} and {y.shape}")
    if t.size < 5:
        raise ValueError("need at least 5 samples to fit a smoothing spline")
    if not (np.all(np.isfinite(t)) and np.all(np.isfinite(y))):
        raise ValueError("t and y must be finite")
    if np.any(np.diff(t) <= 0):
        raise ValueError("t must be strictly increasing (drop restart duplicates with read_hst)")
    if (cutoff_period is None) == (lam is None):
        raise ValueError("give exactly one of cutoff_period and lam")
    if derivative not in (0, 1, 2):
        raise ValueError("derivative must be 0, 1 or 2")
    if cutoff_period is not None:
        lam = smoothing_lam(float(cutoff_period), _median_spacing(t))
    elif not lam > 0:
        raise ValueError("lam must be positive")
    spline = make_smoothing_spline(t, y, lam=float(lam))
    return np.asarray(spline(t, nu=derivative) if derivative else spline(t), dtype=np.float64)
