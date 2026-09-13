"""Reader for Athena++ history files (``*.hst``) and a table of column definitions.

``HST_SCHEMA`` gives the meaning of each column and how it is aggregated: sums over
particles, volume integrals, volume means, or, for ``Pstir``, a plain sum over cells
that must be multiplied by ``dV``.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from ._util import keep_last_written

__all__ = ["read_hst", "HST_SCHEMA", "HstColumn", "ALIASES", "find_hst"]

_HEADER_RE = re.compile(r"\[(\d+)\]=(.*?)(?=\[\d+\]=|$)")


@dataclass(frozen=True)
class HstColumn:
    """Meaning of one history column.

    ``aggregation`` is ``'volume integral'``, ``'volume mean'``, ``'particle sum'``,
    ``'cell sum (no dV)'`` or ``'scalar'``; ``units`` are code units unless stated.
    """

    definition: str
    aggregation: str
    units: str = "code"
    caveat: str = ""


# Definitions follow particles.cpp and pgen/kh_driven*.cpp of the Athena++ fork.
HST_SCHEMA: dict[str, HstColumn] = {
    "time": HstColumn("simulation time", "scalar", "a/U0"),
    "dt": HstColumn("time step", "scalar", "a/U0"),
    "mass": HstColumn("sum rho dV", "volume integral"),
    "1-mom": HstColumn("sum rho v1 dV", "volume integral"),
    "2-mom": HstColumn("sum rho v2 dV", "volume integral"),
    "3-mom": HstColumn("sum rho v3 dV", "volume integral"),
    "1-KE": HstColumn("sum 0.5 rho v1^2 dV", "volume integral"),
    "2-KE": HstColumn("sum 0.5 rho v2^2 dV", "volume integral"),
    "3-KE": HstColumn("sum 0.5 rho v3^2 dV", "volume integral"),
    "1-ME": HstColumn("sum 0.5 B1^2 dV (Athena units, no 4pi)", "volume integral"),
    "2-ME": HstColumn("sum 0.5 B2^2 dV", "volume integral"),
    "3-ME": HstColumn("sum 0.5 B3^2 dV", "volume integral"),
    "np": HstColumn("number of particles N", "particle sum"),
    "vp1": HstColumn("sum_p u_x, with u = p/m = gamma v the reduced momentum", "particle sum"),
    "vp2": HstColumn("sum_p u_y", "particle sum"),
    "vp3": HstColumn("sum_p u_z", "particle sum"),
    "vp1^2": HstColumn("sum_p u_x^2", "particle sum"),
    "vp2^2": HstColumn("sum_p u_y^2", "particle sum"),
    "vp3^2": HstColumn("sum_p u_z^2", "particle sum"),
    "-BxBy": HstColumn("cell mean of s(y) dBx dBy, s=-1 for y<=0 and +1 for y>0, dB = B - <B>_x "
                       "(Maxwell stress folded by half-plane)", "volume mean",
                       caveat="the s(y) fold only makes sense for two-layer setups"),
    "dVxVy": HstColumn("cell mean of -s(y) rho dvx dvy, dv = v - <v>_x (Reynolds stress, folded)", "volume mean",
                       caveat="the s(y) fold only makes sense for two-layer setups"),
    "Pstir": HstColumn("sum over cells of rho v_x a_stir; multiply by dV for the power", "cell sum (no dV)",
                       caveat="the definition changed between problem-generator versions; compare Pstir * dV "
                              "with physics.forcing.stir_power for the run before using it"),
    "KE_cr": HstColumn("sum_p (gamma-1) c^2 = sum_p u^2/(1+gamma); multiply by m_cr for the energy", "particle sum",
                       "U0^2 per unit particle mass (sum of (gamma-1) c^2)"),
    "Gamma": HstColumn("sum_p gamma", "particle sum"),
    "Pideal": HstColumn("sum_p m_cr q_mc v.cE  (total power into particles; already includes m_cr)", "particle sum",
                        caveat="fluid fields are taken at the particle's cell (NGP) while the pusher uses TSC; "
                               "the particle sum is much less affected than single-particle values"),
    "Pideal_x": HstColumn("sum_p m_cr q_mc v_x cE_x", "particle sum"),
    "Pideal_y": HstColumn("sum_p m_cr q_mc v_y cE_y", "particle sum"),
    "Pideal_z": HstColumn("sum_p m_cr q_mc v_z cE_z", "particle sum"),
    "r_g": HstColumn("sum_p u_perp / (q_mc |B|)  (local B, NGP)", "particle sum"),
    "w_c": HstColumn("sum_p q_mc |B| / gamma", "particle sum"),
    "U_par": HstColumn("particle-weighted drift diagnostic", "particle sum",
                       caveat="definition changed between pgen versions (E-weighted signed / unit-vector signed / "
                              "unit-vector absolute); check the pgen of the run before using"),
    "U_apar": HstColumn("particle-weighted drift diagnostic", "particle sum",
                        caveat="see U_par"),
}

# identifier-friendly names; look columns up with df[ALIASES.get(name, name)]
ALIASES = {
    "neg_BxBy": "-BxBy",
    "vp1_sq": "vp1^2",
    "vp2_sq": "vp2^2",
    "vp3_sq": "vp3^2",
    "KE1": "1-KE",
    "KE2": "2-KE",
    "KE3": "3-KE",
    "ME1": "1-ME",
    "ME2": "2-ME",
    "ME3": "3-ME",
}


def _parse_header(line: str, path: Path, lineno: int) -> list[str]:
    found = [(int(i), n.strip()) for i, n in _HEADER_RE.findall(line.lstrip("#"))]
    idx = [i for i, _ in found]
    if idx != list(range(1, len(idx) + 1)):
        raise ValueError(f"{path}:{lineno}: malformed history header (column indices {idx[:5]}...)")
    return [n for _, n in found]


def _split_hst(path: Path) -> tuple[list[str], list[str]]:
    """Column names and data lines of a history file.

    A repeated header from a restart is ignored if identical and raises ValueError if
    the columns differ. An unterminated last line is dropped with a warning.
    """
    with open(path) as fh:
        lines = fh.readlines()
    if lines and not lines[-1].endswith("\n"):
        warnings.warn(f"{path.name}: dropped the incomplete last line (file still being written or truncated)",
                      stacklevel=3)
        lines = lines[:-1]
    names: list[str] | None = None
    data: list[str] = []
    for lineno, line in enumerate(lines, start=1):
        if line.startswith("#"):
            if "[1]=" not in line:
                continue
            found = _parse_header(line, path, lineno)
            if names is None:
                names = found
            elif found != names:
                k = next((i for i, (a, b) in enumerate(zip(names, found)) if a != b), min(len(names), len(found)))
                first = (f"column {k + 1} is {names[k] if k < len(names) else '(none)'!r} before and "
                         f"{found[k] if k < len(found) else '(none)'!r} after")
                raise ValueError(
                    f"{path}:{lineno}: the history columns change here (restart with a different problem "
                    f"generator or output settings?): {len(names)} columns before, {len(found)} after, {first}; "
                    "split the file at this header and read the parts separately"
                )
            continue
        if not line.strip():
            continue
        if names is None:
            break  # data before any header: reported below
        data.append(line)
    if names is None:
        raise ValueError(f"{path}: no '[1]=' column header found before the data")
    return names, data


def read_hst(path: str | Path, drop_duplicates: bool = True) -> pd.DataFrame:
    """Read an Athena++ history file into a DataFrame with the original column names.

    ``path`` is the file or a run directory with a single ``*.hst``. Rows repeated by a
    restart are removed, keeping the most recent ones, and their number is stored in
    ``df.attrs['n_dropped']``. Raises ValueError for a malformed header, a later header with
    different columns, or rows whose length differs from the header.
    """
    p = find_hst(path) if Path(path).is_dir() else Path(path)
    names, lines = _split_hst(p)
    if lines:
        try:
            data = np.loadtxt(lines, ndmin=2)
        except ValueError as err:
            raise ValueError(f"{p}: could not parse the data rows ({err})") from None
    else:
        data = np.empty((0, len(names)))
    if data.shape[1] != len(names):
        raise ValueError(f"{p}: header has {len(names)} columns but data rows have {data.shape[1]}")
    df = pd.DataFrame(data, columns=names)
    n_dropped = 0
    if drop_duplicates and len(df) > 1:
        keep = keep_last_written(df["time"].to_numpy())
        n_dropped = int((~keep).sum())
        if n_dropped:
            warnings.warn(f"{p.name}: dropped {n_dropped} rows duplicated by a restart")
            df = df.loc[keep].reset_index(drop=True)
    df.attrs["path"] = str(p)
    df.attrs["n_dropped"] = n_dropped
    return df


def find_hst(run_dir: str | Path) -> Path:
    """The single ``*.hst`` file in ``run_dir``; raise if there is none or several."""
    files = sorted(Path(run_dir).glob("*.hst"))
    if not files:
        raise FileNotFoundError(f"no .hst file in {run_dir}")
    if len(files) > 1:
        raise ValueError(f"several .hst files in {run_dir}: {[f.name for f in files]}; pass the file path")
    return files[0]
