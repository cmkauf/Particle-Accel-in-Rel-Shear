"""RunConfig: the parameters and derived scales of one simulation, read from its athinput.

A RunConfig is immutable and picklable, so it can be passed explicitly to any function or
worker process that needs physical constants.  Code units are those of the problem
generator: a = rho0 = U0 = 1, time in a/U0, and B in Athena units with magnetic energy
density B^2/2 and Alfven speed B/sqrt(rho)::

    cfg = RunConfig.from_run_dir(423)
    print(cfg.describe())
"""

from __future__ import annotations

import math
import re
import warnings
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from .io.athinput import AthInput, read_athinput
from .physics.forcing import ShearProfile

__all__ = ["RunConfig", "A", "RHO0", "U0", "LEGACY_LAYERS", "parse_run_id"]

A = 1.0     # shear-layer length unit a
RHO0 = 1.0  # reference gas density
U0 = 1.0    # reference shear velocity

LEGACY_LAYERS = (-5.0 * math.pi, 5.0 * math.pi)
"""Layer positions (y1, y2) hard-coded in the problem generator for runs without ``<problem> y1, y2``.

They apply to iprob = 0 runs and do not depend on the box height.
"""

_RUN_ID_RE = re.compile(r"^run0*(\d+)$")


def parse_run_id(path_or_name: str | Path) -> int | None:
    """Run number N of a directory named exactly ``run<N>`` with any zero padding, else None.

    Only the last path component is examined: ``'run0423'`` gives 423, ``'run7_old'`` gives None.
    """
    m = _RUN_ID_RE.match(Path(path_or_name).name)
    return int(m.group(1)) if m else None


def _is_run_reference(run: str) -> bool:
    """True for a bare run number or name (``'423'``, ``'run0423'``) that is not a path on disk."""
    if Path(run).expanduser().exists() or Path(run).name != run:
        return False
    return run.isdigit() or parse_run_id(run) is not None


def _f(x: Any) -> float | None:
    return None if x is None else float(x)


@dataclass(frozen=True)
class RunConfig:
    """Parameters and derived scales of one simulation, built with from_run_dir or from_athinput.

    Attributes
    ----------
    problem_id : ``<job> problem_id``, the base name of the output files.
    run_dir : run directory, or None.
    run_id : run number parsed from the name of run_dir, or None.
    nx : root-grid cells along (x, y, z).
    bounds : domain ((x1min, x1max), (x2min, x2max), (x3min, x3max)) [a].
    meshblock : cells per meshblock, or None without a ``<meshblock>`` block.
    cs : isothermal sound speed [U0].
    c : speed of light [U0]; None without a ``<particles>`` block.
    q_mc : q/(m c), so that q_mc |B| / gamma is the gyrofrequency [U0/a]; None without particles.
    backreaction : whether the particles push back on the gas.
    vp_par : injection reduced momentum u = p/m [U0].
    cr_mass : total CR mass over total gas mass.
    npx : particles per direction, n_par = prod(npx).
    iprob : profile selector of the problem generator, 0 for tanh layers and 1 for a sine.
    M_A : Alfven Mach number U0 / v_A of the shear flow, so B0 = 1/M_A.
    tau : relaxation time of the stirring force [a/U0]; None when stirring is off.
    stir : stirring mode, 0 relaxing towards the reference profile.
    nu_iso : isotropic kinematic viscosity [U0 a].
    eta_ohm : Ohmic resistivity [U0 a].
    shear_amplitude : shear velocity amplitude S [U0].
    shear_amplitude_source : 'argument', 'athinput', 'hst' (initial kinetic energy) or 'default' (S = 1).
    profile : reference shear profile U_ref(y) with amplitude S.
    athinput : the parsed input file, ignored in comparisons.
    """

    # --- identity
    problem_id: str
    run_dir: Path | None
    run_id: int | None
    # --- mesh
    nx: tuple[int, int, int]
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
    meshblock: tuple[int, int, int] | None
    # --- gas
    cs: float | None
    # --- particles
    c: float | None
    q_mc: float | None
    backreaction: bool | None
    vp_par: float
    cr_mass: float
    npx: tuple[int, int, int]
    # --- problem
    iprob: int
    M_A: float
    tau: float | None
    stir: int | None
    nu_iso: float
    eta_ohm: float
    shear_amplitude: float
    shear_amplitude_source: str
    profile: ShearProfile
    athinput: AthInput = field(repr=False, compare=False)

    # ================================================================ builders
    @classmethod
    def from_athinput(cls, athinput: str | Path | AthInput, *, run_dir: str | Path | None = None,
                      hst=None, shear_amplitude: float | None = None,
                      profile_kind: str | None = None,
                      layers: tuple[float, float] | None = None,
                      profile: ShearProfile | None = None) -> "RunConfig":
        """Build from an athinput file, a run directory containing one, or a parsed AthInput.

        Parameters
        ----------
        run_dir : run directory; defaults to the directory of the athinput file.
        hst : history DataFrame or path used to infer S; defaults to the ``*.hst`` in run_dir.
        shear_amplitude : S [U0]; overrides ``<problem> shear_strength``.
        profile_kind : 'double_tanh', 'single_tanh' or 'sin'; overrides the choice from iprob.
        layers : layer positions (y1, y2) [a]; override ``<problem> y1, y2``.
        profile : ShearProfile shape, amplitude replaced by S; not combined with profile_kind or layers.

        Without shear_amplitude or shear_strength, S is inferred from the initial gas kinetic
        energy, 1-KE(0) = S^2 * 0.5 sum U_shape(y_face)^2 dV; a given shear_strength is checked
        against it.  The default profile is 'sin' for iprob = 1 and otherwise a double tanh at
        ``<problem> y1, y2``, or at LEGACY_LAYERS with a warning when those keys are absent.
        Overrides are applied before S is inferred.
        """
        inp = athinput if isinstance(athinput, AthInput) else read_athinput(athinput)
        if run_dir is None and inp.path is not None:
            run_dir = inp.path.parent
        run_dir = Path(run_dir) if run_dir is not None else None

        mesh = inp["mesh"]
        nx = tuple(int(mesh.get(f"nx{d}", 1)) for d in (1, 2, 3))
        bounds = tuple((float(mesh[f"x{d}min"]), float(mesh[f"x{d}max"])) for d in (1, 2, 3))
        mb = inp.blocks.get("meshblock")
        meshblock = tuple(int(mb.get(f"nx{d}", 1)) for d in (1, 2, 3)) if mb else None

        prob = inp.blocks.get("problem", {})
        par = inp.blocks.get("particles", {})
        hydro = inp.blocks.get("hydro", {})
        # defaults of the problem generator: npx_i = nx_i, and 1 along a collapsed dimension
        npx = tuple(int(prob.get(f"npx{d}", nx[d - 1])) if nx[d - 1] > 1 else 1 for d in (1, 2, 3))
        iprob = int(prob.get("iprob", 0))
        Ly = bounds[1][1] - bounds[1][0]

        if profile is not None:
            if profile_kind is not None or layers is not None:
                raise ValueError("pass either profile or profile_kind/layers, not both")
            if not isinstance(profile, ShearProfile):
                raise TypeError(f"profile must be a ShearProfile, got {type(profile).__name__}")
            shape = replace(profile, amplitude=1.0)
        else:
            shape = _profile_shape(prob, iprob, Ly, profile_kind, layers)

        if shear_amplitude is not None:
            S, source = float(shear_amplitude), "argument"
        elif "shear_strength" in prob:
            S, source = float(prob["shear_strength"]), "athinput"
            # the problem generator may ignore shear_strength; the initial kinetic energy shows what it used
            S_hst = _hst_amplitude(shape, nx, bounds, run_dir, hst)
            if S_hst is not None and abs(S_hst - S) > 1e-3 * max(abs(S), 1.0):
                _warn_user(f"{run_dir}: athinput shear_strength = {S:g} but hst 1-KE(0) implies S = {S_hst:.4f}; "
                           "the problem generator may not read shear_strength -- pass shear_amplitude= to override")
        else:
            S, source = _infer_amplitude(shape, nx, bounds, run_dir, hst)

        tau = _f(prob.get("tau", -99.9))
        return cls(
            problem_id=str(inp.problem_id),
            run_dir=run_dir,
            run_id=parse_run_id(run_dir) if run_dir is not None else None,
            nx=nx, bounds=bounds, meshblock=meshblock,
            cs=_f(hydro.get("iso_sound_speed")),
            c=_f(par.get("speed_of_light")),
            q_mc=_f(par.get("charge_over_mass_over_c")),
            backreaction=par.get("backreaction"),
            vp_par=float(prob.get("vp_par", 0.0)),
            cr_mass=float(prob.get("cr_mass", 0.0)),
            npx=npx,
            iprob=iprob,
            M_A=float(prob.get("M_A", 30.0)),
            tau=tau if tau is not None and tau > 0 else None,
            stir=int(prob["stir"]) if "stir" in prob else 0,
            nu_iso=float(prob.get("nu_iso", 0.0)),
            eta_ohm=float(prob.get("eta_ohm", 0.0)),
            shear_amplitude=S,
            shear_amplitude_source=source,
            profile=replace(shape, amplitude=S),
            athinput=inp,
        )

    @classmethod
    def from_run_dir(cls, run_dir: str | Path | int, **kw) -> "RunConfig":
        """Build from a run directory containing ``athinput.*`` and preferably the ``.hst``.

        A run number or name (423, 'run0423') that is not an existing directory is looked up
        with :func:`shearpic.env.resolve_run`.  Keyword arguments go to :meth:`from_athinput`.
        """
        if isinstance(run_dir, bool) or not isinstance(run_dir, (str, Path, int)):
            raise TypeError(f"run_dir must be a path, a run number or a run name, got {run_dir!r}")
        if isinstance(run_dir, int) or (isinstance(run_dir, str) and _is_run_reference(run_dir)):
            from .env import resolve_run

            run_dir = resolve_run(run_dir)
        run_dir = Path(run_dir)
        return cls.from_athinput(read_athinput(run_dir), run_dir=run_dir, **kw)

    # ================================================================ geometry
    @property
    def L(self) -> tuple[float, float, float]:
        """Box lengths (Lx, Ly, Lz) [a]."""
        return tuple(hi - lo for lo, hi in self.bounds)

    Lx = property(lambda self: self.L[0])
    Ly = property(lambda self: self.L[1])
    Lz = property(lambda self: self.L[2])

    @property
    def dx(self) -> tuple[float, float, float]:
        """Cell sizes (dx, dy, dz) [a]."""
        return tuple(L / n for L, n in zip(self.L, self.nx))

    @property
    def dV(self) -> float:
        """Cell volume [a^3]."""
        return math.prod(self.dx)

    @property
    def V(self) -> float:
        """Box volume [a^3]."""
        return math.prod(self.L)

    @property
    def ndim(self) -> int:
        """Number of axes with more than one cell."""
        return sum(n > 1 for n in self.nx)

    def edges(self, axis: int) -> np.ndarray:
        """Cell-face coordinates along ``axis`` (0, 1, 2 = x, y, z) [a], shape (nx[axis] + 1,)."""
        lo, hi = self.bounds[axis]
        return np.linspace(lo, hi, self.nx[axis] + 1)

    def centers(self, axis: int) -> np.ndarray:
        """Cell-centre coordinates along ``axis`` [a], shape (nx[axis],)."""
        e = self.edges(axis)
        return 0.5 * (e[:-1] + e[1:])

    # =============================================================== particles
    @property
    def n_par(self) -> int:
        """Total number of simulation particles N = npx1*npx2*npx3."""
        return math.prod(self.npx)

    @property
    def ppc(self) -> float:
        """Mean particles per cell."""
        return self.n_par / math.prod(self.nx)

    @property
    def m_cr(self) -> float:
        """Mass of one simulation particle, m = cr_mass * rho0 * V / N (as in the C++)."""
        return self.cr_mass * RHO0 * self.V / self.n_par

    # ================================================================= physics
    @property
    def B0(self) -> float:
        """Mean field strength B0 = U0 sqrt(rho0) / M_A (Athena units)."""
        return U0 * math.sqrt(RHO0) / self.M_A

    @property
    def v_A(self) -> float:
        """Alfven speed B0 / sqrt(rho0) [U0]."""
        return self.B0 / math.sqrt(RHO0)

    @property
    def has_particles(self) -> bool:
        """True when ``<particles>`` defines speed_of_light and charge_over_mass_over_c."""
        return self.c is not None and self.q_mc is not None

    def require_particles(self) -> None:
        """Raise ValueError unless the run has particles, i.e. c and q_mc are defined."""
        if not self.has_particles:
            where = f" ({self.run_dir})" if self.run_dir is not None else ""
            raise ValueError(f"this run has no <particles> speed_of_light / charge_over_mass_over_c{where}")

    _require_particles = require_particles  # backward-compatible alias

    @property
    def gamma0(self) -> float:
        """Lorentz factor of the injection momentum vp_par."""
        self.require_particles()
        return math.sqrt(1.0 + (self.vp_par / self.c) ** 2)

    @property
    def E0_per_mass(self) -> float:
        """Injection kinetic energy per unit mass, (gamma0-1) c^2 = vp_par^2 / (1+gamma0)."""
        return self.vp_par**2 / (1.0 + self.gamma0)

    @property
    def Omega0(self) -> float:
        """Gyrofrequency at injection in B0: q_mc B0 / gamma0."""
        return self.q_mc * self.B0 / self.gamma0

    @property
    def T_gyro0(self) -> float:
        """Gyroperiod 2 pi / Omega0 at injection [a/U0]."""
        return 2.0 * math.pi / self.Omega0

    @property
    def r_g0(self) -> float:
        """Gyroradius vp_par / (q_mc B0) of an injected particle moving perpendicular to B0."""
        self.require_particles()
        return self.vp_par / (self.q_mc * self.B0)

    @property
    def r_c0(self) -> float:
        """Isotropic average gyroradius r_c,0 = (pi/4) r_g0, the hst r_g/N at t = 0."""
        return 0.25 * math.pi * self.r_g0

    # ================================================================= outputs
    def output_dt(self, file_type: str, variable: str | None = None) -> float | None:
        """Cadence dt [a/U0] of the first ``<outputN>`` block with this file_type, or None.

        file_type is e.g. 'hst', 'hdf5' or 'rst'; a given variable such as 'prim' must also match.
        """
        o = self.athinput.output(file_type, variable)
        return None if o is None else _f(o.get("dt"))

    @property
    def hst_path(self) -> Path | None:
        """``run_dir/<problem_id>.hst`` if it exists, else None."""
        if self.run_dir is None:
            return None
        p = self.run_dir / f"{self.problem_id}.hst"
        return p if p.exists() else None

    # ================================================================== report
    def summary(self) -> dict[str, Any]:
        """Parameters, derived scales and outputs as a dict."""
        d = {
            "run_id": self.run_id, "problem_id": self.problem_id, "run_dir": str(self.run_dir),
            "nx": self.nx, "L": tuple(round(v, 6) for v in self.L), "dV": self.dV, "V": self.V,
            "M_A": self.M_A, "B0": self.B0, "cs": self.cs, "tau": self.tau, "nu_iso": self.nu_iso,
            "profile": self.profile.kind, "shear_amplitude": self.shear_amplitude,
            "shear_amplitude_source": self.shear_amplitude_source,
        }
        if self.has_particles:
            d.update({
                "c": self.c, "q_mc": self.q_mc, "backreaction": self.backreaction, "vp_par": self.vp_par,
                "n_par": self.n_par, "ppc": self.ppc, "cr_mass": self.cr_mass, "m_cr": self.m_cr,
                "gamma0": self.gamma0, "E0_per_mass": self.E0_per_mass, "Omega0": self.Omega0,
                "T_gyro0": self.T_gyro0, "r_g0": self.r_g0, "r_c0": self.r_c0,
            })
        d["outputs"] = [{k: o.get(k) for k in ("id", "file_type", "variable", "dt")} for o in self.athinput.outputs]
        return d

    def describe(self) -> str:
        """The summary as aligned ``key : value`` lines."""
        return "\n".join(f"{k:>22} : {v}" for k, v in self.summary().items())


def _warn_user(message: str) -> None:
    """Warn at the stack level of the first caller outside the shearpic package."""
    import sys

    pkg = str(Path(__file__).resolve().parent)
    frame, level = sys._getframe(1), 1
    while frame is not None and str(Path(frame.f_code.co_filename).resolve()).startswith(pkg):
        frame, level = frame.f_back, level + 1
    warnings.warn(message, stacklevel=level + 1)


def _profile_shape(prob, iprob: int, Ly: float, profile_kind: str | None,
                   layers: tuple[float, float] | None) -> ShearProfile:
    """Unit-amplitude ShearProfile from ``<problem>`` and the keyword overrides of ``from_athinput``."""
    if profile_kind is None:
        profile_kind = "sin" if iprob == 1 else "double_tanh"
    if layers is not None:
        if len(layers) != 2:
            raise ValueError(f"layers must be (y1, y2), got {layers!r}")
        y1, y2 = (float(v) for v in layers)
    elif "y1" in prob and "y2" in prob:
        y1, y2 = float(prob["y1"]), float(prob["y2"])
    else:
        y1, y2 = float(prob.get("y1", LEGACY_LAYERS[0])), float(prob.get("y2", LEGACY_LAYERS[1]))
        if profile_kind == "double_tanh":
            _warn_user(f"no y1/y2 in <problem>: using the hard-coded layer positions of the pre-y1/y2 problem "
                       f"generator, a double tanh at y1 = {y1:.6g}, y2 = {y2:.6g} (-5 pi, +5 pi); "
                       "pass layers=(y1, y2) or profile_kind to override")
    return ShearProfile(profile_kind, 1.0, A, y1, y2, Ly, int(prob.get("n", 1)))


def _hst_amplitude(shape: ShearProfile, nx, bounds, run_dir: Path | None, hst) -> float | None:
    """S from the initial gas kinetic energy 1-KE(0) = S^2 * 0.5 sum U_shape^2 dV, or None without a t = 0 record."""
    import pandas as pd

    from .io.history import read_hst

    df = None
    try:
        if isinstance(hst, pd.DataFrame):
            df = hst
        elif hst is not None:
            df = read_hst(hst)
        elif run_dir is not None and any(Path(run_dir).glob("*.hst")):
            df = read_hst(run_dir)
    except (OSError, ValueError) as err:
        _warn_user(f"{run_dir}: could not read the history file to check the shear amplitude: {err}")
        return None
    if df is None or "1-KE" not in df.columns or len(df) == 0 or df["time"].iloc[0] != 0.0:
        return None
    y_faces = np.linspace(bounds[1][0], bounds[1][1], nx[1] + 1)[:-1]
    dy = (bounds[1][1] - bounds[1][0]) / nx[1]
    area = (bounds[0][1] - bounds[0][0]) * (bounds[2][1] - bounds[2][0])
    ke_unit = 0.5 * np.sum(shape.U(y_faces) ** 2) * dy * area
    if ke_unit <= 0:
        return None
    return math.sqrt(float(df["1-KE"].iloc[0]) / ke_unit)


def _infer_amplitude(shape: ShearProfile, nx, bounds, run_dir: Path | None, hst) -> tuple[float, str]:
    """S from the history file when the athinput has no ``shear_strength`` (warns)."""
    S = _hst_amplitude(shape, nx, bounds, run_dir, hst)
    if S is None:
        _warn_user(f"{run_dir}: shear amplitude unknown (no shear_strength in athinput, no hst at t=0): assuming S = 1")
        return 1.0, "default"
    _warn_user(f"{run_dir}: shear_strength not in athinput; inferred shear amplitude S = {S:.4f} from hst 1-KE(0)")
    return S, "hst"
