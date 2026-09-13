"""Unit tests for shearpic.config.RunConfig built from synthetic athinput/history files."""

from __future__ import annotations

import dataclasses
import math
import pickle
import warnings

import numpy as np
import pandas as pd
import pytest

from shearpic.config import LEGACY_LAYERS, RunConfig, parse_run_id
from shearpic.io.athinput import parse_athinput
from shearpic.physics.forcing import ShearProfile

MESH = """
<job>
problem_id = synth.run
<output1>
file_type = hst
dt = 0.5
<output2>
file_type = hdf5
variable = prim
dt = 50.0
<analysis>
dt = 10.0
<mesh>
nx1 = 64
x1min = -3.141592653589793
x1max = 3.141592653589793
nx2 = 128
x2min = -12.566370614359172
x2max = 12.566370614359172
nx3 = 1
x3min = -0.5
x3max = 0.5
<meshblock>
nx1 = 16
nx2 = 32
nx3 = 1
<hydro>
iso_sound_speed = 5
<particles>
backreaction = true
charge_over_mass_over_c = 200.0
speed_of_light = 50.0
"""

PROBLEM = """
<problem>
iprob = 0
y1 = -6.283185307179586
y2 = 6.283185307179586
M_A = 10
tau = 0.5
nu_iso = 0.0
vp_par = 50.0
cr_mass = 0.0005
"""


def athinput(problem_extra: str = "", problem: str = PROBLEM, mesh: str = MESH):
    return parse_athinput(mesh + problem + problem_extra)


def write_hst(path, ke1: float):
    names = ["time", "dt", "1-KE", "2-KE"]
    hdr = "# Athena++ history data\n# " + "".join(f"[{i}]={n}".ljust(13) for i, n in enumerate(names, start=1)) + "\n"
    rows = f" {0.0: .10e} {1e-3: .10e} {ke1: .10e} {0.0: .10e}\n {0.5: .10e} {1e-3: .10e} {ke1: .10e} {1e-4: .10e}\n"
    path.write_text(hdr + rows)
    return path


def ke_for_amplitude(S: float) -> float:
    """1-KE(0) = 0.5 S^2 sum U_shape(y_face)^2 dy Lx Lz (rho0 = 1), as initialised by the C++."""
    shape = ShearProfile("double_tanh", 1.0, 1.0, -2 * math.pi, 2 * math.pi)
    faces = np.linspace(-4 * math.pi, 4 * math.pi, 129)[:-1]
    dy = 8 * math.pi / 128
    return 0.5 * S**2 * float(np.sum(shape.U(faces) ** 2)) * dy * (2 * math.pi) * 1.0


# ------------------------------------------------------------------- derived
def test_geometry_and_derived_scales():
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.0\n"))
    assert cfg.nx == (64, 128, 1) and cfg.meshblock == (16, 32, 1)
    assert cfg.L == pytest.approx((2 * math.pi, 8 * math.pi, 1.0))
    assert cfg.dx == pytest.approx((2 * math.pi / 64, 8 * math.pi / 128, 1.0))
    assert cfg.dV == pytest.approx(math.prod(cfg.dx))
    assert cfg.V == pytest.approx(16 * math.pi**2)
    assert cfg.ndim == 2
    np.testing.assert_allclose(cfg.edges(1), np.linspace(-4 * math.pi, 4 * math.pi, 129))
    np.testing.assert_allclose(cfg.centers(0), 0.5 * (cfg.edges(0)[1:] + cfg.edges(0)[:-1]))
    # npx_i defaults to nx_i, except npx = 1 along axes with nx = 1
    assert cfg.npx == (64, 128, 1) and cfg.n_par == 64 * 128 and cfg.ppc == 1.0
    assert cfg.m_cr == pytest.approx(0.0005 * cfg.V / cfg.n_par)
    assert cfg.B0 == pytest.approx(0.1) and cfg.v_A == pytest.approx(0.1)
    assert cfg.cs == 5.0 and cfg.c == 50.0 and cfg.q_mc == 200.0 and cfg.backreaction is True
    assert cfg.gamma0 == pytest.approx(math.sqrt(2.0))
    assert cfg.E0_per_mass == pytest.approx((math.sqrt(2.0) - 1.0) * 2500.0)
    assert cfg.Omega0 == pytest.approx(200.0 * 0.1 / math.sqrt(2.0))
    assert cfg.T_gyro0 == pytest.approx(2 * math.pi / cfg.Omega0)
    assert cfg.r_g0 == pytest.approx(2.5)
    assert cfg.r_c0 == pytest.approx(math.pi / 4 * 2.5)
    assert cfg.tau == 0.5 and cfg.nu_iso == 0.0 and cfg.stir == 0 and cfg.iprob == 0
    assert cfg.output_dt("hst") == 0.5 and cfg.output_dt("hdf5", "prim") == 50.0
    assert cfg.output_dt("vtk") is None
    assert cfg.problem_id == "synth.run" and cfg.run_dir is None and cfg.run_id is None


def test_explicit_npx_and_3d_rules():
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.0\nnpx1 = 256\nnpx2 = 512\nnpx3 = 7\n"))
    assert cfg.npx == (256, 512, 1)  # nx3 == 1 forces npx3 = 1
    assert cfg.n_par == 256 * 512 and cfg.ppc == pytest.approx(16.0)
    assert cfg.m_cr == pytest.approx(0.0005 * cfg.V / (256 * 512))
    mesh3d = MESH.replace("nx3 = 1\nx3min", "nx3 = 8\nx3min")
    cfg3 = RunConfig.from_athinput(athinput("shear_strength = 1.0\n", mesh=mesh3d))
    assert cfg3.nx == (64, 128, 8) and cfg3.npx == (64, 128, 8) and cfg3.ndim == 3


def test_run423_like_golden_values():
    """run423's numbers: c = 50, q_mc = 200, M_A = 10, vp_par = 50, 1024 x 4096 grid, npx = 4096 x 16384."""
    mesh = (MESH.replace("nx1 = 64\nx1min = -3.141592653589793\nx1max = 3.141592653589793",
                         "nx1 = 1024\nx1min = -15.707963267948966\nx1max = 15.707963267948966")
            .replace("nx2 = 128\nx2min = -12.566370614359172\nx2max = 12.566370614359172",
                     "nx2 = 4096\nx2min = -62.83185307179586\nx2max = 62.83185307179586"))
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.0\nnpx1 = 4096\nnpx2 = 16384\nnpx3 = 1\n", mesh=mesh))
    assert cfg.m_cr == pytest.approx(2.94137e-8, rel=1e-5)
    assert cfg.dV == pytest.approx(9.41239e-4, rel=1e-5)
    assert cfg.V == pytest.approx(3947.84, rel=1e-5)
    assert cfg.E0_per_mass == pytest.approx(1035.53, rel=1e-5)
    assert cfg.Omega0 == pytest.approx(14.142, rel=1e-4)
    assert cfg.r_c0 == pytest.approx(1.9635, rel=1e-4)


def test_particle_scales_require_particles_block():
    no_par = MESH.split("<particles>")[0]
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.0\n", mesh=no_par))
    assert cfg.c is None
    with pytest.raises(ValueError, match="no <particles>"):
        _ = cfg.gamma0
    with pytest.raises(ValueError):
        _ = cfg.r_g0
    assert "gamma0" not in cfg.summary()


# ------------------------------------------------------------- shear amplitude
def test_amplitude_from_shear_strength():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no warning when the key exists
        cfg = RunConfig.from_athinput(athinput("shear_strength = 2.5\n"))
    assert cfg.shear_amplitude == 2.5 and cfg.shear_amplitude_source == "athinput"
    assert cfg.profile.amplitude == 2.5 and cfg.profile.kind == "double_tanh"
    assert cfg.profile.y1 == pytest.approx(-2 * math.pi) and cfg.profile.y2 == pytest.approx(2 * math.pi)
    over = RunConfig.from_athinput(athinput("shear_strength = 2.5\n"), shear_amplitude=3.0)
    assert over.shear_amplitude == 3.0 and over.shear_amplitude_source == "argument" and over.profile.amplitude == 3.0


def test_amplitude_inferred_from_hst_file_and_dataframe(tmp_path):
    ke = ke_for_amplitude(2.0)
    # sanity: for well separated layers 1-KE(0) ~ 0.5 S^2 (V - 4 a Lx Lz)
    assert ke == pytest.approx(0.5 * 4.0 * (16 * math.pi**2 - 4 * 2 * math.pi), rel=2e-3)
    hst = write_hst(tmp_path / "synth.run.hst", ke)
    with pytest.warns(UserWarning, match="inferred shear amplitude S = 2.0000"):
        cfg = RunConfig.from_athinput(athinput(), hst=hst)
    assert cfg.shear_amplitude == pytest.approx(2.0, abs=1e-6)
    assert cfg.shear_amplitude_source == "hst" and cfg.profile.amplitude == cfg.shear_amplitude

    df = pd.DataFrame({"time": [0.0, 0.5], "1-KE": [ke, ke]})
    with pytest.warns(UserWarning, match="inferred"):
        cfg_df = RunConfig.from_athinput(athinput(), hst=df)
    assert cfg_df.shear_amplitude == pytest.approx(2.0, abs=1e-6)

    # found automatically next to the athinput in a run directory
    (tmp_path / "athinput.synth").write_text(MESH + PROBLEM)
    with pytest.warns(UserWarning, match="inferred"):
        cfg_dir = RunConfig.from_run_dir(tmp_path)
    assert cfg_dir.shear_amplitude == pytest.approx(2.0, abs=1e-6)
    assert cfg_dir.run_dir == tmp_path and cfg_dir.hst_path == tmp_path / "synth.run.hst"


def test_amplitude_default_when_nothing_available(tmp_path):
    with pytest.warns(UserWarning, match="assuming S = 1"):
        cfg = RunConfig.from_athinput(athinput())
    assert cfg.shear_amplitude == 1.0 and cfg.shear_amplitude_source == "default"
    # an hst that does not start at t = 0 (e.g. only a restart segment) cannot be used
    df = pd.DataFrame({"time": [100.0], "1-KE": [5.0]})
    with pytest.warns(UserWarning, match="assuming S = 1"):
        assert RunConfig.from_athinput(athinput(), hst=df).shear_amplitude_source == "default"


# --------------------------------------------------------------- profile kinds
def test_profile_kind_detection():
    assert RunConfig.from_athinput(athinput("shear_strength = 1\n")).profile.kind == "double_tanh"
    sin_problem = PROBLEM.replace("iprob = 0", "iprob = 1")
    sin = RunConfig.from_athinput(athinput("shear_strength = 1\nn = 2\n", problem=sin_problem))
    assert sin.profile.kind == "sin" and sin.profile.n == 2 and sin.profile.Ly == pytest.approx(8 * math.pi)
    forced = RunConfig.from_athinput(athinput("shear_strength = 1\n"), profile_kind="single_tanh")
    assert forced.profile.kind == "single_tanh" and forced.profile.layer_positions == (0.0,)


NO_LAYERS = "".join(ln for ln in PROBLEM.splitlines(keepends=True) if not ln.startswith(("y1", "y2")))


def test_legacy_iprob0_without_y1_y2_is_double_tanh_at_5pi():
    """Without <problem> y1/y2 (runs 306, 310-319) the pgen places a double tanh at -5 pi and +5 pi."""
    assert LEGACY_LAYERS == pytest.approx((-5 * math.pi, 5 * math.pi))
    with pytest.warns(UserWarning, match="hard-coded layer positions"):
        cfg = RunConfig.from_athinput(athinput("shear_strength = 1\n", problem=NO_LAYERS))
    assert cfg.profile.kind == "double_tanh"
    assert (cfg.profile.y1, cfg.profile.y2) == pytest.approx((-5 * math.pi, 5 * math.pi))
    assert cfg.profile.layer_positions == pytest.approx((-5 * math.pi, 5 * math.pi))
    # the positions do not scale with the box (run306 has Ly = 30 pi)
    tall = MESH.replace("x2min = -12.566370614359172\nx2max = 12.566370614359172",
                        "x2min = -47.12388980384690\nx2max = 47.12388980384690")
    with pytest.warns(UserWarning, match="hard-coded"):
        cfg_tall = RunConfig.from_athinput(athinput("shear_strength = 1\n", problem=NO_LAYERS, mesh=tall))
    assert cfg_tall.Ly == pytest.approx(30 * math.pi)
    assert cfg_tall.profile.layer_positions == pytest.approx((-5 * math.pi, 5 * math.pi))
    # a single layer only on explicit request
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        single = RunConfig.from_athinput(athinput("shear_strength = 1\n", problem=NO_LAYERS), profile_kind="single_tanh")
    assert single.profile.kind == "single_tanh"


def test_layers_and_profile_overrides_apply_before_amplitude_inference(tmp_path):
    # history written for S = 2 with layers at +-2 pi (ke_for_amplitude); the athinput lacks y1/y2
    hst = write_hst(tmp_path / "synth.run.hst", ke_for_amplitude(2.0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wrong = RunConfig.from_athinput(athinput(problem=NO_LAYERS), hst=hst)  # legacy +-5 pi layers
    assert wrong.shear_amplitude != pytest.approx(2.0, abs=1e-3)
    with pytest.warns(UserWarning, match="inferred shear amplitude S = 2.0000"):
        cfg = RunConfig.from_athinput(athinput(problem=NO_LAYERS), hst=hst, layers=(-2 * math.pi, 2 * math.pi))
    assert cfg.shear_amplitude == pytest.approx(2.0, abs=1e-6) and cfg.shear_amplitude_source == "hst"
    assert cfg.profile.kind == "double_tanh" and (cfg.profile.y1, cfg.profile.y2) == pytest.approx((-2 * math.pi,
                                                                                                    2 * math.pi))
    # layers also override <problem> y1, y2
    moved = RunConfig.from_athinput(athinput("shear_strength = 1\n"), layers=(-1.0, 3.0))
    assert moved.profile.layer_positions == (-1.0, 3.0)
    # a complete profile: its shape is used for the inference, its amplitude is replaced by S
    shape = ShearProfile("double_tanh", amplitude=7.0, a=1.0, y1=-2 * math.pi, y2=2 * math.pi)
    with pytest.warns(UserWarning, match="S = 2.0000"):
        by_profile = RunConfig.from_athinput(athinput(problem=NO_LAYERS), hst=hst, profile=shape)
    assert by_profile.profile == dataclasses.replace(shape, amplitude=by_profile.shear_amplitude)
    assert by_profile.shear_amplitude == pytest.approx(2.0, abs=1e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no legacy-layer warning when the profile is given
        RunConfig.from_athinput(athinput("shear_strength = 1\n", problem=NO_LAYERS), profile=shape)
    with pytest.raises(ValueError, match="not both"):
        RunConfig.from_athinput(athinput("shear_strength = 1\n"), profile=shape, layers=(0.0, 1.0))
    with pytest.raises(TypeError):
        RunConfig.from_athinput(athinput("shear_strength = 1\n"), profile="double_tanh")
    with pytest.raises(ValueError, match="layers"):
        RunConfig.from_athinput(athinput("shear_strength = 1\n"), layers=(1.0,))


# -------------------------------------------------------------- immutability etc
def test_frozen_and_picklable(tmp_path):
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.5\n"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        cfg.c = 10.0  # type: ignore[misc]
    clone = pickle.loads(pickle.dumps(cfg))
    assert clone == cfg
    assert clone.m_cr == cfg.m_cr and clone.profile == cfg.profile
    assert clone.athinput.get("output2", "dt") == 50.0
    changed = dataclasses.replace(cfg, M_A=20.0)
    assert changed.B0 == pytest.approx(0.05) and cfg.B0 == pytest.approx(0.1)


def test_describe_and_summary():
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.5\n"))
    text = cfg.describe()
    keys = {line.split(" : ", 1)[0].strip() for line in text.splitlines()}
    for key in ("nx", "dV", "V", "M_A", "B0", "shear_amplitude", "shear_amplitude_source", "c", "q_mc",
                "n_par", "m_cr", "gamma0", "E0_per_mass", "Omega0", "r_g0", "r_c0", "outputs"):
        assert key in keys, key
    s = cfg.summary()
    assert s["profile"] == "double_tanh" and s["shear_amplitude"] == 1.5
    assert [o["dt"] for o in s["outputs"]] == [0.5, 50.0]


def test_parse_run_id():
    assert parse_run_id("/data/run0423") == 423
    assert parse_run_id("run7") == 7
    assert parse_run_id("results/run404/") == 404
    assert parse_run_id("scratch") is None
    # anchored: only names that are exactly run<N>
    assert parse_run_id("myrun7") is None
    assert parse_run_id("/data/oldrun0423") is None
    assert parse_run_id("run7_restart") is None


def test_has_particles_and_require_particles():
    cfg = RunConfig.from_athinput(athinput("shear_strength = 1.0\n"))
    assert cfg.has_particles
    cfg.require_particles()
    cfg._require_particles()  # backward-compatible alias
    no_par = RunConfig.from_athinput(athinput("shear_strength = 1.0\n", mesh=MESH.split("<particles>")[0]))
    assert not no_par.has_particles
    with pytest.raises(ValueError, match="no <particles>"):
        no_par.require_particles()
    with pytest.raises(ValueError, match="no <particles>"):
        no_par._require_particles()


def test_from_run_dir_accepts_run_number_and_name(tmp_path, monkeypatch):
    data = tmp_path / "data"
    run = data / "run0007"
    run.mkdir(parents=True)
    (run / "athinput.synth").write_text(MESH + PROBLEM + "shear_strength = 1.0\n")
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    by_path = RunConfig.from_run_dir(run)
    for ref in (7, "7", "run7", "run0007"):
        cfg = RunConfig.from_run_dir(ref)
        assert cfg.run_dir == run and cfg.run_id == 7 and cfg == by_path, ref
    assert RunConfig.from_run_dir(str(run)).run_dir == run
    with pytest.raises(FileNotFoundError):
        RunConfig.from_run_dir(8)
    with pytest.raises(FileNotFoundError):
        RunConfig.from_run_dir(tmp_path / "missing")
    with pytest.raises(TypeError):
        RunConfig.from_run_dir(True)


def test_docstrings_present():
    for obj in (RunConfig, parse_run_id, RunConfig.output_dt, RunConfig.hst_path, ShearProfile, ShearProfile.U):
        assert obj.__doc__ and len(obj.__doc__) > 40, obj
    for name in ("problem_id", "nx", "bounds", "c", "q_mc", "vp_par", "cr_mass", "npx", "M_A", "tau", "nu_iso",
                 "shear_amplitude", "shear_amplitude_source", "profile"):
        assert f"{name} :" in RunConfig.__doc__, name
