"""Tests of the foundation layer (io, config, physics) against real simulation output.

Uses run423 (HDF5 snapshots and history) and run404 (tracked trajectories); the data are only read.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shearpic.config import RunConfig
from shearpic.io.athdf import Snapshot, list_snapshots
from shearpic.io.history import read_hst
from shearpic.io.trajectory import load_trajectories
from shearpic.physics.forcing import stir_power
from shearpic.physics.relativity import lorentz_factor

pytestmark = pytest.mark.data


# --------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def cfg423(run423) -> RunConfig:
    return RunConfig.from_run_dir(run423)


@pytest.fixture(scope="module")
def hst423(cfg423):
    return read_hst(cfg423.hst_path)


@pytest.fixture(scope="module")
def snaps423(run423):
    return list_snapshots(run423, "out2")


@pytest.fixture(scope="module")
def snap300(snaps423):
    s = Snapshot.open(snaps423[6])
    return s, s.read(["rho", "vel1", "np", "vp1", "Bcc1"], dtype=np.float64)


@pytest.fixture(scope="module")
def trajs404(run404):
    return load_trajectories(run404, backend="thread")


def _row_at(hst, t):
    i = int(np.argmin(np.abs(hst["time"].to_numpy() - t)))
    assert abs(hst["time"].iloc[i] - t) < 1e-6
    return hst.iloc[i]


# ----------------------------------------------------------------- run423 config
def test_run423_config_golden(cfg423):
    c = cfg423
    assert c.shear_amplitude == 1.0 and c.shear_amplitude_source == "athinput"
    assert c.profile.kind == "double_tanh"
    assert c.nx == (1024, 4096, 1) and c.npx == (4096, 16384, 1) and c.n_par == 67108864
    assert c.m_cr == pytest.approx(2.94137e-8, rel=1e-5)
    assert c.dV == pytest.approx(9.41239e-4, rel=1e-5)
    assert c.V == pytest.approx(3947.84, rel=1e-5)
    assert c.B0 == pytest.approx(0.1, rel=1e-12)
    assert c.gamma0 == pytest.approx(math.sqrt(2.0), rel=1e-12)
    assert c.E0_per_mass == pytest.approx(1035.53, rel=1e-5)
    assert c.Omega0 == pytest.approx(14.142, rel=1e-4)
    assert c.r_g0 == pytest.approx(2.5, rel=1e-12)
    assert c.r_c0 == pytest.approx(1.9635, rel=1e-4)
    assert c.output_dt("hst") == 0.5 and c.output_dt("hdf5") == 50.0  # not the <analysis> dt = 10


# ---------------------------------------------------------------- run423 history
def test_run423_history_columns(hst423):
    assert len(hst423.columns) == 50
    for name in ("-BxBy", "vp1^2", "dVxVy", "Pideal_x", "Pideal_y", "Pideal_z", "Emag_wind", "Upz_down"):
        assert name in hst423.columns
    t = hst423["time"].to_numpy()
    assert t[0] == 0.0 and np.all(np.diff(t) > 0)
    assert hst423.attrs["n_dropped"] == 0


def test_run423_history_initial_particle_means(hst423, cfg423):
    r0 = hst423.iloc[0]
    N = r0["np"]
    assert N == pytest.approx(cfg423.n_par, rel=1e-5)
    assert r0["KE_cr"] / N == pytest.approx(1036.2, rel=1e-3)  # ~E0_per_mass = (gamma0-1) c^2
    assert r0["KE_cr"] / N == pytest.approx(cfg423.E0_per_mass, rel=2e-3)
    assert r0["Gamma"] / N == pytest.approx(1.41449, rel=1e-4)
    assert r0["r_g"] / N == pytest.approx(cfg423.r_c0, rel=1e-3)  # isotropic mean gyroradius (pi/4) r_g0
    eps_p = cfg423.m_cr * r0["KE_cr"] / cfg423.V
    eps_k = (r0["1-KE"] + r0["2-KE"] + r0["3-KE"]) / cfg423.V
    assert eps_p == pytest.approx(0.518, rel=1e-3)
    assert eps_k == pytest.approx(0.484, rel=1e-3)


def test_run423_energy_budget(hst423, cfg423):
    """Pideal is the total power into the CRs (already includes m_cr): it balances d(m_cr KE_cr)/dt."""
    h = hst423[(hst423["time"] >= 100.0) & (hst423["time"] <= 600.0)]
    t = h["time"].to_numpy()
    P = h["Pideal"].to_numpy()
    E = cfg423.m_cr * h["KE_cr"].to_numpy()
    work = float(np.sum(0.5 * (P[1:] + P[:-1]) * np.diff(t)))
    assert P.mean() == pytest.approx(2.84, rel=1e-2)
    assert np.polyfit(t, E, 1)[0] == pytest.approx(2.86, rel=2e-2)
    assert 1.0 <= (E[-1] - E[0]) / work <= 1.02
    assert float((h["Pstir"] * cfg423.dV).mean()) == pytest.approx(27.8, rel=1e-2)  # Pstir lacks dV


# -------------------------------------------------------------- run423 snapshots
def test_run423_snapshot_listing(snaps423):
    assert len(snaps423) == 13
    assert Snapshot.open(snaps423[6]).time == pytest.approx(300.0, abs=1e-3)


def test_run423_snapshot_t300_integrals(snap300, hst423, cfg423):
    snap, d = snap300
    assert snap.shape == cfg423.nx and d["rho"].shape == cfg423.nx
    row = _row_at(hst423, 300.0)
    dV = cfg423.dV
    # np is a number density: its volume integral is the particle count
    assert d["np"].sum() * dV == pytest.approx(cfg423.n_par, rel=1e-6)
    # vp1 is the cell-mean reduced momentum <u_x>: sum np vp1 dV = sum_p u_x = hst vp1
    assert (d["np"] * d["vp1"]).sum() * dV == pytest.approx(row["vp1"], rel=1e-5)
    assert (d["np"] * d["vp1"]).sum() * dV == pytest.approx(834032, rel=1e-5)
    assert 0.5 * (d["Bcc1"] ** 2).sum() * dV == pytest.approx(row["1-ME"], rel=1e-5)
    assert 0.5 * (d["Bcc1"] ** 2).sum() * dV == pytest.approx(165.892, rel=1e-5)
    assert 0.5 * (d["rho"] * d["vel1"] ** 2).sum() * dV == pytest.approx(row["1-KE"], rel=1e-5)
    assert 0.5 * (d["rho"] * d["vel1"] ** 2).sum() * dV == pytest.approx(2028.764, rel=1e-5)


def test_run423_stir_power_matches_history(snap300, hst423, cfg423):
    _, d = snap300
    P = stir_power(d["rho"], d["vel1"], cfg423)
    assert P == pytest.approx(16.615, rel=1e-3)
    assert P == pytest.approx(_row_at(hst423, 300.0)["Pstir"] * cfg423.dV, rel=3e-3)


def test_run423_initial_profile_on_lower_faces(snaps423, cfg423):
    vel1 = Snapshot.open(snaps423[0]).read("vel1", dtype=np.float64)["vel1"]
    mean = vel1.mean(axis=(0, 2))
    assert np.abs(mean - cfg423.profile.on_cpp_grid(cfg423.edges(1))).max() < 1e-4
    assert np.abs(mean - cfg423.profile.U(cfg423.centers(1))).max() > 1e-2  # centres are measurably off


def test_run423_h5py_reader_matches_yt(snaps423):
    yt = pytest.importorskip("yt")
    path = snaps423[0]
    ds = yt.load(str(path))
    grid = ds.covering_grid(0, ds.domain_left_edge, ds.domain_dimensions)
    ours = Snapshot.open(path).read(["rho", "Bcc1"], dtype=np.float64)
    for name in ("rho", "Bcc1"):
        ref = np.asarray(grid["athena_pp", name])
        assert ref.shape == ours[name].shape
        assert np.abs(ref - ours[name]).max() == 0.0


def test_run423_snapshot_reader_bit_identical_to_stacked_blocks(snaps423):
    """The block-by-block fill equals a direct reshape of the (lexsorted) meshblocks."""
    import h5py

    path = snaps423[6]
    snap = Snapshot.open(path)
    ours = snap.read(["vel1", "Bcc2"], dtype=None)
    bx, by, bz = snap.block_shape
    nx, ny, nz = snap.shape
    with h5py.File(path, "r", locking=False) as f:
        loc = f["LogicalLocations"][:]
        order = np.lexsort((loc[:, 0], loc[:, 1], loc[:, 2]))
        for name in ("vel1", "Bcc2"):
            ds, idx = snap._locations[name]
            tiled = f[ds][idx][order].reshape(nz // bz, ny // by, nx // bx, bz, by, bx)
            ref = tiled.transpose(0, 3, 1, 4, 2, 5).reshape(nz, ny, nx).transpose(2, 1, 0)
            assert ours[name].dtype == ref.dtype
            assert np.array_equal(ours[name], ref)


# ---------------------------------------------------------- runs without y1/y2
@pytest.mark.parametrize("run_name", ["run310", "run306"])
def test_legacy_runs_double_tanh_at_5pi(data_root, run_name):
    """Without y1/y2 (runs 306, 310-319) the profile is a double tanh at y = -5 pi, +5 pi, also for Ly = 30 pi."""
    import warnings

    run = data_root / run_name
    if not run.is_dir():
        pytest.skip(f"{run_name} not available under {data_root}")
    snaps = list_snapshots(run, "out2")
    if not snaps or not list(run.glob("*.hst")):
        pytest.skip(f"{run_name} has no t=0 snapshot or history file")
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        cfg = RunConfig.from_run_dir(run)
    assert any("hard-coded layer positions" in str(w.message) for w in rec)
    assert cfg.profile.kind == "double_tanh"
    assert cfg.profile.layer_positions == pytest.approx((-5 * math.pi, 5 * math.pi))
    assert cfg.shear_amplitude_source == "hst"
    assert cfg.shear_amplitude == pytest.approx(1.0, abs=1e-3)
    snap = Snapshot.open(snaps[0])
    assert snap.time == 0.0
    mean = snap.read("vel1", dtype=np.float64)["vel1"].mean(axis=(0, 2))
    assert np.abs(mean - cfg.profile.on_cpp_grid(cfg.edges(1))).max() < 1e-4
    # a single layer at y = 0 is off by O(1)
    single = RunConfig.from_run_dir(run, profile_kind="single_tanh", shear_amplitude=1.0)
    assert np.abs(mean - single.profile.on_cpp_grid(cfg.edges(1))).max() > 1.0


# ------------------------------------------------------------ run404 trajectories
def test_run404_config(run404):
    cfg = RunConfig.from_run_dir(run404)
    assert cfg.shear_amplitude_source == "athinput" and cfg.shear_amplitude == 1.0
    assert cfg.c == 50.0 and cfg.q_mc == 200.0


def test_run404_trajectories_read(trajs404):
    assert len(trajs404) == 3
    assert sorted((tr.init_mbid, tr.pid) for tr in trajs404) == [(173, 14337), (264, 19457), (339, 18433)]
    for tr in trajs404:
        assert tr.has_fields  # 13-column files
        assert len(tr) > 1000
        assert np.all(np.diff(tr.t) > 0)
        # ideal motional field: cE = -U x B is perpendicular to B
        cEB = np.abs(np.einsum("ij,ij->i", tr.cE, tr.B))
        assert cEB.max() <= 1e-5 * (np.linalg.norm(tr.cE, axis=1) * np.linalg.norm(tr.B, axis=1)).max()


def test_run404_columns_are_reduced_momentum(trajs404, run404):
    """dx/dt = u_x / gamma, not u_x: the .tab columns 4-6 are u = p/m.

    The x component is used because B0 is along x, so u_x ~ u_par barely changes between
    samples (the perpendicular components gyrate ~2 rad per output interval).
    """
    cfg = RunConfig.from_run_dir(run404)
    for tr in trajs404:
        n = 21
        dt = np.diff(tr.t[:n])
        dx = np.diff(tr.x[:n, 0])
        dx = (dx + 0.5 * cfg.Lx) % cfg.Lx - 0.5 * cfg.Lx  # periodic minimum image
        ux = tr.u[:n, 0]
        vx = ux / lorentz_factor(tr.u[:n], cfg.c)
        res_v = np.median(np.abs(dx / dt - 0.5 * (vx[1:] + vx[:-1])))
        res_u = np.median(np.abs(dx / dt - 0.5 * (ux[1:] + ux[:-1])))
        assert res_v < 0.1 * res_u, (tr.pid, res_v, res_u)
        assert res_v < 0.1
