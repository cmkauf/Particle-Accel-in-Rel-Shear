"""Tests of the analysis layer against reference values from real runs (skipped without data)."""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from shearpic.config import RunConfig
from shearpic.io.history import read_hst
from shearpic.io.spectrum_files import read_legacy_spectrum_dir
from shearpic.io.trajectory import find_trajectory_files, read_trajectory
from shearpic.physics.diagnostics import energy_budget, history_diagnostics
from shearpic.physics.particles import crossing_events_for, cumulative_crossings, trajectory_diagnostics

pytestmark = pytest.mark.data


def _config(run_dir) -> RunConfig:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # amplitude inference notice
        return RunConfig.from_run_dir(run_dir)


# ----------------------------------------------------------------- run369
def test_run369_legacy_spectra(run369):
    with pytest.warns(UserWarning, match="gamma_minus_1"):
        specs = read_legacy_spectrum_dir(run369 / "energy_spectrum_data")
    assert len(specs) == 7
    assert [s.time for s in specs] == [0.0, 100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    for s in specs:
        assert s.variable == "gamma_minus_1" and s.counts is None and s.run_id == 369
        assert s.n_bins == 500 and s.edges[0] == 0.01 and s.edges[-1] == 100.0
        assert np.sum(s.pdf * s.widths) == pytest.approx(1.0, abs=1e-6)
        g = s.to("gamma")
        assert np.sum(g.pdf * g.widths) == pytest.approx(1.0, abs=1e-6)


# ----------------------------------------------------------------- run404
@pytest.fixture(scope="module")
def run404_trajectories(run404):
    files = find_trajectory_files(run404)
    assert files, "run404 should contain trajectory files"
    return [read_trajectory(p) for p in files[:3]]


def test_run404_trajectory_diagnostics_physical(run404, run404_trajectories):
    cfg = _config(run404)
    for traj in run404_trajectories:
        assert traj.has_fields and np.all(np.diff(traj.t) > 0)
        d = trajectory_diagnostics(traj, cfg)
        assert np.all(d["gamma"] >= 1.0)
        assert np.all(np.linalg.norm(d["v"], axis=1) < cfg.c)
        assert d["gamma"].max() > 2.0  # these particles were selected as high-energy
        # momentum convention: dx/dt equals u_x/gamma, not u_x (early samples, x unwrapped)
        n = 50
        dx = np.diff(traj.x[:n, 0])
        dx -= cfg.Lx * np.round(dx / cfg.Lx)
        dxdt = dx / np.diff(traj.t[:n])
        v_mid = 0.5 * (d["v"][1:n, 0] + d["v"][:n - 1, 0])
        u_mid = 0.5 * (traj.u[1:n, 0] + traj.u[:n - 1, 0])
        assert np.median(np.abs(dxdt - v_mid) / np.abs(v_mid)) < 0.02
        assert np.median(np.abs(dxdt - u_mid) / np.abs(u_mid)) > 0.2
        # ideal MHD: cE = -U x B is perpendicular to B
        cE, B = traj.cE, traj.B
        cos_EB = np.einsum("ij,ij->i", cE, B) / (np.linalg.norm(cE, axis=1) * np.linalg.norm(B, axis=1))
        assert np.nanmedian(np.abs(cos_EB)) < 1e-3
        for key in ("pitch_lab", "pitch_drift", "cos_v_cE"):
            vals = d[key][np.isfinite(d[key])]
            assert vals.size > 0.9 * len(traj) and np.all(np.abs(vals) <= 1.0)
        assert np.allclose(d["dgamma_dt"], d["power_per_mass"] / cfg.c**2)


def test_run404_crossings(run404, run404_trajectories):
    cfg = _config(run404)
    for traj in run404_trajectories:
        ev = crossing_events_for(traj, cfg)
        assert 0 < ev.count < len(traj)
        assert np.all(np.isfinite(ev.times)) and np.all(np.diff(ev.times) >= 0)
        assert np.all((ev.times >= traj.t[0]) & (ev.times <= traj.t[-1]))
        assert set(np.unique(ev.direction)) <= {-1, 1}
        cum = cumulative_crossings(ev, len(traj))
        assert cum[-1] == ev.count and np.all(np.diff(cum) >= 0)
        # consecutive crossings of the same layer must alternate in direction
        for layer in range(len(ev.layer_positions)):
            dirs = ev.select(layer=layer).direction
            assert np.all(dirs[1:] != dirs[:-1])
        # a wider hysteresis band can only remove crossings
        assert crossing_events_for(traj, cfg, hysteresis=5.0).count <= ev.count


# ----------------------------------------------------------------- run423
@pytest.fixture(scope="module")
def run423_history(run423):
    return _config(run423), read_hst(run423)


def test_run423_history_diagnostics_golden(run423_history):
    cfg, hst = run423_history
    d = history_diagnostics(hst, cfg)
    first = d.iloc[0]
    assert first["time"] == 0.0
    assert first["eps_p"] == pytest.approx(0.518, rel=2e-3)
    assert first["eps_kin"] == pytest.approx(0.484, rel=2e-3)
    assert first["mean_gamma"] == pytest.approx(1.41449, rel=1e-5)
    assert first["mean_r_g"] == pytest.approx(cfg.r_c0, rel=1e-3)
    window = (d["time"] >= 100.0) & (d["time"] <= 600.0)
    assert d.loc[window, "P_stir"].mean() == pytest.approx(27.8, rel=0.03)
    assert d.loc[window, "P_ideal"].mean() == pytest.approx(2.84, rel=0.02)
    assert d.loc[window, "dEp_dt"].mean() == pytest.approx(2.86, rel=0.02)


def test_run423_energy_budget(run423_history):
    cfg, hst = run423_history
    b = energy_budget(hst, cfg, 100.0, 600.0)
    assert (b["t_start"], b["t_end"]) == (100.0, 600.0)
    assert 1.0 <= b["ratio_Ep_over_P_ideal"] <= 1.02
    assert b["int_P_stir_dt"] / 500.0 == pytest.approx(27.8, rel=0.03)
