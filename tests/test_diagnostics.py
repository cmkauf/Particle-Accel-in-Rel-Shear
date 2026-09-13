"""Unit tests for shearpic.physics.diagnostics on synthetic history DataFrames."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from shearpic.config import RunConfig
from shearpic.io.athinput import parse_athinput
from shearpic.physics.diagnostics import DEFINITIONS, energy_budget, history_diagnostics

ATHINPUT = """
<job>
problem_id = synth
<mesh>
nx1 = 16
x1min = -2.0
x1max = 2.0
nx2 = 32
x2min = -8.0
x2max = 8.0
nx3 = 1
x3min = -0.5
x3max = 0.5
<particles>
speed_of_light = 50.0
charge_over_mass_over_c = 200.0
<problem>
shear_strength = 1.0
y1 = -4.0
y2 = 4.0
vp_par = 50.0
cr_mass = 0.0005
npx1 = 64
npx2 = 128
npx3 = 1
"""


@pytest.fixture(scope="module")
def cfg() -> RunConfig:
    return RunConfig.from_athinput(parse_athinput(ATHINPUT))


def synthetic_hst(cfg: RunConfig, t=None) -> pd.DataFrame:
    """History with analytic time dependence: KE_cr = K0 + k t^2, so Pideal = 2 m_cr k t exactly."""
    t = np.arange(0.0, 20.5, 0.5) if t is None else t
    N = float(cfg.n_par)
    K0, k = 1000.0 * N, 3.0e5
    return pd.DataFrame({
        "time": t,
        "dt": np.full(t.size, 1e-3),
        "1-KE": 10.0 + t, "2-KE": 0.5 * np.ones_like(t), "3-KE": np.zeros_like(t),
        "1-ME": 0.2 + 0.01 * t, "2-ME": 0.1 * np.ones_like(t), "3-ME": np.zeros_like(t),
        "np": np.full(t.size, N),
        "KE_cr": K0 + k * t**2,
        "Gamma": 1.4 * N + 0.01 * N * t,
        "r_g": 2.5 * N * np.ones_like(t),
        "w_c": 14.0 * N * np.ones_like(t),
        "Pideal": 2.0 * cfg.m_cr * k * t,
        "Pideal_x": 0.5 * cfg.m_cr * k * t, "Pideal_y": 0.5 * cfg.m_cr * k * t, "Pideal_z": cfg.m_cr * k * t,
        "Pstir": 7.0 / cfg.dV * np.ones_like(t),
    })


def test_history_diagnostics_definitions(cfg):
    hst = synthetic_hst(cfg)
    d = history_diagnostics(hst, cfg)
    t = hst["time"].to_numpy()
    expected_cols = ["time", "eps_kin", "eps_mag", "eps_p", "mean_gamma", "mean_r_g", "mean_omega_c",
                     "P_ideal", "P_ideal_x", "P_ideal_y", "P_ideal_z", "P_stir", "dEp_dt"]
    assert sorted(d.columns) == sorted(expected_cols)
    assert set(d.columns) <= set(DEFINITIONS) and set(d.attrs["definitions"]) == set(d.columns)
    assert np.allclose(d["eps_kin"], (10.5 + t) / cfg.V)
    assert np.allclose(d["eps_mag"], (0.3 + 0.01 * t) / cfg.V)
    assert np.allclose(d["eps_p"], cfg.m_cr * hst["KE_cr"] / cfg.V)
    assert np.allclose(d["mean_gamma"], 1.4 + 0.01 * t)
    assert np.allclose(d["mean_r_g"], 2.5) and np.allclose(d["mean_omega_c"], 14.0)
    assert np.allclose(d["P_stir"], 7.0)
    assert np.allclose(d["P_ideal"], hst["Pideal"])
    # dEp/dt = d(m_cr KE_cr)/dt = Pideal for this analytic history (2nd order exact in the interior)
    assert np.allclose(d["dEp_dt"].iloc[1:-1], d["P_ideal"].iloc[1:-1], rtol=1e-10, atol=1e-12)


def test_history_diagnostics_only_available_columns(cfg):
    hst = pd.DataFrame({"time": [0.0, 1.0, 2.0], "1-KE": [1.0, 2.0, 3.0], "np": [0.0, 5.0, 5.0],
                        "Gamma": [0.0, 7.0, 8.0]})
    d = history_diagnostics(hst, cfg)
    assert list(d.columns) == ["time", "eps_kin", "mean_gamma"]
    assert np.isnan(d["mean_gamma"].iloc[0])  # np = 0: NaN, no division warning
    assert d["mean_gamma"].iloc[1] == pytest.approx(1.4)


def test_restart_duplicates_are_removed_with_warning(cfg):
    clean = synthetic_hst(cfg)
    # a restart from t = 5 re-writes rows 5.0 ... 8.0 after the first run had reached t = 8.0
    first = clean[clean.time <= 8.0]
    rest = clean[clean.time >= 5.0]
    dirty = pd.concat([first, rest], ignore_index=True)
    with pytest.warns(UserWarning, match="restart"):
        d = history_diagnostics(dirty, cfg)
    ref = history_diagnostics(clean, cfg)
    assert np.all(np.diff(d["time"]) > 0)
    pd.testing.assert_frame_equal(d, ref)
    with pytest.warns(UserWarning, match="restart"):
        b = energy_budget(dirty, cfg, 2.0, 15.0)
    assert b == energy_budget(clean, cfg, 2.0, 15.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        history_diagnostics(clean, cfg)


def test_energy_budget_analytic(cfg):
    hst = synthetic_hst(cfg)
    b = energy_budget(hst, cfg, 1.9, 12.2)  # snaps to the samples inside the window
    assert (b["t_start"], b["t_end"], b["n_samples"]) == (2.0, 12.0, 21)
    k = 3.0e5
    dEp = cfg.m_cr * k * (12.0**2 - 2.0**2)
    assert b["Delta_E_p"] == pytest.approx(dEp, rel=1e-12)
    assert b["int_P_ideal_dt"] == pytest.approx(dEp, rel=1e-12)  # trapezoid is exact for linear power
    assert b["ratio_Ep_over_P_ideal"] == pytest.approx(1.0, rel=1e-10)
    assert b["int_P_stir_dt"] == pytest.approx(7.0 * 10.0)
    assert b["Delta_E_kin"] == pytest.approx(10.0)
    assert b["Delta_E_mag"] == pytest.approx(0.1)
    assert b["Delta_E_gas"] == pytest.approx(10.1)


def test_energy_budget_non_uniform_time_axis(cfg):
    t = np.sort(np.concatenate([np.linspace(0, 10, 7), [0.3, 1.7, 4.4, 9.9]]))
    b = energy_budget(synthetic_hst(cfg, t), cfg, 0.0, 10.0)
    assert b["ratio_Ep_over_P_ideal"] == pytest.approx(1.0, rel=1e-10)


def test_energy_budget_errors_and_missing_columns(cfg):
    hst = synthetic_hst(cfg)
    with pytest.raises(ValueError):
        energy_budget(hst, cfg, 5.0, 5.0)
    with pytest.raises(ValueError, match="fewer than 2"):
        energy_budget(hst, cfg, 100.0, 200.0)
    b = energy_budget(hst[["time", "1-KE"]], cfg, 0.0, 10.0)
    assert b["Delta_E_kin"] == pytest.approx(10.0)
    for key in ("Delta_E_p", "int_P_ideal_dt", "int_P_stir_dt", "Delta_E_mag", "Delta_E_gas", "ratio_Ep_over_P_ideal"):
        assert np.isnan(b[key])
    with pytest.raises(ValueError, match="time"):
        history_diagnostics(hst.drop(columns="time"), cfg)


def _write_hst(path, df: pd.DataFrame):
    names = list(df.columns)
    header = "# Athena++ history data\n# " + "".join(f"[{i}]={n}".ljust(13) for i, n in enumerate(names, 1)) + "\n"
    rows = "".join(" ".join(f"{v: .15e}" for v in row) + "\n" for row in df.to_numpy())
    path.write_text(header + rows)
    return path


def test_history_diagnostics_and_budget_accept_paths(cfg, tmp_path):
    hst = synthetic_hst(cfg)
    f = _write_hst(tmp_path / "synth.hst", hst)
    ref = history_diagnostics(hst, cfg)
    for source in (f, str(f), tmp_path, str(tmp_path)):
        pd.testing.assert_frame_equal(history_diagnostics(source, cfg), ref, rtol=1e-12)
        b = energy_budget(source, cfg, 2.0, 12.0)
        ref_b = energy_budget(hst, cfg, 2.0, 12.0)
        assert b.keys() == ref_b.keys()
        for key in b:
            assert b[key] == pytest.approx(ref_b[key], rel=1e-12), key
    with pytest.raises(TypeError, match="DataFrame or a path"):
        history_diagnostics(hst.to_numpy(), cfg)


@pytest.mark.data
@pytest.mark.parametrize("run, ratio", [("run364", 1.0176), ("run423", 1.0148)])
def test_data_energy_budget_ratio(data_root, run, ratio):
    from pathlib import Path

    run_dir = Path(data_root) / run
    if not run_dir.is_dir():
        pytest.skip(f"{run} not available under {data_root}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cfg = RunConfig.from_run_dir(run_dir)
        budget = energy_budget(cfg.hst_path, cfg, 100.0, 600.0)
    assert budget["n_samples"] == 1001
    assert budget["ratio_Ep_over_P_ideal"] == pytest.approx(ratio, abs=1e-4)
