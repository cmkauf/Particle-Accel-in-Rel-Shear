"""Unit tests for shearpic.physics.particles: crossings, trajectory diagnostics, grid sampling."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from shearpic.config import RunConfig
from shearpic.io.athinput import parse_athinput
from shearpic.io.trajectory import Trajectory
from shearpic.physics import relativity as rel
from shearpic.physics.particles import (
    CrossingEvents,
    crossing_events,
    crossing_events_for,
    cumulative_crossings,
    sample_field_ngp,
    sample_field_tsc,
    trajectory_diagnostics,
)

C, Q_MC = 50.0, 200.0


def make_cfg(nx=(8, 16, 1), bounds=((-2.0, 2.0), (-8.0, 8.0), (-0.5, 0.5))) -> RunConfig:
    text = f"""
<job>
problem_id = synth
<mesh>
nx1 = {nx[0]}
x1min = {bounds[0][0]}
x1max = {bounds[0][1]}
nx2 = {nx[1]}
x2min = {bounds[1][0]}
x2max = {bounds[1][1]}
nx3 = {nx[2]}
x3min = {bounds[2][0]}
x3max = {bounds[2][1]}
<particles>
speed_of_light = {C}
charge_over_mass_over_c = {Q_MC}
<problem>
shear_strength = 1.0
y1 = -4.0
y2 = 4.0
M_A = 10
vp_par = 50.0
cr_mass = 0.0005
"""
    return RunConfig.from_athinput(parse_athinput(text))


# ================================================================= crossings
def test_single_upward_crossing_time_is_interpolated():
    t = np.arange(21, dtype=float)
    y = -10.0 + t * 1.3  # passes y = 0 between samples 7 (-0.9) and 8 (0.4)
    ev = crossing_events(t, y, [0.0], Ly=100.0, hysteresis=1.0)
    assert isinstance(ev, CrossingEvents) and ev.count == 1
    assert ev.direction.tolist() == [1] and ev.layer.tolist() == [0]
    assert ev.index.tolist() == [8]
    assert ev.times[0] == pytest.approx(10.0 / 1.3)
    down = crossing_events(t, -y, [0.0], Ly=100.0, hysteresis=1.0)
    assert down.direction.tolist() == [-1] and down.index.tolist() == [8]


def dithering_path(end_side: float, amplitude: float = 0.5, n_dither: int = 40):
    approach = np.linspace(-5.0, 0.0, 11)
    dither = amplitude * np.sin(np.arange(1, n_dither + 1) * 2.1)
    leave = np.linspace(0.0, 5.0 * end_side, 11)[1:]
    y = np.concatenate([approach, dither, leave])
    return np.arange(y.size, dtype=float), y


def test_hysteresis_suppresses_dithering():
    t, y = dithering_path(end_side=+1)
    ev = crossing_events(t, y, [0.0], Ly=100.0, hysteresis=1.0)
    assert ev.count == 1 and ev.direction[0] == 1
    # the crossing time is the *last* passage through y = 0 before leaving the band
    k = ev.index[0]
    assert y[k - 1] < 0.0 <= y[k] and np.all(y[k:] >= 0.0)
    # without hysteresis every sign change counts, but the net crossing number is still +1
    raw = crossing_events(t, y, [0.0], Ly=100.0, hysteresis=0.0)
    assert raw.count > 10 and raw.direction.sum() == 1
    assert np.all(np.abs(np.diff(raw.direction)) == 2)  # alternating up/down


def test_returning_to_the_same_side_is_not_a_crossing():
    t, y = dithering_path(end_side=-1, amplitude=0.9)
    assert crossing_events(t, y, [0.0], Ly=100.0, hysteresis=1.0).count == 0


def test_excursions_larger_than_the_band_are_counted():
    t = np.linspace(0, 10, 2001)
    y = 3.0 * np.sin(2 * np.pi * t)  # 10 periods, amplitude 3 > h = 1, starts and ends inside the band
    ev = crossing_events(t, y, [0.0], Ly=100.0, hysteresis=1.0)
    # passages at t = 0.5, 1, ..., 9.5; the final return to y = 0 at t = 10 is never confirmed
    assert ev.count == 19
    assert np.allclose(ev.times, 0.5 * np.arange(1, 20), atol=1e-3)
    assert ev.direction.tolist() == [-1, 1] * 9 + [-1]


def test_periodic_wrap_two_layers():
    Ly, layers = 20.0, (-5.0, 5.0)
    t = np.arange(301, dtype=float)
    y_unwrapped = -9.0 + 0.2 * t  # travels 60 = 3 Ly upward
    y_wrapped = (y_unwrapped + 10.0) % Ly - 10.0
    ev = crossing_events(t, y_wrapped, layers, Ly, hysteresis=1.0)
    assert ev.count == 6
    assert ev.layer.tolist() == [0, 1, 0, 1, 0, 1]
    assert np.all(ev.direction == 1)
    expected = [(-5 + 9) / 0.2, (5 + 9) / 0.2, (15 + 9) / 0.2, (25 + 9) / 0.2, (35 + 9) / 0.2, (45 + 9) / 0.2]
    assert np.allclose(ev.times, expected)
    same = crossing_events(t, y_unwrapped, layers, Ly, hysteresis=1.0)
    assert np.array_equal(same.index, ev.index) and np.allclose(same.times, ev.times)


def test_jump_of_the_minimum_image_distance_is_not_a_crossing():
    Ly = 20.0
    t = np.arange(200, dtype=float)
    # dithers across the box edge y = +-10, the point opposite the layer at y = 0
    y = 10.0 + 1.5 * np.sin(0.7 * t)
    y_wrapped = (y + 10.0) % Ly - 10.0
    assert np.any(np.diff(np.sign(y_wrapped)) != 0)
    assert crossing_events(t, y_wrapped, [0.0], Ly, hysteresis=1.0).count == 0
    # same for a layer that is not at the box centre (opposite point at y = -7)
    y2 = -7.0 + 2.0 * np.sin(0.3 * t)
    assert crossing_events(t, (y2 + 10.0) % Ly - 10.0, [3.0], Ly, hysteresis=1.0).count == 0


def test_cumulative_crossings_consistent_with_events():
    t = np.linspace(0, 10, 2001)
    y = 3.0 * np.sin(2 * np.pi * t) - 4.0
    ev = crossing_events(t, y, [-4.0, 4.0], Ly=16.0, hysteresis=1.0)
    cum = cumulative_crossings(ev, t.size)
    assert cum.shape == t.shape and cum.dtype.kind == "i"
    assert cum[-1] == ev.count and np.all(np.diff(cum) >= 0)
    assert np.all(np.diff(cum)[ev.index - 1] >= 1)
    assert cumulative_crossings(ev, t.size, layer=1)[-1] == 0
    assert cumulative_crossings(ev, t.size, direction=1)[-1] == ev.select(direction=1).count
    with pytest.raises(ValueError):
        cumulative_crossings(ev, 5)


def test_crossing_input_validation():
    t = np.arange(5.0)
    with pytest.raises(ValueError):
        crossing_events(t, t[:-1], [0.0], 10.0, 1.0)
    with pytest.raises(ValueError):
        crossing_events(t, t, [0.0], 10.0, 5.0)
    with pytest.raises(ValueError):
        crossing_events(t, np.array([0, 1, np.nan, 2, 3]), [0.0], 10.0, 1.0)
    assert crossing_events([0.0], [1.0], [0.0], 10.0, 1.0).count == 0


def test_crossing_events_for_uses_run_layers():
    cfg = make_cfg()
    t = np.arange(161, dtype=float)
    y = (-7.9 + 0.1 * t + 8.0) % cfg.Ly - 8.0  # from -7.9 to 8.1: crosses y1 = -4 and y2 = 4
    x = np.zeros((t.size, 3))
    x[:, 1] = y
    traj = Trajectory(t=t, x=x, u=np.zeros((t.size, 3)))
    ev = crossing_events_for(traj, cfg)
    assert ev.count == 2 and ev.layer.tolist() == [0, 1]
    assert ev.layer_positions == (-4.0, 4.0) and ev.hysteresis == cfg.profile.a
    sin_cfg = replace(cfg, profile=replace(cfg.profile, kind="sin"))
    with pytest.raises(ValueError):
        crossing_events_for(traj, sin_cfg)


# ===================================================== trajectory diagnostics
def gyration(n=64, u_perp=80.0, u_par=30.0, B0=0.1):
    phase = np.linspace(0, 4 * np.pi, n)
    u = np.stack([u_perp * np.cos(phase), u_perp * np.sin(phase), np.full(n, u_par)], axis=1)
    B = np.tile([0.0, 0.0, B0], (n, 1))
    return np.linspace(0, 1, n), u, B


def test_diagnostics_uniform_field_no_electric_field():
    cfg = make_cfg()
    t, u, B = gyration()
    traj = Trajectory(t=t, x=np.zeros_like(u), u=u, B=B, cE=np.zeros_like(u))
    d = trajectory_diagnostics(traj, cfg)
    gamma = np.sqrt(1 + (80.0**2 + 30.0**2) / C**2)
    assert np.allclose(d["gamma"], gamma)
    assert np.allclose(np.linalg.norm(d["v"], axis=1), np.sqrt(80.0**2 + 30.0**2) / gamma)
    assert np.allclose(d["E_kin_per_mass"], (gamma - 1) * C**2)
    assert np.allclose(d["E_kin"], cfg.m_cr * (gamma - 1) * C**2)
    assert np.allclose(d["u_perp"], 80.0) and np.allclose(d["u_par"], 30.0)
    assert np.allclose(d["r_g"], 80.0 / (Q_MC * 0.1))
    assert np.allclose(d["Omega"], Q_MC * 0.1 / gamma)
    assert np.allclose(d["mu_invariant"], 80.0**2 / (2 * 0.1))
    assert np.allclose(d["pitch_lab"], 30.0 / np.hypot(80.0, 30.0))
    assert np.allclose(d["pitch_drift"], d["pitch_lab"])  # E = 0: frames coincide
    assert np.allclose(d["power_per_mass"], 0.0) and np.allclose(d["dgamma_dt"], 0.0)
    assert np.all(np.isnan(d["cos_v_cE"]))  # undefined for cE = 0, no warning
    assert np.allclose(d["dgamma_dt_fd"], 0.0)


def test_drift_frame_pitch_and_power_sign():
    cfg = make_cfg()
    B0, w = 0.1, 30.0  # gas flows along x with speed w, B along z
    Gw = 1 / np.sqrt(1 - (w / C) ** 2)
    u_par_drift = 40.0  # particle moving exactly along B in the drift frame
    g_drift = np.sqrt(1 + (u_par_drift / C) ** 2)
    u_lab = np.array([[Gw * g_drift * w, 0.0, u_par_drift]])  # inverse boost along x
    B = np.array([[0.0, 0.0, B0]])
    U = np.array([[w, 0.0, 0.0]])
    cE = -np.cross(U, B)
    traj = Trajectory(t=np.array([0.0]), x=np.zeros((1, 3)), u=u_lab, B=B, cE=cE)
    d = trajectory_diagnostics(traj, cfg)
    assert d["pitch_drift"][0] == pytest.approx(1.0, abs=1e-12)
    assert d["pitch_lab"][0] < 0.9
    assert np.isnan(d["dgamma_dt_fd"][0])  # single sample
    # moving along +cE gains energy
    u_gain = np.array([[0.0, 60.0, 0.0]])
    d2 = trajectory_diagnostics(Trajectory(t=np.array([0.0]), x=np.zeros((1, 3)), u=u_gain, B=B, cE=cE), cfg)
    expected = Q_MC * (60.0 / np.sqrt(1 + (60.0 / C) ** 2)) * w * B0
    assert d2["power_per_mass"][0] == pytest.approx(expected)
    assert d2["dgamma_dt"][0] == pytest.approx(expected / C**2)
    assert d2["cos_v_cE"][0] == pytest.approx(1.0)
    # drift speed >= c cannot be boosted: NaN instead of an exception
    weak_B = np.array([[0.0, 0.0, 1e-6]])
    with pytest.warns(RuntimeWarning, match="boost"):
        d3 = trajectory_diagnostics(Trajectory(t=np.array([0.0]), x=np.zeros((1, 3)), u=u_gain, B=weak_B, cE=cE),
                                    cfg)
    assert np.isnan(d3["pitch_drift"][0])
    # mixed samples: only the invalid ones are NaN, the rest equal the exact drift-frame pitch
    n = 5
    B5 = np.tile([0.0, 0.0, B0], (n, 1))
    B5[2] = [0.0, 0.0, 1e-6]
    B5[4] = 0.0
    u5 = np.tile(u_lab, (n, 1))
    cE5 = np.tile(cE, (n, 1))  # the B0-level motional field everywhere: sample 2 drifts faster than c
    cE5[4] = [0.0, 1.0, 0.0]   # cE without B: drift undefined
    with pytest.warns(RuntimeWarning, match="2 of 5"):
        d5 = trajectory_diagnostics(Trajectory(t=np.arange(n, dtype=float), x=np.zeros((n, 3)), u=u5, B=B5, cE=cE5),
                                    cfg)
    assert np.isnan(d5["pitch_drift"][[2, 4]]).all()
    np.testing.assert_allclose(d5["pitch_drift"][[0, 1, 3]], 1.0, atol=1e-12)


def test_diagnostics_seven_column_trajectory_and_energy_rate():
    cfg = make_cfg()
    t = np.linspace(0, 2, 41)
    gamma = 1.5 + 0.25 * t
    u = np.zeros((t.size, 3))
    u[:, 0] = rel.momentum_mag_from_gamma(gamma, C)
    d = trajectory_diagnostics(Trajectory(t=t, x=np.zeros_like(u), u=u), cfg)
    assert np.allclose(d["gamma"], gamma)
    assert np.allclose(d["dgamma_dt_fd"], 0.25)
    for key in ("u_par", "u_perp", "r_g", "Omega", "mu_invariant", "pitch_lab", "pitch_drift", "cos_v_cE",
                "power_per_mass", "dgamma_dt"):
        assert d[key] is None
    with pytest.raises(ValueError, match="no <particles>"):
        trajectory_diagnostics(Trajectory(t=t, x=np.zeros_like(u), u=u), replace(cfg, c=None))


def test_diagnostics_rejects_lists_of_trajectories():
    cfg = make_cfg()
    t, u, B = gyration(n=8)
    traj = Trajectory(t=t, x=np.zeros_like(u), u=u, B=B, cE=np.zeros_like(u))
    for bad in ([traj], (traj, traj), {"t": t, "u": u}):
        with pytest.raises(TypeError, match="pass one Trajectory; loop over lists"):
            trajectory_diagnostics(bad, cfg)


def test_diagnostics_docs_state_sampling_caveats():
    import shearpic.physics.particles as mod

    assert "NGP" in trajectory_diagnostics.__doc__ and "energy" in trajectory_diagnostics.__doc__
    assert "TSC" in mod.__doc__ and "aliased" in mod.__doc__


# =============================================================== grid sampling
def test_ngp_picks_the_containing_cell_and_wraps():
    cfg = make_cfg()
    field = np.arange(8 * 16, dtype=float).reshape(8, 16)
    x = np.array([[-2.0, -8.0, 0.0], [-1.49, -7.01, 0.2], [2.0, 8.0, 0.0], [1.99, 0.01, 0.0]])
    vals = sample_field_ngp(field, x, cfg)
    assert vals.tolist() == [field[0, 0], field[1, 0], field[0, 0], field[7, 8]]
    assert sample_field_ngp(field[:, :, None], x, cfg).tolist() == vals.tolist()
    with pytest.raises(ValueError):
        sample_field_ngp(np.zeros((16, 8)), x, cfg)


@pytest.mark.parametrize("nx", [(8, 16, 1), (8, 16, 6)])
def test_tsc_reproduces_constant_and_linear_fields(nx):
    bounds = ((-2.0, 2.0), (-8.0, 8.0), (-1.5, 1.5))
    cfg = make_cfg(nx, bounds)
    rng = np.random.default_rng(0)
    const = np.full(nx, 3.25, dtype=np.float32)
    x = np.stack([rng.uniform(lo, hi, 500) for lo, hi in bounds], axis=1)
    assert np.allclose(sample_field_tsc(const, x, cfg), 3.25, rtol=1e-12)
    # linear field sampled away from the periodic seam (> 1.5 cells from the boundary)
    X, Y, Z = np.meshgrid(cfg.centers(0), cfg.centers(1), cfg.centers(2), indexing="ij")
    lin = 0.7 * X - 0.3 * Y + (0.2 * Z if nx[2] > 1 else 0.0) + 1.0
    margin = [1.5 * d if n > 1 else 0.0 for d, n in zip(cfg.dx, nx)]
    xin = np.stack([rng.uniform(lo + m, hi - m, 500) for (lo, hi), m in zip(bounds, margin)], axis=1)
    expected = 0.7 * xin[:, 0] - 0.3 * xin[:, 1] + (0.2 * xin[:, 2] if nx[2] > 1 else 0.0) + 1.0
    assert np.allclose(sample_field_tsc(lin, xin, cfg), expected, rtol=0, atol=1e-12)


def test_tsc_is_periodic_and_smooths_like_the_pusher():
    cfg = make_cfg()
    rng = np.random.default_rng(1)
    field = rng.normal(size=(8, 16))
    x = np.stack([rng.uniform(-2, 2, 50), rng.uniform(-8, 8, 50), np.zeros(50)], axis=1)
    shifted = x + np.array([cfg.Lx, -2 * cfg.Ly, 0.3])
    assert np.allclose(sample_field_tsc(field, x, cfg), sample_field_tsc(field, shifted, cfg), atol=1e-12)
    # at the centre of cell (0, j) the weights are 1/8, 3/4, 1/8 per axis, wrapping to cell 7
    j = 5
    xc = np.array([[cfg.centers(0)[0], cfg.centers(1)[j], 0.0]])
    wx = {7: 0.125, 0: 0.75, 1: 0.125}
    wy = {j - 1: 0.125, j: 0.75, j + 1: 0.125}
    expected = sum(wx[i] * wy[k] * field[i, k] for i in wx for k in wy)
    assert sample_field_tsc(field, xc, cfg)[0] == pytest.approx(expected)
    # output keeps the leading shape of the positions
    assert sample_field_tsc(field, x.reshape(5, 10, 3), cfg).shape == (5, 10)
