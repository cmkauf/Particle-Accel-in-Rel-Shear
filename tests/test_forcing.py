"""Unit tests for shearpic.physics.forcing (reference shear profile and stirring force)."""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from shearpic.config import RunConfig
from shearpic.io.athinput import parse_athinput
from shearpic.physics.forcing import ShearProfile, stir_acceleration, stir_power, stir_power_density, x_mean_velocity


def make_cfg(problem: str = "", nx=(8, 64, 1), ybounds=(-16.0, 16.0)) -> RunConfig:
    text = f"""
<job>
problem_id = synth
<mesh>
nx1 = {nx[0]}
x1min = -2.0
x1max = 2.0
nx2 = {nx[1]}
x2min = {ybounds[0]}
x2max = {ybounds[1]}
nx3 = {nx[2]}
x3min = -0.5
x3max = 0.5
<problem>
iprob = 0
shear_strength = 1.5
y1 = -8.0
y2 = 8.0
{problem}
"""
    return RunConfig.from_athinput(parse_athinput(text))


PROFILES = [
    ShearProfile("double_tanh", amplitude=1.7, a=1.0, y1=-5.0, y2=5.0),
    ShearProfile("single_tanh", amplitude=0.8, a=0.7),
    ShearProfile("sin", amplitude=2.0, Ly=20.0, n=2),
]


@pytest.mark.parametrize("prof", PROFILES, ids=lambda p: p.kind)
def test_d2U_matches_finite_differences(prof):
    y = np.linspace(-9.0, 9.0, 721)
    h = 1e-3
    fd2 = (prof.U(y + h) - 2 * prof.U(y) + prof.U(y - h)) / h**2
    np.testing.assert_allclose(prof.d2U(y), fd2, rtol=0, atol=1e-5 * max(1.0, np.abs(fd2).max()))
    assert np.abs(prof.d2U(y)).max() > 0.1  # non-trivial
    np.testing.assert_allclose(prof.viscous_compensation(y, 0.3), -0.3 * prof.d2U(y))


def test_profile_values():
    S = 1.7
    dt = ShearProfile("double_tanh", amplitude=S, a=1.0, y1=-20.0, y2=20.0)
    assert dt.U(0.0) == pytest.approx(S, rel=1e-12)            # jet between the layers
    assert dt.U(-60.0) == pytest.approx(-S, rel=1e-12)         # wind outside
    assert dt.U(60.0) == pytest.approx(-S, rel=1e-12)
    assert dt.U(-20.0) == pytest.approx(0.0, abs=1e-12)        # zero at the layer centres
    assert dt.U(20.0) == pytest.approx(0.0, abs=1e-12)
    assert np.all(np.abs(dt.U(np.linspace(-100, 100, 1001))) <= S * (1 + 1e-12))
    st = ShearProfile("single_tanh", amplitude=2.0, a=0.5)
    assert st.U(0.25) == pytest.approx(2.0 * math.tanh(0.5))
    sn = ShearProfile("sin", amplitude=3.0, Ly=10.0, n=1)
    assert sn.U(2.5) == pytest.approx(3.0)
    assert dt.layer_positions == (-20.0, 20.0) and st.layer_positions == (0.0,) and sn.layer_positions == ()
    # |U| = 0.9 S exactly at the layer half-width
    half = 0.5 * dt.layer_width(0.9)
    assert abs(dt.U(20.0 - half)) == pytest.approx(0.9 * S, rel=1e-6)
    with pytest.raises(ValueError):
        ShearProfile("cosh")
    with pytest.raises(dataclasses.FrozenInstanceError):
        dt.amplitude = 2.0  # type: ignore[misc]


def test_on_cpp_grid_uses_lower_faces():
    prof = ShearProfile("double_tanh", amplitude=1.0, y1=-8.0, y2=8.0)
    edges = np.linspace(-16.0, 16.0, 65)
    vals = prof.on_cpp_grid(edges)
    assert vals.shape == (64,)
    np.testing.assert_array_equal(vals, prof.U(edges[:-1]))
    centres = 0.5 * (edges[:-1] + edges[1:])
    assert np.abs(vals - prof.U(centres)).max() > 0.1  # the half-cell offset matters inside the layers


def test_x_mean_velocity():
    vx = np.random.default_rng(0).normal(size=(4, 6, 3))
    np.testing.assert_allclose(x_mean_velocity(vx), vx.mean(axis=(0, 2)))
    np.testing.assert_allclose(x_mean_velocity(vx[:, :, 0]), vx[:, :, 0].mean(axis=0))


def test_x_mean_velocity_delegates_to_fields_x_average(monkeypatch):
    import shearpic.physics.forcing as forcing_mod

    calls = []
    monkeypatch.setattr(forcing_mod, "x_average", lambda f: calls.append(f) or np.zeros(3))
    x_mean_velocity(np.ones((2, 3, 1)))
    assert len(calls) == 1


def _vx_from_profile(cfg, factor=1.0, faces=True, noise=0.0):
    e = cfg.edges(1)
    y = e[:-1] if faces else 0.5 * (e[:-1] + e[1:])
    prof = factor * cfg.profile.U(y)
    vx = np.broadcast_to(prof[None, :, None], cfg.nx).copy()
    if noise:
        pert = np.random.default_rng(1).normal(size=cfg.nx)
        vx += noise * (pert - pert.mean(axis=(0, 2), keepdims=True))  # zero x-mean perturbation
    return vx


def test_stir_acceleration_zero_at_reference_profile():
    cfg = make_cfg("tau = 0.5\nnu_iso = 0.0")
    assert cfg.tau == 0.5 and cfg.profile.kind == "double_tanh" and cfg.shear_amplitude == 1.5
    vx = _vx_from_profile(cfg, noise=0.3)  # fluctuations do not change the x-mean
    np.testing.assert_allclose(stir_acceleration(vx, cfg), 0.0, atol=1e-13)
    # sampled at cell centres the same field is *not* the reference
    assert np.abs(stir_acceleration(vx, cfg, cpp_faces=False)).max() > 0.1


def test_stir_acceleration_relaxation_and_viscous_term():
    cfg = make_cfg("tau = 0.25\nnu_iso = 0.02")
    y = cfg.edges(1)[:-1]
    vx = _vx_from_profile(cfg, factor=0.6)
    expected = (0.4 * cfg.profile.U(y)) / 0.25 + 0.02 * (-cfg.profile.d2U(y))
    np.testing.assert_allclose(stir_acceleration(vx, cfg), expected, rtol=1e-12, atol=1e-14)
    # at the reference profile only the viscous compensation remains
    np.testing.assert_allclose(stir_acceleration(_vx_from_profile(cfg), cfg), -0.02 * cfg.profile.d2U(y), atol=1e-13)


def test_tau_handling():
    for problem in ("", "tau = -99.9", "tau = 0.0"):
        cfg = make_cfg(problem)
        assert cfg.tau is None
        a = stir_acceleration(np.ones(cfg.nx), cfg)
        assert a.shape == (cfg.nx[1],) and np.all(a == 0)
        assert stir_power(np.ones(cfg.nx), np.ones(cfg.nx), cfg) == 0.0
    cfg = make_cfg("tau = 0.5\nstir = 1")
    with pytest.raises(NotImplementedError):
        stir_acceleration(np.ones(cfg.nx), cfg)


def test_stir_power_sign_and_integral():
    cfg = make_cfg("tau = 0.5")
    rho = np.full(cfg.nx, 1.3)
    slow = _vx_from_profile(cfg, factor=0.5)   # flow weaker than the reference: stirring does positive work
    fast = _vx_from_profile(cfg, factor=1.5)   # flow stronger: stirring removes energy
    P_slow = stir_power(rho, slow, cfg)
    P_fast = stir_power(rho, fast, cfg)
    assert P_slow > 0 and P_fast < 0
    y = cfg.edges(1)[:-1]
    U2 = cfg.profile.U(y) ** 2
    # rho vx a = rho (f U) ((1 - f) U / tau), summed over x cells and times dV
    expected = lambda f: 1.3 * f * (1 - f) / 0.5 * U2.sum() * cfg.nx[0] * cfg.dV  # noqa: E731
    assert P_slow == pytest.approx(expected(0.5), rel=1e-12)
    assert P_fast == pytest.approx(expected(1.5), rel=1e-12)
    dens = stir_power_density(rho, slow, cfg)
    assert dens.shape == cfg.nx and np.all(dens >= -1e-15)
    assert dens.sum() * cfg.dV == pytest.approx(P_slow)
    # 2D (squeezed) arrays give the same total
    assert stir_power(rho[:, :, 0], slow[:, :, 0], cfg) == pytest.approx(P_slow, rel=1e-12)
