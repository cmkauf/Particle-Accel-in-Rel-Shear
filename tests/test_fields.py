"""Unit tests for shearpic.physics.fields (periodic derivative operators and grid diagnostics)."""

from __future__ import annotations

import numpy as np
import pytest

from shearpic.config import RunConfig
from shearpic.io.athinput import parse_athinput
from shearpic.physics import fields as F
from shearpic.physics.relativity import drift_velocity


def make_cfg(nx=(16, 32, 1), bounds=((-2.0, 2.0), (-8.0, 8.0), (-0.5, 0.5)), y1=-4.0, y2=4.0) -> RunConfig:
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
speed_of_light = 50.0
charge_over_mass_over_c = 200.0
<problem>
shear_strength = 1.0
y1 = {y1}
y2 = {y2}
vp_par = 50.0
cr_mass = 0.0005
"""
    return RunConfig.from_athinput(parse_athinput(text))


def periodic_grid(n, L=2 * np.pi):
    dx = L / n
    return np.arange(n) * dx, dx


# ------------------------------------------------------------------ derivatives
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_ddx_second_order_convergence(axis):
    errors = []
    for n in (16, 32, 64):
        x, dx = periodic_grid(n)
        shape = [1, 1, 1]
        shape[axis] = n
        f = np.broadcast_to(np.sin(2 * x).reshape(shape), [n if i == axis else 3 for i in range(3)])
        exact = np.broadcast_to((2 * np.cos(2 * x)).reshape(shape), f.shape)
        errors.append(np.abs(F.ddx(f, dx, axis) - exact).max())
    ratios = np.array(errors[:-1]) / np.array(errors[1:])
    np.testing.assert_allclose(ratios, 4.0, rtol=0.03)


def test_ddx_exact_discrete_derivative_and_edges():
    n = 64
    x, dx = periodic_grid(n)
    k = 2
    f = np.cos(k * x)
    d = F.ddx(f, dx, 0)
    # central differences of cos(kx) are exactly -(sin(k dx)/dx) sin(kx), including the edges
    np.testing.assert_allclose(d, -np.sin(k * dx) / dx * np.sin(k * x), rtol=0, atol=1e-12)
    exact = -k * np.sin(k * x)
    err = np.abs(d - exact)
    assert err[[0, -1]].max() <= err.max() + 1e-12  # edges are no worse than the interior
    # np.gradient falls back to first-order one-sided differences at the ends: O(dx) errors there
    g_err = np.abs(np.gradient(f, dx) - exact)
    assert g_err[[0, -1]].max() > 10 * err.max()
    np.testing.assert_allclose(g_err[1:-1], err[1:-1], atol=1e-12)  # identical in the interior


def test_ddx_length_one_axis_is_zero():
    f = np.random.default_rng(0).normal(size=(4, 5, 1))
    assert np.all(F.ddx(f, 0.1, 2) == 0) and F.ddx(f, 0.1, 2).shape == f.shape
    f2 = f[:, :, 0]
    assert np.all(F.ddx(f2, 0.1, 2) == 0)  # axis beyond ndim: 2D array treated as z-invariant


def _grid2d(nx=24, ny=40, Lx=2 * np.pi, Ly=4 * np.pi):
    dx, dy = Lx / nx, Ly / ny
    x = (np.arange(nx) * dx)[:, None, None]
    y = (np.arange(ny) * dy)[None, :, None]
    return x, y, (dx, dy, 1.0)


def test_curl_known_25d_field_including_Bz():
    x, y, dxs = _grid2d()
    dx, dy, _ = dxs
    kx, ky = 2.0, 1.5
    zeros = np.zeros((x.size, y.size, 1))
    Fx = np.sin(ky * y) + zeros
    Fy = np.cos(kx * x) + zeros
    Fz = np.sin(kx * x) * np.cos(ky * y)
    cx, cy, cz = F.curl((Fx, Fy, Fz), dxs)
    sx, sy = np.sin(kx * dx) / dx, np.sin(ky * dy) / dy  # exact discrete wavenumbers
    np.testing.assert_allclose(cx, -sy * np.sin(kx * x) * np.sin(ky * y), atol=1e-12)   # dFz/dy
    np.testing.assert_allclose(cy, -sx * np.cos(kx * x) * np.cos(ky * y), atol=1e-12)   # -dFz/dx
    np.testing.assert_allclose(cz, -sx * np.sin(kx * x) - sy * np.cos(ky * y), atol=1e-12)
    # and close to the continuum answer
    np.testing.assert_allclose(cz, -kx * np.sin(kx * x) - ky * np.cos(ky * y), atol=0.2)
    np.testing.assert_allclose(F.current((Fx, Fy, Fz), dxs)[0], cx)
    np.testing.assert_allclose(F.vorticity((Fx, Fy, Fz), dxs)[2], cz)


def test_divergence_of_curl_vanishes_3d():
    rng = np.random.default_rng(3)
    A = tuple(rng.normal(size=(10, 12, 8)) for _ in range(3))
    dxs = (0.3, 0.2, 0.5)
    div = F.divergence(F.curl(A, dxs), dxs)
    assert np.abs(div).max() < 1e-12 * max(np.abs(a).max() for a in A) / min(dxs) ** 2
    g = F.gradient(A[0], dxs)
    assert np.abs(np.stack(F.curl(g, dxs))).max() < 1e-11  # curl grad = 0 as well


def test_divergence_known_field():
    x, y, dxs = _grid2d()
    Fx = np.sin(x) + 0 * y
    Fy = np.cos(0.5 * y) + 0 * x
    div = F.divergence((Fx, Fy, np.zeros_like(Fx)), dxs)
    expected = np.sin(dxs[0]) / dxs[0] * np.cos(x) - np.sin(0.5 * dxs[1]) / dxs[1] * np.sin(0.5 * y)
    np.testing.assert_allclose(div, expected, atol=1e-12)


def test_spacing_from_runconfig_and_two_component_vectors():
    cfg = make_cfg(nx=(16, 32, 1))
    rng = np.random.default_rng(11)
    vx, vy = rng.normal(size=cfg.nx), rng.normal(size=cfg.nx)
    zeros = np.zeros(cfg.nx)
    ref = F.curl((vx, vy, zeros), cfg.dx)
    for spacing in (cfg, cfg.dx, list(cfg.dx), cfg.dx[:2]):
        for got, want in zip(F.curl((vx, vy), spacing), ref):
            np.testing.assert_array_equal(got, want)
            assert got.shape == cfg.nx
        np.testing.assert_array_equal(F.vorticity((vx, vy), spacing)[2], ref[2])
        np.testing.assert_array_equal(F.current((vx, vy), spacing)[2], ref[2])
        np.testing.assert_array_equal(F.divergence((vx, vy), spacing), F.divergence((vx, vy, zeros), cfg.dx))
        for got, want in zip(F.gradient(vx, spacing), F.gradient(vx, cfg.dx)):
            np.testing.assert_array_equal(got, want)
    # ddx takes a float, a sequence or the config
    np.testing.assert_array_equal(F.ddx(vx, cfg, 1), F.ddx(vx, cfg.dx[1], 1))
    np.testing.assert_array_equal(F.ddx(vx, cfg.dx, 0), F.ddx(vx, cfg.dx[0], 0))
    with pytest.raises(ValueError, match="2 or 3 components"):
        F.curl((vx,), cfg)
    with pytest.raises(TypeError, match="grid spacing"):
        F.gradient(vx, 0.5)


def test_vorticity_z_and_current_z_match_curl():
    x, y, dxs = _grid2d()
    vx = np.sin(1.5 * y) + 0 * x
    vy = np.cos(2.0 * x) + 0 * y
    full = F.curl((vx, vy, np.zeros_like(vx)), dxs)[2]
    wz = F.vorticity_z(vx, vy, dxs)
    assert wz.shape == vx.shape
    np.testing.assert_array_equal(wz, full)
    np.testing.assert_array_equal(F.current_z(vx, vy, dxs), full)
    # 2D (x, y) arrays work as well
    np.testing.assert_allclose(F.vorticity_z(vx[:, :, 0], vy[:, :, 0], dxs[:2]), full[:, :, 0])
    cfg = make_cfg(nx=(16, 32, 1))
    X, Y = np.meshgrid(cfg.centers(0), cfg.centers(1), indexing="ij")
    np.testing.assert_allclose(F.current_z(np.sin(np.pi * Y / 8)[:, :, None], np.zeros(cfg.nx), cfg),
                               -F.ddx(np.sin(np.pi * Y / 8)[:, :, None], cfg, 1))


# ------------------------------------------------------------------ algebraic ops
def test_magnitude_and_motional_cE_sign():
    assert F.magnitude(np.array(3.0), np.array(4.0)) == 5.0
    one, zero = np.ones(2), np.zeros(2)
    cE = F.motional_cE((one, zero, zero), (zero, one, zero))  # -x_hat cross y_hat = -z_hat
    np.testing.assert_array_equal(np.stack(cE), [zero, zero, -one])
    rng = np.random.default_rng(7)
    v = rng.normal(size=(3, 6))
    B = rng.normal(size=(3, 6))
    cE = np.stack(F.motional_cE(tuple(v), tuple(B)))
    np.testing.assert_allclose(cE.T, -np.cross(v.T, B.T), atol=1e-14)
    # the E x B drift of the motional field is U_perp
    w = drift_velocity(cE.T, B.T)
    v_perp = v.T - (np.sum(v * B, 0) / np.sum(B * B, 0))[:, None] * B.T
    np.testing.assert_allclose(w, v_perp, atol=1e-12)


def test_x_average_and_fluctuation():
    rng = np.random.default_rng(1)
    f = rng.normal(size=(6, 5, 3))
    np.testing.assert_allclose(F.x_average(f), f.mean(axis=(0, 2)))
    df = F.fluctuation(f)
    assert df.shape == f.shape
    np.testing.assert_allclose(df.mean(axis=(0, 2)), 0.0, atol=1e-14)
    np.testing.assert_allclose(f - df, np.broadcast_to(F.x_average(f)[None, :, None], f.shape))
    f2 = f[:, :, 0]
    np.testing.assert_allclose(F.x_average(f2), f2.mean(axis=0))
    np.testing.assert_allclose(F.fluctuation(f2).mean(axis=0), 0.0, atol=1e-14)
    # a pure function of y has no fluctuation
    prof = np.broadcast_to(np.arange(5.0)[None, :, None], (6, 5, 3))
    assert np.abs(F.fluctuation(prof)).max() == 0


def test_stresses():
    x, y, _ = _grid2d(nx=16, ny=8)
    base = np.broadcast_to(np.cos(y), (16, 8, 1))
    vx = base + np.sin(x) * np.ones_like(y)
    vy = np.sin(x) * np.ones_like(y)
    rho = np.full(vx.shape, 2.0)
    np.testing.assert_allclose(F.reynolds_stress_xy(rho, vx, vy), 2.0 * np.sin(x) ** 2 * np.ones_like(y), atol=1e-12)
    np.testing.assert_allclose(F.maxwell_stress_xy(vx, vy), -np.sin(x) ** 2 * np.ones_like(y), atol=1e-12)


def test_energy_densities_and_integrate():
    cfg = make_cfg()
    rng = np.random.default_rng(5)
    rho = rng.uniform(0.5, 1.5, cfg.nx)
    v = tuple(rng.normal(size=cfg.nx) for _ in range(3))
    B = tuple(rng.normal(size=cfg.nx) for _ in range(3))
    ek = F.kinetic_energy_density(rho, v)
    eb = F.magnetic_energy_density(B)
    np.testing.assert_allclose(ek, 0.5 * rho * (v[0] ** 2 + v[1] ** 2 + v[2] ** 2))
    np.testing.assert_allclose(eb, 0.5 * (B[0] ** 2 + B[1] ** 2 + B[2] ** 2))
    assert cfg.dV == pytest.approx(4.0 / 16 * 16.0 / 32 * 1.0)
    assert F.integrate(np.ones(cfg.nx), cfg) == pytest.approx(cfg.V)
    assert F.integrate(ek, cfg) == pytest.approx(ek.sum() * cfg.dV)
    # hst-style component energy: 1-KE = sum 0.5 rho v1^2 dV
    ke1 = F.integrate(F.kinetic_energy_density(rho, (v[0],)), cfg)
    assert ke1 == pytest.approx(np.sum(0.5 * rho * v[0] ** 2) * cfg.dV)


def test_cr_moments():
    cfg = make_cfg()
    n = np.full(cfg.nx, 3.0)
    u = (np.full(cfg.nx, 2.0), np.zeros(cfg.nx), np.full(cfg.nx, -1.0))
    np.testing.assert_allclose(F.cr_mass_density(n, cfg), 3.0 * cfg.m_cr)
    mom = F.cr_momentum_density(n, u, cfg)
    np.testing.assert_allclose(mom[0], 6.0 * cfg.m_cr)
    np.testing.assert_allclose(mom[2], -3.0 * cfg.m_cr)
    # total particle count sum(np dV) equals N when np is uniform at N/V
    assert F.integrate(np.full(cfg.nx, cfg.n_par / cfg.V), cfg) == pytest.approx(cfg.n_par)


# -------------------------------------------------------------------- layer mask
def test_layer_mask_periodic():
    # layers at y = -8 (= +8 periodically) and 0 in a box y in [-8, 8)
    cfg = make_cfg(y1=-8.0, y2=0.0)
    y = cfg.centers(1)
    width = cfg.profile.layer_width(0.9)
    assert width == pytest.approx(2 * np.arctanh(0.9))
    mask = F.layer_mask(cfg, cutoff=0.9)
    assert mask.shape == y.shape and mask.dtype == bool
    near_bottom = np.abs(y - (-8.0)) < width / 2
    near_top = np.abs(y - 8.0) < width / 2   # periodic image of the y1 layer
    near_mid = np.abs(y) < width / 2
    np.testing.assert_array_equal(mask, near_bottom | near_top | near_mid)
    assert mask[0] and mask[-1] and mask[len(y) // 2]
    assert not mask[len(y) // 4]
    # explicit coordinates and a smaller cutoff (narrower layers)
    yy = np.array([-7.9, 7.9, 0.2, 4.0, -4.0])
    np.testing.assert_array_equal(F.layer_mask(cfg, 0.9, y=yy), [True, True, True, False, False])
    assert F.layer_mask(cfg, 0.1).sum() < mask.sum()
    with pytest.raises(ValueError):
        F.layer_mask(cfg, cutoff=1.0)
