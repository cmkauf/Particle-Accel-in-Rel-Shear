"""Unit tests for shearpic.physics.relativity.

Besides algebraic identities, two tests integrate du/dt = q_mc (cE + v x B), v = u / gamma, with RK4
and check ``dgamma_dt`` and ``magnetic_moment_per_mass`` along the numerical orbit.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from shearpic.physics import relativity as rel

C = 50.0
RNG = np.random.default_rng(1234)


# ------------------------------------------------------------------ kinematics
def test_gamma_velocity_momentum_round_trip():
    u = RNG.normal(0.0, 3 * C, size=(200, 3))
    g = rel.lorentz_factor(u, C)
    v = rel.velocity(u, C)
    assert np.all(rel.norm(v) < C)
    np.testing.assert_allclose(rel.momentum_from_velocity(v, C), u, rtol=1e-9)
    np.testing.assert_allclose(g, 1.0 / np.sqrt(1.0 - rel.dot(v, v) / C**2), rtol=1e-9)
    np.testing.assert_allclose(rel.lorentz_factor_mag(rel.norm(u), C), g, rtol=1e-14)
    np.testing.assert_allclose(rel.momentum_mag_from_gamma(g, C), rel.norm(u), rtol=1e-9)
    assert rel.momentum_mag_from_gamma(1.0 - 1e-15, C) == 0.0  # no NaN from round-off


def test_velocity_never_reaches_c():
    u = np.array([[1e8 * C, 0.0, 0.0], [0.0, 0.0, 0.0]])
    v = rel.velocity(u, C)
    assert rel.norm(v)[0] <= C and rel.norm(v)[1] == 0.0
    with pytest.raises(ValueError):
        rel.momentum_from_velocity([C, 0.0, 0.0], C)


def test_kinetic_energy_limits_and_stable_form():
    # non-relativistic: (gamma-1) c^2 -> u^2 / 2
    u_small = np.array([1e-3 * C, 0.0, 0.0])
    ke = rel.kinetic_energy_per_mass(u_small, C)
    assert ke == pytest.approx(0.5 * (1e-3 * C) ** 2, rel=1e-6)
    # ultra-relativistic: (gamma-1) c^2 -> |u| c
    u_big = np.array([0.0, 1e6 * C, 0.0])
    assert rel.kinetic_energy_per_mass(u_big, C) == pytest.approx(1e6 * C * C, rel=2e-6)
    # stable form equals (gamma - 1) c^2 where the latter has no cancellation problem
    u = RNG.normal(0.0, C, size=(100, 3))
    np.testing.assert_allclose(rel.kinetic_energy_per_mass(u, C), (rel.lorentz_factor(u, C) - 1.0) * C**2, rtol=1e-10)
    np.testing.assert_allclose(rel.kinetic_energy_per_mass_mag(rel.norm(u), C), rel.kinetic_energy_per_mass(u, C),
                               rtol=1e-14)
    # tiny u: naive (gamma-1)c^2 underflows to 0, the stable form does not
    tiny = 1e-9 * C
    assert (rel.lorentz_factor_mag(tiny, C) - 1.0) * C**2 == 0.0
    assert rel.kinetic_energy_per_mass_mag(tiny, C) == pytest.approx(0.5 * tiny**2, rel=1e-12)
    # injection energy of run423: u = c -> (sqrt2 - 1) c^2 = 1035.53
    assert rel.kinetic_energy_per_mass_mag(50.0, 50.0) == pytest.approx(1035.53, rel=1e-5)


def test_vector_helpers():
    a = np.array([1.0, 2.0, 2.0])
    assert rel.norm(a) == 3.0
    assert rel.dot(a, [1.0, 0.0, 0.0]) == 1.0
    assert rel.cos_angle([1, 0, 0], [0, 5, 0]) == pytest.approx(0.0)
    assert rel.cos_angle([1, 1, 0], [-2, -2, 0]) == pytest.approx(-1.0)
    assert np.isnan(rel.cos_angle([0, 0, 0], [1, 0, 0]))


# ------------------------------------------------------------------ gyromotion
def test_parallel_perp():
    B = np.array([0.0, 0.0, 2.0])
    par, perp = rel.parallel_perp([3.0, 4.0, -5.0], B)
    assert par == pytest.approx(-5.0) and perp == pytest.approx(5.0)
    vecs = RNG.normal(size=(50, 3))
    Bs = RNG.normal(size=(50, 3))
    par, perp = rel.parallel_perp(vecs, Bs)
    np.testing.assert_allclose(par**2 + perp**2, rel.dot(vecs, vecs), rtol=1e-12)
    np.testing.assert_allclose(perp, rel.norm(np.cross(vecs, Bs)) / rel.norm(Bs), rtol=1e-9)
    assert np.all(perp >= 0)


def test_gyroradius_and_frequency_known_field():
    q_mc, B0 = 200.0, 0.1
    B = np.array([B0, 0.0, 0.0])
    u = np.array([30.0, 40.0, 0.0])  # u_par = 30, u_perp = 40, |u| = 50 = c -> gamma = sqrt2
    assert rel.gyroradius(u, B, q_mc) == pytest.approx(40.0 / (q_mc * B0))
    assert rel.gyrofrequency(u, B, q_mc, C) == pytest.approx(q_mc * B0 / math.sqrt(2.0))
    # run423 reference scales: r_g0 = vp_par / (q_mc B0) = 2.5, Omega0 = 14.142
    assert rel.gyroradius([0.0, 50.0, 0.0], [0.1, 0, 0], 200.0) == pytest.approx(2.5)
    assert rel.gyrofrequency([0.0, 50.0, 0.0], [0.1, 0, 0], 200.0, 50.0) == pytest.approx(14.142, rel=1e-4)
    # r_g = v_perp / Omega (the usual definition) for any momentum
    us = RNG.normal(0, 3 * C, (20, 3))
    _, v_perp = rel.parallel_perp(rel.velocity(us, C), B)
    np.testing.assert_allclose(rel.gyroradius(us, B, q_mc), v_perp / rel.gyrofrequency(us, B, q_mc, C), rtol=1e-10)
    assert np.isnan(rel.gyroradius(u, [0.0, 0.0, 0.0], q_mc))


def test_magnetic_moment_definition():
    B = np.array([0.0, 3.0, 4.0])  # |B| = 5
    u = np.array([2.0, 0.0, 0.0])  # fully perpendicular
    assert rel.magnetic_moment_per_mass(u, B) == pytest.approx(4.0 / 10.0)
    assert rel.magnetic_moment_per_mass(0.5 * B, B) == pytest.approx(0.0, abs=1e-12)


# ------------------------------------------------------------- frames & drifts
def test_drift_velocity_is_U_perp():
    U = RNG.normal(size=(30, 3))
    B = RNG.normal(size=(30, 3))
    cE = -np.cross(U, B)
    w = rel.drift_velocity(cE, B)
    U_perp = U - (rel.dot(U, B) / rel.dot(B, B))[:, None] * B
    np.testing.assert_allclose(w, U_perp, rtol=1e-9, atol=1e-12)
    assert np.all(np.isnan(rel.drift_velocity([1.0, 0, 0], [0.0, 0, 0])))


def test_boost_comoving_particle_is_at_rest():
    w = np.array([0.3, -0.4, 0.2]) * C
    u = rel.momentum_from_velocity(w, C)
    np.testing.assert_allclose(rel.boost(u, w, C), 0.0, atol=1e-10 * C)
    # many particles, one frame; and one particle per frame
    ws = RNG.uniform(-0.5, 0.5, (40, 3)) * C
    us = rel.momentum_from_velocity(ws, C)
    np.testing.assert_allclose(rel.boost(us, ws, C), 0.0, atol=1e-9 * C)


def test_boost_limits_and_invariants():
    u = RNG.normal(0, C, (25, 3))
    np.testing.assert_array_equal(rel.boost(u, np.zeros(3), C), u)  # w = 0
    w = np.array([1e-4 * C, 0.0, 0.0])
    np.testing.assert_allclose(rel.boost(u, w, C), u - rel.lorentz_factor(u, C)[:, None] * w,
                               rtol=1e-7, atol=1e-7)  # Galilean limit u' = u - gamma w
    w = np.array([0.0, 0.6 * C, 0.8 * 0.5 * C])
    up = rel.boost(u, w, C)
    n = w / rel.norm(w)
    np.testing.assert_allclose(np.cross(up, n), np.cross(u, n), rtol=1e-12, atol=1e-9)  # u_perp unchanged
    # the 4-momentum norm gamma'^2 - u'^2/c^2 = 1 holds by construction; check energy transformation
    Gam = 1.0 / math.sqrt(1.0 - rel.dot(w, w) / C**2)
    gamma_p = Gam * (rel.lorentz_factor(u, C) - rel.dot(u, w) / C**2)
    np.testing.assert_allclose(rel.lorentz_factor(up, C), gamma_p, rtol=1e-10)
    with pytest.raises(ValueError):
        rel.boost(u, [C, 0.0, 0.0], C)


def test_boost_invalid_elements_give_nan_not_exception():
    u = RNG.normal(0, C, (6, 3))
    w = np.tile([0.3 * C, 0.0, 0.0], (6, 1))
    w[1] = [C, 0.0, 0.0]          # |w| = c
    w[3] = [0.0, 2.0 * C, 0.0]    # superluminal drift (weak B)
    w[4] = [np.nan, np.nan, np.nan]  # undefined drift (B = 0)
    good = [0, 2, 5]
    with pytest.warns(RuntimeWarning, match="3 of 6") as rec:
        out = rel.boost(u, w, C)
    assert len(rec) == 1  # warned once
    assert out.shape == (6, 3)
    assert np.all(np.isnan(out[[1, 3, 4]]))
    np.testing.assert_allclose(out[good], rel.boost(u[good], w[good], C), rtol=1e-14)
    # a single invalid frame velocity is still an error ...
    with pytest.raises(ValueError):
        rel.boost(u, [2 * C, 0.0, 0.0], C)
    # ... except a non-finite one, which just propagates as NaN
    with pytest.warns(RuntimeWarning):
        assert np.all(np.isnan(rel.boost(u, [np.nan, 0.0, 0.0], C)))
    # pitch_cosine(frame='drift') follows the same rules
    B = np.tile([0.0, 0.0, 1.0], (4, 1))
    B[2] = [0.0, 0.0, 1e-9]  # drift speed >> c
    B[3] = 0.0               # no field
    cE = np.tile([0.0, 0.2 * C, 0.0], (4, 1))
    with pytest.warns(RuntimeWarning, match="2 of 4"):
        mu = rel.pitch_cosine(u[:4], B, cE=cE, c=C, frame="drift")
    assert np.all(np.isfinite(mu[:2])) and np.all(np.isnan(mu[2:]))
    np.testing.assert_allclose(mu[:2], rel.pitch_cosine(u[:2], B[:2], cE=cE[:2], c=C, frame="drift"))
    with pytest.raises(ValueError):
        rel.pitch_cosine(u, [0.0, 0.0, 1e-9], cE=[0.0, 0.2 * C, 0.0], c=C, frame="drift")


def test_boost_composition_collinear_matches_velocity_addition():
    u = RNG.normal(0, 2 * C, (10, 3))
    e = np.array([1.0, 2.0, -2.0]) / 3.0
    w1, w2 = 0.5 * C, -0.7 * C
    two_step = rel.boost(rel.boost(u, w1 * e, C), w2 * e, C)
    w12 = (w1 + w2) / (1.0 + w1 * w2 / C**2)  # relativistic velocity addition
    np.testing.assert_allclose(two_step, rel.boost(u, w12 * e, C), rtol=1e-10, atol=1e-9)
    # boosting back by -w' returns the lab momentum (w' = -w seen from the boosted frame)
    np.testing.assert_allclose(rel.boost(rel.boost(u, w1 * e, C), -w1 * e, C), u, rtol=1e-10, atol=1e-9)


def test_pitch_cosine_frames():
    B = RNG.normal(size=(20, 3))
    u = RNG.normal(0, C, (20, 3))
    lab = rel.pitch_cosine(u, B)
    np.testing.assert_allclose(lab, rel.dot(u, B) / (rel.norm(u) * rel.norm(B)), rtol=1e-12)
    # E = 0: drift frame is the lab frame
    np.testing.assert_allclose(rel.pitch_cosine(u, B, cE=np.zeros(3), c=C, frame="drift"), lab, rtol=1e-12)
    # a particle streaming along B in the drift frame (no gyration) drifts with U in the lab:
    # lab pitch < 1 because of the drift, drift-frame pitch is exactly 1
    B1 = np.array([0.0, 0.0, 1.0])
    U = np.array([0.4 * C, 0.0, 0.0])
    cE = -np.cross(U, B1)
    u_lab = rel.boost(np.array([0.0, 0.0, 1.5 * C]), -U, C)
    np.testing.assert_allclose(rel.velocity(u_lab, C)[:2], U[:2], rtol=1e-12)  # moves with the drift across B
    assert rel.pitch_cosine(u_lab, B1) < 0.9
    assert rel.pitch_cosine(u_lab, B1, cE=cE, c=C, frame="drift") == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError):
        rel.pitch_cosine(u, B, frame="drift")
    with pytest.raises(ValueError):
        rel.pitch_cosine(u, B, cE=np.zeros(3), c=C, frame="plasma")


def test_pitch_cosine_drift_frame_gyration_is_isotropic_pitch():
    """A particle gyrating with pitch cosine mu in the drift frame has lab mu shifted; the drift frame recovers it."""
    B = np.array([0.0, 0.0, 1.0])
    U = np.array([0.0, 0.3 * C, 0.0])
    cE = -np.cross(U, B)
    mu0, p = 0.35, 2.0 * C
    phases = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    u_rest = p * np.stack([np.sqrt(1 - mu0**2) * np.cos(phases), np.sqrt(1 - mu0**2) * np.sin(phases),
                           np.full_like(phases, mu0)], axis=1)
    u_lab = rel.boost(u_rest, -U, C)  # drift frame -> lab frame
    np.testing.assert_allclose(rel.pitch_cosine(u_lab, B, cE=cE, c=C, frame="drift"), mu0, rtol=1e-10)
    assert np.ptp(rel.pitch_cosine(u_lab, B)) > 0.05  # lab pitch varies with gyrophase


# ----------------------------------------------------------------- energy/power
def test_power_and_dgamma_dt_definitions():
    u = RNG.normal(0, C, (10, 3))
    cE = RNG.normal(0, 1.0, (10, 3))
    q_mc = 200.0
    P = rel.power_per_mass(u, cE, q_mc, C)
    np.testing.assert_allclose(P, q_mc * rel.dot(rel.velocity(u, C), cE), rtol=1e-12)
    np.testing.assert_allclose(rel.dgamma_dt(u, cE, q_mc, C), P / C**2, rtol=1e-12)


def _rk4(rhs, y, h, n):
    out = [y]
    for _ in range(n):
        k1 = rhs(y)
        k2 = rhs(y + 0.5 * h * k1)
        k3 = rhs(y + 0.5 * h * k2)
        k4 = rhs(y + h * k3)
        y = y + h / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        out.append(y)
    return np.array(out)


def test_dgamma_dt_matches_pusher_in_crossed_fields():
    """c^2 [gamma(T) - gamma(0)] = int q_mc v . cE dt for crossed uniform E and B (|cE| < c|B|)."""
    c, q_mc = 10.0, 2.0
    B = np.array([0.2, 0.0, 1.0])
    cE = np.array([0.0, 6.0, 0.0])  # E.B = 0, drift speed 6 = 0.6 c
    assert rel.norm(cE) < c * rel.norm(B)

    def rhs(y):  # y = (u, W) with dW/dt = q_mc v.cE (work per unit mass)
        u = y[:3]
        v = u / math.sqrt(1.0 + u @ u / c**2)
        return np.concatenate([q_mc * (cE + np.cross(v, B)), [q_mc * v @ cE]])

    h, n = 2e-3, 3000  # ~10 gyro-periods, ~300 steps per period
    traj = _rk4(rhs, np.array([3.0, -8.0, 2.0, 0.0]), h, n)
    u, W = traj[:, :3], traj[:, 3]
    gamma = rel.lorentz_factor(u, c)
    assert np.ptp(gamma) > 0.5  # the energy really changes a lot along the orbit
    work = c**2 * (gamma - gamma[0])
    np.testing.assert_allclose(work, W, rtol=0, atol=1e-6 * np.abs(W).max())
    # finite differences of gamma along the orbit vs the analytic rate
    fd = (gamma[2:] - gamma[:-2]) / (2 * h)
    rate = rel.dgamma_dt(u[1:-1], cE, q_mc, c)
    np.testing.assert_allclose(fd, rate, rtol=0, atol=1e-4 * np.abs(rate).max())
    # vectorised: same rate with broadcast (n,3) u against (3,) cE
    np.testing.assert_allclose(rel.dgamma_dt(u, cE, q_mc, c), [rel.dgamma_dt(ui, cE, q_mc, c) for ui in u])


def test_relativistic_magnetic_moment_is_adiabatic_invariant():
    """Betatron acceleration: B_z(t) ramps up slowly with the induced cE = -(dB/dt/2)(-y, x, 0).

    u_perp^2 / (2|B|) stays constant while gamma grows by ~70 %; u_perp^2 / (2 gamma |B|) does not.
    """
    c, q_mc = 1.0, 1.0
    B_start, B_end, T = 1.0, 3.0, 600.0
    Bdot = (B_end - B_start) / T

    def fields(t, x, y):
        Bz = B_start + Bdot * t
        return np.array([0.0, 0.0, Bz]), np.array([0.5 * Bdot * y, -0.5 * Bdot * x, 0.0])

    def rhs(s):  # s = (t, x, y, ux, uy, uz)
        t, x, y, u = s[0], s[1], s[2], s[3:]
        B, cE = fields(t, x, y)
        v = u / math.sqrt(1.0 + u @ u / c**2)
        return np.concatenate([[1.0, v[0], v[1]], q_mc * (cE + np.cross(v, B))])

    u0 = np.array([0.0, 3.0, 0.5])
    s0 = np.concatenate([[0.0, 3.0 / (q_mc * B_start), 0.0], u0])  # gyro-centre at the origin
    steps = 6000
    traj = _rk4(rhs, s0, T / steps, steps)
    t, u = traj[:, 0], traj[:, 3:]
    B = np.stack([fields(tt, xx, yy)[0] for tt, xx, yy in traj[:, :3]])
    mu = rel.magnetic_moment_per_mass(u, B)
    gamma = rel.lorentz_factor(u, c)
    assert gamma[-1] / gamma[0] > 1.6
    assert abs(mu[-1] / mu[0] - 1.0) < 0.02
    assert np.abs(mu / mu[0] - 1.0).max() < 0.05
    mu_old = mu / gamma  # u_perp^2 / (2 gamma |B|), not an invariant
    assert abs(mu_old[-1] / mu_old[0] - 1.0) > 0.3
    # the parallel momentum is untouched (E and dB/dt are perpendicular to B)
    np.testing.assert_allclose(u[:, 2], u0[2], rtol=1e-10)
    assert t[-1] == pytest.approx(T)


# ----------------------------------------------------------------- broadcasting
def test_broadcasting_n3_against_3():
    u = RNG.normal(0, C, (7, 3))
    B = np.array([0.1, 0.0, 0.0])
    cE = np.array([0.0, 0.01, 0.0])
    per_row = lambda f: np.array([f(ui) for ui in u])  # noqa: E731
    np.testing.assert_allclose(rel.gyroradius(u, B, 200.0), per_row(lambda ui: rel.gyroradius(ui, B, 200.0)))
    np.testing.assert_allclose(rel.gyrofrequency(u, B, 200.0, C),
                               per_row(lambda ui: rel.gyrofrequency(ui, B, 200.0, C)))
    np.testing.assert_allclose(rel.magnetic_moment_per_mass(u, B),
                               per_row(lambda ui: rel.magnetic_moment_per_mass(ui, B)))
    np.testing.assert_allclose(rel.pitch_cosine(u, B, cE, C, "drift"),
                               per_row(lambda ui: rel.pitch_cosine(ui, B, cE, C, "drift")))
    np.testing.assert_allclose(rel.boost(u, [0.1 * C, 0, 0], C), per_row(lambda ui: rel.boost(ui, [0.1 * C, 0, 0], C)))
    assert rel.drift_velocity(np.tile(cE, (7, 1)), B).shape == (7, 3)
    assert rel.drift_velocity(cE, np.tile(B, (7, 1))).shape == (7, 3)
    assert rel.lorentz_factor(np.zeros((4, 5, 3)), C).shape == (4, 5)
    assert rel.velocity(np.zeros((4, 5, 3)), C).shape == (4, 5, 3)
    # scalar inputs give scalars
    assert np.ndim(rel.gyroradius([1.0, 1.0, 0.0], B, 1.0)) == 0
