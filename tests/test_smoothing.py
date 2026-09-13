"""Tests of shearpic.physics.diagnostics.smooth_series (smoothing-spline low-pass filter)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import make_smoothing_spline

from shearpic.physics.diagnostics import smooth_series, smoothing_cutoff_period, smoothing_lam


def _gain(t, period, **kw):
    """Amplitude gain of the smoother for a sinusoid, measured away from the ends."""
    y = np.sin(2 * np.pi * t / period + 0.4)
    s = smooth_series(t, y, **kw)
    mid = (t > t[0] + 0.2 * (t[-1] - t[0])) & (t < t[-1] - 0.2 * (t[-1] - t[0]))
    return float(np.dot(s[mid], y[mid]) / np.dot(y[mid], y[mid]))


@pytest.mark.parametrize("dt", [0.5, 0.1, 2.0])
@pytest.mark.parametrize("cutoff", [16.7, 40.0])
def test_half_amplitude_at_cutoff_period(dt, cutoff):
    t = np.arange(0.0, 60 * cutoff, dt)
    assert _gain(t, cutoff, cutoff_period=cutoff) == pytest.approx(0.5, rel=0.02)


def test_transfer_function_shape():
    t = np.arange(0.0, 600.0 + 1e-9, 0.5)
    P_c = 16.7
    for period in (10.0, 20.0, 40.0, 100.0):
        H = 1.0 / (1.0 + (P_c / period) ** 4)       # 1/(1 + lam dt omega^4) with lam = (P_c/2pi)^4/dt
        assert _gain(t, period, cutoff_period=P_c) == pytest.approx(H, abs=0.01)


def test_lam_equivalence_at_lam_100():
    t = np.arange(0.0, 600.0 + 1e-9, 0.5)
    rng = np.random.default_rng(0)
    y = np.cumsum(rng.normal(size=t.size))
    np.testing.assert_allclose(smooth_series(t, y, lam=100.0), make_smoothing_spline(t, y, lam=100.0)(t), rtol=1e-12)
    assert smoothing_cutoff_period(100.0, 0.5) == pytest.approx(16.708, abs=1e-3)
    assert smoothing_lam(smoothing_cutoff_period(100.0, 0.5), 0.5) == pytest.approx(100.0, rel=1e-12)
    # cutoff 16.708 is the same filter as lam = 100 at dt = 0.5
    np.testing.assert_allclose(smooth_series(t, y, cutoff_period=smoothing_cutoff_period(100.0, 0.5)),
                               smooth_series(t, y, lam=100.0), rtol=1e-9, atol=1e-9)


def test_same_physical_smoothing_on_different_cadence():
    period, P_c = 30.0, 16.7
    g1 = _gain(np.arange(0.0, 1500.0, 0.5), period, cutoff_period=P_c)
    g2 = _gain(np.arange(0.0, 1500.0, 0.1), period, cutoff_period=P_c)
    assert g1 == pytest.approx(g2, abs=0.01)


def test_derivative_is_analytic_spline_derivative():
    t = np.arange(0.0, 600.0 + 1e-9, 0.5)
    y = 3.0 * t + 5.0 * np.sin(2 * np.pi * t / 200.0)
    d = smooth_series(t, y, cutoff_period=16.7, derivative=1)
    exact = 3.0 + 5.0 * 2 * np.pi / 200.0 * np.cos(2 * np.pi * t / 200.0)
    mid = (t > 50) & (t < 550)
    assert np.max(np.abs(d[mid] - exact[mid])) < 2e-3
    lam = smoothing_lam(16.7, 0.5)
    spl = make_smoothing_spline(t, y, lam=lam)
    np.testing.assert_allclose(d, spl.derivative()(t), rtol=1e-10, atol=1e-12)
    # a straight line is reproduced exactly, including its slope
    np.testing.assert_allclose(smooth_series(t, 2.0 * t - 1, cutoff_period=50.0, derivative=1), 2.0, atol=1e-8)


def test_non_uniform_times_use_median_spacing():
    rng = np.random.default_rng(1)
    t = np.cumsum(0.5 + 0.05 * rng.random(2000))
    assert _gain(t, 16.7, cutoff_period=16.7) == pytest.approx(0.5, abs=0.03)


@pytest.mark.parametrize("bad", [
    dict(t=[0, 1, 1, 2, 3, 4], y=[0, 1, 2, 3, 4, 5], cutoff_period=2.0),     # duplicate time
    dict(t=[0, 2, 1, 3, 4, 5], y=[0, 1, 2, 3, 4, 5], cutoff_period=2.0),     # decreasing
    dict(t=[0, 1, 2, 3, 4, 5], y=[0, 1, 2, 3, 4], cutoff_period=2.0),        # length mismatch
    dict(t=[0, 1, 2, 3, 4, 5], y=[0, 1, np.nan, 3, 4, 5], cutoff_period=2.0),
    dict(t=[0, 1, 2, 3, 4, 5], y=[0, 1, 2, 3, 4, 5]),                          # neither
    dict(t=[0, 1, 2, 3, 4, 5], y=[0, 1, 2, 3, 4, 5], cutoff_period=2.0, lam=1.0),
    dict(t=[0, 1, 2, 3, 4, 5], y=[0, 1, 2, 3, 4, 5], cutoff_period=-1.0),
    dict(t=[0, 1, 2], y=[0, 1, 2], cutoff_period=2.0),                          # too short
])
def test_invalid_input_raises(bad):
    with pytest.raises(ValueError):
        smooth_series(**bad)


def test_invalid_derivative_order():
    t = np.arange(10.0)
    with pytest.raises(ValueError):
        smooth_series(t, t, cutoff_period=2.0, derivative=3)
