"""Unit tests for shearpic.physics.turbulence and shearpic.io.field_spectra (synthetic fields)."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from shearpic.io.field_spectra import load_field_spectra, save_field_spectra, snapshot_mhd_spectra
from shearpic.physics.turbulence import (FieldSpectrum, average_spectra, fit_power_law, fluctuations, local_slopes,
                                         mhd_energy_spectra, nyquist_wavenumber, power_law_amplitude, rebin_log,
                                         shell_spectrum, shell_width, sum_spectra)

# anisotropic box like run364 (Lx = 10 pi, Ly = 40 pi), coarser grid
LX, LY = 10 * np.pi, 40 * np.pi
NX, NY = 32, 128


def centres(n, L):
    return (np.arange(n) + 0.5) * L / n - L / 2


def grid(nx=NX, ny=NY, Lx=LX, Ly=LY):
    return np.meshgrid(centres(nx, Lx), centres(ny, Ly), indexing="ij")


# ----------------------------------------------------------------------- shells
def test_default_shells_and_nyquist():
    assert shell_width((NX, NY), (LX, LY)) == pytest.approx(0.2)
    assert shell_width((NX, NY, 1), (LX, LY, 1.0)) == pytest.approx(0.2)   # nz = 1 axis ignored
    assert nyquist_wavenumber((NX, NY), (LX, LY)) == pytest.approx(min(np.pi * NX / LX, np.pi * NY / LY))
    s = shell_spectrum([np.zeros((NX, NY))], (LX, LY))
    assert s.k_edges[0] == 0 and s.k_edges[1] == pytest.approx(0.1) and np.allclose(np.diff(s.k_edges[1:]), 0.2)
    assert s.k[1] == pytest.approx(0.2) and s.k[5] == pytest.approx(1.0)
    assert s.n_modes.sum() <= NX * NY and s.total == 0 and s.parseval_error() == 0


@pytest.mark.parametrize("k_max", ["nyquist", "corner", 1.3])
def test_parseval_random_fields(k_max):
    rng = np.random.default_rng(1)
    comps = [rng.normal(size=(NX, NY)) * (i + 1) for i in range(3)]
    s = shell_spectrum(comps, (LX, LY), k_max=k_max, prefactor=0.5)
    expected = 0.5 * sum(np.mean(c ** 2) for c in comps)
    assert s.total == pytest.approx(expected, rel=1e-12)
    assert s.energy_in_bins + s.energy_beyond_kmax == pytest.approx(expected, rel=1e-12)
    if k_max == "corner":
        assert s.energy_beyond_kmax == 0 and s.n_modes.sum() == NX * NY
    else:
        assert s.energy_beyond_kmax > 0
    assert s.k[-1] <= (nyquist_wavenumber((NX, NY), (LX, LY)) if k_max == "nyquist" else 1e9) + 1e-12


def test_parseval_3d_with_nz_1_and_3d_box():
    rng = np.random.default_rng(2)
    f = rng.normal(size=(NX, NY, 1))
    s3 = shell_spectrum([f], (LX, LY, 1.0), k_max="corner")
    s2 = shell_spectrum([f[:, :, 0]], (LX, LY), k_max="corner")
    np.testing.assert_allclose(s3.E, s2.E, rtol=1e-12)
    g = rng.normal(size=(8, 16, 12))
    s = shell_spectrum([g], (1.0, 2.0, 1.5), k_max="corner")
    assert s.total == pytest.approx(0.5 * np.mean(g ** 2), rel=1e-12) and s.n_modes.sum() == g.size


def test_single_mode_lands_in_right_shell_anisotropic_box():
    X, Y = grid()
    mx, my, A = 3, 8, 0.7                       # kx = 0.6, ky = 0.4 -> |k| = 0.721 -> shell m = 4 (0.7 .. 0.9)
    kx, ky = 2 * np.pi * mx / LX, 2 * np.pi * my / LY
    f = A * np.cos(kx * X + ky * Y + 0.3)
    s = shell_spectrum([f], (LX, LY))
    m = int(np.argmax(s.E))
    assert m == 4 and s.k[m] == pytest.approx(0.8)
    assert s.bin_energy[m] == pytest.approx(0.25 * A ** 2, rel=1e-12)      # 0.5 <f^2> = A^2/4
    assert s.E[m] == pytest.approx(0.25 * A ** 2 / 0.2, rel=1e-12)
    assert np.sum(s.bin_energy) - s.bin_energy[m] < 1e-25


def test_mode_on_shell_edge_goes_to_upper_shell():
    X, Y = grid()
    f = np.sin(2 * np.pi * 2 / LY * Y)          # kx = 0, ky = 0.1 = dk/2: exactly on the edge of shells 0 and 1
    s = shell_spectrum([f], (LX, LY))
    assert np.argmax(s.bin_energy) == 1 and s.bin_energy[1] == pytest.approx(0.25, rel=1e-12)


def test_isotropic_power_law_slope_recovered():
    n, L, alpha = 256, 2 * np.pi, -5.0 / 3.0
    k1 = np.fft.fftfreq(n, d=1.0 / n)
    KX, KY = np.meshgrid(k1, k1, indexing="ij")
    K = np.hypot(KX, KY)
    amp = np.zeros_like(K)
    amp[K > 0] = K[K > 0] ** ((alpha - 1.0) / 2.0)       # 2D: E(k) ~ k |F_k|^2
    rng = np.random.default_rng(3)
    F = amp * np.exp(2j * np.pi * rng.random(K.shape))
    f = np.fft.ifft2(F).real
    s = shell_spectrum([f], (L, L))
    slope, err = fit_power_law(s, 4.0, 100.0)
    assert slope == pytest.approx(alpha, abs=0.08)
    assert 0 < err < 0.05
    assert fit_power_law(rebin_log(s, 8), 4.0, 100.0)[0] == pytest.approx(alpha, abs=0.1)


# ----------------------------------------------------------------- fluctuations
def _mhd_fields(seed=0, nx=NX, ny=NY):
    rng = np.random.default_rng(seed)
    X, Y = grid(nx, ny)
    rho = 1.0 + 0.1 * np.cos(2 * np.pi * X / LX) * np.sin(2 * np.pi * Y / LY) + 0.02 * rng.random((nx, ny))
    vx = np.tanh(Y / 3.0) + 0.1 * rng.normal(size=(nx, ny))
    vy = 0.05 * np.sin(2 * np.pi * 3 * X / LX) + 0.05 * rng.normal(size=(nx, ny))
    vz = 0.02 * rng.normal(size=(nx, ny))
    Bx = 0.1 + 0.05 * np.cos(Y) + 0.03 * rng.normal(size=(nx, ny))
    By = 0.03 * rng.normal(size=(nx, ny))
    Bz = 0.01 * rng.normal(size=(nx, ny))
    return rho, (vx, vy, vz), (Bx, By, Bz)


def test_fluctuations_energy_and_kx0_removal():
    rho, v, B = _mhd_fields()
    w, b = fluctuations(rho, v, B)
    favre = [(rho * u).mean(axis=0) / rho.mean(axis=0) for u in v]
    eps_k = 0.5 * np.mean(rho * sum((u - m[None, :]) ** 2 for u, m in zip(v, favre)))
    assert 0.5 * np.mean(sum(c ** 2 for c in w)) == pytest.approx(eps_k, rel=1e-13)
    # B - <B>_x has no kx = 0 modes at all
    for c in b:
        assert np.abs(np.fft.fft2(c, norm="forward")[0, :]).max() < 1e-15
    # Reynolds mean without weight removes kx = 0 exactly; the Favre-weighted field keeps a small residual
    wr, _ = fluctuations(rho, v, None, velocity_mean="reynolds", weight="none")
    assert np.abs(np.fft.fft2(wr[0], norm="forward")[0, :]).max() < 1e-15
    residual = sum(np.sum(np.abs(np.fft.fft2(c, norm="forward")[0, :]) ** 2) for c in w)
    assert 0 < residual < 1e-2 * np.mean(sum(c ** 2 for c in w))
    wm, _ = fluctuations(rho, v, None, weight="mean_rho")
    assert 0.5 * np.mean(sum(c ** 2 for c in wm)) == pytest.approx(
        0.5 * rho.mean() * np.mean(sum((u - m[None, :]) ** 2 for u, m in zip(v, favre))), rel=1e-12)


def test_fluctuations_3d_average_over_x_and_z():
    rng = np.random.default_rng(5)
    rho = 1 + 0.1 * rng.random((6, 8, 4))
    B = [rng.normal(size=(6, 8, 4)) + np.arange(8)[None, :, None] for _ in range(3)]
    _, b = fluctuations(rho, [rng.normal(size=(6, 8, 4))], B)
    np.testing.assert_allclose(b[0].mean(axis=(0, 2)), 0, atol=1e-14)


def test_fluctuations_validation():
    rho, v, B = _mhd_fields()
    with pytest.raises(ValueError):
        fluctuations(rho, v, B, velocity_mean="bad")
    with pytest.raises(ValueError):
        fluctuations(rho, v, B, weight="bad")
    with pytest.raises(ValueError):
        fluctuations(-rho, v, B)
    with pytest.raises(ValueError):
        fluctuations(rho, (v[0][:-1],), B)


def test_mhd_energy_spectra_consistency():
    rho, v, B = _mhd_fields()
    spec = mhd_energy_spectra(rho, v, B, (LX, LY, 1.0), time=3.5)
    w, b = fluctuations(rho, v, B)
    assert spec["kin"].total == pytest.approx(0.5 * np.mean(sum(c ** 2 for c in w)), rel=1e-12)
    assert spec["mag"].total == pytest.approx(0.5 * np.mean(sum(c ** 2 for c in b)), rel=1e-12)
    np.testing.assert_allclose(spec["tot"].E, spec["kin"].E + spec["mag"].E)
    assert spec["tot"].time == 3.5 and spec["tot"].kind == "tot"
    assert spec["mag"].E[0] < 1e-25                          # shell 0 holds only kx = 0 modes


# ------------------------------------------------------------------ operations
def _toy(E, time=None, kind="x", edges=None):
    E = np.asarray(E, float)
    edges = np.arange(E.size + 1, dtype=float) if edges is None else edges
    return FieldSpectrum(k_edges=edges, E=E, n_modes=np.ones(E.size, int), total=float(np.sum(E * np.diff(edges))),
                         time=time, kind=kind)


def test_rebin_log_conserves_energy_and_modes():
    rho, v, B = _mhd_fields(nx=64, ny=256)
    s = mhd_energy_spectra(rho, v, B, (LX, LY))["tot"]
    for n in (1, 4, 8, 20):
        r = rebin_log(s, n)
        assert r.energy_in_bins == pytest.approx(s.energy_in_bins, rel=1e-13)
        assert r.n_modes.sum() == s.n_modes.sum() and r.total == s.total
        assert np.all(np.isin(r.k_edges, s.k_edges))                   # whole shells only
        assert r.k_edges[0] == 0 and r.k_edges[1] == s.k_edges[1] and r.k_edges[-1] == s.k_edges[-1]
        decades = np.log10(s.k_edges[-1] / s.k_edges[1])
        assert r.E.size - 1 <= np.ceil(n * decades) + 1e-9                # at most n bins per decade
    assert rebin_log(s, 1000).E.size == s.E.size                       # finer than shells: unchanged
    np.testing.assert_allclose(rebin_log(s, 1000).E, s.E)
    with pytest.raises(ValueError):
        rebin_log(s, 0)


def test_average_and_sum_spectra():
    a, b = _toy([1, 2, 3], time=1.0, kind="kin"), _toy([3, 2, 1], time=1.0, kind="mag")
    avg = average_spectra([a, b])
    np.testing.assert_allclose(avg.E, 2.0)
    assert avg.time is None and avg.kind == "" and avg.total == pytest.approx(6.0)
    tot = sum_spectra(a, b, kind="tot")
    np.testing.assert_allclose(tot.E, 4.0)
    assert tot.time == 1.0 and tot.total == pytest.approx(12.0)
    assert average_spectra([a, a]).kind == "kin"
    with pytest.raises(ValueError):
        average_spectra([a, _toy([1, 2, 3], edges=np.array([0, 1, 2, 4.0]))])
    with pytest.raises(ValueError):
        average_spectra([])


def test_fit_power_law_exact_and_errors():
    edges = np.linspace(0.5, 100.5, 101)
    k = 0.5 * (edges[:-1] + edges[1:])
    E = 3.0 * k ** -2.0
    E[10] = 0.0                                   # zero bins are excluded
    s = _toy(E, edges=edges)
    slope, err = fit_power_law(s, 2.0, 50.0)
    assert slope == pytest.approx(-2.0, abs=1e-12) and err < 1e-10
    assert power_law_amplitude(s, slope, 2.0, 50.0) == pytest.approx(3.0, rel=1e-10)
    with pytest.raises(ValueError):
        fit_power_law(s, 2.0, 3.0)               # two bins only
    with pytest.raises(ValueError):
        fit_power_law(s, 5.0, 2.0)


def _broken_power_law(dk=0.2, k_max=100.0, k_break=10.0, s1=-1.0, s2=-3.0):
    """Linear shells centred on m dk (edges 0, dk/2, 3dk/2, ...) with E ~ k^s1 below k_break and k^s2 above."""
    n = int(round(k_max / dk)) + 1
    edges = np.concatenate(([0.0], (np.arange(n) + 0.5) * dk))
    k = 0.5 * (edges[:-1] + edges[1:])
    E = np.where(k < k_break, (k / k_break) ** s1, (k / k_break) ** s2)
    E[0] = 0.0
    return _toy(E, edges=edges), k


def test_fit_power_law_weighting_log_vs_shell():
    s, k = _broken_power_law()
    sel = (k >= 1.0) & (k <= 100.0)
    x, y = np.log(k[sel]), np.log(s.E[sel])
    for weighting, w2 in (("log", s.dk[sel] / k[sel]), ("shell", np.ones(sel.sum()))):
        (ref, _), cov = np.polyfit(x, y, 1, w=np.sqrt(w2), cov=True)   # polyfit weights multiply the residuals
        slope, err = fit_power_law(s, 1.0, 100.0, weighting=weighting)
        assert slope == pytest.approx(ref, abs=1e-12)
        assert err == pytest.approx(np.sqrt(cov[0, 0]), rel=1e-9)
    log = fit_power_law(s, 1.0, 100.0)[0]                           # default: equal weight per log interval
    shell = fit_power_law(s, 1.0, 100.0, weighting="shell")[0]
    # one decade at -1 and one at -3: equal weight per decade gives about the mean; per shell the
    # 90% of shells above k = 10 dominate
    assert log == pytest.approx(-2.0, abs=0.15) and shell < -2.5 and log == fit_power_law(s, 1.0, 100.0, weighting="log")[0]
    # equal weight per log k on linear shells ~ equal-weight fit of the same spectrum on log bins
    assert log == pytest.approx(fit_power_law(rebin_log(s, 20), 1.0, 100.0, weighting="shell")[0], abs=0.1)
    with pytest.raises(ValueError, match="weighting"):
        fit_power_law(s, 1.0, 100.0, weighting="linear")


def test_power_law_amplitude_uses_weighting():
    s, k = _broken_power_law()
    sel = (k >= 1.0) & (k <= 100.0)
    for weighting, w in (("log", s.dk[sel] / k[sel]), ("shell", np.ones(sel.sum()))):
        expected = np.exp(np.sum(w * (np.log(s.E[sel]) + 2.0 * np.log(k[sel]))) / w.sum())
        assert power_law_amplitude(s, -2.0, 1.0, 100.0, weighting=weighting) == pytest.approx(expected, rel=1e-12)
    assert power_law_amplitude(s, -2.0, 1.0, 100.0) == power_law_amplitude(s, -2.0, 1.0, 100.0, weighting="log")


def test_local_slopes_of_broken_power_law():
    s, _ = _broken_power_law()
    rows = local_slopes(s, 0.5)                                     # half-decade windows, quarter-decade steps
    assert [(round(r["k_lo"], 3), round(r["k_hi"], 3)) for r in rows[:2]] == [(0.316, 1.0), (0.562, 1.778)]
    assert rows[-1]["k_hi"] == pytest.approx(100.0) and all(r["n_bins"] >= 3 for r in rows)
    for r in rows:
        if r["k_hi"] <= 10.0:
            assert r["slope"] == pytest.approx(-1.0, abs=1e-9)
        elif r["k_lo"] >= 10.0 * 1.01:
            assert r["slope"] == pytest.approx(-3.0, abs=1e-9)
        assert r["k_centre"] == pytest.approx(np.sqrt(r["k_lo"] * r["k_hi"]))
    slopes = [r["slope"] for r in rows]
    assert np.all(np.diff(slopes) <= 1e-9)                          # steepening spectrum: slopes never increase
    explicit = local_slopes(s, [(3, 6), (20, 30), (0.15, 0.25)])
    assert explicit[0]["slope"] == pytest.approx(-1.0) and explicit[1]["slope"] == pytest.approx(-3.0)
    assert np.isnan(explicit[2]["slope"]) and explicit[2]["n_bins"] == 1
    bounded = local_slopes(s, 0.5, k_min=3.0, k_max=30.0)
    assert bounded[0]["k_lo"] >= 3.0 and bounded[-1]["k_hi"] <= 30.0 + 1e-9
    with pytest.raises(ValueError):
        local_slopes(s, -0.5)
    with pytest.raises(ValueError):
        local_slopes(s, 0.5, min_bins=2)


def test_field_spectrum_validation():
    with pytest.raises(ValueError):
        FieldSpectrum(k_edges=[0, 1, 1], E=[1, 1], n_modes=[1, 1], total=2)
    with pytest.raises(ValueError):
        FieldSpectrum(k_edges=[0, 1, 2], E=[1], n_modes=[1], total=1)


# ------------------------------------------------------------------------ io
def test_save_load_single_and_series(tmp_path):
    rho, v, B = _mhd_fields()
    series = [mhd_energy_spectra(rho, v, B, (LX, LY), time=t) for t in (1.0, 2.0)]
    p = save_field_spectra(tmp_path / "s.npz", series, {"run": 364, "value": np.float64(np.nan), "p": Path("x")})
    loaded, meta = load_field_spectra(p)
    assert meta == {"run": 364, "value": None, "p": "x"}
    assert [s["kin"].time for s in loaded] == [1.0, 2.0]
    for a, b in zip(series, loaded):
        for name in ("kin", "mag", "tot"):
            np.testing.assert_array_equal(a[name].E, b[name].E)
            np.testing.assert_array_equal(a[name].n_modes, b[name].n_modes)
            assert a[name].total == b[name].total and a[name].kind == b[name].kind
            assert a[name].energy_beyond_kmax == b[name].energy_beyond_kmax
    avg = {name: average_spectra([s[name] for s in series]) for name in ("kin", "mag")}
    p2 = save_field_spectra(tmp_path / "avg.product", avg)            # suffix kept as given
    single, meta2 = load_field_spectra(p2)
    assert p2.name == "avg.product" and meta2 == {} and single["kin"].time is None
    np.testing.assert_array_equal(single["mag"].E, avg["mag"].E)
    with np.load(p, allow_pickle=False) as f:                       # no pickled objects inside
        assert "metadata_json" in f.files


def test_save_rejects_inconsistent_series(tmp_path):
    a = {"kin": _toy([1, 2, 3], time=1.0)}
    b = {"kin": _toy([1, 2], time=2.0)}
    with pytest.raises(ValueError):
        save_field_spectra(tmp_path / "x.npz", [a, b])
    with pytest.raises(ValueError):
        save_field_spectra(tmp_path / "x.npz", [a, {"mag": _toy([1, 2, 3])}])
    with pytest.raises(ValueError):
        save_field_spectra(tmp_path / "x.npz", {"bad__name": _toy([1])})
    with pytest.raises(ValueError):
        save_field_spectra(tmp_path / "x.npz", [])


# ------------------------------------------------------------- snapshot worker
def write_mhd_athdf(path: Path, fields: dict[str, np.ndarray], bounds, time: float, mb=None) -> Path:
    """Minimal uniform-mesh .athdf holding (x, y, z)-indexed ``fields`` split into meshblocks."""
    names = list(fields)
    nx = fields[names[0]].shape
    mb = mb or nx
    n1, n2, n3 = (n // b for n, b in zip(nx, mb))
    locs = np.array([(i, j, k) for k in range(n3) for j in range(n2) for i in range(n1)], dtype=">i8")
    data = np.empty((len(names), len(locs), mb[2], mb[1], mb[0]), dtype=np.float32)
    for b_, (i, j, k) in enumerate(locs):
        for m, name in enumerate(names):
            blk = fields[name][i * mb[0]:(i + 1) * mb[0], j * mb[1]:(j + 1) * mb[1], k * mb[2]:(k + 1) * mb[2]]
            data[m, b_] = blk.transpose(2, 1, 0)
    with h5py.File(path, "w") as h:
        a = h.attrs
        a["DatasetNames"] = np.array([b"prim"], dtype="S21")
        a["NumVariables"] = np.array([len(names)], dtype=">i4")
        a["VariableNames"] = np.array([n.encode() for n in names], dtype="S21")
        a["RootGridSize"] = np.array(nx, dtype=">i4")
        a["MeshBlockSize"] = np.array(mb, dtype=">i4")
        for d in range(3):
            a[f"RootGridX{d + 1}"] = np.array([*bounds[d], 1.0], dtype=np.float32)
        a["Time"] = time
        a["NumCycles"] = 1
        a["NumMeshBlocks"] = len(locs)
        a["MaxLevel"] = 0
        h["prim"] = data
        h["LogicalLocations"] = locs
        h["Levels"] = np.zeros(len(locs), dtype=">i4")
    return path


def test_snapshot_mhd_spectra_matches_arrays(tmp_path):
    rho, v, B = _mhd_fields()
    fields = {"rho": rho, "vel1": v[0], "vel2": v[1], "vel3": v[2], "Bcc1": B[0], "Bcc2": B[1], "Bcc3": B[2]}
    fields = {k: val.astype(np.float32)[:, :, None] for k, val in fields.items()}
    path = write_mhd_athdf(tmp_path / "t.out2.00001.athdf", fields,
                           ((-LX / 2, LX / 2), (-LY / 2, LY / 2), (-0.5, 0.5)), time=2.0, mb=(16, 32, 1))

    class Cfg:  # anything with box lengths
        L = (LX, LY, 1.0)

    res = snapshot_mhd_spectra(path, Cfg())
    f64 = {k: val[:, :, 0].astype(np.float64) for k, val in fields.items()}
    ref = mhd_energy_spectra(f64["rho"], (f64["vel1"], f64["vel2"], f64["vel3"]), (f64["Bcc1"], f64["Bcc2"], f64["Bcc3"]),
                             Cfg())
    assert res["time"] == pytest.approx(2.0)
    np.testing.assert_allclose(res["spectra"]["kin"].E, ref["kin"].E, rtol=1e-12)
    assert res["grid"]["eps_kin_fluct"] == pytest.approx(res["spectra"]["kin"].total, rel=1e-12)
    assert res["grid"]["eps_mag_fluct"] == pytest.approx(res["spectra"]["mag"].total, rel=1e-12)
    assert res["grid"]["eps_mag"] == pytest.approx(0.5 * np.mean(sum(f64[f"Bcc{i}"] ** 2 for i in (1, 2, 3))), rel=1e-12)


def test_snapshot_mhd_spectra_refuses_dataless(tmp_path, monkeypatch):
    from shearpic.io.athdf import DatalessFileError

    monkeypatch.setattr("shearpic.io._util.is_dataless", lambda p: True)
    with pytest.raises(DatalessFileError):
        snapshot_mhd_spectra(tmp_path / "missing.athdf", object())


def test_parseval_assertion_fires_on_binning_bug(monkeypatch):
    import shearpic.physics.turbulence as turb

    real = np.bincount
    monkeypatch.setattr(turb.np, "bincount", lambda idx, weights=None, minlength=0:
                        real(idx, weights=None if weights is None else weights * 1.001, minlength=minlength))
    with pytest.raises(AssertionError, match="Parseval"):
        shell_spectrum([np.random.default_rng(0).normal(size=(8, 8))], (1.0, 1.0))
