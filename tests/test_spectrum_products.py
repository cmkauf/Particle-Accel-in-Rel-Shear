"""Spectrum products: means and bounds, shape statistics, metadata round trips, legacy counts, file selection."""

from __future__ import annotations

import math
import warnings

import numpy as np
import pandas as pd
import pytest

from shearpic.io.spectrum_files import (load_spectrum, read_legacy_histogram_csv, save_spectrum,
                                        spectrum_counts_from_pdf)
from shearpic.physics.spectra import (Spectrum, cutoff_value, energy_values, histogram, lin_edges, log_edges,
                                      parse_edges, particle_snapshot_spectrum, power_law_index)

C = 50.0


def momenta(n=50000, seed=0):
    rng = np.random.default_rng(seed)
    d = rng.normal(size=(n, 3))
    d /= np.linalg.norm(d, axis=1)[:, None]
    return d * (C * np.exp(rng.normal(0.0, 0.6, n)))[:, None]


# ------------------------------------------------------------------ edges
def test_parse_edges():
    np.testing.assert_array_equal(parse_edges("log:1:100:500"), np.logspace(0, 2, 501))
    e = parse_edges("lin:-600:600:240")
    assert e.size == 241 and e[0] == -600 and e[-1] == 600 and np.allclose(np.diff(e), 5.0)
    np.testing.assert_array_equal(lin_edges(0, 1, 4), [0, 0.25, 0.5, 0.75, 1.0])
    for bad in ("log:1:100", "cubic:1:2:3", "lin:a:b:c", "lin:2:1:4"):
        with pytest.raises(ValueError):
            parse_edges(bad)


# ------------------------------------------------------------------ means
def test_mean_and_bounds_bracket_the_exact_mean():
    u = momenta()
    g1 = energy_values(u, C, "gamma_minus_1")
    spec = histogram(energy_values(u, C, "gamma"), log_edges(1.0, 1000.0, 400), "gamma")
    lo, hi = spec.mean_bounds("gamma_minus_1")
    assert lo <= g1.mean() <= hi
    assert spec.mean("gamma_minus_1") == pytest.approx(g1.mean(), rel=1e-3)
    assert lo <= spec.mean("gamma_minus_1") <= hi
    p = energy_values(u, C, "p_over_mc")
    plo, phi = spec.mean_bounds("p_over_mc")
    assert plo <= p.mean() <= phi and spec.mean("p_over_mc") == pytest.approx(p.mean(), rel=1e-3)
    # the same histogram in another variable gives the same means
    assert spec.to("p_over_mc").mean("gamma_minus_1") == pytest.approx(spec.mean("gamma_minus_1"), rel=1e-12)
    # legacy pdf-only spectra
    legacy = Spectrum("gamma", spec.edges, pdf=spec.dN_dx("total"))
    assert legacy.mean("gamma_minus_1") == pytest.approx(spec.mean("gamma_minus_1"), rel=1e-12)
    assert legacy.mean_bounds("gamma_minus_1") == pytest.approx((lo, hi), rel=1e-12)


def test_mean_with_out_of_range_particles():
    spec = histogram([1.5, 2.5, 50.0, 0.5], [1.0, 2.0, 3.0], "gamma_minus_1")
    assert (spec.underflow, spec.overflow) == (1.0, 1.0)
    with pytest.raises(ValueError, match="unknown"):
        spec.mean()
    assert spec.mean(in_range=True) == pytest.approx(2.0)
    lo, hi = spec.mean_bounds()
    assert lo == pytest.approx((1 + 2 + 0 + 3) / 4) and hi == math.inf
    only_under = histogram([1.5, 0.5], [1.0, 2.0], "gamma_minus_1")
    assert only_under.mean_bounds() == pytest.approx(((1 + 0) / 2, (2 + 1) / 2))


# ------------------------------------------------------------------ statistics
def test_power_law_index_and_cutoff_on_an_exact_power_law():
    edges = log_edges(1.0, 100.0, 200)
    x = np.sqrt(edges[:-1] * edges[1:])
    f = x**-2.5 * np.exp(-x / 30.0)
    spec = Spectrum("gamma", edges, counts=np.round(f * np.diff(edges) * 1e12))
    assert power_law_index(spec, 2.0, 4.0) == pytest.approx(2.5 + 3.0 / 30.0, rel=0.02)
    level = spec.dN_dx("total")[150] * 1.0001
    g = cutoff_value(spec, level)
    assert x[149] < g < x[150] + 1e-9
    assert math.isnan(cutoff_value(spec, 1e-300))
    with pytest.raises(ValueError, match="peak"):
        cutoff_value(spec, 1e9)
    with pytest.raises(ValueError, match="fewer than 2"):
        power_law_index(spec, 200.0, 300.0)
    # an empty bin right after the level: the upper edge of the last occupied bin
    counts = np.zeros(10)
    counts[:3] = [100, 50, 20]
    s2 = Spectrum("gamma", log_edges(1, 10, 10), counts=counts)
    assert cutoff_value(s2, s2.dN_dx()[2] * 0.5) == pytest.approx(s2.edges[3])


# ------------------------------------------------------------------ io
def test_save_load_with_metadata(tmp_path):
    s = histogram([1.5, 2.5], [1.0, 2.0, 3.0], "gamma", time=100.0, run_id=364, c=C)
    meta = {"output_number": np.int64(3), "edges": np.array([1.0, 2.0]), "nested": {"x": float("nan")}}
    p = save_spectrum(tmp_path / "a" / "spectrum_gamma_00003.npz", s, metadata=meta)
    assert sorted(f.name for f in p.parent.iterdir()) == ["spectrum_gamma_00003.npz"]
    back, m = load_spectrum(p, with_metadata=True)
    assert m == {"output_number": 3, "edges": [1.0, 2.0], "nested": {"x": None}}
    assert np.array_equal(back.counts, s.counts) and back.time == 100.0
    assert load_spectrum(save_spectrum(tmp_path / "b.npz", s), with_metadata=True)[1] == {}
    with np.load(p, allow_pickle=False) as f:
        assert "metadata_json" in f.files
    with pytest.raises(TypeError):
        save_spectrum(tmp_path / "c.npz", s, metadata=[1, 2])


def test_load_metadata_reads_only_the_metadata(tmp_path):
    from shearpic.io.spectrum_files import load_metadata, save_phase_space
    from shearpic.physics.phase_space import phase_space_histogram

    s = histogram([1.5, 2.5], [1.0, 2.0, 3.0], "gamma", time=100.0)
    stream = {"basename": "prob", "file_id": "out4", "kind": "tab"}
    p = save_spectrum(tmp_path / "s.npz", s, metadata={"stream": stream, "file_time": 100.0})
    assert load_metadata(p) == {"stream": stream, "file_time": 100.0}
    h = phase_space_histogram([0.5], [0.5], [0.0, 1.0], [0.0, 1.0], component="x")
    q = save_phase_space(tmp_path / "h.npz", {"x": h}, metadata={"stream": stream})
    assert load_metadata(q) == {"stream": stream}
    assert load_metadata(save_spectrum(tmp_path / "n.npz", s)) == {}
    np.savez(tmp_path / "other.npz", a=np.zeros(2))
    with pytest.raises(ValueError, match="not a shearpic file"):
        load_metadata(tmp_path / "other.npz")


def _legacy_csv(path, values, edges):
    density, _ = np.histogram(values, bins=edges, density=True)
    pd.DataFrame({"bin_centers": 0.5 * (edges[1:] + edges[:-1]), "density": density}).to_csv(path, index=False)


def test_spectrum_counts_from_pdf(tmp_path):
    g = energy_values(momenta(20000, seed=3), C, "gamma")
    edges = np.logspace(0, 2, 501)
    _legacy_csv(tmp_path / "histogram_frame_00001_t_100.0.csv", g, edges)
    legacy = read_legacy_histogram_csv(tmp_path / "histogram_frame_00001_t_100.0.csv", variable="gamma")
    ref = histogram(g, edges, "gamma")
    spec = spectrum_counts_from_pdf(legacy, 20000)
    assert np.array_equal(spec.counts, ref.counts) and spec.overflow == ref.overflow and spec.n_total == 20000
    assert spec.time == 100.0
    # particles above the gamma range: overflow (gamma >= 1, so nothing can be below 1)
    spec2 = spectrum_counts_from_pdf(legacy, 20000 + 5)
    assert spec2.overflow == ref.overflow + 5
    with pytest.raises(ValueError, match="more than n_total"):
        spectrum_counts_from_pdf(legacy, int(ref.counts.sum()) - 1)
    e = np.logspace(-2, 2, 5)
    g1 = Spectrum("gamma_minus_1", e, pdf=np.array([1.0, 2.0, 3.0, 4.0]) / 10 / np.diff(e))
    with pytest.raises(ValueError, match="underflow from overflow"):  # first edge 0.01 > 0: 15 particles unplaced
        spectrum_counts_from_pdf(g1, 25)
    assert spectrum_counts_from_pdf(g1, 10).overflow == 0


# ------------------------------------------------------------------ files
def _write_tab(path, time, u):
    lines = [f"# Athena++ particle data at time = {time:.18g}", "born_meshblock  particle_id  x  y  z  vx  vy  vz"]
    lines += [f"0  {k}  0  0  0  {u[k, 0]:.18g}  {u[k, 1]:.18g}  {u[k, 2]:.18g}" for k in range(u.shape[0])]
    path.write_text("\n".join(lines) + "\n")


def test_particle_snapshot_spectrum_streams_and_expected_n(tmp_path):
    run = tmp_path / "run0009"
    run.mkdir()
    u = momenta(40, seed=4)
    for b in range(2):
        _write_tab(run / f"prob.block{b}.out4.00001.par.tab", 10.0, u[b * 20:(b + 1) * 20])
        _write_tab(run / f"prob.block{b}.out5.00001.par.tab", 10.0, u[b * 20:(b + 1) * 20])
        _write_tab(run / f"old.block{b}.out4.00001.par.tab", 10.0, u[b * 20:(b + 1) * 20])
    edges = log_edges(1.0, 100.0, 50)
    with pytest.raises(ValueError, match="several particle outputs"):
        particle_snapshot_spectrum(run, 1, C, edges, kind="tab", backend="serial")
    with pytest.raises(ValueError, match="several particle outputs"):  # out4 of two problem ids
        particle_snapshot_spectrum(run, 1, C, edges, kind="tab", file_id="out4", backend="serial")
    spec = particle_snapshot_spectrum(run, 1, C, edges, kind="tab", file_id="out4", basename="prob",
                                      backend="serial", expected_n=40)
    assert spec.n_total == 40 and spec.time == 10.0 and spec.run_id == 9
    assert np.array_equal(spec.counts, histogram(energy_values(u, C, "gamma"), edges, "gamma").counts)
    with pytest.raises(ValueError, match="expected 64"):
        particle_snapshot_spectrum(run, 1, C, edges, kind="tab", file_id="out5", backend="serial", expected_n=64)


# ------------------------------------------------------------------ plotting
def test_plot_spectrum_stairs():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shearpic.plotting.lines import plot_spectrum

    counts = np.array([0, 5, 10, 0, 2], dtype=float)
    spec = Spectrum("gamma", log_edges(1, 10, 5), counts=counts)
    fig, ax = plt.subplots()
    line = plot_spectrum(ax, spec, style="stairs")
    x, y = line.get_xdata(), line.get_ydata()
    np.testing.assert_allclose(x, np.repeat(spec.edges, 2)[1:-1])
    f = spec.dN_dx("total")
    assert np.isnan(y[0]) and np.isnan(y[1]) and y[2] == y[3] == f[1] and np.isnan(y[6])
    line2 = plot_spectrum(ax, spec, style="stairs", floor=1e-6)
    assert line2.get_ydata()[0] == 1e-6 and line2.get_ydata()[6] == 1e-6
    line3 = plot_spectrum(ax, spec, style="line")
    assert np.isnan(line3.get_ydata()[0]) and line3.get_ydata().size == 5
    with pytest.raises(ValueError, match="style"):
        plot_spectrum(ax, spec, style="bars")
    with pytest.raises(ValueError, match="floor"):
        plot_spectrum(ax, spec, style="stairs", floor=0.0)
    zero_first = histogram([1.2, 1.5, 3.0], [1.0, 2.0, 4.0], "gamma").to("gamma_minus_1")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        l4 = plot_spectrum(ax, zero_first, style="stairs")
    assert l4.get_xdata()[0] == pytest.approx(0.5)  # starts at the arithmetic centre of [0, 1]
    plt.close(fig)


# ------------------------------------------------------------------ data
@pytest.mark.data
def test_run364_legacy_spectra_statistics(data_root):
    d = data_root / "run364" / "energy_spectrum_data"
    wanted = {0.0: None, 600.0: (4.332, 7.938), 1000.0: (2.313, 9.826), 2400.0: (0.672, 15.230)}
    found = {}
    for p in d.glob("histogram_frame_*_t_*.csv"):
        spec = read_legacy_histogram_csv(p, variable="gamma")
        if spec.time in wanted:
            found[spec.time] = spectrum_counts_from_pdf(spec, 67108864)  # exact integer counts, nothing lost
    if len(found) < len(wanted):
        pytest.skip("run364 legacy spectra not available")
    assert found[0.0].overflow == 0 and found[0.0].mean("gamma_minus_1") == pytest.approx(0.41458, abs=1e-5)
    lo, hi = found[600.0].mean_bounds("gamma_minus_1")
    assert lo < 0.69837 < hi  # history KE_cr / (N c^2) at t = 600
    for t, (alpha, gcut) in ((k, v) for k, v in wanted.items() if v):
        assert power_law_index(found[t], 3.0, 6.0) == pytest.approx(alpha, abs=1e-3)
        assert cutoff_value(found[t], 1e-4) == pytest.approx(gcut, abs=1e-3)
