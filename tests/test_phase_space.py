"""Tests for shearpic.physics.phase_space and the phase-space readers/writers of shearpic.io.spectrum_files."""

from __future__ import annotations

import warnings
from dataclasses import replace

import numpy as np
import pytest

from shearpic.io.athinput import parse_athinput
from shearpic.io.particles import PARBIN_RECORD
from shearpic.io.spectrum_files import (load_phase_space, read_legacy_phase_space_npz, recover_integer_counts,
                                        save_phase_space)
from shearpic.physics.phase_space import (PhaseSpaceHist, layer_asymmetry, particle_snapshot_phase_space,
                                          phase_space_histogram)

C = 50.0
U_EDGES = np.linspace(-150.0, 150.0, 31)
Y_EDGES = np.linspace(-2.0, 2.0, 9)

ATHINPUT = """
<job>
problem_id = synth
<mesh>
nx1 = 8
x1min = -2.0
x1max = 2.0
nx2 = 8
x2min = -2.0
x2max = 2.0
nx3 = 1
x3min = -0.5
x3max = 0.5
<meshblock>
nx1 = 4
nx2 = 4
nx3 = 1
<particles>
speed_of_light = 50.0
charge_over_mass_over_c = 200.0
<problem>
shear_strength = 1.0
y1 = -1.0
y2 = 1.0
vp_par = 50.0
cr_mass = 0.0005
npx1 = 8
npx2 = 8
"""


def particles(n=4000, seed=0, scale=80.0):
    rng = np.random.default_rng(seed)
    u = rng.normal(0.0, scale, size=(n, 3))
    x = np.column_stack([rng.uniform(-2, 2, n), rng.uniform(-2, 2, n), rng.uniform(-0.5, 0.5, n)])
    return x, u


def write_blocks(run_dir, file_number, time, blocks, kind, file_id="parts", basename="synth", first_gid=0):
    """Per-meshblock particle files in the formats of Particles::FormattedTableOutput / BinaryOutput."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for gid, (x, u) in enumerate(blocks, start=first_gid):
        name = run_dir / f"{basename}.block{gid}.{file_id}.{file_number:05d}.par.{kind}"
        pid = np.arange(u.shape[0])
        if kind == "tab":
            lines = [f"# Athena++ particle data at time = {time:.18g}",
                     "born_meshblock  particle_id  x  y  z  vx  vy  vz"]
            lines += ["  ".join([str(gid), str(int(pid[k]))] + [f"{v:.18g}" for v in (*x[k], *u[k])])
                      for k in range(pid.size)]
            name.write_text("\n".join(lines) + "\n")
        else:
            with open(name, "wb") as fh:
                np.array([0, 1, 0, 1, 0, 1, -2, 2, -2, 2, -0.5, 0.5], dtype="<f4").tofile(fh)
                np.array([time, 0.01], dtype="<f4").tofile(fh)
                np.array([pid.size], dtype="<i8").tofile(fh)
                rec = np.zeros(pid.size, dtype=PARBIN_RECORD)
                for j, key in enumerate(("x", "y", "z")):
                    rec[key] = x[:, j]
                for j, key in enumerate(("ux", "uy", "uz")):
                    rec[key] = u[:, j]
                rec["pid"], rec["init_mbid"] = pid, gid
                rec.tofile(fh)


# ------------------------------------------------------------------ histogram
def test_histogram_matches_numpy_and_keeps_out_of_range():
    x, u = particles()
    h = phase_space_histogram(u[:, 2], x[:, 1], U_EDGES, Y_EDGES, component="z", time=3.0, c=C)
    ref, _, _ = np.histogram2d(u[:, 2], x[:, 1], bins=[U_EDGES, Y_EDGES])
    assert h.counts.dtype == np.int64 and np.array_equal(h.counts, ref.astype(np.int64))
    assert h.n_total == 4000
    outside = (np.abs(u[:, 2]) > 150)
    assert h.n_total - h.n_in_range == outside.sum() > 0
    assert h.out_of_range_fraction() == pytest.approx(outside.mean())
    iy = np.clip(np.searchsorted(Y_EDGES, x[:, 1], side="right") - 1, 0, 7)
    assert np.array_equal(h.out_of_range_y, np.bincount(iy[outside], minlength=8))
    assert np.array_equal(h.n_y, np.bincount(iy, minlength=8))
    np.testing.assert_allclose(h.sum_u_y, np.bincount(iy, weights=u[:, 2], minlength=8))
    np.testing.assert_allclose(h.sum_u2_y, np.bincount(iy, weights=u[:, 2] ** 2, minlength=8))
    # the last edge belongs to the last bin; values outside y are counted only in n_total
    h2 = phase_space_histogram([150.0, 0.0, 1.0], [2.0, 5.0, -2.0], U_EDGES, Y_EDGES, component="x")
    assert h2.counts[-1, -1] == 1 and h2.n_total == 3 and h2.n_in_range == 2 and h2.n_y.sum() == 2


def test_histogram_nonfinite_and_validation():
    with pytest.warns(UserWarning, match="non-finite"):
        h = phase_space_histogram([np.nan, 1.0], [0.0, 0.0], U_EDGES, Y_EDGES, component="y")
    assert h.n_total == 1
    with pytest.raises(ValueError, match="component"):
        phase_space_histogram([1.0], [0.0], U_EDGES, Y_EDGES, component="w")
    with pytest.raises(ValueError, match="shape"):
        phase_space_histogram([1.0, 2.0], [0.0], U_EDGES, Y_EDGES, component="x")
    with pytest.raises(ValueError, match="increasing"):
        PhaseSpaceHist("x", U_EDGES[::-1], Y_EDGES, np.zeros((30, 8), int), 0)
    with pytest.raises(ValueError, match="integer"):
        PhaseSpaceHist("x", U_EDGES, Y_EDGES, np.full((30, 8), 0.5), 100)
    with pytest.raises(ValueError, match="smaller"):
        PhaseSpaceHist("x", U_EDGES, Y_EDGES, np.ones((30, 8), int), 10)
    with pytest.raises(ValueError, match="n_y"):
        PhaseSpaceHist("x", U_EDGES, Y_EDGES, np.ones((30, 8), int), 1000, out_of_range_y=np.zeros(8, int),
                       n_y=np.ones(8, int))


def test_density_normalisations_and_jacobian():
    x, u = particles(seed=2)
    h = phase_space_histogram(u[:, 0], x[:, 1], U_EDGES, Y_EDGES, component="x", c=C)
    area_u = np.diff(U_EDGES)[:, None] * np.diff(Y_EDGES)[None, :]
    H, xe, ye = h.density("in_range")
    assert np.sum(H * area_u) == pytest.approx(1.0)
    ref, _, _ = np.histogram2d(u[:, 0], x[:, 1], bins=[U_EDGES, Y_EDGES], density=True)
    np.testing.assert_allclose(H, ref, rtol=1e-12)  # identical to numpy's density=True
    Ht, xe_t, _ = h.density("total")
    assert np.sum(Ht * area_u) == pytest.approx(1.0 - h.out_of_range_fraction())
    Hm, xm, ym = h.density("total", unit="mc")
    np.testing.assert_allclose(xm, U_EDGES / C)
    np.testing.assert_allclose(Hm, Ht * C)
    area_mc = np.diff(xm)[:, None] * np.diff(ym)[None, :]
    assert np.sum(Hm * area_mc) == pytest.approx(np.sum(Ht * area_u))  # same probability in both units
    Hn, _, _ = h.density(None)
    np.testing.assert_allclose(Hn * area_u, h.counts)
    with pytest.raises(ValueError, match="speed of light"):
        replace(h, c=None).density(unit="mc")
    with pytest.raises(ValueError, match="unit"):
        h.density(unit="gamma")


def test_moments_exact_and_binned():
    x, u = particles(seed=4, scale=40.0)  # few particles out of range
    u[:, 1] += 3.0 * x[:, 1]
    h = phase_space_histogram(u[:, 1], x[:, 1], U_EDGES, Y_EDGES, component="y")
    iy = np.clip(np.searchsorted(Y_EDGES, x[:, 1], side="right") - 1, 0, 7)
    mean, var = h.moments()
    assert h.moments_source() == "sums"
    for j in range(8):
        sel = iy == j
        assert mean[j] == pytest.approx(u[sel, 1].mean())
        assert var[j] == pytest.approx(u[sel, 1].var(), rel=1e-9)
    mean_b, var_b = h.moments("bins")
    np.testing.assert_allclose(mean_b, mean, atol=3.0)  # bin width 10: centre rounding
    legacy = PhaseSpaceHist("y", U_EDGES, Y_EDGES, h.counts, h.n_total)
    assert legacy.moments_source() == "bins"
    with pytest.raises(ValueError, match="legacy"):
        legacy.moments("sums")
    assert legacy.mean_in_y_range(-2, 2) == pytest.approx(
        (h.counts * 0.5 * (U_EDGES[1:] + U_EDGES[:-1])[:, None]).sum() / h.counts.sum())
    assert legacy.out_of_range_fraction_y() is None


def test_add_reduces_blocks_exactly():
    x, u = particles(n=3000, seed=5)
    parts = [phase_space_histogram(u[s, 2], x[s, 1], U_EDGES, Y_EDGES, component="z", time=1.0, c=C)
             for s in (slice(0, 1000), slice(1000, 3000))]
    total = sum(parts)
    ref = phase_space_histogram(u[:, 2], x[:, 1], U_EDGES, Y_EDGES, component="z", time=1.0, c=C)
    for name in ("counts", "out_of_range_y", "n_y"):
        assert np.array_equal(getattr(total, name), getattr(ref, name))
    np.testing.assert_allclose(total.sum_u2_y, ref.sum_u2_y)
    assert total.n_total == 3000 and total.c == C
    with pytest.raises(ValueError, match="different times"):
        parts[0] + replace(parts[1], time=2.0)
    with pytest.raises(ValueError, match="u_z and u_x"):
        parts[0] + replace(parts[1], component="x")
    no_sums = replace(parts[1], sum_u_y=None)
    assert (parts[0] + no_sums).sum_u_y is None


def test_layer_asymmetry_on_a_linear_profile():
    rng = np.random.default_rng(7)
    n = 200000
    y = rng.uniform(-8, 8, n)
    uz = rng.normal(0, 5, n) - 2.0 * (y - 2.0)  # d<u_z>/dy = -2 across the layer at y = 2
    ye = np.linspace(-8, 8, 33)
    h = phase_space_histogram(uz, y, np.linspace(-200, 200, 81), ye, component="z")
    res = layer_asymmetry(h, 2.0, half_width=2.0, fit_half_width=2.0)
    assert res["slope"] == pytest.approx(-2.0, rel=0.02)
    assert res["mean_below"] == pytest.approx(2.0, rel=0.05) and res["mean_above"] == pytest.approx(-2.0, rel=0.05)
    assert res["source"] == "sums"
    # periodic: a layer at the box edge averages across the boundary
    res_edge = layer_asymmetry(h, 8.0, half_width=1.0, fit_half_width=1.0)
    assert np.isfinite(res_edge["mean_below"]) and np.isfinite(res_edge["mean_above"])


# ----------------------------------------------------------------- io
def test_save_load_round_trip_and_atomic(tmp_path):
    x, u = particles(seed=8)
    hists = {k: phase_space_histogram(u[:, j], x[:, 1], U_EDGES, Y_EDGES, component=k, time=12.5, run_id=364, c=C)
             for j, k in enumerate("xyz")}
    hists["y"] = PhaseSpaceHist("y", U_EDGES, Y_EDGES, hists["y"].counts, hists["y"].n_total, run_id="runA")
    p = save_phase_space(tmp_path / "ps.npz", hists, metadata={"a": np.float64(1.5), "path": tmp_path})
    assert [f.name for f in tmp_path.iterdir()] == ["ps.npz"]  # no temporary file left
    back, meta = load_phase_space(p, with_metadata=True)
    assert meta == {"a": 1.5, "path": str(tmp_path)}
    for k in "xyz":
        a, b = hists[k], back[k]
        assert np.array_equal(a.counts, b.counts) and a.n_total == b.n_total
        assert (a.time, a.run_id, a.c) == (b.time, b.run_id, b.c)
        for name in ("out_of_range_y", "n_y", "sum_u_y", "sum_u2_y"):
            va, vb = getattr(a, name), getattr(b, name)
            assert (va is None and vb is None) or np.array_equal(va, vb)
    assert back["y"].run_id == "runA" and back["x"].run_id == 364
    with pytest.raises(ValueError, match="does not match"):
        save_phase_space(tmp_path / "bad.npz", {"x": hists["y"]})
    np.savez(tmp_path / "foreign.npz", x=np.arange(3))
    with pytest.raises(ValueError, match="format_version"):
        load_phase_space(tmp_path / "foreign.npz")


def test_recover_integer_counts():
    rng = np.random.default_rng(9)
    counts = rng.integers(0, 50, size=(20, 10))
    counts[3, 3] = 1
    q = counts / counts.sum()
    rec, n = recover_integer_counts(q)
    assert np.array_equal(rec, counts) and n == counts.sum()
    # no bin with a single particle: found by the search over the minimum count
    c2 = counts.copy() * 1 + 2
    c2[0, 0] = 3
    rec2, n2 = recover_integer_counts(c2 / c2.sum())
    assert np.array_equal(rec2, c2) and n2 == c2.sum()
    # a narrow distribution needs the hint
    big = np.array([18885704, 22765265, 21813617, 3644278])
    with pytest.raises(ValueError, match="integer counts"):
        recover_integer_counts(big / big.sum(), max_min_count=50)
    rec3, n3 = recover_integer_counts(big / big.sum(), n_hint=int(big.sum()))
    assert np.array_equal(rec3, big)
    with pytest.raises(ValueError, match="integer counts"):
        recover_integer_counts(rng.random(30))  # not a histogram
    with pytest.raises(ValueError, match="empty"):
        recover_integer_counts(np.zeros(4))


def test_read_legacy_phase_space_npz(tmp_path):
    x, u = particles(n=20000, seed=10, scale=90.0)
    edge_p, edge_y = np.linspace(-150, 150, 61), np.linspace(-2, 2, 101)
    data = {"edge_p": edge_p, "edge_y": edge_y, "y_min": -2.0, "y_max": 2.0}
    for j, k in enumerate("xyz"):
        data[f"hist_p{k}"] = np.histogram2d(u[:, j], x[:, 1], bins=[edge_p, edge_y], density=True)[0]
    np.savez(tmp_path / "phase_space_histograms.npz", **data)
    hists = read_legacy_phase_space_npz(tmp_path / "phase_space_histograms.npz", n_total=20000, time=1200.0,
                                        run_id=364, c=C)
    for j, k in enumerate("xyz"):
        ref = phase_space_histogram(u[:, j], x[:, 1], edge_p, edge_y, component=k)
        h = hists[k]
        assert np.array_equal(h.counts, ref.counts)
        assert h.n_total == 20000 and h.n_in_range == ref.n_in_range < 20000
        assert h.out_of_range_fraction() == pytest.approx(ref.out_of_range_fraction())
        assert h.out_of_range_y is None and h.n_y is None and (h.time, h.c, h.run_id) == (1200.0, C, 364)
    with pytest.raises(ValueError, match="more than n_total"):
        read_legacy_phase_space_npz(tmp_path / "phase_space_histograms.npz", n_total=100, time=None)
    bad = dict(data, hist_px=data["hist_px"] * (1 + 1e-3 * np.random.default_rng(0).random(data["hist_px"].shape)))
    np.savez(tmp_path / "bad.npz", **bad)
    with pytest.raises(ValueError, match="integer counts"):
        read_legacy_phase_space_npz(tmp_path / "bad.npz", n_total=20000, time=None)
    np.savez(tmp_path / "other.npz", edge_p=edge_p)
    with pytest.raises(ValueError, match="not a legacy"):
        read_legacy_phase_space_npz(tmp_path / "other.npz", n_total=1, time=None)


# ------------------------------------------------------- snapshot map-reduce
def _blocks(n_blocks=4, n_per=150, seed=11):
    out = []
    for b in range(n_blocks):
        x, u = particles(n=n_per + 7 * b, seed=seed + b, scale=70.0)
        out.append((x, u.astype(np.float32).astype(np.float64)))  # float32-exact for the .par.bin writer
    return out


@pytest.mark.parametrize("kind", ["tab", "bin"])
@pytest.mark.parametrize("backend", ["serial", "process"])
def test_particle_snapshot_phase_space_matches_direct(tmp_path, kind, backend):
    from shearpic.config import RunConfig

    run_dir = tmp_path / "run0042"
    blocks = _blocks()
    blocks = [(x.astype(np.float32).astype(np.float64), u) for x, u in blocks]
    write_blocks(run_dir, 3, 150.0, blocks, kind)
    cfg = RunConfig.from_athinput(parse_athinput(ATHINPUT), run_dir=run_dir)
    n = sum(b[1].shape[0] for b in blocks)
    hists = particle_snapshot_phase_space(run_dir, 3, cfg, U_EDGES, Y_EDGES, kind=kind, backend=backend,
                                          n_workers=2 if backend == "process" else 1, expected_n=n)
    x_all = np.concatenate([b[0] for b in blocks])
    u_all = np.concatenate([b[1] for b in blocks])
    for j, k in enumerate("xyz"):
        ref = phase_space_histogram(u_all[:, j], x_all[:, 1], U_EDGES, Y_EDGES, component=k)
        h = hists[k]
        assert np.array_equal(h.counts, ref.counts) and np.array_equal(h.n_y, ref.n_y)
        np.testing.assert_allclose(h.sum_u2_y, ref.sum_u2_y, rtol=1e-12)
        assert h.n_total == n and h.time == pytest.approx(150.0) and h.c == C and h.run_id == 42
    with pytest.raises(ValueError, match="expected 9"):
        particle_snapshot_phase_space(run_dir, 3, cfg, U_EDGES, Y_EDGES, kind=kind, backend="serial", expected_n=9)


def test_particle_snapshot_phase_space_errors_and_fields(tmp_path, monkeypatch):
    import shearpic.io.particles as pio

    run_dir = tmp_path / "run7"
    blocks = _blocks(n_blocks=2, n_per=30)
    with pytest.raises(FileNotFoundError):
        particle_snapshot_phase_space(run_dir, 0, C, U_EDGES, Y_EDGES, backend="serial")
    write_blocks(run_dir, 0, 1.0, blocks[:1], "tab")
    write_blocks(run_dir, 0, 2.0, blocks[1:], "tab", first_gid=1)
    with pytest.raises(ValueError, match="different times"):
        particle_snapshot_phase_space(run_dir, 0, C, U_EDGES, Y_EDGES, backend="serial")
    other = tmp_path / "run8"
    write_blocks(other, 0, 1.0, blocks, "tab", file_id="out4")
    write_blocks(other, 0, 1.0, blocks, "tab", file_id="out5")
    with pytest.raises(ValueError, match="several particle outputs"):
        particle_snapshot_phase_space(other, 0, C, U_EDGES, Y_EDGES, backend="serial")
    requested = []
    real = pio.read_particle_block

    def recording(path, fields=pio.PARTICLE_FIELDS):
        requested.append(tuple(fields))
        return real(path, fields)

    monkeypatch.setattr(pio, "read_particle_block", recording)
    hists = particle_snapshot_phase_space(other, 0, C, U_EDGES, Y_EDGES, components=("z",), file_id="out5",
                                          backend="serial")
    assert list(hists) == ["z"] and requested == [("x", "u")] * 2
    with pytest.raises(ValueError, match="components"):
        particle_snapshot_phase_space(other, 0, C, U_EDGES, Y_EDGES, components=("q",), file_id="out5")


def test_plot_phase_space_interpolation_and_return():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shearpic.plotting.lines import plot_phase_space

    x, u = particles(seed=12)
    h = phase_space_histogram(u[:, 0], x[:, 1], U_EDGES, Y_EDGES, component="x", c=C)
    H, xe, ye = h.density("total", unit="mc")
    fig, ax = plt.subplots()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        im = plot_phase_space(ax, H, xe, ye, cbar=None)
    assert im.get_interpolation() == "nearest" and list(im.get_extent()) == [-3.0, 3.0, -2.0, 2.0]
    im2 = plot_phase_space(ax, H, xe, ye, cbar="right", interpolation="bilinear", rasterized=True)
    assert im2.get_interpolation() == "bilinear" and im2.colorbar is not None and im2.get_rasterized()
    plt.close(fig)


# ----------------------------------------------------------------- data
@pytest.mark.data
def test_run364_legacy_phase_space_counts(data_root):
    path = data_root / "run364" / "phase_space_histograms.npz"
    if not path.exists():
        pytest.skip("run364/phase_space_histograms.npz not available")
    hists = read_legacy_phase_space_npz(path, n_total=67108864, time=1200.0, run_id=364, c=50.0)
    assert [hists[k].n_in_range for k in "xyz"] == [64049625, 64047831, 64352486]
    assert hists["x"].out_of_range_fraction() == pytest.approx(0.04559, abs=1e-5)
    H, xe, ye = hists["z"].density("total", unit="mc")
    assert (xe[0], xe[-1]) == (-3.0, 3.0)
    area = np.diff(xe)[:, None] * np.diff(ye)[None, :]
    assert np.sum(H * area) == pytest.approx(64352486 / 67108864)
    lower, upper = (layer_asymmetry(hists["z"], L, half_width=10.0, fit_half_width=4.0) for L in (-31.4159, 31.4159))
    assert lower["mean_below"] > 0 > lower["mean_above"] and upper["mean_below"] > 0 > upper["mean_above"]
    assert lower["slope"] == pytest.approx(-1.75, abs=0.01) and upper["slope"] == pytest.approx(-0.91, abs=0.01)
