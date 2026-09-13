"""Unit tests for shearpic.physics.spectra and shearpic.io.spectrum_files (synthetic data)."""

from __future__ import annotations

import pickle
import sys
import threading
import types
import warnings
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from shearpic.io.athinput import parse_athinput
from shearpic.io.particles import PARBIN_RECORD
from shearpic.io.spectrum_files import (
    load_spectrum,
    read_legacy_histogram_csv,
    read_legacy_spectrum_dir,
    reconstruct_log_edges,
    save_spectrum,
)
from shearpic.physics import relativity as rel
from shearpic.physics.spectra import (
    VARIABLES,
    Spectrum,
    convert_values,
    energy_values,
    histogram,
    log_edges,
    particle_snapshot_spectrum,
)

C = 50.0


def random_momenta(n=20000, seed=0, scale=3.0):
    rng = np.random.default_rng(seed)
    direction = rng.normal(size=(n, 3))
    direction /= np.linalg.norm(direction, axis=1)[:, None]
    mag = C * np.exp(rng.normal(0.0, 1.0, n)) * scale / 3.0
    return direction * mag[:, None]


# ------------------------------------------------------------------ basics
def test_log_edges_exact_ends_and_bin_count():
    e = log_edges(0.01, 100.0, 400)
    assert e.size == 401
    assert e[0] == 0.01 and e[-1] == 100.0
    assert np.allclose(np.diff(np.log10(e)), 0.01)
    with pytest.raises(ValueError):
        log_edges(0.0, 1.0, 10)
    with pytest.raises(ValueError):
        log_edges(1.0, 10.0, 0)


def test_energy_values_match_relativity_and_are_accurate_at_small_u():
    u = random_momenta(100)
    gamma = rel.lorentz_factor(u, C)
    assert np.allclose(energy_values(u, C, "gamma"), gamma, rtol=1e-14)
    assert np.allclose(energy_values(u, C, "gamma_minus_1") * C**2, rel.kinetic_energy_per_mass(u, C), rtol=1e-13)
    assert np.allclose(energy_values(u, C, "p_over_mc"), np.linalg.norm(u, axis=1) / C, rtol=1e-14)
    # u = 1e-6 c: gamma - 1 = 5e-13 would lose ~4 digits if computed as gamma - 1
    tiny = np.array([[1e-6 * C, 0.0, 0.0]])
    assert energy_values(tiny, C, "gamma_minus_1")[0] == pytest.approx(0.5e-12 - 0.125e-24, rel=1e-12)
    # a RunConfig-like object is accepted in place of c
    assert np.allclose(energy_values(u, SimpleNamespace(c=C), "gamma"), gamma)
    with pytest.raises(ValueError):
        energy_values(u, C, "energy")


def test_histogram_under_overflow_and_right_edge():
    values = np.array([0.5, 1.0, 1.5, 2.0, 3.0, 3.5, np.nan])
    with pytest.warns(UserWarning, match="non-finite"):
        s = histogram(values, [1.0, 2.0, 3.0], "gamma")
    assert np.array_equal(s.counts, [2.0, 2.0])  # 3.0 == last edge goes into the last bin
    assert (s.underflow, s.overflow, s.n_total) == (1.0, 1.0, 6.0)
    ref, _ = np.histogram(values[np.isfinite(values)], [1.0, 2.0, 3.0])
    assert np.array_equal(s.counts, ref)


def test_histogram_weights():
    values = np.array([0.5, 1.2, 1.7, 2.5, 4.0])
    w = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    s = histogram(values, [1.0, 2.0, 3.0], "gamma", weights=w, time=3.0, run_id=7, c=C)
    assert np.array_equal(s.counts, [5.0, 4.0])
    assert (s.underflow, s.overflow, s.n_total) == (1.0, 5.0, 15.0)
    assert (s.time, s.run_id, s.c) == (3.0, 7, C)


def test_histogram_agrees_with_numpy_on_random_data():
    u = random_momenta()
    g = energy_values(u, C, "gamma")
    edges = log_edges(1.0, 20.0, 64)
    s = histogram(g, edges, "gamma")
    ref, _ = np.histogram(g, edges)
    assert np.array_equal(s.counts, ref)
    assert s.n_total == g.size
    assert s.underflow == 0 and s.overflow == np.sum(g > 20.0)


# ------------------------------------------------------ change of variables
@pytest.mark.parametrize("target", VARIABLES)
def test_change_of_variable_preserves_counts_exactly(target):
    u = random_momenta()
    edges = log_edges(1.05, 30.0, 80)
    s = histogram(energy_values(u, C, "gamma"), edges, "gamma")
    t = s.to(target)
    assert t.variable == target
    assert np.array_equal(t.counts, s.counts)
    assert (t.underflow, t.overflow, t.n_total) == (s.underflow, s.overflow, s.n_total)
    # histogramming the target variable directly on the mapped edges gives the same counts
    direct = histogram(energy_values(u, C, target), t.edges, target)
    assert np.array_equal(direct.counts, t.counts)
    # and the mapping round-trips
    assert np.allclose(t.to("gamma").edges, edges, rtol=1e-13, atol=0)


def test_convert_values_exact_relations_and_domain():
    g = np.array([1.0, 1.5, 10.0])
    assert np.allclose(convert_values(g, "gamma", "p_over_mc"), np.sqrt(g**2 - 1))
    assert np.allclose(convert_values(g, "gamma", "gamma_minus_1"), g - 1)
    p = np.array([0.0, 0.5, 3.0])
    assert np.allclose(convert_values(p, "p_over_mc", "gamma_minus_1"), np.sqrt(1 + p**2) - 1, rtol=1e-14)
    # p = 1e-8: sqrt(1 + p^2) - 1 cancels to 0 in floating point, the stable form does not
    assert convert_values([1e-8], "p_over_mc", "gamma_minus_1")[0] == pytest.approx(0.5e-16, rel=1e-8)
    # round-off below the domain is clipped, real violations raise
    assert convert_values([1.0 - 1e-15], "gamma", "gamma_minus_1")[0] == 0.0
    with pytest.raises(ValueError):
        convert_values([0.5], "gamma", "p_over_mc")


def test_pdf_transforms_with_jacobian():
    u = random_momenta()
    edges = log_edges(1.0, 50.0, 100)
    s = histogram(energy_values(u, C, "gamma"), edges, "gamma")
    legacy = Spectrum("gamma", edges, pdf=s.dN_dx("in_range"))
    for target in ("gamma_minus_1", "p_over_mc"):
        t = legacy.to(target)
        assert np.sum(t.pdf * t.widths) == pytest.approx(1.0, rel=1e-12)
        assert np.allclose(t.pdf, s.to(target).dN_dx("in_range"), rtol=1e-12)
    back = legacy.to("p_over_mc").to("gamma")
    assert np.allclose(back.pdf, legacy.pdf, rtol=1e-10)


def test_dN_dx_normalisations():
    values = np.concatenate([np.full(10, 0.5), np.full(30, 1.5), np.full(60, 2.5)])
    s = histogram(values, [1.0, 2.0, 4.0], "gamma")
    assert np.allclose(s.dN_dx(None), [30.0, 30.0])
    assert np.allclose(s.dN_dx("total"), [0.3, 0.3])
    assert np.sum(s.dN_dx("total") * s.widths) == pytest.approx(0.9)
    assert np.sum(s.dN_dx("in_range") * s.widths) == pytest.approx(1.0)
    pdf_only = Spectrum("gamma", s.edges, pdf=2 * s.dN_dx("in_range"))
    assert np.allclose(pdf_only.dN_dx("total"), pdf_only.pdf)
    assert np.sum(pdf_only.dN_dx("in_range") * s.widths) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="counts are unknown"):
        pdf_only.dN_dx(None)
    with pytest.raises(ValueError):
        s.dN_dx("bogus")


def test_centers_and_validation():
    s = Spectrum("gamma_minus_1", [0.0, 1.0, 4.0], counts=[1, 2])
    assert np.allclose(s.centers(), [0.5, 2.5])
    with pytest.raises(ValueError):
        s.centers("geometric")
    s2 = Spectrum("gamma", [1.0, 4.0, 16.0], counts=[1, 2])
    assert np.allclose(s2.centers("geometric"), [2.0, 8.0])
    assert s2.n_total == 3.0 and s2.underflow == 0.0
    with pytest.raises(ValueError):
        Spectrum("gamma", [1.0, 1.0, 2.0], counts=[1, 1])
    with pytest.raises(ValueError):
        Spectrum("gamma", [0.5, 1.0, 2.0], counts=[1, 1])  # gamma < 1
    with pytest.raises(ValueError):
        Spectrum("gamma", [1.0, 2.0], counts=[1, 1])
    with pytest.raises(ValueError):
        Spectrum("gamma", [1.0, 2.0])
    with pytest.raises(ValueError):
        Spectrum("energy", [1.0, 2.0], counts=[1])


def test_add_map_reduce():
    u = random_momenta(9000)
    edges = log_edges(1.0, 10.0, 32)
    g = energy_values(u, C, "gamma")
    parts = [histogram(g[i::3], edges, "gamma", time=5.0, run_id=1, c=C) for i in range(3)]
    total = sum(parts)
    whole = histogram(g, edges, "gamma")
    assert np.array_equal(total.counts, whole.counts)
    assert (total.n_total, total.underflow, total.overflow) == (whole.n_total, whole.underflow, whole.overflow)
    assert (total.time, total.run_id, total.c) == (5.0, 1, C)
    with pytest.raises(ValueError, match="edges"):
        parts[0] + histogram(g, log_edges(1.0, 10.0, 31), "gamma")
    with pytest.raises(ValueError, match="convert"):
        parts[0] + parts[1].to("p_over_mc")
    with pytest.raises(ValueError, match="different times"):
        parts[0] + replace(parts[1], time=6.0)
    with pytest.raises(ValueError, match="pdf-only"):
        parts[0] + Spectrum("gamma", edges, pdf=np.ones(32))
    assert (parts[0] + replace(parts[1], time=None, run_id=2)).run_id is None


# ------------------------------------------------------------ npz round trip
def test_save_load_round_trip(tmp_path):
    s = histogram([1.5, 2.5, 7.0], [1.0, 2.0, 3.0], "gamma", time=12.5, run_id=423, c=C)
    p = save_spectrum(tmp_path / "sub" / "spec.npz", s)
    r = load_spectrum(p)
    assert r.variable == s.variable and np.array_equal(r.edges, s.edges) and np.array_equal(r.counts, s.counts)
    assert (r.n_total, r.underflow, r.overflow, r.time, r.run_id, r.c) == (3.0, 0.0, 1.0, 12.5, 423, C)
    assert r.pdf is None
    legacy = Spectrum("gamma_minus_1", [0.01, 0.1, 1.0], pdf=[1.0, 2.0], run_id="run0001")
    p2 = save_spectrum(tmp_path / "legacy.spec", legacy)  # suffix kept as given
    assert p2.name == "legacy.spec" and p2.exists()
    r2 = load_spectrum(p2)
    assert r2.counts is None and np.array_equal(r2.pdf, legacy.pdf)
    assert (r2.time, r2.run_id, r2.c, r2.n_total) == (None, "run0001", None, None)
    with np.load(p2) as f:
        assert int(f["format_version"]) == 1


def test_load_rejects_foreign_npz(tmp_path):
    p = tmp_path / "x.npz"
    np.savez(p, edges=np.arange(3.0))
    with pytest.raises(ValueError, match="format_version"):
        load_spectrum(p)


# ------------------------------------------------------------- legacy CSVs
def write_legacy_csv(path, edges, values):
    density, _ = np.histogram(values, bins=edges, density=True)
    centers = 0.5 * (edges[1:] + edges[:-1])
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"bin_centers": centers, "density": density}).to_csv(path, index=False)
    return density


def test_legacy_csv_gamma_minus_1_round_trip(tmp_path):
    edges = np.logspace(-2, 2, 501)
    rng = np.random.default_rng(1)
    values = np.exp(rng.normal(0, 1, 5000))
    path = tmp_path / "run0369" / "energy_spectrum_data" / "histogram_frame_00002_t_200.0.csv"
    density = write_legacy_csv(path, edges, values)
    with pytest.warns(UserWarning, match="gamma_minus_1"):
        s = read_legacy_histogram_csv(path)
    assert s.variable == "gamma_minus_1"
    assert np.array_equal(s.edges, edges)  # round exponents are regenerated exactly
    assert np.allclose(s.pdf, density) and s.counts is None
    assert s.time == 200.0 and s.run_id == 369
    assert np.sum(s.pdf * s.widths) == pytest.approx(1.0, abs=1e-12)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert read_legacy_histogram_csv(path, variable="gamma_minus_1").variable == "gamma_minus_1"


def test_legacy_csv_gamma_detection_and_errors(tmp_path):
    edges = np.logspace(0, 2, 501)
    path = tmp_path / "histogram_frame_00000_t_0.0.csv"
    write_legacy_csv(path, edges, np.linspace(1.1, 90, 1000))
    with pytest.warns(UserWarning):
        s = read_legacy_histogram_csv(path)
    assert s.variable == "gamma" and s.edges[0] == 1.0 and s.time == 0.0
    with pytest.raises(ValueError):
        read_legacy_histogram_csv(path, variable="bogus")
    bad = tmp_path / "histogram_frame_00001_t_1.0.csv"
    write_legacy_csv(bad, np.linspace(1, 10, 51), np.linspace(1.1, 9, 100))
    with pytest.raises(ValueError, match="not log-spaced"):
        read_legacy_histogram_csv(bad)


def test_reconstruct_log_edges_non_round_exponents():
    edges = np.logspace(-1.2345678, 1.1, 51)
    centers = 0.5 * (edges[1:] + edges[:-1])
    assert np.allclose(reconstruct_log_edges(centers), edges, rtol=1e-12, atol=0)


def test_legacy_dir_sorted_by_frame_and_ignores_summaries(tmp_path):
    edges = np.logspace(-2, 2, 101)
    rng = np.random.default_rng(2)
    for frame, t in ((10, 1000.0), (2, 200.0), (1, 100.0)):
        write_legacy_csv(tmp_path / f"histogram_frame_{frame:05d}_t_{t}.csv", edges, np.exp(rng.normal(size=500)))
    (tmp_path / "frame_summary.csv").write_text("frame_num,time\n0,garbage\n")
    (tmp_path / "histogram_metadata.csv").write_text("num_frames\n3\n")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = read_legacy_spectrum_dir(tmp_path)
    assert [s.time for s in specs] == [100.0, 200.0, 1000.0]
    assert len([w for w in caught if "auto-detected" in str(w.message)]) == 1
    with pytest.raises(FileNotFoundError):
        read_legacy_spectrum_dir(tmp_path / "empty_does_not_exist")


# ----------------------------------------------------- snapshot map-reduce
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
"""


def write_blocks(run_dir, file_number, time, blocks, kind, first_gid=0):
    """Write per-meshblock particle files in the exact formats of Particles::*Output."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for gid, (x, u, pid) in enumerate(blocks, start=first_gid):
        name = run_dir / f"synth.block{gid}.parts.{file_number:05d}.par.{kind}"
        if kind == "tab":  # FormattedTableOutput: setprecision(18), two-space separated
            lines = [f"# Athena++ particle data at time = {time:.18g}",
                     "born_meshblock  particle_id  x  y  z  vx  vy  vz"]
            lines += ["  ".join([str(gid), str(int(pid[k]))] + [f"{v:.18g}" for v in (*x[k], *u[k])])
                      for k in range(pid.size)]
            name.write_text("\n".join(lines) + "\n")
        else:  # BinaryOutput: 12 f4 box, 2 f4 (time, dt), i8 n, 44-byte records
            with open(name, "wb") as fh:
                np.array([0, 1, 0, 1, 0, 1, -2, 2, -2, 2, -0.5, 0.5], dtype="<f4").tofile(fh)
                np.array([time, 0.01], dtype="<f4").tofile(fh)
                np.array([pid.size], dtype="<i8").tofile(fh)
                rec = np.zeros(pid.size, dtype=PARBIN_RECORD)
                for j, key in enumerate(("x", "y", "z")):
                    rec[key] = x[:, j]
                for j, key in enumerate(("ux", "uy", "uz")):
                    rec[key] = u[:, j]
                rec["dpar"], rec["property"], rec["pid"], rec["init_mbid"] = 1.0, 0.0, pid, gid
                rec.tofile(fh)


def make_blocks(n_blocks=4, n_per=300, seed=3):
    rng = np.random.default_rng(seed)
    blocks = []
    for b in range(n_blocks):
        u = random_momenta(n_per + 17 * b, seed=seed + b).astype(np.float32).astype(np.float64)
        x = rng.uniform(-2, 2, size=(u.shape[0], 3))
        blocks.append((x, u, np.arange(u.shape[0], dtype=np.int64) + 1000 * b))
    return blocks


@pytest.mark.parametrize("kind", ["tab", "bin"])
@pytest.mark.parametrize("backend", ["serial", "process"])
def test_particle_snapshot_spectrum_matches_direct_histogram(tmp_path, kind, backend):
    from shearpic.config import RunConfig

    run_dir = tmp_path / "run0042"
    blocks = make_blocks()
    write_blocks(run_dir, 3, 150.0, blocks, kind)
    cfg = RunConfig.from_athinput(parse_athinput(ATHINPUT), run_dir=run_dir)
    edges = log_edges(1.0, 10.0, 40)
    spec = particle_snapshot_spectrum(run_dir, 3, cfg, edges, variable="gamma", backend=backend,
                                      n_workers=2 if backend == "process" else 1)
    u_all = np.concatenate([b[1] for b in blocks])
    ref = histogram(energy_values(u_all, C, "gamma"), edges, "gamma")
    assert np.array_equal(spec.counts, ref.counts)
    assert spec.n_total == u_all.shape[0]
    assert (spec.underflow, spec.overflow) == (ref.underflow, ref.overflow)
    assert spec.time == pytest.approx(150.0)
    assert spec.c == C and spec.run_id == 42
    # plain float c and another variable work too
    spec_p = particle_snapshot_spectrum(run_dir, 3, C, spec.to("p_over_mc").edges, variable="p_over_mc",
                                        backend="serial")
    assert np.array_equal(spec_p.counts, spec.counts)


def test_particle_snapshot_spectrum_errors(tmp_path):
    run_dir = tmp_path / "run7"
    blocks = make_blocks(n_blocks=2, n_per=50)
    edges = log_edges(1.0, 10.0, 10)
    with pytest.raises(FileNotFoundError):
        particle_snapshot_spectrum(run_dir, 0, C, edges, backend="serial")
    write_blocks(run_dir, 0, 1.0, blocks, "tab")
    write_blocks(run_dir, 0, 1.0, blocks, "bin")
    with pytest.raises(ValueError, match="both"):
        particle_snapshot_spectrum(run_dir, 0, C, edges, backend="serial")
    assert particle_snapshot_spectrum(run_dir, 0, C, edges, kind="bin", backend="serial").n_total == 100 + 17
    other = tmp_path / "run8"
    write_blocks(other, 1, 1.0, blocks[:1], "tab")
    write_blocks(other, 1, 2.0, blocks[1:], "tab", first_gid=1)
    with pytest.raises(ValueError, match="different times"):
        particle_snapshot_spectrum(other, 1, C, edges, backend="serial")


def test_block_spectrum_reads_only_momenta(tmp_path, monkeypatch):
    import shearpic.io.particles as pio

    run_dir = tmp_path / "run0042"
    blocks = make_blocks(n_blocks=3, n_per=40)
    write_blocks(run_dir, 3, 150.0, blocks, "bin")
    requested = []
    real = pio.read_particle_block

    def recording(path, fields=pio.PARTICLE_FIELDS):
        requested.append(tuple(fields))
        return real(path, fields)

    monkeypatch.setattr(pio, "read_particle_block", recording)
    edges = log_edges(1.0, 10.0, 20)
    spec = particle_snapshot_spectrum(run_dir, 3, C, edges, backend="serial")
    assert requested == [("u",)] * 3
    u_all = np.concatenate([b[1] for b in blocks])
    assert np.array_equal(spec.counts, histogram(energy_values(u_all, C, "gamma"), edges, "gamma").counts)


# ------------------------------------------------------- on_error, nesting, MPI
def _truncate(path, nbytes=10):
    data = path.read_bytes()
    path.write_bytes(data[:-nbytes])


@pytest.mark.parametrize("backend", ["serial", "thread", "process"])
def test_particle_snapshot_spectrum_on_error(tmp_path, backend):
    run_dir = tmp_path / "run0042"
    blocks = make_blocks(n_blocks=4, n_per=60)
    write_blocks(run_dir, 3, 150.0, blocks, "bin")
    _truncate(run_dir / "synth.block2.parts.00003.par.bin")
    edges = log_edges(1.0, 10.0, 20)
    with pytest.raises(RuntimeError, match="block2"):
        particle_snapshot_spectrum(run_dir, 3, C, edges, backend=backend, n_workers=2)
    with pytest.warns(UserWarning, match=r"skipped 1 of 4 .*block2"):
        spec = particle_snapshot_spectrum(run_dir, 3, C, edges, backend=backend, n_workers=2, on_error="collect")
    u_ok = np.concatenate([b[1] for k, b in enumerate(blocks) if k != 2])
    assert np.array_equal(spec.counts, histogram(energy_values(u_ok, C, "gamma"), edges, "gamma").counts)
    assert spec.n_total == u_ok.shape[0] and spec.run_id == 42
    for k in (0, 1, 3):
        _truncate(run_dir / f"synth.block{k}.parts.00003.par.bin")
    with pytest.raises(RuntimeError, match="all 4 particle files"):
        particle_snapshot_spectrum(run_dir, 3, C, edges, backend="serial", on_error="collect")
    with pytest.raises(ValueError, match="on_error"):
        particle_snapshot_spectrum(run_dir, 3, C, edges, on_error="ignore")


def test_particle_snapshot_spectrum_nested_in_outer_map_runs_serially(tmp_path, monkeypatch):
    import shearpic.parallel as par

    run_dir = tmp_path / "run0042"
    blocks = make_blocks(n_blocks=3, n_per=50)
    for number, t in ((1, 10.0), (2, 20.0)):
        write_blocks(run_dir, number, t, blocks, "tab")
    decisions = []
    real = par._choose_backend

    def recording(backend, workers, n_items):
        chosen = real(backend, workers, n_items)
        decisions.append((backend, workers, chosen))
        return chosen

    monkeypatch.setattr(par, "_choose_backend", recording)
    edges = log_edges(1.0, 10.0, 20)
    specs = par.parallel_map(lambda k: particle_snapshot_spectrum(run_dir, k, C, edges), [1, 2],
                             backend="thread", n_workers=2)
    assert [s.time for s in specs] == [10.0, 20.0]
    inner = decisions[1:]  # decisions[0] is the outer map
    assert len(inner) == 2 and all(d == ("auto", 1, "serial") for d in inner)  # not N x N processes
    assert np.array_equal(specs[0].counts, specs[1].counts)


class _FakeCommWorld:
    """COMM_WORLD for MPI ranks simulated by threads (gather pickles and meets at a barrier)."""

    def __init__(self, size, timeout=5.0):
        self.size, self._rank = size, threading.local()
        self._barrier, self._slots = threading.Barrier(size, timeout=timeout), [None] * size

    def Get_rank(self):  # noqa: N802
        return self._rank.value

    def Get_size(self):  # noqa: N802
        return self.size

    def gather(self, obj, root=0):
        self._slots[self._rank.value] = pickle.loads(pickle.dumps(obj))
        self._barrier.wait()  # BrokenBarrierError if another rank never gets here
        return list(self._slots) if self._rank.value == root else None


def _run_fake_mpi(monkeypatch, size, call):
    world = _FakeCommWorld(size)
    mpi = types.ModuleType("mpi4py.MPI")
    mpi.COMM_WORLD = world
    pkg = types.ModuleType("mpi4py")
    pkg.MPI = mpi
    monkeypatch.setitem(sys.modules, "mpi4py", pkg)
    monkeypatch.setitem(sys.modules, "mpi4py.MPI", mpi)
    outcomes = [None] * size

    def rank_main(rank):
        world._rank.value = rank
        try:
            outcomes[rank] = ("ok", call())
        except BaseException as err:  # noqa: BLE001
            outcomes[rank] = ("error", err)

    threads = [threading.Thread(target=rank_main, args=(r,), daemon=True) for r in range(size)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)
    assert not any(t.is_alive() for t in threads), "a simulated MPI rank hung"
    return outcomes


def test_particle_snapshot_spectrum_mpi_ranks(tmp_path, monkeypatch):
    run_dir = tmp_path / "run0042"
    blocks = make_blocks(n_blocks=5, n_per=40)
    write_blocks(run_dir, 3, 150.0, blocks, "bin")
    edges = log_edges(1.0, 10.0, 20)
    ref = particle_snapshot_spectrum(run_dir, 3, C, edges, backend="serial")
    out = _run_fake_mpi(monkeypatch, 2, lambda: particle_snapshot_spectrum(run_dir, 3, C, edges, backend="mpi"))
    assert out[0][0] == "ok" and np.array_equal(out[0][1].counts, ref.counts)
    assert out[0][1].n_total == ref.n_total and out[0][1].time == ref.time
    assert out[1] == ("ok", None)
    # a bad block on rank 1 (block3): rank 0 raises after the gather, rank 1 returns None, nobody hangs
    _truncate(run_dir / "synth.block3.parts.00003.par.bin")
    out = _run_fake_mpi(monkeypatch, 2, lambda: particle_snapshot_spectrum(run_dir, 3, C, edges, backend="mpi"))
    assert out[0][0] == "error" and isinstance(out[0][1], RuntimeError) and "block3" in str(out[0][1])
    assert out[1] == ("ok", None)
    with pytest.warns(UserWarning, match="skipped 1 of 5"):
        out = _run_fake_mpi(monkeypatch, 2, lambda: particle_snapshot_spectrum(run_dir, 3, C, edges, backend="mpi",
                                                                               on_error="collect"))
    assert out[0][0] == "ok" and out[0][1].n_total == ref.n_total - blocks[3][1].shape[0]
    assert out[1] == ("ok", None)
