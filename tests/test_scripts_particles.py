"""Smoke and end-to-end tests of analysis/energy_spectrum.py, analysis/plot_spectrum.py and analysis/phase_space.py.

The fast tests build a tiny synthetic run (4 meshblocks, 64 particles, three par.tab outputs, a history file and
a tracked-particle trajectory) in ``tmp_path``.  The ``data`` tests reproduce Figures 5 and 6 from the local
reference files (run364 legacy spectra and phase-space histograms, run355 trajectory).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
C = 50.0
N_BLOCK = 16


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"_analysis_{name}", ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def es():
    return load_script("energy_spectrum")


@pytest.fixture(scope="module")
def ps():
    return load_script("plot_spectrum")


@pytest.fixture(scope="module")
def phs():
    return load_script("phase_space")


ATHINPUT = """<job>
problem_id = synth

<output1>
file_type  = hst
dt         = 0.5

<output4>
file_type  = tab
dt         = 100.0

<time>
tlim       = 200.0

<mesh>
nx1        = 8
x1min      = -2.0
x1max      = 2.0
nx2        = 8
x2min      = -4.0
x2max      = 4.0
nx3        = 1
x3min      = -0.5
x3max      = 0.5

<meshblock>
nx1        = 4
nx2        = 4
nx3        = 1

<particles>
charge_over_mass_over_c = 200.0
speed_of_light = 50.0

<problem>
iprob = 0
shear_strength = 1.0
y1 = -2.0
y2 = 2.0
M_A = 10
tau = 0.5
npx1 = 8
npx2 = 8
npx3 = 1
vp_par = 50.0
cr_mass = 0.0005
"""

TIMES = {0: 0.0, 1: 100.00031, 2: 200.0002}


def block_particles(number: int, gid: int):
    rng = np.random.default_rng(100 * number + gid)
    scale = 40.0 * (1 + number)
    u = rng.normal(0.0, scale, size=(N_BLOCK, 3))
    x = np.column_stack([rng.uniform(-2, 2, N_BLOCK), rng.uniform(-4, 4, N_BLOCK), np.zeros(N_BLOCK)])
    return x, u


def write_output(run_dir: Path, number: int, time: float, gids=range(4)):
    for gid in gids:
        x, u = block_particles(number, gid)
        lines = [f"# Athena++ particle data at time = {time:.18g}", "born_meshblock  particle_id  x  y  z  vx  vy  vz"]
        lines += ["  ".join([str(gid), str(k)] + [f"{v:.18g}" for v in (*x[k], *u[k])]) for k in range(N_BLOCK)]
        (run_dir / f"synth.block{gid}.out4.{number:05d}.par.tab").write_text("\n".join(lines) + "\n")


def all_particles(number: int):
    parts = [block_particles(number, g) for g in range(4)]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


def write_hst(run_dir: Path, scale: float = 1.0):
    from shearpic.physics.relativity import kinetic_energy_per_mass

    names = ["time", "dt", "np", "KE_cr"]
    rows = []
    for number, t in TIMES.items():
        _, u = all_particles(number)
        rows.append([t, 1e-3, 64.0, scale * float(np.sum(kinetic_energy_per_mass(u, C)))])
    header = "# Athena++ history data\n# " + "  ".join(f"[{i + 1}]={n}" for i, n in enumerate(names)) + "\n"
    body = "".join("  ".join(f"{v:.12e}" for v in r) + "\n" for r in rows)
    (run_dir / "synth.hst").write_text(header + body)


@pytest.fixture
def run(tmp_path):
    root = tmp_path / "data"
    run_dir = root / "run0042"
    run_dir.mkdir(parents=True)
    (run_dir / "athinput.kh_org").write_text(ATHINPUT)
    for number, t in TIMES.items():
        write_output(run_dir, number, t)
    write_hst(run_dir)
    return run_dir


def common_args(tmp_path, run_dir):
    return ["--data-root", str(run_dir.parent), "--products", str(tmp_path / "products"), "--out",
            str(tmp_path / "figs"), "--backend", "serial", "--formats", "png", "--dpi", "50"]


# ================================================================ energy_spectrum.py
def test_parse_outputs(es):
    avail = [0, 1, 2, 5]
    assert es.parse_outputs("all", avail) == avail
    assert es.parse_outputs("1:4", avail) == [1, 2]
    assert es.parse_outputs(":1,5", avail) == [0, 1, 5]
    with pytest.raises(ValueError, match="does not exist"):
        es.parse_outputs("3", avail)
    assert es.format_numbers([0, 1, 2]) == "0..2" and es.format_numbers([1, 5]) == "1, 5"
    # reversed or empty selections are errors that name the available outputs
    with pytest.raises(ValueError, match=r"'2:1' is reversed .*available: 0, 1, 2, 5"):
        es.parse_outputs("2:1", avail)
    with pytest.raises(ValueError, match=r"'3:4' contains no particle output .*available: 0, 1, 2, 5"):
        es.parse_outputs("3:4", avail)
    with pytest.raises(ValueError, match="selects no particle output"):
        es.parse_outputs(" , ", avail)
    with pytest.raises(ValueError, match="cannot parse"):
        es.parse_outputs("a:3", avail)
    assert es.parse_outputs("all", []) == []


def test_energy_spectrum_compute_skip_force_and_checks(es, run, tmp_path, capsys):
    from shearpic.io.spectrum_files import load_spectrum
    from shearpic.physics.spectra import energy_values, histogram, log_edges

    edges_spec = "log:1:100:2000"  # fine bins: the midpoint estimate of <gamma-1> is within 1e-3
    args = ["--run", "run42", "--edges", edges_spec, "--hst-check", *common_args(tmp_path, run)]
    assert es.main(args) == 0
    products = tmp_path / "products"
    files = sorted(p.name for p in products.iterdir())
    assert files == [f"spectrum_gamma_synth.out4.tab_{n:05d}.npz" for n in TIMES]
    for number, t in TIMES.items():
        spec, meta = load_spectrum(products / f"spectrum_gamma_synth.out4.tab_{number:05d}.npz", with_metadata=True)
        _, u = all_particles(number)
        ref = histogram(energy_values(u, C, "gamma"), log_edges(1, 100, 2000), "gamma")
        assert np.array_equal(spec.counts, ref.counts) and spec.overflow == ref.overflow and spec.n_total == 64
        assert spec.time == pytest.approx(t) and spec.c == C and spec.run_id == 42
        assert meta["output_number"] == number
        assert meta["stream"] == {"basename": "synth", "file_id": "out4", "kind": "tab"}
        assert meta["file_time"] == pytest.approx(t)
        assert meta["n_block_files"] == 4 and meta["hst_check"]["status"] == "ok"
        assert meta["run"]["athinput_sha256"] and meta["edges_spec"] == edges_spec
    out = capsys.readouterr().out
    assert "strictly increasing" in out and out.count("computed") == 3
    # rerun: every output is skipped; --force recomputes
    assert es.main(args) == 0
    out = capsys.readouterr().out
    assert out.count("skipped (valid product)") == 3
    # --hst-check still checks the kept products (the hst column is filled, not '-')
    kept = [line.split() for line in out.splitlines() if line.endswith("skipped (valid product)")]
    assert len(kept) == 3 and all(tokens[-4] != "-" for tokens in kept)
    write_hst(run, scale=1.01)
    assert es.main(args) == 1
    assert "hst check failed (existing product)" in capsys.readouterr().err
    write_hst(run)
    assert es.main([*args, "--force", "--outputs", "1"]) == 0
    assert capsys.readouterr().out.count("computed") == 1
    # a wrong particle number is an error, not a product
    assert es.main(["--run", "run42", "--edges", "log:1:10:20", "--expected-n", "65", *common_args(tmp_path, run)]) == 1
    assert "expected 65" in capsys.readouterr().err
    assert not (products / "x").exists()


def test_energy_spectrum_failures(es, run, tmp_path, capsys):
    base = ["--run", "run42", "--edges", "log:1:100:2000", *common_args(tmp_path, run)]
    write_hst(run, scale=1.01)  # history disagrees by 1%
    assert es.main([*base, "--hst-check"]) == 1
    assert "hst check failed" in capsys.readouterr().err
    assert not list((tmp_path / "products").glob("*.npz"))  # nothing written
    write_hst(run)
    (run / "synth.block3.out4.00002.par.tab").unlink()  # a missing block
    assert es.main(base) == 1
    assert "3 block files, the mesh has 4" in capsys.readouterr().err
    write_output(run, 2, TIMES[2])
    write_output(run, 3, 150.0)  # time goes backwards
    assert es.main(base) == 1
    assert "does not exceed" in capsys.readouterr().err


def test_energy_spectrum_mpi_rank_helper(es, monkeypatch):
    monkeypatch.delenv("PMI_SIZE", raising=False)
    assert es.mpi_rank("auto") == 0 and es.mpi_rank("serial") == 0


def test_hst_check_logic(es):
    from shearpic.physics.spectra import histogram

    spec = histogram([1.5, 2.5], [1.0, 2.0, 3.0], "gamma_minus_1")
    assert es.hst_check(spec, None) == {"status": "not covered"}
    ok = es.hst_check(spec, {"time": 0.0, "mean_gamma_minus_1": 2.0})
    assert ok["status"] == "ok" and ok["rel_diff"] == 0.0
    assert es.hst_check(spec, {"time": 0.0, "mean_gamma_minus_1": 2.01})["status"] == "FAILED"


# ================================================================ plot_spectrum.py
def test_plot_spectrum_from_npz_products(es, ps, run, tmp_path, capsys):
    assert es.main(["--run", "run42", "--edges", "log:1:10:40", *common_args(tmp_path, run)]) == 0
    capsys.readouterr()
    args = ["--run", "run42", "--source", "npz", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
            "--cutoff-level", "0.05", *common_args(tmp_path, run)]
    assert ps.main(args) == 0
    out = capsys.readouterr().out
    assert "Fig. 5 statistics" in out and "wrote" in out
    side = json.loads((tmp_path / "figs" / "particle_spectrum.json").read_text())
    rows = side["statistics"]["per_output"]
    assert [r["time"] for r in rows] == pytest.approx(list(TIMES.values()))
    assert side["inputs"]["source"] == "npz"
    assert rows[1]["hst_mean_gamma_minus_1"] == pytest.approx(rows[1]["mean_gamma_minus_1"], rel=0.05)
    assert (tmp_path / "figs" / "particle_spectrum.png").stat().st_size > 1000
    assert ps.main([*args, "--variable", "p_over_mc", "--style", "line", "--name", "p"]) == 0
    assert (tmp_path / "figs" / "p.png").exists()


def test_plot_spectrum_legacy_csvs(ps, run, tmp_path):
    from shearpic.physics.spectra import energy_values

    d = run / "energy_spectrum_data"
    d.mkdir()
    edges = np.logspace(0, 1, 41)
    for number, t in TIMES.items():
        _, u = all_particles(number)
        density, _ = np.histogram(energy_values(u, C, "gamma"), bins=edges, density=True)
        pd.DataFrame({"bin_centers": 0.5 * (edges[1:] + edges[:-1]), "density": density}).to_csv(
            d / f"histogram_frame_{number:05d}_t_{round(t):.1f}.csv", index=False)
    pd.DataFrame({"num_frames": [4], "frame_numbers": ["[0, 1, 2, 3]"], "times": ["[]"], "nbins": [40]}).to_csv(
        d / "histogram_metadata.csv", index=False)
    args = ["--run", "run42", "--source", "legacy", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
            *common_args(tmp_path, run)]
    with pytest.warns(UserWarning, match=r"frames \[3\]"):
        assert ps.main(args) == 0
    side = json.loads((tmp_path / "figs" / "particle_spectrum.json").read_text())
    assert side["inputs"]["missing_frames"] == [3]
    # counts recovered exactly (some particles are above gamma = 10: overflow)
    over = [r["overflow"] for r in side["statistics"]["per_output"]]
    x2, u2 = all_particles(2)
    assert over[2] == int(np.sum(energy_values(u2, C, "gamma") > 10.0))
    assert side["inputs"]["counts_recovered"] is True


def test_plot_spectrum_time_checks(ps):
    from shearpic.physics.spectra import Spectrum

    s = [Spectrum("gamma", [1, 2], counts=[1], time=t) for t in (0.0, 100.0, 100.0)]
    with pytest.raises(ValueError, match="strictly increasing"):
        ps.check_times(s, [0, 1, 2])
    with pytest.raises(ValueError, match="without a time"):
        ps.check_times([Spectrum("gamma", [1, 2], counts=[1])], [0])
    assert list(ps.check_times(s[:2], [0, 1])) == [0.0, 100.0]


def test_cutoff_growth(ps):
    rows = [{"time": t, "gamma_cut": 1.0 + 0.01 * t} for t in (0.0, 300.0, 600.0, 1000.0, 1500.0, 2000.0)]
    g = ps.cutoff_growth(rows)
    assert g["rate_300_1000"] == pytest.approx(0.01) and g["linear_fit_from_1000"]["slope"] == pytest.approx(0.01)
    assert g["linear_fit_from_300"]["max_abs_residual"] < 1e-9


# ================================================================ phase_space.py
def write_trajectory(run_dir: Path) -> str:
    rel = "high_energy_particles/trajectory_initmbid_1_pid_2.tab"
    path = run_dir / rel
    path.parent.mkdir(parents=True)
    t = np.arange(0.0, 20.0, 0.05)
    B0, q, cc = 0.5, 10.0, 25.0
    u_perp = 60.0
    gamma = np.sqrt(1 + u_perp**2 / cc**2)
    om = q * B0 / gamma
    y = 3.0 + (u_perp / (q * B0)) * np.sin(om * t)
    y = (y + 4.0) % 8.0 - 4.0  # periodic box, wraps once in a while
    cols = [t, np.zeros_like(t), y, np.zeros_like(t), 5 + 0 * t, u_perp * np.cos(om * t), -u_perp * np.sin(om * t),
            B0 + 0 * t, 0 * t, 0 * t, 0 * t, 0 * t, 0 * t]
    np.savetxt(path, np.column_stack(cols), fmt="%.10e")
    return rel


@pytest.fixture
def traj_run(tmp_path):
    run_dir = tmp_path / "data" / "run0043"
    run_dir.mkdir(parents=True)
    text = (ATHINPUT.replace("speed_of_light = 50.0", "speed_of_light = 25.0")
            .replace("charge_over_mass_over_c = 200.0", "charge_over_mass_over_c = 10.0")
            .replace("M_A = 10", "M_A = 2").replace("shear_strength = 1.0", "shear_strength = 2.0"))
    (run_dir / "athinput.kh_org").write_text(text)
    return run_dir, write_trajectory(run_dir)


def test_phase_space_helpers(phs):
    assert phs.nice_limit(3.106) == 3.5 and phs.nice_limit(3.0) == 3.5 and phs.nice_limit(2.7) == 3.0
    assert phs.axis_ticks(-3.5, 3.5) == [-2.0, 0.0, 2.0]
    # no tick label at a shared column edge
    assert phs.axis_ticks(-2, 2) == [-1.0, 0.0, 1.0] and phs.axis_ticks(-12, 12) == [-5.0, 0.0, 5.0]
    for lo, hi in ((-3, 3), (-6, 6), (-8, 8), (-3.5, 12), (0, 1), (-150, 150)):
        ticks = phs.axis_ticks(lo, hi)
        assert 2 <= len(ticks) <= 5 and all(lo + 0.12 * (hi - lo) - 1e-9 <= v <= hi - 0.12 * (hi - lo) + 1e-9
                                            for v in ticks)
    assert phs.row_limits(3.1, (-3, 3), None, None, False) == ((-3.5, 3.5), (-3.5, 3.5))
    assert phs.row_limits(3.1, (-3, 3), None, None, True) == ((-3.5, 3.5), (-3, 3))
    assert phs.row_limits(1.0, (-12, 12), (-2, 2), [-4, 4], False) == ((-4, 4), (-4, 4))
    seg, keep = phs._segments(np.array([0.0, 1.0, 2.0, 3.0]), np.array([3.5, 3.9, -3.9, -3.5]), 8.0)
    assert keep.tolist() == [True, False, True] and seg.shape == (2, 2, 2)


def test_phase_space_compute_and_plot(phs, run, traj_run, tmp_path, capsys):
    from shearpic.io.spectrum_files import load_phase_space
    from shearpic.physics.phase_space import phase_space_histogram

    traj_dir, traj_file = traj_run
    common = common_args(tmp_path, run)
    compute = ["compute", "--run", "run42", "--time", "100", "--u-edges", "lin:-150:150:30", "--y-bins", "8",
               *common]
    assert phs.main(compute) == 0
    product = tmp_path / "products" / "phase_space_synth.out4.tab_00001.npz"
    hists, meta = load_phase_space(product, with_metadata=True)
    x, u = all_particles(1)
    for j, k in enumerate("xyz"):
        ref = phase_space_histogram(u[:, j], x[:, 1], np.linspace(-150, 150, 31), np.linspace(-4, 4, 9), component=k)
        assert np.array_equal(hists[k].counts, ref.counts) and np.array_equal(hists[k].n_y, ref.n_y)
        assert hists[k].n_total == 64 and hists[k].c == C
        assert meta["out_of_range_fraction"][k] == pytest.approx(ref.out_of_range_fraction())
    assert meta["file_time"] == pytest.approx(TIMES[1]) and meta["output_number"] == 1
    capsys.readouterr()
    assert phs.main(compute) == 0
    assert "exists and matches" in capsys.readouterr().out
    with pytest.raises(FileNotFoundError, match="no particle output within"):
        phs.main(["compute", "--run", "run42", "--time", "400", "--y-bins", "8", *common])

    plot = ["plot", "--hist-source", "npz", "--hist-run", "run42", "--time", "100", "--traj-run", "run43",
            "--traj-file", traj_file, "--t-window", "5", "15", "--xlim-hist", "-3", "3", "--min-particles-per-bin", "0.5",
            *common]
    assert phs.main(plot) == 0
    out = capsys.readouterr().out
    assert "layer asymmetry" in out and "caption note" in out
    side = json.loads((tmp_path / "figs" / "phase_space.json").read_text())
    orbit = side["statistics"]["orbit"]
    assert orbit["gamma_min"] == pytest.approx(np.sqrt(1 + (5**2 + 60**2) / 25**2), rel=1e-9)
    assert orbit["t_first"] >= 5 and orbit["t_last"] <= 15
    assert side["statistics"]["histogram"]["moments_source"]["z"] == "sums"
    assert side["layout"]["xlim_hist"] == side["layout"]["xlim_traj"]  # shared axis per column
    assert side["colour_scale"]["vmin"] == pytest.approx(0.5 * C / (64 * 10.0 * 1.0))
    assert (tmp_path / "figs" / "phase_space.png").exists()


def test_phase_space_plot_legacy_and_all(phs, run, traj_run, tmp_path):
    traj_dir, traj_file = traj_run
    x, u = all_particles(2)
    edge_p, edge_y = np.linspace(-150, 150, 61), np.linspace(-4, 4, 11)
    data = {f"hist_p{k}": np.histogram2d(u[:, j], x[:, 1], bins=[edge_p, edge_y], density=True)[0]
            for j, k in enumerate("xyz")}
    np.savez(run / "phase_space_histograms.npz", edge_p=edge_p, edge_y=edge_y, y_min=-4.0, y_max=4.0, **data)
    common = common_args(tmp_path, run)
    args = ["plot", "--hist-source", "legacy", "--hist-run", "run42", "--time", "200", "--n-total", "64",
            "--traj-run", "run43", "--traj-file", traj_file, "--t-window", "0", "10", "--separate-xlim", "--xlim-traj", "-4", "4",
            "--name", "legacy", *common]
    with pytest.warns(UserWarning, match="no bin holds 10 particles"):
        assert phs.main(args) == 0
    side = json.loads((tmp_path / "figs" / "legacy.json").read_text())
    oor = side["statistics"]["histogram"]["out_of_range_fraction"]
    assert oor["x"] == pytest.approx(np.mean(np.abs(u[:, 0]) > 150))
    assert side["statistics"]["histogram"]["moments_source"]["x"] == "bins"
    assert not side["layout"]["shared_x"]
    # compute + plot in one go
    assert phs.main(["all", "--run", "run42", "--time", "0", "--u-edges", "lin:-300:300:60", "--y-bins", "8",
                     "--traj-run", "run43", "--traj-file", traj_file, "--t-window", "0", "10", "--name", "all",
                     *common]) == 0
    assert (tmp_path / "products" / "phase_space_synth.out4.tab_00000.npz").exists()
    assert (tmp_path / "figs" / "all.png").exists()
    side = json.loads((tmp_path / "figs" / "all.json").read_text())
    assert side["histogram"]["selection"] == {"explicit": True}  # the product just computed
    assert side["layout"]["xlim_hist_source"].startswith("visible")


# ================================================================ data


@pytest.mark.data
def test_fig5_from_run364_legacy(ps, data_root, tmp_path):
    if not (data_root / "run364" / "energy_spectrum_data").is_dir():
        pytest.skip("run364 legacy spectra not available")
    assert ps.main(["--data-root", str(data_root), "--out", str(tmp_path), "--formats", "png", "--dpi", "40"]) == 0
    side = json.loads((tmp_path / "particle_spectrum.json").read_text())
    rows = {r["time"]: r for r in side["statistics"]["per_output"]}
    assert len(rows) == 25 and side["inputs"]["counts_recovered"] and side["inputs"]["missing_frames"] == []
    assert rows[2400.0]["alpha"] == pytest.approx(0.672, abs=1e-3)
    assert rows[1800.0]["gamma_cut"] == pytest.approx(12.590, abs=1e-3)
    assert rows[600.0]["hst_mean_gamma_minus_1"] == pytest.approx(0.69837, abs=2e-5)
    assert side["statistics"]["gamma_cut_growth"]["rate_1000_2400"] == pytest.approx(0.00386, abs=2e-5)
    assert side["colour_scale"]["vmax"] == 2400.0
    # page size with tight_layout at font.size 10 and bbox 'tight' with a 0.1 in pad
    fitz = pytest.importorskip("fitz")
    assert ps.main(["--data-root", str(data_root), "--out", str(tmp_path), "--formats", "pdf"]) == 0
    with fitz.open(tmp_path / "particle_spectrum.pdf") as doc:
        rect = doc[0].rect
    assert rect.width == pytest.approx(713.8, abs=3.0) and rect.height == pytest.approx(567.5, abs=3.0)


@pytest.mark.data
def test_fig6_from_run364_and_run355(phs, data_root, tmp_path):
    for need in ("run364/phase_space_histograms.npz",
                 "run355/high_energy_particles/trajectory_initmbid_766_pid_30721.tab"):
        if not (data_root / need).exists():
            pytest.skip(f"{need} not available")
    assert phs.main(["plot", "--data-root", str(data_root), "--out", str(tmp_path), "--formats", "png",
                     "--dpi", "40"]) == 0
    side = json.loads((tmp_path / "phase_space.json").read_text())
    h, o = side["statistics"]["histogram"], side["statistics"]["orbit"]
    assert h["out_of_range_fraction"]["x"] == pytest.approx(0.045586, abs=1e-6)
    assert h["out_of_range_fraction"]["z"] == pytest.approx(0.041073, abs=1e-6)
    for layer in h["layers"]["z"]:
        assert layer["mean_below"] > 0 > layer["mean_above"] and layer["slope"] < 0
    assert (o["gamma_min"], o["gamma_max"]) == (pytest.approx(2.770, abs=1e-3), pytest.approx(3.308, abs=1e-3))
    assert o["n_samples"] == 209 and o["band_layer"] == pytest.approx(31.4159, abs=1e-4)
    assert side["layout"]["xlim_hist"] == [-3.5, 3.5] and side["layout"]["hist_x_range"] == [-3.0, 3.0]


def test_energy_spectrum_under_simulated_mpi(es, run, tmp_path, monkeypatch, capsys):
    """Two simulated MPI ranks: identical work lists, same collectives, only rank 0 writes; a failure hangs nobody."""
    from test_spectra import _run_fake_mpi

    from shearpic.io.spectrum_files import load_spectrum

    serial = tmp_path / "serial"
    base = ["--run", "run42", "--edges", "log:1:100:200", "--data-root", str(run.parent), "--formats", "png"]
    assert es.main([*base, "--products", str(serial), "--backend", "serial"]) == 0
    mpi_dir = tmp_path / "mpi"
    out = _run_fake_mpi(monkeypatch, 2, lambda: es.main([*base, "--products", str(mpi_dir), "--backend", "mpi"]))
    assert out == [("ok", 0), ("ok", 0)]
    for n in TIMES:
        a, b = (load_spectrum(d / f"spectrum_gamma_synth.out4.tab_{n:05d}.npz") for d in (serial, mpi_dir))
        assert np.array_equal(a.counts, b.counts) and a.time == b.time
    # a truncated block of output 1: rank 0 records the failure after the gather, both ranks finish output 2
    (run / "synth.block1.out4.00001.par.tab").write_text("# Athena++ particle data at time = 100\nbad header\n")
    fresh = tmp_path / "mpi2"
    out = _run_fake_mpi(monkeypatch, 2, lambda: es.main([*base, "--products", str(fresh), "--backend", "mpi"]))
    assert out[0] == ("ok", 1) and out[1][0] == "ok"
    assert sorted(p.name for p in fresh.glob("*.npz")) == ["spectrum_gamma_synth.out4.tab_00000.npz",
                                                          "spectrum_gamma_synth.out4.tab_00002.npz"]


# ================================================================ stream and time identity of products
BIN_TIMES = {0: 0.0, 1: 10.0, 2: 20.0}


def bin_particles(number: int, gid: int):
    """out5 bin stream: output 0 holds the same particles as out4.00000, later outputs different ones."""
    return block_particles(0 if number == 0 else 10 + number, gid)


def write_bin_output(run_dir: Path, number: int, time: float, dt: float = 10.0, gids=range(4)):
    from shearpic.io.particles import PARBIN_RECORD

    for gid in gids:
        x, u = bin_particles(number, gid)
        rec = np.zeros(N_BLOCK, dtype=PARBIN_RECORD)
        for j, k in enumerate("xyz"):
            rec[k] = x[:, j]
            rec["u" + k] = u[:, j]
        rec["pid"] = np.arange(N_BLOCK)
        rec["init_mbid"] = gid
        head = np.array([-2, 2, -4, 4, -0.5, 0.5, -2, 2, -4, 4, -0.5, 0.5, time, dt], dtype="<f4").tobytes()
        head += np.array([N_BLOCK], dtype="<i8").tobytes()
        (run_dir / f"synth.block{gid}.out5.{number:05d}.par.bin").write_bytes(head + rec.tobytes())


@pytest.fixture
def two_stream_run(run):
    """Two particle streams numbered 00000..00002: out4 par.tab every 100 (t = 0, 100, 200) and out5 par.bin
    every 10 (t = 0, 10, 20)."""
    text = ATHINPUT.replace("<time>", "<output5>\nfile_type  = bin\ndt         = 10.0\n\n<time>")
    (run / "athinput.kh_org").write_text(text)
    for number, t in BIN_TIMES.items():
        write_bin_output(run, number, t)
    return run


def test_two_particle_streams_of_different_cadence(es, ps, phs, two_stream_run, traj_run, tmp_path, capsys):
    from shearpic.io.spectrum_files import load_spectrum
    from shearpic.physics.spectra import energy_values, histogram, parse_edges

    run = two_stream_run
    _, traj_file = traj_run
    common = common_args(tmp_path, run)
    products = tmp_path / "products"
    base = ["--run", "run42", "--edges", "log:1:100:2000", *common]

    # energy spectra: the bin stream first, then the tab stream must be computed, not "skipped (valid product)"
    assert es.main([*base, "--kind", "bin"]) == 0
    assert capsys.readouterr().out.count("computed") == 3
    assert es.main([*base, "--kind", "tab", "--hst-check"]) == 0
    out = capsys.readouterr().out
    assert out.count("computed") == 3 and "skipped" not in out and "times 0 .. 200" in out
    assert sorted(p.name for p in products.glob("spectrum_*.npz")) == sorted(
        [f"spectrum_gamma_synth.out4.tab_{n:05d}.npz" for n in TIMES]
        + [f"spectrum_gamma_synth.out5.bin_{n:05d}.npz" for n in BIN_TIMES])
    edges = parse_edges("log:1:100:2000")
    for n in (1, 2):
        tab, tmeta = load_spectrum(products / f"spectrum_gamma_synth.out4.tab_{n:05d}.npz", with_metadata=True)
        binned, bmeta = load_spectrum(products / f"spectrum_gamma_synth.out5.bin_{n:05d}.npz", with_metadata=True)
        assert tab.time == pytest.approx(TIMES[n]) and tmeta["stream"]["kind"] == "tab"
        assert binned.time == pytest.approx(BIN_TIMES[n]) and bmeta["stream"] == {"basename": "synth",
                                                                                    "file_id": "out5", "kind": "bin"}
        u_tab = np.concatenate([block_particles(n, g)[1] for g in range(4)])
        u_bin = np.concatenate([bin_particles(n, g)[1] for g in range(4)]).astype(np.float32).astype(np.float64)
        assert np.array_equal(tab.counts, histogram(energy_values(u_tab, C, "gamma"), edges, "gamma").counts)
        assert np.array_equal(binned.counts, histogram(energy_values(u_bin, C, "gamma"), edges, "gamma").counts)
        assert tmeta["hst_check"]["status"] == "ok"
    assert es.main([*base, "--kind", "tab"]) == 0  # now each stream's products are valid for that stream
    assert capsys.readouterr().out.count("skipped (valid product)") == 3
    # a product renamed onto the other stream's name is not accepted as valid
    ok, why = es.check_existing(products / "spectrum_gamma_synth.out5.bin_00001.npz", "gamma", edges, 64,
                                stream=("synth", "out4", "tab"))
    assert not ok and why.startswith("different stream")

    # Fig 5 from npz: two streams -> choose one; each gives its own times
    pargs = ["--run", "run42", "--source", "npz", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
             "--cutoff-level", "0.05", *common]
    with pytest.raises(ValueError, match="several particle streams"):
        ps.main(pargs)
    assert ps.main([*pargs, "--kind", "tab"]) == 0
    side = json.loads((tmp_path / "figs" / "particle_spectrum.json").read_text())
    assert [r["time"] for r in side["statistics"]["per_output"]] == pytest.approx(list(TIMES.values()))
    assert side["inputs"]["stream"] == {"basename": "synth", "file_id": "out4", "kind": "tab"}
    assert side["inputs"]["other_streams_present"] == ["synth.out5.bin"]
    assert ps.main([*pargs, "--file-id", "out5", "--t-range", "5", "25", "--name", "bin"]) == 0
    side = json.loads((tmp_path / "figs" / "bin.json").read_text())
    assert [r["time"] for r in side["statistics"]["per_output"]] == pytest.approx([10.0, 20.0])
    with pytest.raises(FileNotFoundError, match="present: "):
        ps.main([*pargs, "--basename", "other"])
    capsys.readouterr()

    # Fig 6: after t = 20 from the bin stream, t = 200 from the tab stream is computed, not taken as existing
    compute = ["compute", "--run", "run42", "--u-edges", "lin:-150:150:30", "--y-bins", "8", *common]
    assert phs.main([*compute, "--time", "20", "--kind", "bin"]) == 0
    capsys.readouterr()
    assert phs.main([*compute, "--time", "200", "--kind", "tab"]) == 0
    assert "exists and matches" not in capsys.readouterr().out
    assert (products / "phase_space_synth.out4.tab_00002.npz").exists()
    assert (products / "phase_space_synth.out5.bin_00002.npz").exists()
    plot = ["plot", "--hist-source", "npz", "--hist-run", "run42", "--traj-run", "run43", "--traj-file", traj_file,
            "--t-window", "5", "15", "--min-particles-per-bin", "0.5", *common]
    with pytest.raises(ValueError, match="several particle streams"):
        phs.main([*plot, "--time", "200"])
    assert phs.main([*plot, "--time", "200", "--kind", "tab"]) == 0
    side = json.loads((tmp_path / "figs" / "phase_space.json").read_text())
    assert side["histogram"]["time"] == pytest.approx(TIMES[2])
    assert side["histogram"]["selection"]["stream"]["kind"] == "tab"
    assert side["histogram"]["selection"]["tolerance"] == pytest.approx(50.0)
    # the bin stream's cadence is 10: t = 200 is not within 5 of its t = 20 product
    with pytest.raises(FileNotFoundError, match=r"within 5 of t = 200 .*t = 20"):
        phs.main([*plot, "--time", "200", "--kind", "bin"])
    assert phs.main([*plot, "--time", "21", "--kind", "bin", "--name", "bin6"]) == 0
    assert json.loads((tmp_path / "figs" / "bin6.json").read_text())["histogram"]["time"] == pytest.approx(20.0)


def test_energy_spectrum_recomputes_rewritten_outputs_and_other_physics(es, ps, run, tmp_path, capsys):
    base = ["--run", "run42", "--edges", "log:1:100:200", *common_args(tmp_path, run)]
    assert es.main(base) == 0
    capsys.readouterr()
    # an output rewritten with another header time (e.g. a rerun of the run) is recomputed
    write_output(run, 1, 100.5)
    assert es.main(base) == 0
    out = capsys.readouterr().out
    assert "recomputing spectrum_gamma_synth.out4.tab_00001.npz: different time" in out
    assert out.count("skipped (valid product)") == 2 and out.count("computed") == 1
    # tlim edited at a restart: every product is kept and the note is printed once, not per product
    (run / "athinput.kh_org").write_text(ATHINPUT.replace("tlim       = 200.0", "tlim       = 900.0"))
    assert es.main(base) == 0
    cap = capsys.readouterr()
    assert cap.out.count("skipped (valid product)") == 3 and cap.err.count("different sha256") == 1
    pargs = ["--run", "run42", "--source", "npz", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
             "--cutoff-level", "0.05", *common_args(tmp_path, run)]
    assert ps.main(pargs) == 0  # the plot stage accepts it too
    assert "different sha256" in capsys.readouterr().err
    # a different speed of light: the plot refuses the products, the producer recomputes them
    (run / "athinput.kh_org").write_text(ATHINPUT.replace("speed_of_light = 50.0", "speed_of_light = 60.0"))
    with pytest.raises(ValueError, match="different run .*c: 50"):
        ps.main(pargs)
    assert es.main(base) == 0
    out = capsys.readouterr().out
    assert out.count("computed") == 3 and "different run parameters" in out


def test_plot_spectrum_reads_old_style_product_names(es, ps, run, tmp_path):
    common = common_args(tmp_path, run)
    assert es.main(["--run", "run42", "--edges", "log:1:10:40", *common]) == 0
    products = tmp_path / "products"
    for n in TIMES:  # names written before the stream was part of the name
        (products / f"spectrum_gamma_synth.out4.tab_{n:05d}.npz").rename(products / f"spectrum_gamma_{n:05d}.npz")
    pargs = ["--run", "run42", "--source", "npz", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
             "--cutoff-level", "0.05", *common]
    assert ps.main(pargs) == 0
    side = json.loads((tmp_path / "figs" / "particle_spectrum.json").read_text())
    assert side["inputs"]["old_style_names"] == [f"spectrum_gamma_{n:05d}.npz" for n in TIMES]
    assert side["inputs"]["stream"]["kind"] == "tab"
    # a recomputed output under the new name wins over the old name
    assert es.main(["--run", "run42", "--edges", "log:1:10:40", "--outputs", "1", *common]) == 0
    assert ps.main(pargs) == 0
    side = json.loads((tmp_path / "figs" / "particle_spectrum.json").read_text())
    assert "spectrum_gamma_synth.out4.tab_00001.npz" in side["inputs"]["files"]
    assert "spectrum_gamma_00001.npz" not in side["inputs"]["files"]
    # a product renamed to another stream's name is refused
    (products / "spectrum_gamma_00002.npz").rename(products / "spectrum_gamma_synth.out5.bin_00002.npz")
    with pytest.raises(ValueError, match="renamed file"):
        ps.main(pargs)


def test_plot_stages_do_not_create_product_directories(ps, phs, run, traj_run, tmp_path):
    _, traj_file = traj_run
    common = common_args(tmp_path, run)
    with pytest.raises(FileNotFoundError, match="does not exist"):
        ps.main(["--run", "run42", "--source", "npz", "--bin-variable", "gamma", *common])
    with pytest.raises(FileNotFoundError, match="does not exist"):
        phs.main(["plot", "--hist-source", "npz", "--hist-run", "run42", "--time", "100", "--traj-run", "run43",
                  "--traj-file", traj_file, *common])
    assert not (tmp_path / "products").exists()


def _script_env():
    import os

    env = dict(os.environ)
    env["PYTHONPATH"] = str(ANALYSIS.parent / "src") + os.pathsep + env.get("PYTHONPATH", "")
    env.pop("SHEARPIC_DEBUG", None)
    env["MPLBACKEND"] = "Agg"
    return env


def test_scripts_report_user_errors_in_one_line(run, traj_run, tmp_path):
    import subprocess
    import sys

    _, traj_file = traj_run
    common = common_args(tmp_path, run)
    r = subprocess.run([sys.executable, str(ANALYSIS / "energy_spectrum.py"), "--run", "run42", "--outputs", "2:1",
                        *common], capture_output=True, text=True, env=_script_env(), timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr
    assert "error: output range '2:1' is reversed" in r.stderr and "available: 0..2" in r.stderr
    r = subprocess.run([sys.executable, str(ANALYSIS / "phase_space.py"), "plot", "--hist-source", "npz",
                        "--hist-run", "run42", "--time", "100", "--traj-run", "run43", "--traj-file", traj_file,
                        *common], capture_output=True, text=True, env=_script_env(), timeout=120)
    assert r.returncode == 2 and "Traceback" not in r.stderr and "error: products directory" in r.stderr
    assert not (tmp_path / "products").exists()
    for name in ("energy_spectrum", "plot_spectrum", "phase_space"):
        assert 'if __name__ == "__main__":\n    run_cli(main)' in (ANALYSIS / f"{name}.py").read_text()


def test_wrong_preset_is_a_readable_error(es, ps, phs, tmp_path):
    with pytest.raises(KeyError, match="preset 'fig4_steady_state' has no run; pass --run"):
        es.main(["--preset", "fig4_steady_state"])
    with pytest.raises(KeyError, match="has no variable"):
        ps.main(["--preset", "fig4_steady_state", "--run", "run42"])
    with pytest.raises(KeyError, match=r"fig4_steady_state \(histogram\)' has no run"):
        phs.main(["compute", "--preset", "fig4_steady_state"])
    with pytest.raises(KeyError, match=r"fig4_steady_state \(histogram\)' has no run"):
        phs.main(["plot", "--preset", "fig4_steady_state"])


def _capture_saved_figure(module, monkeypatch):
    import matplotlib

    saved = {}
    real = module.save_figure

    def spy(fig, directory, name, **kw):
        saved[name] = {"fig": fig, "pad_inches": matplotlib.rcParams["savefig.pad_inches"],
                       "texts": [t.get_text() for ax in fig.axes for t in ax.texts],
                       "positions": [ax.get_position().bounds for ax in fig.axes]}
        return real(fig, directory, name, **kw)

    monkeypatch.setattr(module, "save_figure", spy)
    return saved


def test_phase_space_npz_limits_follow_the_product_and_annotate_cuts(phs, run, traj_run, tmp_path, monkeypatch):
    _, traj_file = traj_run
    common = common_args(tmp_path, run)
    # a product as wide as the HPC default (+-600 U0 = +-12 mc at c = 50) of particles with |u|/c up to ~7
    assert phs.main(["compute", "--run", "run42", "--time", "200", "--u-edges", "lin:-600:600:120", "--y-bins",
                     "8", *common]) == 0
    x, u = all_particles(2)
    p = u / C
    assert np.mean(np.abs(p) > 3.5) > 0.05  # a +-3.5 axis would hide these particles
    saved = _capture_saved_figure(phs, monkeypatch)
    plot = ["plot", "--hist-source", "npz", "--hist-run", "run42", "--time", "200", "--traj-run", "run43",
            "--traj-file", traj_file, "--t-window", "5", "15", "--min-particles-per-bin", "0.5", *common]
    assert phs.main(plot) == 0
    lay = json.loads((tmp_path / "figs" / "phase_space.json").read_text())["layout"]
    edges = np.linspace(-12, 12, 121)
    k = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, 119)
    need = float(max(np.abs(edges[k]).max(), np.abs(edges[k + 1]).max()))  # every occupied bin is >= vmin
    step = 0.5 if need <= 5 else (1.0 if need <= 10 else 2.0)
    lim = step * np.ceil(need / step - 1e-9)
    assert lay["xlim_hist_source"].startswith("visible") and lay["xlim_hist"] == [-lim, lim]
    assert lay["fraction_outside_displayed_x"] == {"x": 0.0, "y": 0.0, "z": 0.0}
    assert not any("%" in t for t in saved["phase_space"]["texts"])  # nothing cut, nothing annotated
    # full histogram range on request
    assert phs.main([*plot, "--xlim-hist-auto", "range", "--name", "range"]) == 0
    assert json.loads((tmp_path / "figs" / "range.json").read_text())["layout"]["xlim_hist"] == [-12.0, 12.0]
    # explicit limits that cut the data: exact fractions per side in the metadata and printed at the panel edges
    assert phs.main([*plot, "--xlim-hist", "-2", "2", "--separate-xlim", "--name", "narrow"]) == 0
    lay = json.loads((tmp_path / "figs" / "narrow.json").read_text())["layout"]
    for j, comp in enumerate("xyz"):
        left, right = lay["fraction_beyond_axis_in_histogram"][comp]
        centre = 0.5 * (edges[k[:, j]] + edges[k[:, j] + 1])
        assert left == pytest.approx(np.mean(centre < -2)) and right == pytest.approx(np.mean(centre > 2))
        assert lay["fraction_outside_displayed_x"][comp] == pytest.approx(np.mean(np.abs(centre) > 2))
        assert left + right > 0
    texts = [t for t in saved["narrow"]["texts"] if "%" in t]
    assert len(texts) == sum(int(np.any(p[:, j] < -2.0)) + int(np.any(p[:, j] > 2.0)) for j in range(3))
    assert all(t.startswith(r"$\leftarrow") or t.endswith(r"\rightarrow$") for t in texts)
    assert not any(r"\geq" in t for t in texts)  # every particle is inside the histogram range


def test_phase_space_helpers_for_limits(phs):
    H = np.zeros((6, 2))
    xe = np.linspace(-3.0, 3.0, 7)
    H[1, 0] = 1.0  # bin [-2, -1] above vmin
    H[5, 1] = 0.1  # below vmin
    assert phs.visible_range({"x": (H, xe, None)}, 0.5, (-3.0, 3.0)) == (-2.0, 2.0)
    assert phs.visible_range({"x": (np.zeros((6, 2)), xe, None)}, 0.5, (-3.0, 3.0)) == (-3.0, 3.0)
    H2 = np.zeros((120, 2))
    H2[3, 0] = 1.0  # |edge| up to 11.6 -> multiple of 2 -> 12, clipped to the histogram range
    assert phs.visible_range({"x": (H2, np.linspace(-12, 12, 121), None)}, 0.5, (-12.0, 12.0)) == (-12.0, 12.0)
    assert phs.percent_text(0.0123) == "1.2%" and phs.percent_text(1e-7) == "<0.001%"
    assert phs.percent_text(0.5) == "50%"
    assert phs.PRODUCT_RE.match("phase_space_org.stir.feedback.out4.tab_00012.npz")["tag"] == \
        "org.stir.feedback.out4.tab"
    assert phs.PRODUCT_RE.match("phase_space_00012.npz")["tag"] is None
    assert phs.PRODUCT_RE.match("phase_space_histograms.npz") is None
    assert phs.product_name(("org.stir.feedback", "out4", "tab"), 12) == "phase_space_org.stir.feedback.out4.tab_00012.npz"


def test_fig5_layout_uses_tight_layout(ps, es, run, tmp_path, monkeypatch):
    assert es.main(["--run", "run42", "--edges", "log:1:10:40", *common_args(tmp_path, run)]) == 0
    saved = _capture_saved_figure(ps, monkeypatch)
    assert ps.main(["--run", "run42", "--source", "npz", "--bin-variable", "gamma", "--slope-range", "1.1", "3",
                    "--cutoff-level", "0.05", *common_args(tmp_path, run), "--formats", "pdf"]) == 0
    rec = saved["particle_spectrum"]
    assert rec["pad_inches"] == 0.1  # bbox_inches='tight' with the default pad
    x0, y0, w, h = rec["positions"][0]  # the data axes (the colorbar axes come second)
    assert w > 0.8 and x0 + w > 0.95  # tight_layout ran (default subplot params: w = 0.775, right edge 0.9)
