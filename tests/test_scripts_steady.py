"""Tests of analysis/steady_state.py (Figs. 3, 4, 7): synthetic runs in tmp_path, plus real-data runs.

    pytest tests/test_scripts_steady.py                                   # synthetic smoke tests
    PARTICLE_ACCEL_TEST_DATA=/path/to/results pytest -m data tests/test_scripts_steady.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"

ATHINPUT = """<job>
problem_id = synth.{kind}

<output1>
file_type = hst
dt = 0.5

<output2>
file_type = hdf5
variable = prim
dt = 10.0

<mesh>
nx1 = 32
x1min = -15.707963267948966
x1max = 15.707963267948966
nx2 = 128
x2min = -62.83185307179586
x2max = 62.83185307179586
nx3 = 1
x3min = -0.5
x3max = 0.5

<particles>
speed_of_light = 50.0
charge_over_mass_over_c = 200.0

<problem>
shear_strength = 1.0
y1 = -31.41592653589793
y2 = 31.41592653589793
vp_par = 50.0
cr_mass = 0.0005
tau = {tau}
npx1 = 64
npx2 = 256
npx3 = 1
"""

HST_COLUMNS = ["time", "dt", "1-KE", "2-KE", "3-KE", "1-ME", "2-ME", "3-ME", "np", "Pstir", "KE_cr",
               "Pideal", "Pideal_x", "Pideal_y", "Pideal_z"]
PRESETS = """schema_version: 1
fig3_turbulence:
  run: run0001
  output: out2
  times: [20, 30]
  fit_range: [0.5, 3.0]
  fit_weighting: log
  systematic_fit_ranges: [[0.5, 2.0], [0.5, 3.0], [1.0, 3.0]]
  local_slope_window: 0.5
  max_local_slope_change: 0.5
  n_per_decade: 8
  ratio_ranges: [[0.5, 2.0], [30.0, 80.0]]
  xlim: [0.15708, null]
  ylim: null
  ratio_ylim: [1.0e-2, 9.9]
  guide_offset: 4.0
  figure: turbulence
fig4_steady_state:
  driven: run0001
  decaying: run0002
  t_max: 50
  smoothing_period: 4.0
  ylim_top: [0.001, 0.61]
  ylim_bottom: [0.49, 0.88]
  ytick_top: 0.1
  ytick_bottom: 0.05
  figure: steady_state
fig7_power:
  run: run0001
  t_steady: [10, 50]
  smoothing_period: 4.0
  xlim: [0, 60]
  ylim: [-1, 4]
  ratio_threshold: 0.98
  figure: power
"""


# ------------------------------------------------------------------ fixtures
@pytest.fixture(scope="module")
def script():
    if str(ANALYSIS) not in sys.path:
        sys.path.insert(0, str(ANALYSIS))
    spec = importlib.util.spec_from_file_location("steady_state_script", ANALYSIS / "steady_state.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_hst(path: Path, t: np.ndarray, cfg_m_cr: float, N: float, driven: bool) -> None:
    K0, k = 1036.2 * N, 0.04 * N          # eps_p: 0.518 -> 0.568 over t = 50
    ke_cr = K0 + k * t ** 2
    P = 2.0 * cfg_m_cr * k * t
    data = {
        "time": t, "dt": np.full(t.size, 1e-3),
        "1-KE": 1911.0 + (60.0 if driven else -20.0) * np.tanh(t / 20.0), "2-KE": 5.0 + 0 * t, "3-KE": 0 * t,
        "1-ME": 19.7 + 100 * (1 - np.exp(-t / 10)), "2-ME": 5.0 + 0 * t, "3-ME": 0 * t,
        "np": np.full(t.size, N), "Pstir": np.full(t.size, 28.0 / (np.pi * 10 * 40 * np.pi / (32 * 128))),
        "KE_cr": ke_cr, "Pideal": P, "Pideal_x": 0.01 * P, "Pideal_y": -0.01 * P, "Pideal_z": P,
    }
    header = "# Athena++ history data\n# " + " ".join(f"[{i + 1}]={n}" for i, n in enumerate(HST_COLUMNS)) + "\n"
    rows = np.column_stack([data[c] for c in HST_COLUMNS])
    with open(path, "w") as fh:
        fh.write(header)
        np.savetxt(fh, rows, fmt="%.10e")


def _write_snapshot(path: Path, time: float, seed: int) -> None:
    nx, ny = 32, 128
    Lx, Ly = 10 * np.pi, 40 * np.pi
    x = (np.arange(nx) + 0.5) * Lx / nx - Lx / 2
    y = (np.arange(ny) + 0.5) * Ly / ny - Ly / 2
    X, Y = np.meshgrid(x, y, indexing="ij")
    rng = np.random.default_rng(seed)
    fields = {
        "rho": 1.0 + 0.05 * np.cos(2 * np.pi * X / Lx) + 0.01 * rng.random((nx, ny)),
        "vel1": -1 + np.tanh((Y + 31.4) / 1) - np.tanh((Y - 31.4) / 1) + 0.1 * rng.normal(size=(nx, ny)),
        "vel2": 0.1 * np.sin(2 * np.pi * 2 * X / Lx + Y / 4) + 0.05 * rng.normal(size=(nx, ny)),
        "vel3": 0.02 * rng.normal(size=(nx, ny)),
        "Bcc1": 0.1 + 0.05 * np.cos(2 * np.pi * 3 * X / Lx) + 0.03 * rng.normal(size=(nx, ny)),
        "Bcc2": 0.05 * rng.normal(size=(nx, ny)),
        "Bcc3": 0.01 * rng.normal(size=(nx, ny)),
    }
    names = list(fields)
    data = np.stack([fields[n].astype(np.float32).T[None, None] for n in names])  # (nvar, nblock, nz, ny, nx)
    with h5py.File(path, "w") as h:
        a = h.attrs
        a["DatasetNames"] = np.array([b"prim"], dtype="S21")
        a["NumVariables"] = np.array([len(names)], dtype=">i4")
        a["VariableNames"] = np.array([n.encode() for n in names], dtype="S21")
        a["RootGridSize"] = np.array((nx, ny, 1), dtype=">i4")
        a["MeshBlockSize"] = np.array((nx, ny, 1), dtype=">i4")
        for d, (lo, hi) in enumerate(((-Lx / 2, Lx / 2), (-Ly / 2, Ly / 2), (-0.5, 0.5))):
            a[f"RootGridX{d + 1}"] = np.array([lo, hi, 1.0], dtype=np.float32)
        a["Time"] = time
        a["NumCycles"] = 10
        a["NumMeshBlocks"] = 1
        a["MaxLevel"] = 0
        h["prim"] = data
        h["LogicalLocations"] = np.zeros((1, 3), dtype=">i8")
        h["Levels"] = np.zeros(1, dtype=">i4")


@pytest.fixture
def synthetic(tmp_path):
    """Data root with a driven run0001 (hst + 3 snapshots) and a decaying run0002 (hst on another cadence)."""
    from shearpic.config import RunConfig

    root = tmp_path / "data"
    driven, decaying = root / "run0001", root / "run0002"
    for d, kind, tau in ((driven, "stir", 0.5), (decaying, "decay", -0.5)):
        d.mkdir(parents=True)
        (d / "athinput.kh").write_text(ATHINPUT.format(kind=kind, tau=tau))
    cfg = RunConfig.from_run_dir(driven)
    _write_hst(driven / "synth.stir.hst", np.arange(0.0, 60.0 + 1e-9, 0.5), cfg.m_cr, float(cfg.n_par), True)
    _write_hst(decaying / "synth.decay.hst", np.arange(0.0, 55.0 + 1e-9, 0.25), cfg.m_cr, float(cfg.n_par), False)
    for n, t in ((2, 20.0), (3, 30.0), (4, 40.0)):
        _write_snapshot(driven / f"synth.stir.out2.{n:05d}.athdf", t + 3e-4, seed=n)
    presets = tmp_path / "presets.yaml"
    presets.write_text(PRESETS)
    return {"root": root, "driven": driven, "decaying": decaying, "presets": presets, "cfg": cfg,
            "products": tmp_path / "products", "out": tmp_path / "figures"}


def _common(s, fmt="png"):
    return ["--presets", str(s["presets"]), "--data-root", str(s["root"]), "--products", str(s["products"]),
            "--out", str(s["out"]), "--formats", fmt, "--dpi", "60", "--backend", "serial"]


def _snapshot_of(directory: Path) -> dict[str, tuple[int, float]]:
    return {str(p.relative_to(directory)): (p.stat().st_size, p.stat().st_mtime) for p in directory.rglob("*")}


# ------------------------------------------------------------------ helpers
def test_formatter_for_step(script):
    assert script._formatter_for_step(0.05) == "%.2f"
    assert script._formatter_for_step(0.1) == "%.1f"
    assert script._formatter_for_step(0.25) == "%.2f"
    assert script._formatter_for_step(1.0) == "%.0f"
    assert script._formatter_for_step(2.5) == "%.1f"


def test_parser_subcommands(script):
    parser = script.build_parser()
    a = parser.parse_args(["turbulence", "--stage", "compute", "--times", "200", "300"])
    assert a.command == "turbulence" and a.times == [200.0, 300.0] and a.preset == "fig3_turbulence"
    assert parser.parse_args(["energy"]).preset == "fig4_steady_state"
    assert parser.parse_args(["power", "--t-steady", "100", "600"]).t_steady == [100.0, 600.0]
    with pytest.raises(SystemExit):
        parser.parse_args(["turbulence", "--stage", "bogus"])


def test_figure_presets_exist(script):
    from common import load_preset

    for name, keys in (("fig3_turbulence", ("run", "times", "fit_range")), ("fig4_steady_state", ("driven", "decaying")),
                       ("fig7_power", ("run", "t_steady"))):
        p = load_preset(name)
        assert all(k in p for k in keys)


# ------------------------------------------------------------------ Fig. 3
def test_turbulence_compute_and_plot(script, synthetic, capsys):
    s = synthetic
    before = _snapshot_of(s["root"])
    assert script.main(["turbulence", "--stage", "all", *_common(s, "png,pdf")]) == 0
    assert _snapshot_of(s["root"]) == before                          # raw run directories untouched
    product = s["products"] / "turbulence_spectra.out2.npz"
    assert product.exists()
    from shearpic.io.field_spectra import load_field_spectra

    series, meta = load_field_spectra(product)
    assert [round(r["time"]) for r in meta["snapshots"]] == [20, 30]
    assert meta["estimator"]["dk"] == pytest.approx(0.2) and meta["run"]["nx"] == [32, 128, 1]
    for spec, rec in zip(series, meta["snapshots"]):
        assert spec["kin"].total == pytest.approx(rec["grid"]["eps_kin_fluct"], rel=1e-10)
        assert spec["mag"].total == pytest.approx(rec["grid"]["eps_mag_fluct"], rel=1e-10)
    for ext in ("png", "pdf", "json"):
        assert (s["out"] / f"turbulence.{ext}").exists()
    info = json.loads((s["out"] / "turbulence.json").read_text())
    st = info["statistics"]
    assert np.isfinite(st["slope_tot"]) and len(st["slope_tot_snapshots"]) == 2
    assert st["fit_weighting"] == "log" and st["fit_range"] == [0.5, 3.0]
    sysm = st["slope_tot_systematic"]
    assert len(sysm["fits"]) == 6 and sysm["min"] <= st["slope_tot"] <= sysm["max"]
    assert st["local_slopes_tot"] and all({"k_lo", "k_hi", "slope", "n_bins"} <= set(r) for r in st["local_slopes_tot"])
    assert st["ratio_logmean_30_80"] is None                          # empty range -> NaN -> null
    assert st["k_peak_kin"] > 0 and st["k_nyquist"] == pytest.approx(3.2)
    assert "Parseval" in capsys.readouterr().out


def test_turbulence_products_merge_and_select_by_time(script, synthetic):
    s = synthetic
    script.main(["turbulence", "--stage", "compute", "--times", "40", *_common(s)])
    script.main(["turbulence", "--stage", "compute", "--times", "20", "30", *_common(s)])
    from shearpic.io.field_spectra import load_field_spectra

    series, meta = load_field_spectra(s["products"] / "turbulence_spectra.out2.npz")
    assert [round(r["time"]) for r in meta["snapshots"]] == [20, 30, 40] and len(series) == 3
    assert script.main(["turbulence", "--stage", "plot", "--t-range", "25", "45", *_common(s)]) == 0
    info = json.loads((s["out"] / "turbulence.json").read_text())
    assert [round(t) for t in info["statistics"]["times"]] == [30, 40]
    with pytest.raises(ValueError, match="t = 50"):
        script.main(["turbulence", "--stage", "plot", "--times", "50", *_common(s)])


def test_turbulence_refuses_online_only_snapshot(script, synthetic, monkeypatch):
    from shearpic.io.athdf import DatalessFileError

    s = synthetic
    target = s["driven"] / "synth.stir.out2.00003.athdf"
    monkeypatch.setattr("shearpic.io.athdf.is_dataless", lambda p: Path(p) == target)
    with pytest.raises(DatalessFileError):
        script.main(["turbulence", "--stage", "compute", "--times", "30", *_common(s)])
    assert not (s["products"] / "turbulence_spectra.out2.npz").exists()


def test_plot_stage_needs_product(script, synthetic):
    with pytest.raises(FileNotFoundError, match="compute"):
        script.main(["turbulence", "--stage", "plot", *_common(synthetic)])


def _spectrum_series(times, slope_lo=-1.0, slope_hi=-3.0, k_break=10.0, dk=0.2, k_max=100.0):
    """Linear-shell spectra (edges 0, dk/2, 3dk/2, ...) with a broken power law and E_m/E_k = k."""
    from shearpic.physics.turbulence import FieldSpectrum, sum_spectra

    n = int(round(k_max / dk)) + 1
    edges = np.concatenate(([0.0], (np.arange(n) + 0.5) * dk))
    k = 0.5 * (edges[:-1] + edges[1:])
    base = np.where(k < k_break, (k / k_break) ** slope_lo, (k / k_break) ** slope_hi)
    base[0] = 0.0
    series, records = [], []
    for i, t in enumerate(times):
        kin = base / (1.0 + k) * (1 + 0.01 * i)
        mag = kin * k
        make = lambda E, kind: FieldSpectrum(k_edges=edges, E=E, n_modes=np.ones(n, int),  # noqa: E731
                                             total=float(np.sum(E * np.diff(edges))), time=t, kind=kind)
        spec = {"kin": make(kin, "kin"), "mag": make(mag, "mag")}
        spec["tot"] = sum_spectra(spec["kin"], spec["mag"], kind="tot")
        series.append(spec)
        grid = {"eps_kin_fluct": spec["kin"].total, "eps_mag_fluct": spec["mag"].total, "eps_kin": 0.5, "eps_mag": 0.05}
        records.append({"time": t, "file": f"x.out2.{i:05d}.athdf", "grid": grid})
    return series, records, k


class _Cfg:
    nx = (1024, 4096, 1)                # run364: k_Nyquist = 102.4
    L = (10 * np.pi, 40 * np.pi, 1.0)
    Lx = 10 * np.pi

    def output_dt(self, kind):
        return {"hst": 0.5, "hdf5": 10.0}.get(kind)


def test_turbulence_statistics_log_weighted_slopes_and_systematics(script):
    from shearpic.physics.turbulence import average_spectra, fit_power_law

    series, records, k = _spectrum_series([200.0, 300.0])
    preset = {**script.FIG3_DEFAULTS, "fit_range": [3.0, 15.0], "ratio_ranges": [[2.0, 30.0]]}
    st = script.turbulence_statistics(series, records, _Cfg(), preset)
    avg = average_spectra([sp["tot"] for sp in series])
    assert st["fit_weighting"] == "log"
    assert st["slope_tot"] == pytest.approx(fit_power_law(avg, 3.0, 15.0, weighting="log")[0], abs=1e-12)
    assert st["slope_tot_shell_weighting"] == pytest.approx(fit_power_law(avg, 3.0, 15.0, weighting="shell")[0])
    assert st["slope_tot_shell_weighting"] < st["slope_tot"]          # per-shell weight favours the steep end
    fits = st["slope_tot_systematic"]["fits"]
    assert {(f["k_min"], f["k_max"], f["weighting"]) for f in fits} == {
        (a, b, w) for a, b in preset["systematic_fit_ranges"] for w in ("log", "shell")}
    assert st["slope_tot_systematic"]["min"] == pytest.approx(min(f["slope"] for f in fits))
    assert st["slope_tot_systematic"]["max"] == pytest.approx(max(f["slope"] for f in fits))
    # local slopes: -1 below the break, -3 above; the fit-range ends differ by 2 -> flagged
    rows = st["local_slopes_tot"]
    assert rows[0]["k_lo"] >= 2 * np.pi / _Cfg.Lx - 1e-12 and rows[-1]["k_hi"] <= 102.4
    ends = st["slope_tot_fit_range_ends"]
    assert st["fit_range_ends_windows"][0] == pytest.approx([3.0, 3.0 * 10 ** 0.5])
    assert ends[1] < ends[0] and st["fit_range_local_slope_change"] == pytest.approx(ends[0] - ends[1])
    assert st["fit_range_single_power_law"] is (st["fit_range_local_slope_change"] <= 0.5)
    single = script.turbulence_statistics(*_spectrum_series([200.0], slope_hi=-1.0)[:2], _Cfg(), preset)
    assert single["fit_range_single_power_law"] and single["slope_tot"] == pytest.approx(-1.0, abs=1e-9)   # tot = base
    # E_m/E_k = k: mean per log k over 2 < k < 30 is sum(dk/k * k)/sum(dk/k)
    sel = (k > 2.0) & (k < 30.0)
    assert st["ratio_logmean_2_30"] == pytest.approx(np.sum(np.ones(sel.sum())) / np.sum(1.0 / k[sel]), rel=1e-9)
    assert "ratio_mean_2_30" not in st


def test_turbulence_hst_check_needs_a_covering_history_row(script):
    import pandas as pd

    series, records, _ = _spectrum_series([200.0, 400.0, 700.0])
    t = np.arange(0.0, 600.0 + 1e-9, 0.5)
    hst = pd.DataFrame({"time": t, "eps_kin": 0.5 + t * 1e-5, "eps_mag": 0.05 + t * 1e-6})
    st = script.turbulence_statistics(series, records, _Cfg(), script.FIG3_DEFAULTS, hst)
    rows = {round(r["time"]): r for r in st["parseval"]}
    assert rows[400]["eps_kin_hst"] == pytest.approx(0.5 + 400 * 1e-5) and rows[400]["hst_time"] == 400.0
    assert rows[700]["eps_kin_hst"] is None and rows[700]["eps_mag_hst"] is None and rows[700]["hst_time"] is None
    # a gap in the history: the nearest rows (199.5, 200.5) are more than dt/2 = 0.25 away
    gap = hst[np.abs(hst["time"] - 200.0) > 0.3]
    st2 = script.turbulence_statistics(series[:1], records[:1], _Cfg(), script.FIG3_DEFAULTS, gap)
    assert st2["parseval"][0]["eps_kin_hst"] is None
    near = hst.assign(time=hst["time"] + 0.2)                     # 0.2 < dt/2: still covered
    st3 = script.turbulence_statistics(series[:1], records[:1], _Cfg(), script.FIG3_DEFAULTS, near)
    assert st3["parseval"][0]["hst_time"] == pytest.approx(200.2)


def test_turbulence_plot_prints_hst_not_covered(script, synthetic, capsys):
    s = synthetic
    hst = s["driven"] / "synth.stir.hst"
    lines = hst.read_text().splitlines()
    kept = [ln for ln in lines if ln.startswith("#") or float(ln.split()[0]) <= 25.0]
    hst.write_text("\n".join(kept) + "\n")                          # history ends at t = 25
    assert script.main(["turbulence", "--stage", "all", *_common(s)]) == 0
    out = capsys.readouterr().out
    t20 = next(ln for ln in out.splitlines() if "Parseval t = 20" in ln)
    t30 = next(ln for ln in out.splitlines() if "Parseval t = 30" in ln)
    assert "(hst 0." in t20 and "hst not covered" in t30 and "(hst 0." not in t30
    st = json.loads((s["out"] / "turbulence.json").read_text())["statistics"]
    assert st["parseval"][1]["eps_kin_hst"] is None


def test_wavelength_axis_ticks_at_round_wavelengths(script, synthetic):
    import matplotlib.pyplot as plt

    major, minor = script.wavelength_ticks([2 * np.pi / 40, 102.4])
    assert major == pytest.approx([2 * np.pi / 10, 2 * np.pi, 2 * np.pi / 0.1])
    assert np.isclose(minor, 2 * np.pi / 0.2).any() and np.isclose(minor, 2 * np.pi / 30).any()
    assert all(2 * np.pi / 40 <= kk <= 102.4 for kk in major + minor)
    series, records, _ = _spectrum_series([200.0, 300.0])
    from shearpic.physics.turbulence import average_spectra

    preset = {**script.FIG3_DEFAULTS, "xlim": [0.15708, None]}
    st = script.turbulence_statistics(series, records, _Cfg(), preset)
    avg = {n: average_spectra([sp[n] for sp in series]) for n in ("kin", "mag", "tot")}
    fig, _ = script.draw_turbulence(avg, _Cfg(), st, preset)
    fig.canvas.draw()
    ax1, ax3, ax2 = fig.axes
    lo, hi = ax2.get_xlim()
    ticks = [(loc, lab.get_text()) for loc, lab in zip(ax2.get_xticks(), ax2.get_xticklabels()) if lo <= loc <= hi]
    assert [lab for _, lab in ticks] == ["10", "1", "0.1"]
    assert [loc for loc, _ in ticks] == pytest.approx([2 * np.pi / 10, 2 * np.pi, 20 * np.pi])
    texts = [t.get_text() for t in ax1.texts]
    assert texts == [script.guide_label(st["slope_tot"], 3.0, 15.0)]
    assert r"\leq k \leq 15" in texts[0] and f"k^{{{st['slope_tot']:.1f}}}" in texts[0]
    plt.close(fig)


def test_turbulence_refuses_duplicate_requested_times(script, synthetic):
    s = synthetic
    with pytest.raises(ValueError, match="(?i)both|same|duplicate"):
        script.main(["turbulence", "--stage", "compute", "--times", "20", "20", *_common(s)])
    with pytest.raises(ValueError, match="(?i)both|same|duplicate"):
        script.main(["turbulence", "--stage", "compute", "--times", "20", "24", "--tol", "5", *_common(s)])
    assert not (s["products"] / "turbulence_spectra.out2.npz").exists()


def _duplicate_product(s):
    from shearpic.io.field_spectra import load_field_spectra, save_field_spectra

    path = s["products"] / "turbulence_spectra.out2.npz"
    series, meta = load_field_spectra(path)
    rec = dict(meta["snapshots"][0], file="synth.stir.out2.00099.athdf")   # same time, another file (restart)
    meta["snapshots"] = [meta["snapshots"][0], rec, *meta["snapshots"][1:]]
    save_field_spectra(path, [series[0], series[0], *series[1:]], meta)
    return path


def test_turbulence_refuses_duplicate_times_in_products(script, synthetic):
    s = synthetic
    assert script.main(["turbulence", "--stage", "compute", "--times", "20", "30", *_common(s)]) == 0
    path = _duplicate_product(s)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="duplicate snapshot time"):
        script.main(["turbulence", "--stage", "plot", "--t-range", "15", "45", *_common(s)])
    with pytest.raises(ValueError, match="duplicate snapshot time"):
        script.main(["turbulence", "--stage", "compute", "--times", "40", *_common(s)])   # merge keeps both t = 20
    assert path.read_bytes() == before                             # nothing saved
    # recomputing t = 20 replaces both entries, and the product is clean again
    assert script.main(["turbulence", "--stage", "compute", "--times", "20", *_common(s)]) == 0
    assert script.main(["turbulence", "--stage", "plot", "--t-range", "15", "45", *_common(s)]) == 0


def test_turbulence_plot_time_matching_tolerance(script, synthetic):
    s = synthetic
    assert script.main(["turbulence", "--stage", "compute", "--times", "20", "30", *_common(s)]) == 0
    with pytest.raises(ValueError, match="t = 21"):                # beyond the default tolerance
        script.main(["turbulence", "--stage", "plot", "--times", "21", *_common(s)])
    with pytest.raises(ValueError, match="both match"):
        script.main(["turbulence", "--stage", "plot", "--times", "20", "21", "--tol", "1.5", *_common(s)])
    assert script.main(["turbulence", "--stage", "plot", "--times", "21", "--tol", "1.5", *_common(s)]) == 0
    info = json.loads((s["out"] / "turbulence.json").read_text())
    assert [round(t) for t in info["statistics"]["times"]] == [20]


def test_turbulence_plot_checks_run_parameters_not_athinput_hash(script, synthetic, capsys):
    s = synthetic
    assert script.main(["turbulence", "--stage", "compute", *_common(s)]) == 0
    athinput = s["driven"] / "athinput.kh"
    text = athinput.read_text()
    athinput.write_text(text.replace("<job>", "<job>\n# restart: tlim edited"))   # hash changes, physics does not
    capsys.readouterr()
    assert script.main(["turbulence", "--stage", "plot", *_common(s)]) == 0
    assert "different sha256" in capsys.readouterr().err
    athinput.write_text(text.replace("speed_of_light = 50.0", "speed_of_light = 25.0"))
    with pytest.raises(ValueError, match="different run"):
        script.main(["turbulence", "--stage", "plot", *_common(s)])


def test_plot_stage_leaves_no_empty_product_directory(script, synthetic):
    s = synthetic
    with pytest.raises(FileNotFoundError):
        script.main(["turbulence", "--stage", "plot", *_common(s)])
    assert not s["products"].exists()


def test_subcommand_with_wrong_preset_gives_readable_error(script, synthetic, capsys):
    from common import run_cli

    s = synthetic
    with pytest.raises(KeyError, match="has no run.*fig7_power"):
        script.main(["power", "--preset", "fig4_steady_state", *_common(s)])
    with pytest.raises(KeyError, match="fig3_turbulence"):
        script.main(["turbulence", "--stage", "plot", "--preset", "fig7_power", *_common(s)])
    with pytest.raises(KeyError, match="fig4_steady_state"):
        script.main(["energy", "--preset", "fig7_power", *_common(s)])
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        run_cli(script.main, ["power", "--preset", "fig4_steady_state", *_common(s)])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: preset 'fig4_steady_state' has no run") and "Traceback" not in err
    with pytest.raises(SystemExit) as exc:
        run_cli(script.main, ["power", *_common(s)])
    assert exc.value.code == 0


# ------------------------------------------------------------------ Fig. 4
def test_energy_figure(script, synthetic):
    s = synthetic
    assert script.main(["energy", *_common(s)]) == 0
    info = json.loads((s["out"] / "steady_state.json").read_text())
    st = info["statistics"]
    V = s["cfg"].V
    assert st["driven"]["eps_kin_t0"] == pytest.approx(1916.0 / V, rel=1e-9)
    assert st["driven"]["t_last"] == 50.0 and st["decaying"]["t_last"] == 50.0
    assert st["decaying"]["dt_median"] == 0.25 and st["driven"]["dt_median"] == 0.5
    assert st["decaying"]["lam"] == pytest.approx(2 * st["driven"]["lam"])  # same period, half the dt
    assert (s["out"] / "steady_state.png").exists()


def test_energy_tick_labels_have_enough_decimals(script, synthetic):
    import matplotlib.pyplot as plt

    s = synthetic
    args = script.build_parser().parse_args(["energy", *_common(s)])
    preset = script._preset(args, script.FIG4_DEFAULTS)
    driven = script.energy_series("run0001", args, 50.0, 4.0)
    decaying = script.energy_series("run0002", args, 50.0, 4.0)
    fig = script.draw_energy(driven, decaying, preset, 50.0)
    fig.canvas.draw()

    def visible_labels(ax):
        lo, hi = ax.get_ylim()
        return [lab.get_text() for loc, lab in zip(ax.get_yticks(), ax.get_yticklabels()) if lo <= loc <= hi]

    bottom = visible_labels(fig.axes[1])
    assert bottom == ["0.50", "0.55", "0.60", "0.65", "0.70", "0.75", "0.80", "0.85"]
    assert visible_labels(fig.axes[0]) == ["0.1", "0.2", "0.3", "0.4", "0.5", "0.6"]
    assert fig.axes[1].get_xlabel() == r"$U_0 t/a$"
    plt.close(fig)


def test_energy_reports_curves_outside_panel_limits(script, synthetic, capsys):
    s = synthetic
    assert script.main(["energy", *_common(s)]) == 0
    st = json.loads((s["out"] / "steady_state.json").read_text())["statistics"]
    for panel in ("top", "bottom"):
        assert set(st["panels"][panel]["fraction_outside_ylim"].values()) == {0.0}
        assert st["panels"][panel]["autoscaled"] is False
    assert st["panels"]["bottom"]["ylim"] == [0.49, 0.88]
    assert "warning" not in capsys.readouterr().err

    clipped = s["presets"].read_text().replace("ylim_bottom: [0.49, 0.88]", "ylim_bottom: [0.49, 0.55]")
    s["presets"].write_text(clipped)
    assert script.main(["energy", *_common(s)]) == 0
    st = json.loads((s["out"] / "steady_state.json").read_text())["statistics"]
    args = script.build_parser().parse_args(["energy", *_common(s)])
    driven = script.energy_series("run0001", args, 50.0, 4.0)
    decaying = script.energy_series("run0002", args, 50.0, 4.0)
    frac = st["panels"]["bottom"]["fraction_outside_ylim"]
    for label, d in (("driven eps_p", driven), ("decaying eps_p", decaying)):
        v = d["eps_p_smooth"][d["t"] <= 50.0]
        expected = float(np.mean((v < 0.49) | (v > 0.55)))
        assert frac[label] == pytest.approx(expected) and 0 < expected < 1
    assert set(st["panels"]["top"]["fraction_outside_ylim"].values()) == {0.0}
    captured = capsys.readouterr()
    assert "warning: Fig. 4 bottom panel" in captured.err and "driven eps_p" in captured.err
    assert "bottom panel ylim" in captured.out


def test_energy_null_limits_autoscale(script, synthetic):
    import matplotlib.pyplot as plt

    s = synthetic
    s["presets"].write_text(s["presets"].read_text().replace("ylim_top: [0.001, 0.61]", "ylim_top: null")
                            .replace("ylim_bottom: [0.49, 0.88]", "ylim_bottom: null"))
    assert script.main(["energy", *_common(s)]) == 0
    st = json.loads((s["out"] / "steady_state.json").read_text())["statistics"]
    args = script.build_parser().parse_args(["energy", *_common(s)])
    preset = script._preset(args, script.FIG4_DEFAULTS)
    driven = script.energy_series("run0001", args, 50.0, 4.0)
    decaying = script.energy_series("run0002", args, 50.0, 4.0)
    for panel, keys in (("top", ("eps_kin", "eps_mag")), ("bottom", ("eps_p",))):
        p = st["panels"][panel]
        assert p["autoscaled"] is True and set(p["fraction_outside_ylim"].values()) == {0.0}
        vals = np.concatenate([d[k + "_smooth"][d["t"] <= 50.0] for d in (driven, decaying) for k in keys])
        span = vals.max() - vals.min()
        assert p["ylim"] == pytest.approx([max(vals.min() - 0.05 * span, 0.0), vals.max() + 0.05 * span])
    fig = script.draw_energy(driven, decaying, preset, 50.0)
    assert fig.axes[1].get_ylim() == pytest.approx(tuple(st["panels"]["bottom"]["ylim"]))
    assert st["panels"]["top"]["ylim"][0] >= 0.0                     # energy densities: no negative axis
    # no tick label on the edge the two panels share (hspace = 0), so labels cannot overprint
    top_lo, top_hi = fig.axes[0].get_ylim()
    bot_lo, bot_hi = fig.axes[1].get_ylim()
    assert all(abs(v - top_lo) > 0.02 * (top_hi - top_lo) for v in fig.axes[0].get_yticks())
    assert all(abs(v - bot_hi) > 0.02 * (bot_hi - bot_lo) for v in fig.axes[1].get_yticks())
    plt.close(fig)


def test_auto_ylim_padding_and_non_negative_clamp(script):
    assert script.auto_ylim([np.array([1.0, 2.0])]) == pytest.approx([0.95, 2.05])
    assert script.auto_ylim([np.array([0.01, 1.01])]) == pytest.approx([0.0, 1.06])      # 0.01 - 0.05 < 0 -> 0
    assert script.auto_ylim([np.array([-1.0, 1.0])]) == pytest.approx([-1.1, 1.1])       # signed data unclamped
    assert script.auto_ylim([np.array([3.0, 3.0])]) == pytest.approx([2.985, 3.015])     # flat curve


# ------------------------------------------------------------------ Fig. 7
def test_power_figure_and_statistics(script, synthetic):
    s = synthetic
    assert script.main(["power", *_common(s, "png,pdf")]) == 0
    info = json.loads((s["out"] / "power.json").read_text())
    st = info["statistics"]
    # analytic history: E_p = m_cr (K0 + k t^2) -> dE_p/dt = 2 m_cr k t = Pideal = Pideal_z
    assert st["mean_dEp_dt_spline"] == pytest.approx(st["mean_P_ideal_raw"], rel=2e-3)
    assert st["Delta_E_p_over_Delta_t"] == pytest.approx(st["mean_P_ideal_raw"], rel=1e-3)
    assert st["ratio_Delta_E_p_over_int_P_ideal"] == pytest.approx(1.0, abs=1e-3)
    assert st["Pz_over_P_ideal_raw_means"] == pytest.approx(1.0)
    assert st["mean_Px_over_mean_Pz_raw"] == pytest.approx(0.01) and st["mean_Py_over_mean_Pz_raw"] == pytest.approx(-0.01)
    assert st["mean_P_stir"] == pytest.approx(28.0, rel=1e-6)
    assert st["t_steady"] == [10.0, 50.0]
    assert "rho_0 U_0^3 a^2" in "".join(info["caption_notes"]).replace("rho0 U0^3 a^2", "rho_0 U_0^3 a^2")


def test_power_axis_label_units(script, synthetic):
    import matplotlib.pyplot as plt

    s = synthetic
    args = script.build_parser().parse_args(["power", *_common(s)])
    preset = script._preset(args, script.FIG7_DEFAULTS)
    fig = script.draw_power(script.power_series("run0001", args, 4.0), preset)
    ax = fig.axes[0]
    assert ax.get_ylabel() == r"$P\ [\rho_0 U_0^3 a^2]$"
    assert ax.get_ylim() == (-1.0, 4.0) and [t.get_text() for t in ax.get_legend().get_texts()][0] == r"$dE_p / dt$"
    plt.close(fig)


# ------------------------------------------------------------------ real data
def _real_run(data_root: Path, name: str) -> Path:
    p = data_root / name
    if not p.is_dir():
        pytest.skip(f"{name} not available under {data_root}")
    return p


@pytest.mark.data
def test_data_energy_and_power_match_reference_numbers(script, data_root, tmp_path):
    _real_run(data_root, "run364")
    _real_run(data_root, "run370")
    common = ["--data-root", str(data_root), "--out", str(tmp_path / "fig"), "--products", str(tmp_path / "prod"),
              "--formats", "png", "--dpi", "50"]
    with pytest.warns(UserWarning, match="shear_strength"):
        assert script.main(["energy", *common]) == 0
        assert script.main(["power", *common]) == 0
    e = json.loads((tmp_path / "fig" / "steady_state.json").read_text())["statistics"]
    assert e["driven"]["eps_kin_t0"] == pytest.approx(0.4841, abs=1e-4)
    assert e["driven"]["eps_p_t0"] == pytest.approx(0.5181, abs=1e-4)
    assert e["driven"]["eps_p_tmax_smooth"] == pytest.approx(0.8728, abs=1e-4)
    assert e["decaying"]["eps_p_tmax_smooth"] == pytest.approx(0.5405, abs=1e-4)
    assert e["driven"]["eps_p_min_smooth"] == pytest.approx(0.4995, abs=1e-4)
    assert e["driven"]["t_eps_p_min_smooth"] == pytest.approx(55.5, abs=0.01)
    assert e["driven"]["lam"] == pytest.approx(100.0, rel=0.005)
    p = json.loads((tmp_path / "fig" / "power.json").read_text())["statistics"]
    assert p["mean_dEp_dt_spline"] == pytest.approx(2.822, abs=2e-3)
    assert p["mean_P_ideal_raw"] == pytest.approx(2.774, abs=2e-3)
    assert p["Pz_over_P_ideal_raw_means"] == pytest.approx(0.9949, abs=2e-4)
    assert p["mean_Pz_smooth_over_dEp_dt"] == pytest.approx(0.9782, abs=5e-4)
    assert p["ratio_Delta_E_p_over_int_P_ideal"] == pytest.approx(1.0176, abs=5e-4)
    assert p["mean_P_stir"] == pytest.approx(28.24, abs=0.01)
    assert p["min_dEp_dt_smooth"] == pytest.approx(-6.8, abs=0.05) and p["dEp_dt_clipped_by_ylim"]
    assert p["max_abs_spline_minus_gradient"] < 0.02


@pytest.mark.data
def test_data_turbulence_single_snapshot(script, data_root, tmp_path):
    run = _real_run(data_root, "run364")
    from shearpic.io._util import is_dataless

    snap = run / "org.stir.feedback.out2.00004.athdf"
    if not snap.exists() or is_dataless(snap):
        pytest.skip("run364 out2.00004 (t = 400) is not available offline")
    common = ["--data-root", str(data_root), "--out", str(tmp_path / "fig"), "--products", str(tmp_path / "prod"),
              "--formats", "png", "--dpi", "50", "--backend", "serial", "--times", "400"]
    with pytest.warns(UserWarning, match="shear_strength"):
        assert script.main(["turbulence", "--stage", "all", *common]) == 0
    st = json.loads((tmp_path / "fig" / "turbulence.json").read_text())["statistics"]
    row = st["parseval"][0]
    assert row["time"] == pytest.approx(400.0, abs=1e-3)
    assert row["kin_spectrum"] == pytest.approx(0.06402, abs=1e-5)       # 0.5 <rho u''^2>
    assert row["mag_spectrum"] == pytest.approx(0.04495, abs=1e-5)       # 0.5 <B'^2>
    assert row["eps_kin_grid"] == pytest.approx(row["eps_kin_hst"], rel=1e-4)
    assert row["eps_mag_grid"] == pytest.approx(row["eps_mag_hst"], rel=1e-4)
    assert st["k_peak_kin"] == pytest.approx(0.2)
    assert st["fit_range"] == [3.0, 15.0] and st["fit_weighting"] == "log"
    assert -1.9 < st["slope_tot"] < -1.3
    assert st["ratio_logmean_2_30"] == pytest.approx(1.9, abs=0.4)


@pytest.mark.data
def test_data_turbulence_run364_slopes_are_quoted_with_their_systematics(script, data_root, tmp_path):
    """Fig. 3 on run364 t = 200-500: log-weighted slope over 3-15, its range/weighting spread, local slopes."""
    run = _real_run(data_root, "run364")
    from shearpic.io._util import is_dataless

    snaps = [run / f"org.stir.feedback.out2.{n:05d}.athdf" for n in (2, 3, 4, 5)]
    if any(not p.exists() or is_dataless(p) for p in snaps):
        pytest.skip("run364 out2.00002-00005 (t = 200-500) are not all available offline")
    common = ["--data-root", str(data_root), "--out", str(tmp_path / "fig"), "--products", str(tmp_path / "prod"),
              "--formats", "pdf", "--dpi", "50", "--workers", "2"]
    with pytest.warns(UserWarning, match="shear_strength"):
        assert script.main(["turbulence", "--stage", "all", *common]) == 0
    st = json.loads((tmp_path / "fig" / "turbulence.json").read_text())["statistics"]
    assert [round(t) for t in st["times"]] == [200, 300, 400, 500]
    assert st["slope_tot"] == pytest.approx(-1.616, abs=2e-3)                   # log-weighted, 3 <= k <= 15
    assert st["slope_tot_shell_weighting"] == pytest.approx(-1.663, abs=2e-3)
    assert st["slope_tot_systematic"]["min"] == pytest.approx(-1.899, abs=2e-3)  # [5, 20], per-shell weights
    assert st["slope_tot_systematic"]["max"] == pytest.approx(-1.497, abs=2e-3)  # [3, 10], per-log-k weights
    assert st["slope_kin"] == pytest.approx(-1.330, abs=2e-3) and st["slope_mag"] == pytest.approx(-1.759, abs=2e-3)
    local = {(round(r["k_lo"], 2), round(r["k_hi"], 2)): r["slope"] for r in st["local_slopes_tot"]}
    assert local[(3.16, 10.0)] == pytest.approx(-1.51, abs=0.01)
    assert local[(10.0, 31.62)] == pytest.approx(-2.45, abs=0.01)
    steep = [r["slope"] for r in st["local_slopes_tot"] if r["k_lo"] >= 1.7]
    assert np.all(np.diff(steep) < 0)                                           # steepens monotonically from k ~ 2
    assert st["fit_range_single_power_law"] and st["fit_range_local_slope_change"] == pytest.approx(0.25, abs=0.01)
    assert st["ratio_logmean_2_30"] == pytest.approx(1.931, abs=2e-3)
    assert st["ratio_energy_2_30"] == pytest.approx(2.066, abs=2e-3)
    assert all(row["eps_kin_hst"] == pytest.approx(row["eps_kin_grid"], rel=1e-4) for row in st["parseval"])
    info = json.loads((tmp_path / "fig" / "turbulence.json").read_text())
    assert info["display"]["guide_label"] == r"$\propto k^{-1.6}$ ($3 \leq k \leq 15$)"
