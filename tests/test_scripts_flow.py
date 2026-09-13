"""Tests of analysis/shear_profile.py (Figs. 1 and 2): synthetic runs in tmp_path plus real-data regressions."""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path

import h5py
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from shearpic.config import RunConfig  # noqa: E402
from shearpic.io.athdf import DatalessFileError  # noqa: E402
from shearpic.physics.forcing import ShearProfile  # noqa: E402

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"

NX, MB = (32, 128, 1), (16, 32, 1)
BOUNDS = ((-4.0, 4.0), (-16.0, 16.0), (-0.5, 0.5))

ATHINPUT = """\
<job>
problem_id = synth
<output1>
file_type = hst
dt = 0.5
<output2>
file_type = hdf5
variable = prim
dt = 10.0
<mesh>
nx1 = 32
x1min = -4.0
x1max = 4.0
ix1_bc = periodic
ox1_bc = periodic
nx2 = 128
x2min = -16.0
x2max = 16.0
ix2_bc = periodic
ox2_bc = periodic
nx3 = 1
x3min = -0.5
x3max = 0.5
<meshblock>
nx1 = 16
nx2 = 32
nx3 = 1
<hydro>
iso_sound_speed = 5.0
<particles>
speed_of_light = {c}
charge_over_mass_over_c = {q_mc}
<problem>
iprob = 0
shear_strength = {S}
y1 = -8.0
y2 = 8.0
M_A = {M_A}
tau = 0.5
vp_par = 50.0
cr_mass = 0.0005
npx1 = 32
npx2 = 128
"""


# ------------------------------------------------------------------ helpers
def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"analysis_script_{name}", ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def script():
    return load_script("shear_profile")


def write_athdf(path: Path, fields: dict[str, np.ndarray], time: float, cycle: int = 0) -> Path:
    """Minimal Athena++ .athdf (datasets 'prim' and 'B') from (nx, ny, nz) arrays."""
    n1, n2, n3 = (n // b for n, b in zip(NX, MB))
    locs = np.array([(i, j, k) for k in range(n3) for j in range(n2) for i in range(n1)], dtype=">i8")
    groups = {"prim": ["rho", "vel1", "vel2", "vel3"], "B": ["Bcc1", "Bcc2", "Bcc3"]}
    with h5py.File(path, "w") as h:
        a = h.attrs
        a["DatasetNames"] = np.array([b"prim", b"B"], dtype="S21")
        a["NumVariables"] = np.array([4, 3], dtype=">i4")
        a["VariableNames"] = np.array([n.encode() for g in groups.values() for n in g], dtype="S21")
        a["RootGridSize"] = np.array(NX, dtype=">i4")
        a["MeshBlockSize"] = np.array(MB, dtype=">i4")
        for d in range(3):
            a[f"RootGridX{d + 1}"] = np.array([*BOUNDS[d], 1.0], dtype=np.float32)
        a["Time"] = np.float32(time)
        a["NumCycles"] = cycle
        a["NumMeshBlocks"] = len(locs)
        a["MaxLevel"] = 0
        for ds, names in groups.items():
            data = np.empty((len(names), len(locs), MB[2], MB[1], MB[0]), dtype=np.float32)
            for b, (i, j, k) in enumerate(locs):
                for m, name in enumerate(names):
                    block = fields[name][i * MB[0]:(i + 1) * MB[0], j * MB[1]:(j + 1) * MB[1], k * MB[2]:(k + 1) * MB[2]]
                    data[m, b] = block.transpose(2, 1, 0)
            h[ds] = data
        h["LogicalLocations"] = locs
        h["Levels"] = np.zeros(len(locs), dtype=">i4")
    return path


def grid():
    x = BOUNDS[0][0] + (np.arange(NX[0]) + 0.5) * (BOUNDS[0][1] - BOUNDS[0][0]) / NX[0]
    y_faces = np.linspace(BOUNDS[1][0], BOUNDS[1][1], NX[1] + 1)
    y = 0.5 * (y_faces[:-1] + y_faces[1:])
    X, Y = np.meshgrid(x, y, indexing="ij")
    return X[..., None], Y[..., None], y_faces


def layer_bump(y):
    return np.exp(-((y + 8.0) ** 2)) - np.exp(-((y - 8.0) ** 2))


def synthetic_fields(t: float, S: float = 1.0, ic_factor: float = 1.0) -> dict[str, np.ndarray]:
    X, Y, y_faces = grid()
    shape = ShearProfile("double_tanh", 1.0, 1.0, -8.0, 8.0)
    U_ref = S * shape.U(y_faces[:-1])[None, :, None]
    amp = 0.02 * t
    kx = 2 * np.pi / 8.0
    return {
        "rho": (1.0 + 0.3 * np.sin(kx * X) * np.exp(-((np.abs(Y) - 8.0) ** 2) / 4) + 0.001 * t) * np.ones_like(Y),
        "vel1": ic_factor * U_ref + amp * layer_bump(Y) + 0.1 * np.sin(kx * X),
        "vel2": 0.2 * np.cos(kx * X) * np.exp(-((np.abs(Y) - 8.0) ** 2)),
        "vel3": 0.01 * np.ones_like(X * Y),
        "Bcc1": 0.1 + 2.0 * np.sin(kx * X + 0.2 * Y) ** 2,
        "Bcc2": 0.5 * np.cos(kx * X) * np.ones_like(Y),
        "Bcc3": np.zeros_like(X * Y),
    }


def make_run(root: Path, name: str = "run0001", *, times=(0.0, 10.0004, 20.0002), numbers=None, S: float = 1.0,
             M_A: float = 10.0, c: float = 50.0, q_mc: float = 200.0, ic_factor: float = 1.0) -> Path:
    run = root / name
    run.mkdir(parents=True)
    (run / "athinput.synth").write_text(ATHINPUT.format(S=S, M_A=M_A, c=c, q_mc=q_mc))
    _, _, y_faces = grid()
    shape = ShearProfile("double_tanh", 1.0, 1.0, -8.0, 8.0)
    dy = (BOUNDS[1][1] - BOUNDS[1][0]) / NX[1]
    ke1 = 0.5 * S**2 * float(np.sum(shape.U(y_faces[:-1]) ** 2)) * dy * 8.0 * 1.0
    names = ["time", "dt", "1-KE", "2-KE"]
    hdr = "# Athena++ history data\n# " + "".join(f"[{i}]={n}".ljust(13) for i, n in enumerate(names, 1)) + "\n"
    rows = "".join(f" {t: .10e} {1e-3: .10e} {ke1: .10e} {0.0: .10e}\n" for t in (0.0, 0.5, 1.0))
    (run / "synth.hst").write_text(hdr + rows)
    numbers = numbers if numbers is not None else range(len(times))
    for n, t in zip(numbers, times):
        write_athdf(run / f"synth.out2.{n:05d}.athdf", synthetic_fields(t, S, ic_factor if t == 0 else 1.0),
                    time=t, cycle=100 * n)
    return run


def common_args(root: Path, tmp_path: Path, run: str = "run0001") -> list[str]:
    return ["--run", run, "--data-root", str(root), "--products", str(tmp_path / "products"), "--out",
            str(tmp_path / "figures"), "--formats", "png", "--dpi", "40", "--backend", "serial"]


# =========================================================== Fig. 1: profile
def test_profile_compute_and_plot(script, tmp_path, capsys):
    root = tmp_path / "data"
    run = make_run(root)
    res = script.run(["profile", "--times", "0", "10", "20", *common_args(root, tmp_path)])
    product = tmp_path / "products" / "shear_profile_out2_t0-10-20.npz"
    assert res["product"] == product and product.exists()
    arrays, meta = script.load_product(product)
    np.testing.assert_allclose(arrays["time"], [0.0, 10.0004, 20.0002], atol=1e-4)
    # the product is the unweighted x-average of vel1 of each snapshot
    for k, t in enumerate((0.0, 10.0004, 20.0002)):
        vx = synthetic_fields(t)["vel1"].astype(np.float32).astype(np.float64)
        np.testing.assert_allclose(arrays["profiles"][k], vx.mean(axis=(0, 2)), atol=1e-12)
    cfg = RunConfig.from_run_dir(run)
    np.testing.assert_allclose(arrays["U_ref_faces"], cfg.profile.on_cpp_grid(cfg.edges(1)))
    np.testing.assert_allclose(arrays["y"], cfg.centers(1))
    assert meta["files"] == ["synth.out2.00000.athdf", "synth.out2.00001.athdf", "synth.out2.00002.athdf"]
    assert meta["run"]["athinput_sha256"] == cfg.athinput.sha256 and meta["shearpic_version"]
    assert math.isclose(meta["shear_amplitude_hst"], 1.0, rel_tol=1e-9)

    stats = res["statistics"]
    assert stats["initial_check_passed"] and stats["S_check_passed"]
    assert stats["initial_max_abs_dev"] < 1e-6
    # deviation = 0.02 t exp(-(y -+ 8)^2) plus float32 rounding
    for r in stats["per_time"]:
        assert math.isclose(r["max_abs_dev"], 0.02 * r["time"], rel_tol=0.02, abs_tol=1e-5)
        assert math.isclose(r["jet_mean"], 1.0, abs_tol=1e-4) and math.isclose(r["wind_mean"], -1.0, abs_tol=1e-4)
    figure = tmp_path / "figures" / "shear_profile.png"
    assert figure.exists()
    side = json.loads((tmp_path / "figures" / "shear_profile.json").read_text())
    assert side["statistics"]["per_time"][2]["time"] == pytest.approx(20.0002, abs=1e-4)
    assert side["run"]["shear_amplitude"] == 1.0
    out = capsys.readouterr().out
    assert "max|<u_x>_x(0) - U_ref(faces)|" in out and "jet / wind mean" in out

    # plot stage alone needs only the product and athinput/hst
    for p in run.glob("*.athdf"):
        p.unlink()
    figure.unlink()
    script.run(["profile", "--stage", "plot", "--times", "0", "10", "20", *common_args(root, tmp_path)])
    assert figure.exists()


def test_profile_figure_layout(script, tmp_path):
    root = tmp_path / "data"
    run = make_run(root)
    script.run(["profile", "--stage", "compute", "--times", "0", "20", *common_args(root, tmp_path)])
    arrays, _ = script.load_product(tmp_path / "products" / "shear_profile_out2_t0-20.npz")
    cfg = RunConfig.from_run_dir(run)
    fig = script.plot_profiles(cfg, arrays)
    ax1, ax2 = fig.axes
    assert tuple(fig.get_size_inches()) == (10.0, 9.0)
    assert ax1.get_xlim() == (-1.15, 1.15) and ax2.get_ylim() == (-16.0, 16.0)
    assert [t.get_text() for t in ax2.get_legend().get_texts()] == ["$U_0 t/a$ = 0", "$U_0 t/a$ = 20"]
    assert r"\langle u_x" in ax2.get_xlabel() and ax1.get_title() == "(a)"
    line_a = ax1.get_lines()[0]
    np.testing.assert_allclose(line_a.get_xdata(), cfg.profile.U(line_a.get_ydata()))
    colors = [l.get_color() for l in ax2.get_lines()]
    np.testing.assert_allclose(colors[0][:3], (1.0, 0.8, 0.8)) and np.testing.assert_allclose(colors[-1][:3],
                                                                                              (0.2, 0.0, 0.0))
    plt.close(fig)


def test_profile_checks_fail_for_a_wrong_initial_state(script, tmp_path):
    root = tmp_path / "data"
    make_run(root, ic_factor=1.2)          # t = 0 profile 20% off the reference
    with pytest.raises(RuntimeError, match="deviates from U_ref"):
        script.run(["profile", "--times", "0", "10", *common_args(root, tmp_path)])


def test_profile_statistics_definitions(script):
    U = np.array([-1.0, -0.5, 0.0, 0.5, 1.0, 1.0])
    prof = np.vstack([U, U + np.array([0, 0.1, -0.2, 0, 0.01, -0.01])])
    st = script.profile_statistics(np.array([0.0, 50.0]), prof, U, 1.0, 1.0004, dt=10.0, bulk_cutoff=0.99)
    assert st["initial_max_abs_dev"] == 0.0 and st["initial_check_passed"] and st["S_check_passed"]
    assert st["per_time"][1]["max_abs_dev"] == pytest.approx(0.2)
    assert st["per_time"][1]["jet_mean"] == pytest.approx(1.0) and st["per_time"][1]["wind_mean"] == -1.0
    st = script.profile_statistics(np.array([40.0]), prof[:1], U, 1.0, 1.1, dt=10.0)
    assert st["initial_max_abs_dev"] is None and not st["S_check_passed"]


def test_snapshots_selected_by_time_and_dataless_refused(script, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_run(root, times=(0.0, 20.0001, 30.0), numbers=(0, 2, 3))   # file 00001 missing
    script.run(["profile", "--stage", "compute", "--times", "20", "0", *common_args(root, tmp_path)])
    arrays, meta = script.load_product(tmp_path / "products" / "shear_profile_out2_t20-0.npz")
    assert meta["files"] == ["synth.out2.00002.athdf", "synth.out2.00000.athdf"]
    np.testing.assert_allclose(arrays["time"], [20.0001, 0.0], atol=1e-4)
    with pytest.raises(FileNotFoundError, match="t = 10"):
        script.run(["profile", "--stage", "compute", "--times", "10", *common_args(root, tmp_path)])

    import shearpic.io.athdf as athdf

    monkeypatch.setattr(athdf, "is_dataless", lambda p: Path(p).name.endswith("00003.athdf"))
    with pytest.raises(DatalessFileError, match="online-only"):
        script.run(["profile", "--stage", "compute", "--times", "30", *common_args(root, tmp_path)])
    with pytest.warns(UserWarning, match="online-only placeholder"):
        script.run(["profile", "--stage", "compute", "--times", "30", "--allow-download",
                     *common_args(root, tmp_path)])


def test_product_from_another_run_is_refused(script, tmp_path, capsys):
    root = tmp_path / "data"
    run = make_run(root, times=(0.0,))
    args = ["profile", "--times", "0", *common_args(root, tmp_path)]
    script.run(args[:1] + ["--stage", "compute"] + args[1:])
    athinput = run / "athinput.synth"
    original = athinput.read_text()
    # an edit that does not change the physics (e.g. tlim at a restart, a comment): plotted, with a note
    athinput.write_text(original.replace("<job>", "<job>\n# restarted with a longer tlim") + "<time>\ntlim = 900\n")
    capsys.readouterr()
    script.run(args[:1] + ["--stage", "plot"] + args[1:])
    assert "different sha256" in capsys.readouterr().err
    # a physics parameter differs: refused
    athinput.write_text(original.replace("tau = 0.5", "tau = 0.25"))
    with pytest.raises(ValueError, match="different run.*tau"):
        script.run(args[:1] + ["--stage", "plot"] + args[1:])


def test_snapshot_times_off_the_cadence_are_refused(script, tmp_path, capsys):
    """A requested time between outputs is an error rather than a duplicate of the nearest snapshot."""
    root = tmp_path / "data"
    make_run(root)                                         # outputs at t = 0, 10.0004, 20.0002 (dt = 10)
    base = ["profile", "--stage", "compute", *common_args(root, tmp_path)]
    with pytest.raises(FileNotFoundError, match=r"t = 5 .*nearest local snapshot: synth.out2.00000.athdf"):
        script.run([*base, "--times", "0", "5"])
    with pytest.raises(ValueError, match=r"t = 0 and t = 5 both resolve to synth.out2.00000.athdf"):
        script.run([*base, "--times", "0", "5", "--tol", "5"])
    assert not (tmp_path / "products").exists()
    # a wider explicit tolerance accepts a single off-cadence time, and the plot stage honours it
    res = script.run(["profile", "--times", "0", "12", "--tol", "2.5", *common_args(root, tmp_path)])
    np.testing.assert_allclose(script.load_product(res["product"])[0]["time"], [0.0, 10.0004], atol=1e-4)
    with pytest.raises(ValueError, match="not the requested"):
        script.run(["profile", "--stage", "plot", "--times", "0", "12", "--tol", "0.01", *common_args(root, tmp_path)])
    # the command-line entry point turns the error into one line with exit status 2
    capsys.readouterr()
    with pytest.raises(SystemExit) as exc:
        script.run_cli(script.main, [*base, "--times", "0", "5"])
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert err.startswith("error: no out2 snapshot within 0.01 of t = 5") and "Traceback" not in err


def test_plot_stage_leaves_no_empty_product_directories(script, tmp_path, monkeypatch):
    root = tmp_path / "data"
    make_run(root, times=(0.0,))
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "outputs"))
    args = ["--run", "run0001", "--data-root", str(root), "--out", str(tmp_path / "figures"), "--formats", "png"]
    for cmd in (["profile", "--times", "0"], ["flow", "--times", "0"]):
        with pytest.raises(FileNotFoundError, match="PARTICLE_ACCEL_OUTPUT"):
            script.run([cmd[0], "--stage", "plot", *cmd[1:], *args])
    assert not (tmp_path / "outputs").exists()


def test_wrong_preset_gives_a_readable_error(script, tmp_path, capsys):
    root = tmp_path / "data"
    make_run(root, times=(0.0,))
    with pytest.raises(KeyError, match="preset 'fig4_steady_state' has no run, times; use --preset fig1_shear_profile"):
        script.run(["profile", "--preset", "fig4_steady_state", "--data-root", str(root)])
    with pytest.raises(SystemExit) as exc:
        script.run_cli(script.main, ["flow", "--preset", "fig1_shear_profile", "--data-root", str(root)])
    assert exc.value.code == 2 and "has no limits, cmaps" in capsys.readouterr().err


# ============================================================== Fig. 2: flow
FLOW_LIC = ["--kernel-length", "8", "--niter", "2", "--lic-downsample", "2", "--field-downsample", "2"]


def test_flow_compute_and_plot(script, tmp_path, capsys):
    root = tmp_path / "data"
    run = make_run(root)
    res = script.run(["flow", "--times", "10", "20", *FLOW_LIC, *common_args(root, tmp_path)])
    p10 = tmp_path / "products" / "flow_maps_out2_t10.npz"
    assert res["products"][0] == p10 and p10.exists()
    arrays, meta = script.load_product(p10)
    f = {k: v.astype(np.float32).astype(np.float64)[:, :, 0] for k, v in synthetic_fields(10.0004).items()}
    from shearpic.plotting.lic import block_mean

    assert arrays["rho"].shape == (16, 64) and arrays["tex_u"].shape == (16, 64) and arrays["tex_b"].shape == (16, 64)
    np.testing.assert_allclose(arrays["rho"], block_mean(f["rho"], 2), rtol=1e-6)
    np.testing.assert_allclose(arrays["umag"], block_mean(np.sqrt(f["vel1"]**2 + f["vel2"]**2 + f["vel3"]**2), 2),
                               rtol=1e-6)
    np.testing.assert_allclose(arrays["bmag"], block_mean(np.sqrt(f["Bcc1"]**2 + f["Bcc2"]**2 + f["Bcc3"]**2), 2),
                               rtol=1e-6)
    assert 0.0 <= arrays["tex_b"].min() and arrays["tex_b"].max() <= 1.0
    np.testing.assert_allclose(arrays["x_faces"], np.linspace(-4, 4, 17), atol=1e-12)
    np.testing.assert_allclose(arrays["tex_y_faces"], np.linspace(-16, 16, 65), atol=1e-12)
    assert float(arrays["time"]) == pytest.approx(10.0004, abs=1e-4)
    assert meta["periodic"] == [True, True] and meta["lic"]["mode_b"] == "polarization"
    assert meta["lic"]["mode_u"] == "velocity" and meta["lic"]["kernel_length"] == 8
    full = meta["full_resolution_statistics"]["bmag"]
    bmag = np.sqrt(f["Bcc1"]**2 + f["Bcc2"]**2 + f["Bcc3"]**2)
    assert full["max"] == pytest.approx(bmag.max(), rel=1e-6)
    assert full["clipped"]["below"] == pytest.approx(np.mean(bmag < 0.15))
    assert full["clipped"]["above"] == pytest.approx(np.mean(bmag > 3.15))

    assert (tmp_path / "figures" / "flow_evolution.png").exists()
    side = json.loads((tmp_path / "figures" / "flow_evolution.json").read_text())
    assert side["caption_note"].startswith("run1: S = 1 ") and "S U0 / v_A = 10)" in side["caption_note"]
    assert len(side["clipping"]) == 2 and set(side["clipping"][0]["displayed"]) == {"rho", "umag", "bmag"}
    shown = side["clipping"][1]["displayed"]["rho"]
    rho20 = block_mean(synthetic_fields(20.0002)["rho"].astype(np.float32)[:, :, 0].astype(np.float64), 2)
    assert shown["above"] == pytest.approx(np.mean(rho20 > 1.1))
    out = capsys.readouterr().out
    assert "below/above" in out and "M_A effective" in out

    # plot stage from products only
    for p in run.glob("*.athdf"):
        p.unlink()
    script.run(["flow", "--stage", "plot", "--times", "10", "20", *FLOW_LIC, *common_args(root, tmp_path)])
    with pytest.raises(FileNotFoundError, match="--stage compute"):
        script.run(["flow", "--stage", "plot", "--times", "30", *FLOW_LIC, *common_args(root, tmp_path)])


def test_flow_panels_share_one_scale(script, tmp_path):
    root = tmp_path / "data"
    run = make_run(root)
    script.run(["flow", "--stage", "compute", "--times", "0", "20", *FLOW_LIC, *common_args(root, tmp_path)])
    cfg = RunConfig.from_run_dir(run)
    products = [script.load_product(tmp_path / "products" / f"flow_maps_out2_t{t}.npz")[0] for t in (0, 20)]
    settings = script.flow_settings(script.build_parser().parse_args(["flow", *FLOW_LIC]),
                                    script.load_preset("fig2_flow_evolution"))
    fig = script.plot_flow(cfg, products, settings)
    fig.canvas.draw()
    panels = [a for a in fig.axes if a.get_xlim() == (-4.0, 4.0)]
    assert len(panels) == 6
    boxes = [a.get_window_extent() for a in panels]
    for b in boxes:
        assert b.width == pytest.approx(boxes[0].width) and b.height == pytest.approx(boxes[0].height)
        assert b.height / b.width == pytest.approx(32.0 / 8.0, rel=1e-3)     # equal physical aspect
    # colorbars: one per column, horizontal, on top, with the preset limits and labels
    cbars = [a for a in fig.axes if a not in panels]
    assert len(cbars) == 3
    labels = [c.xaxis.label.get_text() for c in cbars]
    assert labels == [r"$\rho\ [\rho_0]$", r"$|u|\ [U_0]$", r"$|B|\ [\sqrt{\rho_0}\,U_0]$"]
    assert panels[0].images[0].get_clim() == (0.5, 1.1)
    assert [c.get_xlim() for c in cbars] == [(0.5, 1.1), (0.0, 5.0), (0.15, 3.15)]
    # each LIC panel is a single pre-blended RGBA image with texture alpha 0.1
    for a in (panels[1], panels[2], panels[4], panels[5]):
        assert len(a.images) == 1 and a.images[0].lic_alpha == pytest.approx(0.1)
        assert np.asarray(a.images[0].get_array()).shape == (64, 16, 4)
    texts = [t.get_text() for t in panels[0].texts] + [t.get_text() for t in panels[3].texts]
    assert texts == [r"$t = 0\ a/U_0$", r"$t = 20\ a/U_0$"]
    assert panels[3].get_xlabel() == "x/a" and panels[0].get_xlabel() == "" and panels[0].get_ylabel() == "y/a"
    plt.close(fig)


def _panel_luminance(page_png, pdf_path, index):
    """Grey levels of the ``index``-th map panel (PDF image order) cropped from a rendering of the page."""
    import fitz
    from PIL import Image

    page = fitz.open(pdf_path)[0]
    boxes = sorted((i["bbox"] for i in page.get_image_info() if i["bbox"][3] - i["bbox"][1] > 40),
                   key=lambda b: (round(b[1]), b[0]))
    im = Image.open(page_png).convert("L")
    sx, sy = im.size[0] / page.rect.width, im.size[1] / page.rect.height
    x0, y0, x1, y1 = boxes[index]
    return np.asarray(im.crop((int(x0 * sx) + 2, int(y0 * sy) + 2, int(x1 * sx) - 2, int(y1 * sy) - 2)), dtype=float)


def test_flow_lic_texture_has_the_same_contrast_in_pdf_and_png(script, tmp_path):
    """The LIC texture changes the |u| panel by the same amount in the PDF and in the PNG."""
    fitz = pytest.importorskip("fitz")
    root = tmp_path / "data"
    make_run(root)
    script.run(["flow", "--stage", "compute", "--times", "10", *FLOW_LIC, *common_args(root, tmp_path)])
    signal = {}
    for alpha in ("0.1", "0"):
        out = tmp_path / f"alpha{alpha}"
        args = common_args(root, tmp_path)
        args[args.index("--out") + 1], args[args.index("--formats") + 1], args[args.index("--dpi") + 1] = \
            str(out), "pdf,png", "72"
        script.run(["flow", "--stage", "plot", "--times", "10", *FLOW_LIC, "--lic-alpha", alpha, *args])
        pdf = out / "flow_evolution.pdf"
        page = fitz.open(pdf)[0]
        assert len(page.get_images()) == 3 + 3        # one image per map panel plus the colorbars
        png_of_pdf = out / "pdf_render.png"
        page.get_pixmap(dpi=72).save(png_of_pdf)
        signal[alpha] = {"pdf": _panel_luminance(png_of_pdf, pdf, 1), "png": _panel_luminance(out / "flow_evolution.png",
                                                                                                pdf, 1)}
    shape = signal["0"]["pdf"].shape
    diff = {fmt: np.abs(signal["0.1"][fmt][:shape[0], :shape[1]] - signal["0"][fmt][:shape[0], :shape[1]]).mean()
            for fmt in ("pdf", "png")}
    assert diff["png"] > 3.0                             # the texture changes the |u| panel visibly ...
    assert 0.6 * diff["png"] < diff["pdf"] < 1.6 * diff["png"]   # ... by the same amount in the PDF


def test_saved_pages_use_the_default_padding(script, tmp_path):
    """Figs. 1 and 2 are saved with bbox_inches='tight' and matplotlib's default pad of 0.1 in."""
    fitz = pytest.importorskip("fitz")
    from shearpic.plotting.style import paper_style

    root = tmp_path / "data"
    make_run(root)
    args = common_args(root, tmp_path)
    args[args.index("--formats") + 1] = "pdf"
    script.run(["profile", "--times", "0", "10", *args])
    assert script.FIG1_RC["savefig.pad_inches"] == 0.1 and script.FIG2_RC["savefig.pad_inches"] == 0.1
    with paper_style(script.FIG1_RC):
        fig = script.plot_profiles(RunConfig.from_run_dir(root / "run0001"),
                                   script.load_product(tmp_path / "products" / "shear_profile_out2_t0-10.npz")[0])
        fig.savefig(tmp_path / "nopad.pdf", bbox_inches="tight", pad_inches=0.0)
    plt.close(fig)
    page = fitz.open(tmp_path / "figures" / "shear_profile.pdf")[0].rect
    bare = fitz.open(tmp_path / "nopad.pdf")[0].rect
    assert page.width - bare.width == pytest.approx(14.4, abs=0.2)
    assert page.height - bare.height == pytest.approx(14.4, abs=0.2)


def test_flow_parameters_and_caption_note(script, tmp_path):
    run = make_run(tmp_path / "data", "run0358", times=(), S=2.0, M_A=2.0, c=25.0, q_mc=10.0)
    cfg = RunConfig.from_run_dir(run)
    p = script.flow_parameters(cfg)
    assert p["B0"] == pytest.approx(0.5) and p["M_A_effective"] == pytest.approx(4.0)
    note = script.caption_note(cfg, p)
    assert note.startswith("run358: S = 2 ") and "S U0 / v_A = 4" in note and "c = 25 U0" in note
    assert script.periodic_axes(cfg) == (True, True)
    assert script.stream_dt(cfg, "out2") == 10.0 and script.format_time(40.0003) == "40"


def test_help_mentions_stages(script, capsys):
    with pytest.raises(SystemExit):
        script.main(["flow", "--help"])
    out = capsys.readouterr().out
    assert "--stage" in out and "--allow-download" in out and "--kernel-length" in out and "--tol" in out


# ============================================================ real data
def _local_snapshots(run: Path, numbers):
    from shearpic.io._util import is_dataless

    paths = [run / f"org.stir.feedback.out2.{n:05d}.athdf" for n in numbers]
    if not all(p.exists() and not is_dataless(p) for p in paths):
        pytest.skip(f"snapshots {numbers} of {run.name} are not available offline")


@pytest.mark.data
def test_run396_profiles(script, data_root, tmp_path):
    run = data_root / "run396"
    if not run.is_dir():
        pytest.skip("run396 not available")
    _local_snapshots(run, (0, 10))
    res = script.run(["profile", "--times", "0", "100", "--run", "run396", "--data-root", str(data_root),
                       "--products", str(tmp_path / "p"), "--out", str(tmp_path / "f"), "--formats", "png",
                       "--dpi", "40", "--workers", "2"])
    st = res["statistics"]
    assert st["initial_max_abs_dev"] < 1e-4
    assert st["S_hst"] == pytest.approx(1.0, abs=1e-5) and st["S_check_passed"]
    last = st["per_time"][-1]
    assert last["time"] == pytest.approx(100.0, abs=0.01)
    assert last["max_abs_dev"] == pytest.approx(0.2142, abs=0.005) and last["rms_dev"] == pytest.approx(0.0318, abs=0.002)
    assert abs(last["jet_mean"] - 1.0) < 0.01 and abs(last["wind_mean"] + 1.0) < 0.01
    assert (tmp_path / "f" / "shear_profile.png").exists()


@pytest.mark.data
def test_run358_flow_t40(script, data_root, tmp_path):
    run = data_root / "run358"
    if not run.is_dir():
        pytest.skip("run358 not available")
    _local_snapshots(run, (2,))
    res = script.run(["flow", "--times", "40", "--run", "run358", "--data-root", str(data_root), "--products",
                       str(tmp_path / "p"), "--out", str(tmp_path / "f"), "--formats", "png", "--dpi", "30",
                       "--kernel-length", "8", "--niter", "2", "--backend", "serial"])
    arrays, meta = script.load_product(res["products"][0])
    assert arrays["rho"].shape == (512, 2048) and arrays["tex_u"].shape == (512, 2048)
    assert meta["file"] == "org.stir.feedback.out2.00002.athdf" and meta["time"] == pytest.approx(40.0, abs=0.01)
    full = meta["full_resolution_statistics"]
    pct = full["rho"]["percentiles"]
    assert pct["p1"] == pytest.approx(0.824, abs=0.003) and pct["p50"] == pytest.approx(1.005, abs=0.002)
    assert pct["p99"] == pytest.approx(1.071, abs=0.003)
    assert full["bmag"]["clipped"]["below"] == pytest.approx(0.0067, abs=0.001)
    assert full["bmag"]["clipped"]["above"] == pytest.approx(0.0065, abs=0.001)
    p = res["flow_parameters"]
    assert p["shear_amplitude"] == 2.0 and p["B0"] == pytest.approx(0.5) and p["M_A_effective"] == pytest.approx(4.0)
    assert (tmp_path / "f" / "flow_evolution.png").exists()
