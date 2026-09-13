"""Tests of analysis/pack_small_data.py and analysis/reproduce_figures.py.

Synthetic tests build tiny fake run directories in ``tmp_path`` (the packer only copies files
and runs the figure scripts, so their content does not have to be real) and replace the
subprocess runner where a real figure script would be needed.  The ``data`` tests build real
bundles from ``PARTICLE_ACCEL_TEST_DATA`` into ``tmp_path`` and reproduce figures from them.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ANALYSIS = Path(__file__).resolve().parents[1] / "analysis"
REPO = ANALYSIS.parent


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(f"analysis_script_{name}", ANALYSIS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def pack():
    return load_script("pack_small_data")


@pytest.fixture(scope="module")
def repro():
    return load_script("reproduce_figures")


@pytest.fixture(autouse=True)
def _no_bytecode(monkeypatch):
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- fake runs
def make_fake_runs(root: Path) -> dict[str, Path]:
    """Run directories with the layout the presets of paper_figures.yaml expect (dummy content)."""
    files = {}

    def put(rel: str, data: bytes) -> None:
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files[rel] = p

    for run, problem in (("run396", "org.stir.feedback"), ("run358", "org.stir.feedback"),
                         ("run364", "org.stir.feedback"), ("run370", "org.decay.feedback"),
                         ("run355", "org.stir.feedback")):
        put(f"{run}/athinput.kh_org", f"<job>\nproblem_id = {problem}  # {run}\n".encode())
        put(f"{run}/{problem}.hst", f"# fake history of {run}\n".encode() * 50)
        put(f"{run}/athena_mpi.sh", b"#!/bin/bash\n")                       # not needed by any figure
    put("run364/org.stir.feedback.out2.00004.athdf", os.urandom(4096))     # raw snapshot: never bundled
    for n in range(3):
        put(f"run364/energy_spectrum_data/histogram_frame_{n:05d}_t_{100.0 * n}.csv", b"bin_centers,density\n1,0\n")
    put("run364/energy_spectrum_data/histogram_metadata.csv", b"nbins,frame_numbers\n500,\"[0, 1, 2]\"\n")
    put("run364/energy_spectrum_data/frame_summary.csv", b"frame,time\n")
    put("run364/phase_space_histograms.npz", os.urandom(1000))
    put("run355/high_energy_particles/trajectory_initmbid_766_pid_30721.tab", b"0 1 2 3 4 5 6\n" * 20)
    put("run355/high_energy_particles/trajectory_initmbid_1_pid_1.tab", b"not needed\n")
    return files


def fake_compute_runner(calls: list, *, fail: set[str] = frozenset()):
    """Runner that records the command and writes the products a compute stage would write."""

    def runner(cmd, *, env=None, log=None, echo=True, timeout=None):
        calls.append({"cmd": list(cmd), "env": dict(env or {}), "log": log})
        script, sub = Path(cmd[1]).name, cmd[2]
        products = Path(cmd[cmd.index("--products") + 1])
        products.mkdir(parents=True, exist_ok=True)
        if sub in fail:
            return 1, "Traceback (most recent call last):\nRuntimeError: boom\n"
        if script == "shear_profile.py" and sub == "profile":
            (products / "shear_profile_out2_t0-20-40-60-80-100.npz").write_bytes(b"profile")
        elif script == "shear_profile.py" and sub == "flow":
            for t in (40, 80):
                (products / f"flow_maps_out2_t{t}.npz").write_bytes(b"flow" * 10)
        elif script == "steady_state.py" and sub == "turbulence":
            from shearpic.io.field_spectra import save_field_spectra
            from shearpic.physics.turbulence import FieldSpectrum

            spec = {"tot": FieldSpectrum(k_edges=np.array([0.0, 0.1, 0.3]), E=np.ones(2), n_modes=np.array([1, 2]),
                                         total=0.3)}
            save_field_spectra(products / "turbulence_spectra.out2.npz", [spec] * 4,
                               {"snapshots": [{"time": t + 3e-4} for t in (200, 300, 400, 500)]})
        return 0, "done\n"

    return runner


# ============================================================================ pack_small_data
def test_pack_copies_only_needed_files_and_writes_manifest(pack, tmp_path):
    root = tmp_path / "results"
    files = make_fake_runs(root)
    before = {rel: (sha(p), p.stat().st_mtime_ns) for rel, p in files.items()}
    dest = tmp_path / "bundle"

    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "4-7", "--quiet"]) == 0

    copied = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    data_files = [p for p in copied if p.startswith("data/")]
    assert "data/run364/org.stir.feedback.out2.00004.athdf" not in copied
    assert not any(p.endswith(".athdf") or p.endswith("athena_mpi.sh") for p in copied)
    assert "data/run355/high_energy_particles/trajectory_initmbid_1_pid_1.tab" not in copied
    assert set(data_files) == {
        "data/run364/athinput.kh_org", "data/run364/org.stir.feedback.hst",
        "data/run370/athinput.kh_org", "data/run370/org.decay.feedback.hst",
        "data/run355/athinput.kh_org", "data/run355/org.stir.feedback.hst",
        "data/run355/high_energy_particles/trajectory_initmbid_766_pid_30721.tab",
        "data/run364/phase_space_histograms.npz",
        "data/run364/energy_spectrum_data/histogram_metadata.csv",
        "data/run364/energy_spectrum_data/frame_summary.csv",
        *(f"data/run364/energy_spectrum_data/histogram_frame_{n:05d}_t_{100.0 * n}.csv" for n in range(3)),
    }
    for rel in data_files:  # identical copies; the raw data are untouched
        assert sha(dest / rel) == sha(root / rel[len("data/"):])
    assert {rel: (sha(p), p.stat().st_mtime_ns) for rel, p in files.items()} == before

    manifest = json.loads((dest / "MANIFEST.json").read_text())
    by_path = {f["path"]: f for f in manifest["files"]}
    assert set(by_path) == set(data_files)
    hst = by_path["data/run364/org.stir.feedback.hst"]
    assert hst["figures"] == ["fig4", "fig5", "fig6", "fig7"]
    assert hst["sha256"] == sha(files["run364/org.stir.feedback.hst"])
    assert hst["size"] == files["run364/org.stir.feedback.hst"].stat().st_size
    assert hst["source"] == str(files["run364/org.stir.feedback.hst"])
    assert by_path["data/run370/athinput.kh_org"]["figures"] == ["fig4"]
    assert set(manifest["figures"]) == {"fig4", "fig5", "fig6", "fig7"}
    assert all(f["complete"] for f in manifest["figures"].values())
    assert manifest["total_size"] == sum(f["size"] for f in manifest["files"])
    assert manifest["usage"]["environment"]["PARTICLE_ACCEL_DATA"] == str((dest / "data").resolve())

    # re-running changes nothing
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "4-7", "--quiet"]) == 0
    manifest2 = json.loads((dest / "MANIFEST.json").read_text())
    assert {f["path"]: f["status"] for f in manifest2["files"]} == {p: "unchanged" for p in data_files}
    assert manifest2["created_utc"] == manifest["created_utc"]


def test_pack_runs_compute_stages_into_the_bundle(pack, tmp_path):
    root = tmp_path / "results"
    make_fake_runs(root)
    dest = tmp_path / "bundle"
    calls: list = []
    runner = fake_compute_runner(calls)

    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "1-3", "--workers", "2"],
                     runner=runner) == 0
    assert [(Path(c["cmd"][1]).name, c["cmd"][2]) for c in calls] == [
        ("shear_profile.py", "profile"), ("shear_profile.py", "flow"), ("steady_state.py", "turbulence")]
    for c, run in zip(calls, ("run0396", "run0358", "run0364")):
        cmd = c["cmd"]
        assert cmd[3:5] == ["--stage", "compute"]
        assert cmd[cmd.index("--products") + 1] == str((dest / "outputs" / run / "products").resolve())
        assert cmd[cmd.index("--data-root") + 1] == str(root)
        assert cmd[cmd.index("--workers") + 1] == "2"
        assert "--allow-download" not in cmd
        assert c["env"]["PARTICLE_ACCEL_OUTPUT"] == str((dest / "outputs").resolve())
        assert c["env"]["PARTICLE_ACCEL_DATA"] == str(root)
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    by_path = {f["path"]: f for f in manifest["files"]}
    prof = by_path["outputs/run0396/products/shear_profile_out2_t0-20-40-60-80-100.npz"]
    assert prof["kind"] == "product" and prof["figures"] == ["fig1"] and prof["status"] == "computed"
    assert prof["computed_by"] == "python analysis/shear_profile.py profile --stage compute"
    assert by_path["outputs/run0358/products/flow_maps_out2_t80.npz"]["figures"] == ["fig2"]
    assert by_path["data/run364/athinput.kh_org"]["figures"] == ["fig3"]
    assert manifest["figures"]["fig2"]["compute"]["status"] == "computed"

    # products exist: nothing is recomputed; --recompute reruns; adding figures merges the manifest
    calls.clear()
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "1-3"], runner=runner) == 0
    assert calls == []
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "1", "--recompute"],
                     runner=runner) == 0
    assert len(calls) == 1
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "7"], runner=runner) == 0
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert set(manifest["figures"]) == {"fig1", "fig2", "fig3", "fig7"}
    by_path = {f["path"]: f for f in manifest["files"]}
    assert by_path["data/run364/athinput.kh_org"]["figures"] == ["fig3", "fig7"]
    assert "outputs/run0358/products/flow_maps_out2_t40.npz" in by_path


def test_pack_fig3_product_missing_times_is_recomputed(pack, tmp_path):
    root = tmp_path / "results"
    make_fake_runs(root)
    dest = tmp_path / "bundle"
    calls: list = []
    presets = tmp_path / "presets.yaml"
    presets.write_text((ANALYSIS / "paper_figures.yaml").read_text())
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "3", "--presets", str(presets)],
                     runner=fake_compute_runner(calls)) == 0
    text = presets.read_text().replace("times: [200, 300, 400, 500]", "times: [200, 300, 400, 500, 600]")
    presets.write_text(text)
    calls.clear()
    # the fake product lacks t = 600, so the stage runs again and the product is still incomplete -> failure
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "3", "--presets", str(presets)],
                     runner=fake_compute_runner(calls)) == 1
    assert len(calls) == 1
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert not manifest["figures"]["fig3"]["complete"]


def test_pack_reports_failed_compute_and_no_compute(pack, tmp_path):
    root = tmp_path / "results"
    make_fake_runs(root)
    dest = tmp_path / "bundle"
    calls: list = []
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "1,2"],
                     runner=fake_compute_runner(calls, fail={"flow"})) == 1
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["figures"]["fig1"]["complete"]
    assert not manifest["figures"]["fig2"]["complete"]
    assert any("compute stage failed" in p for p in manifest["figures"]["fig2"]["problems"])

    dest2 = tmp_path / "bundle2"
    calls.clear()
    assert pack.main(["--dest", str(dest2), "--data-root", str(root), "--figures", "1", "--no-compute"],
                     runner=fake_compute_runner(calls)) == 1
    assert calls == []
    assert (dest2 / "data/run396/athinput.kh_org").is_file()


def test_pack_skips_online_only_placeholders(pack, tmp_path, monkeypatch):
    root = tmp_path / "results"
    make_fake_runs(root)
    dest = tmp_path / "bundle"
    placeholder = "histogram_frame_00001_t_100.0.csv"
    monkeypatch.setattr(pack, "is_dataless", lambda p: Path(p).name == placeholder)

    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "5"]) == 1
    assert not (dest / "data/run364/energy_spectrum_data" / placeholder).exists()
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert [s["path"] for s in manifest["skipped"]] == [f"data/run364/energy_spectrum_data/{placeholder}"]
    assert manifest["skipped"][0]["figures"] == ["fig5"]
    assert not manifest["figures"]["fig5"]["complete"]
    assert any("skipped-online-only" in p for p in manifest["figures"]["fig5"]["problems"])

    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "5", "--allow-download"]) == 0
    assert (dest / "data/run364/energy_spectrum_data" / placeholder).is_file()
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert manifest["skipped"] == [] and manifest["figures"]["fig5"]["complete"]


def test_pack_dry_run_and_missing_inputs(pack, tmp_path):
    root = tmp_path / "results"
    files = make_fake_runs(root)
    dest = tmp_path / "bundle"
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--dry-run"], runner=None) == 0
    assert not dest.exists()

    files["run355/high_energy_particles/trajectory_initmbid_766_pid_30721.tab"].unlink()
    assert pack.main(["--dest", str(dest), "--data-root", str(root), "--figures", "6"]) == 1
    manifest = json.loads((dest / "MANIFEST.json").read_text())
    assert any("trajectory_initmbid_766_pid_30721.tab" in p for p in manifest["figures"]["fig6"]["problems"])


def test_pack_refuses_destinations_in_data_or_repository(pack, tmp_path):
    root = tmp_path / "results"
    make_fake_runs(root)
    with pytest.raises(SystemExit, match="data root"):
        pack.main(["--dest", str(root / "bundle"), "--data-root", str(root)])
    with pytest.raises(SystemExit, match="data root"):
        pack.main(["--dest", str(tmp_path), "--data-root", str(tmp_path / "data")])
    inside = REPO / "outputs" / "never-created-by-tests"
    with pytest.raises(SystemExit, match="git repository"):
        pack.main(["--dest", str(inside), "--data-root", str(root)])
    assert not inside.exists()
    assert not (root / "bundle").exists()


def test_figure_inputs_match_script_product_names(repro, tmp_path):
    from common import load_preset

    steady = load_script("steady_state")
    fi = repro.figure_inputs("fig3", load_preset("fig3_turbulence"))
    assert fi.computed == [("run364", steady.turbulence_product_path(tmp_path, str(tmp_path / "p"), "out2").name)]
    fi = repro.figure_inputs("fig1", load_preset("fig1_shear_profile"))
    assert fi.computed == [("run396", "shear_profile_out2_t0-20-40-60-80-100.npz")]
    fi = repro.figure_inputs("fig2", load_preset("fig2_flow_evolution"))
    assert fi.computed == [("run358", "flow_maps_out2_t40.npz"), ("run358", "flow_maps_out2_t80.npz")]
    fi = repro.figure_inputs("fig6", load_preset("fig6_phase_space"))
    assert fi.runs == ["run364", "run355"]
    assert ("run355", "high_energy_particles/trajectory_initmbid_766_pid_30721.tab") in fi.files
    npz = dict(load_preset("fig5_particle_spectrum"), source="npz")
    assert repro.figure_inputs("fig5", npz).hpc_products == [("run364", "spectrum_gamma_*.npz")]
    assert repro.product_run_name("run364") == "run0364"
    # the names of the HPC products match the globs
    import fnmatch

    stream = ("org.stir.feedback", "out4", "tab")
    spectrum = load_script("energy_spectrum").product_path(tmp_path, "gamma", stream, 12).name
    assert fnmatch.fnmatch(spectrum, repro.figure_inputs("fig5", npz).hpc_products[0][1])
    hist = dict(load_preset("fig6_phase_space"))
    hist["histogram"] = dict(hist["histogram"], source="npz")
    (run, pattern), = repro.figure_inputs("fig6", hist).hpc_products
    assert run == "run364" and fnmatch.fnmatch(load_script("phase_space").product_name(stream, 24), pattern)


# ============================================================================ reproduce_figures
def test_parse_figures(repro):
    assert repro.parse_figures("all") == [f"fig{n}" for n in range(1, 8)]
    assert repro.parse_figures("7,1, fig4") == ["fig1", "fig4", "fig7"]
    assert repro.parse_figures("4-6") == ["fig4", "fig5", "fig6"]
    assert repro.parse_figures(["fig2", "3"]) == ["fig2", "fig3"]
    for bad in ("8", "fig", "1-9", ""):
        with pytest.raises(ValueError):
            repro.parse_figures(bad)


def test_parse_printed_tables_round_trips_print_table(repro, capsys):
    from common import print_table

    print("shearpic 0.1.0; some line : not a table")
    print_table("Fig. 9 statistics (run 1; a : b)", {"t = 40: rho p1/p50/p99": "0.823 / 1.005", "alpha": 1.8134,
                                                      "N": 67108864, "note": "x : y", "empty": ""})
    print("wrote something")
    print_table("Fig. 9 statistics (run 1; a : b)", {"k": -2.5e-3})
    tables = repro.parse_printed_tables(capsys.readouterr().out)
    assert list(tables) == ["Fig. 9 statistics (run 1; a : b)", "Fig. 9 statistics (run 1; a : b) (2)"]
    t = tables["Fig. 9 statistics (run 1; a : b)"]
    assert t == {"t = 40: rho p1/p50/p99": "0.823 / 1.005", "alpha": 1.8134, "N": 67108864, "note": "x : y",
                 "empty": ""}
    assert tables["Fig. 9 statistics (run 1; a : b) (2)"] == {"k": -2.5e-3}


def fake_figure_runner(calls: list, *, fail: set[str] = frozenset()):
    """Runner that behaves like a figure script: prints a table, writes the figure's JSON sidecar."""
    from common import load_preset

    def runner(cmd, *, env=None, log=None, echo=True, timeout=None):
        calls.append(list(cmd))
        preset = load_preset(cmd[cmd.index("--preset") + 1], cmd[cmd.index("--presets") + 1])
        out = Path(cmd[cmd.index("--out") + 1])
        if log is not None:
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text("log\n")
        if preset["figure"] in fail:
            return 1, "Traceback (most recent call last):\nFileNotFoundError: product missing\n"
        (out / f"{preset['figure']}.json").write_text(json.dumps({
            "created_utc": "x", "command": "y", "host": "h", "outputs": [f"{preset['figure']}.png"],
            "statistics": {"value": 2.8216}, "caption_notes": ["note"]}))
        return 0, f"\nFig statistics ({preset['figure']})\n  mean : 2.8216\n  n    : 3\nwrote it\n"

    return runner


def test_reproduce_collects_statistics_and_fails_on_errors(repro, tmp_path):
    out = tmp_path / "figs"
    root = tmp_path / "results"
    make_fake_runs(root)
    calls: list = []
    code = repro.main(["--figures", "4,7", "--out", str(out), "--data-root", str(root), "--formats", "png",
                       "--dpi", "50", "--quiet"], runner=fake_figure_runner(calls))
    assert code == 0
    assert [Path(c[1]).name for c in calls] == ["steady_state.py", "steady_state.py"]
    assert calls[0][2] == "energy" and calls[1][2] == "power"
    for c in calls:
        assert c[c.index("--out") + 1] == str(out.resolve())
        assert c[c.index("--formats") + 1] == "png" and c[c.index("--dpi") + 1] == "50"
        assert c[c.index("--data-root") + 1] == str(root)
    stats = json.loads((out / "paper_statistics.json").read_text())
    fig7 = stats["figures"]["fig7"]
    assert fig7["status"] == "ok" and fig7["exit_code"] == 0
    assert fig7["printed_tables"] == {"Fig statistics (power)": {"mean": 2.8216, "n": 3}}
    assert fig7["metadata"] == {"outputs": ["power.png"], "statistics": {"value": 2.8216}, "caption_notes": ["note"]}
    assert fig7["preflight_problems"] == []

    calls.clear()
    code = repro.main(["--figures", "5-6", "--out", str(out), "--data-root", str(root), "--quiet"],
                      runner=fake_figure_runner(calls, fail={"particle_spectrum"}))
    assert code == 1
    assert [Path(c[1]).name for c in calls] == ["plot_spectrum.py", "phase_space.py"]  # keeps going
    assert calls[1][2] == "plot"
    stats = json.loads((out / "paper_statistics.json").read_text())
    assert set(stats["figures"]) == {"fig4", "fig5", "fig6", "fig7"}  # merged with the first call
    assert stats["figures"]["fig5"]["status"] == "failed"
    assert "FileNotFoundError: product missing" in stats["figures"]["fig5"]["error_tail"]

    calls.clear()
    assert repro.main(["--figures", "5-7", "--out", str(out), "--data-root", str(root), "--fail-fast", "--quiet"],
                      runner=fake_figure_runner(calls, fail={"particle_spectrum"})) == 1
    assert len(calls) == 1


def test_reproduce_stops_before_running_when_a_run_is_missing(repro, tmp_path, capsys, monkeypatch):
    """A missing data root, run or unusable output root stops before any script runs or anything is written."""
    monkeypatch.delenv("PARTICLE_ACCEL_OUTPUT", raising=False)
    out = tmp_path / "figs"
    calls: list = []
    # no data root at all
    assert repro.main(["--figures", "4,7", "--out", str(out), "--data-root", str(tmp_path / "none"), "--quiet"],
                      runner=fake_figure_runner(calls)) == 2
    err = capsys.readouterr().err
    assert "error: data root" in err and "does not exist" in err
    assert "note: PARTICLE_ACCEL_OUTPUT is not set" in err
    # a data root without run370 (Fig. 4's decaying run)
    root = tmp_path / "results"
    make_fake_runs(root)
    import shutil

    shutil.rmtree(root / "run370")
    assert repro.main(["--figures", "4,7", "--out", str(out), "--data-root", str(root), "--quiet"],
                      runner=fake_figure_runner(calls)) == 2
    err = capsys.readouterr().err
    lines = [line for line in err.splitlines() if line.startswith("error:")]
    assert len(lines) == 1 and "run run370 (needed by fig4) not found under" in lines[0]
    assert calls == [] and not out.exists()
    # an output root that is a file
    (tmp_path / "file").write_text("x")
    assert repro.main(["--figures", "7", "--data-root", str(root), "--output-root", str(tmp_path / "file" / "o"),
                       "--quiet"], runner=fake_figure_runner(calls)) == 2
    assert "is not a directory" in capsys.readouterr().err and calls == []
    # the note is not printed when the output root is given
    assert repro.main(["--figures", "7", "--out", str(out), "--data-root", str(root), "--output-root",
                       str(tmp_path / "o"), "--quiet"], runner=fake_figure_runner(calls)) == 0
    assert "PARTICLE_ACCEL_OUTPUT is not set" not in capsys.readouterr().err
    assert json.loads((out / "paper_statistics.json").read_text())["figures"]["fig7"]["status"] == "ok"


def test_reproduce_compute_bundle_check_and_list(repro, tmp_path, capsys):
    bundle = tmp_path / "bundle"
    root = bundle / "data"
    make_fake_runs(root)
    calls: list = []
    assert repro.main(["--figures", "1", "--compute", "--bundle", str(bundle), "--quiet"],
                      runner=fake_figure_runner(calls)) == 0
    cmd = calls[0]
    assert cmd[2:5] == ["profile", "--stage", "all"]
    assert cmd[cmd.index("--data-root") + 1] == str(root.resolve())
    assert (bundle / "outputs" / "paper_figures" / "paper_statistics.json").is_file()

    assert repro.main(["--figures", "4,7", "--bundle", str(bundle), "--check"]) == 0
    assert repro.main(["--figures", "1", "--bundle", str(bundle), "--check"]) == 1
    assert "missing product" in capsys.readouterr().out
    assert repro.main(["--list"]) == 0
    assert "phase_space.py plot" in capsys.readouterr().out
    assert repro.main(["--figures", "9"]) == 2


def test_run_command_streams_logs_and_times_out(repro, tmp_path):
    log = tmp_path / "logs" / "x.log"
    code, text = repro.run_command([sys.executable, "-c", "import sys; print('hello'); sys.exit(3)"], log=log,
                                   echo=False)
    assert code == 3 and text == "hello\n"
    assert log.read_text().splitlines()[1] == "hello"
    code, text = repro.run_command([sys.executable, "-c", "import time; time.sleep(30)"], echo=False, timeout=0.5)
    assert code != 0 and "killed after 0.5 s" in text
    code, text = repro.run_command([str(tmp_path / "does-not-exist")], echo=False)
    assert code == 127


# ============================================================================ real data
def _require_local(paths):
    from shearpic.io._util import is_dataless

    for p in paths:
        if not Path(p).exists():
            pytest.skip(f"{p} not available")
        if is_dataless(p):
            pytest.skip(f"{p} is an online-only placeholder")


@pytest.mark.data
def test_bundle_figures_4_to_7_from_real_data(pack, repro, data_root, tmp_path):
    needed = [data_root / "run364" / "org.stir.feedback.hst", data_root / "run370" / "org.decay.feedback.hst",
              data_root / "run355" / "org.stir.feedback.hst", data_root / "run364" / "phase_space_histograms.npz",
              data_root / "run355" / "high_energy_particles" / "trajectory_initmbid_766_pid_30721.tab",
              *sorted((data_root / "run364" / "energy_spectrum_data").glob("*.csv"))]
    _require_local(needed)
    bundle = tmp_path / "bundle"
    assert pack.main(["--dest", str(bundle), "--data-root", str(data_root), "--figures", "4-7", "--quiet"]) == 0
    manifest = json.loads((bundle / "MANIFEST.json").read_text())
    assert manifest["total_size"] < 5e6
    assert not list(bundle.rglob("*.athdf"))
    for f in manifest["files"]:
        assert f["sha256"] == sha(f["source"])

    out = tmp_path / "figs"
    code = repro.main(["--bundle", str(bundle), "--figures", "4-7", "--out", str(out), "--formats", "png",
                       "--dpi", "60", "--workers", "1", "--quiet"])
    assert code == 0, (out / "paper_statistics.json").read_text()
    assert not (bundle / "outputs").exists() or not any((bundle / "outputs").rglob("*.npz"))
    for name in ("steady_state", "particle_spectrum", "phase_space", "power"):
        assert (out / f"{name}.png").is_file()
    stats = json.loads((out / "paper_statistics.json").read_text())["figures"]
    assert all(stats[k]["status"] == "ok" for k in ("fig4", "fig5", "fig6", "fig7"))
    fig7 = next(t for n, t in stats["fig7"]["printed_tables"].items() if n.startswith("Fig. 7 statistics"))
    assert fig7["<dE_p/dt> (spline derivative)"].startswith("2.8216")
    fig6 = next(t for n, t in stats["fig6"]["printed_tables"].items() if n.startswith("Fig. 6 ensemble"))
    assert fig6["out-of-range fraction u_x"] == pytest.approx(0.0455862, abs=1e-6)
    fig5 = stats["fig5"]["printed_tables"]["Fig. 5 inputs (legacy)"]
    assert fig5["N (cfg.n_par)"] == 67108864 and fig5["outputs"] == 25
    fig4 = next(t for n, t in stats["fig4"]["printed_tables"].items() if n.startswith("Fig. 4 statistics"))
    assert fig4["driven eps_p(t=0), (t=600)"].startswith("0.5181, 0.8730")
    assert stats["fig7"]["metadata"]["statistics"]["mean_dEp_dt_spline"] == pytest.approx(2.8216, abs=1e-4)


@pytest.mark.data
def test_bundle_compute_stages_figures_1_and_3_from_real_data(pack, repro, data_root, tmp_path):
    snaps = [data_root / "run396" / f"org.stir.feedback.out2.{n:05d}.athdf" for n in (0, 2, 4, 6, 8, 10)]
    snaps += [data_root / "run364" / f"org.stir.feedback.out2.{n:05d}.athdf" for n in (2, 3, 4, 5)]
    _require_local(snaps)
    bundle = tmp_path / "bundle"
    assert pack.main(["--dest", str(bundle), "--data-root", str(data_root), "--figures", "1,3", "--workers", "2",
                      "--quiet"]) == 0
    products = sorted(p.relative_to(bundle).as_posix() for p in (bundle / "outputs").rglob("*.npz"))
    assert products == ["outputs/run0364/products/turbulence_spectra.out2.npz",
                        "outputs/run0396/products/shear_profile_out2_t0-20-40-60-80-100.npz"]
    assert not list((bundle / "data").rglob("*.athdf"))

    out = tmp_path / "figs"
    assert repro.main(["--bundle", str(bundle), "--figures", "1,3", "--out", str(out), "--formats", "png",
                       "--dpi", "60", "--quiet"]) == 0
    stats = json.loads((out / "paper_statistics.json").read_text())["figures"]
    fig1 = next(t for n, t in stats["fig1"]["printed_tables"].items() if n.startswith("Fig. 1 statistics"))
    assert fig1["S from hst 1-KE(0)"] == pytest.approx(1.0, abs=1e-5)
    fig3 = next(t for n, t in stats["fig3"]["printed_tables"].items() if n.startswith("Fig. 3 statistics"))
    # the fit range and weighting belong to steady_state.py; the bundle only has to reproduce its table
    (slope_key,) = [k for k in fig3 if k.startswith("slope E_k+E_m on ")]
    assert -2.5 < float(str(fig3[slope_key]).split()[0]) < -1.0
