"""End-to-end tests of the ``shearpic`` command line (shearpic.cli.main) in tmp directories."""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from shearpic import env as envmod
from shearpic.cli import main, parse_run_spec
from shearpic.registry import load_registry

TEMPLATE = """\
<comment>
problem   = Driven KH instability
configure = --prob kh_driven -b --eos isothermal --p charged -mpi -hdf5

<job>
problem_id = kh.cli  # problem ID: base name of output files

<output1>
file_type  = hst        # History data dump
dt         = 0.5

<output2>
file_type  = hdf5       # Binary data dump
variable   = prim
dt         = 50.0

<output3>
file_type  = rst        # restart dump
dt         = 100.0

<time>
cfl_number = 0.4
nlim       = -1
tlim       = 600.0

<mesh>
nx1        = 32
x1min      = -1.5707963267948966
x1max      = 1.5707963267948966
nx2        = 64
x2min      = -6.283185307179586
x2max      = 6.283185307179586
nx3        = 1
x3min      = -0.5
x3max      = 0.5

<meshblock>
nx1        = 16
nx2        = 16
nx3        = 1

<hydro>
iso_sound_speed = 5

<particles>
backreaction = true
charge_over_mass_over_c = 200.0
speed_of_light = 50.0

<problem>
iprob = 0
shear_strength = 1.0
y1 = -3.141592653589793
y2 = 3.141592653589793
M_A = 10
tau = 0.5
npx1 = 64
npx2 = 128
npx3 = 1
vp_par = 50.0
cr_mass = 0.0005
"""

REAL_REGISTRY = Path(__file__).resolve().parents[1] / "runs.yaml"


def _sha(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.exists() else None


@pytest.fixture(autouse=True)
def _real_registry_untouched():
    before = _sha(REAL_REGISTRY)
    yield
    assert _sha(REAL_REGISTRY) == before, "a test modified the repository's runs.yaml"


@pytest.fixture
def ws(tmp_path, monkeypatch):
    reg_file = tmp_path / "runs.yaml"
    reg_file.write_text("# test registry\nschema_version: 1\nrun_dir_format: run{id:04d}\nruns: []\nexperiments: {}\n")
    data = tmp_path / "data"
    data.mkdir()
    template = tmp_path / "athinput.kh_org"
    template.write_text(TEMPLATE)
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(reg_file))
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "local")
    return {"reg": reg_file, "data": data, "template": template, "tmp": tmp_path}


def _age(root: Path, hours: float = 2.0) -> Path:
    """Backdate every file below ``root``, as for a simulation that finished a while ago."""
    t = time.time() - hours * 3600
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), (t, t))
    return root


def run(capsys, *argv):
    code = main([str(a) for a in argv])
    out, err = capsys.readouterr()
    assert "Traceback" not in out + err
    return code, out, err


# --------------------------------------------------------------------- basics
def test_env_on_stampede3(capsys, monkeypatch):
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "stampede3")
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "48")
    code, out, _ = run(capsys, "env")
    assert code == 0
    assert "machine      : stampede3" in out
    assert "SLURM job    : 1" in out
    assert "login node   : False" in out
    cpus = int(next(line for line in out.splitlines() if line.startswith("cpus usable")).split(":")[1])
    assert 1 <= cpus <= 48


def test_env_reports_registry(capsys, monkeypatch, tmp_path):
    reg = tmp_path / "runs.yaml"
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(reg))
    code, out, _ = run(capsys, "env")
    assert code == 0 and f"registry     : {reg} (MISSING: create it with `shearpic runs init`)" in out
    reg.write_text("runs: []\n")
    assert f"registry     : {reg} (exists)" in run(capsys, "env")[1]
    other = tmp_path / "other.yaml"
    assert f"registry     : {other} (MISSING" in run(capsys, "env", "--registry", other)[1]
    monkeypatch.delenv("PARTICLE_ACCEL_REGISTRY")
    monkeypatch.setattr(envmod, "repo_root", lambda: None)  # installed without a checkout
    envmod._detect_cached.cache_clear()
    try:
        code, out, _ = run(capsys, "env")
    finally:
        envmod._detect_cached.cache_clear()
    assert code == 0 and "registry     : not found" in out and "PARTICLE_ACCEL_REGISTRY" in out
    code, _, err = run(capsys, "runs", "list")
    assert code == 2 and "set PARTICLE_ACCEL_REGISTRY or pass --registry" in err


def test_usage_errors_return_2(capsys, ws):
    assert run(capsys)[0] == 2
    code, _, err = run(capsys, "runs")
    assert code == 2 and "ACTION" in err
    assert run(capsys, "runs", "set-status")[0] == 2
    assert run(capsys, "--help")[0] == 0
    code, _, err = run(capsys, "runs", "show", "12")
    assert code == 2 and err.startswith("error: run 12 is not registered")
    code, _, err = run(capsys, "runs", "show", "banana")
    assert code == 2 and "not a run id" in err


def test_parse_run_spec():
    assert parse_run_spec("3:fiducial") == {"run": 3, "label": "fiducial", "color": None, "linestyle": None}
    assert parse_run_spec("run0003:$c=50$:C0:--") == {"run": 3, "label": "$c=50$", "color": "C0", "linestyle": "--"}
    assert parse_run_spec("3:a:tab:red::")["color"] == "tab:red"
    assert parse_run_spec("3:a:tab:red::")["linestyle"] == ":"
    assert parse_run_spec("3:a::-.") == {"run": 3, "label": "a", "color": None, "linestyle": "-."}
    with pytest.raises(ValueError):
        parse_run_spec("3")


# ------------------------------------------------------------------ workflow
def test_runs_workflow(capsys, ws):
    code, out, err = run(capsys, "runs", "new", "--purpose", "fiducial c=50", "--template", ws["template"],
                         "--tag", "cscan", "--pgen-commit", "deadbeef")
    assert code == 0, err
    assert "created run 1 [planned]" in out and (ws["data"] / "run0001" / "athinput.kh_org").exists()

    code, out, err = run(capsys, "runs", "new", "--purpose", "c=100", "--from", "1",
                         "--set", "particles/speed_of_light=100", "--set", "problem/M_A=20", "--tag", "cscan")
    assert code == 0, err
    assert "changes from parent run 1:" in out
    assert "particles/speed_of_light: 50.0 -> 100.0" in out and "problem/M_A: 10 -> 20" in out

    code, out, _ = run(capsys, "runs", "new", "--purpose", "dry", "--from", "2", "--set", "problem/tau=1", "--dry-run")
    assert code == 0 and "would create run 3" in out and "nothing was written" in out
    assert not (ws["data"] / "run0003").exists()

    code, out, err = run(capsys, "runs", "new", "--purpose", "typo", "--from", "1", "--set", "problem/M_AA=3",
                         "--dry-run")
    assert code == 0 and "warning: <problem> M_AA is not in" in err

    code, out, _ = run(capsys, "runs", "list")
    lines = out.splitlines()
    assert lines[0].split() == ["id", "status", "created", "c", "q_mc", "M_A", "S", "purpose"]
    assert lines[2].split()[:7] == ["1", "planned", lines[2].split()[2], "50", "200", "10", "1"]
    assert lines[3].split()[3] == "100" and "c=100" in lines[3]

    code, out, _ = run(capsys, "runs", "set-status", "2", "submitted", "--job-id", "777", "--note", "queued",
                       "--partition", "skx", "--nodes", "2")
    assert code == 0 and "run 2: planned -> submitted" in out
    code, out, _ = run(capsys, "runs", "list", "--status", "submitted")
    assert len(out.splitlines()) == 3 and "c=100" in out
    code, out, _ = run(capsys, "runs", "list", "--tag", "nothing")
    assert "no runs match" in out

    code, out, _ = run(capsys, "runs", "note", "run0002", "restarted after node failure")
    assert code == 0
    code, out, _ = run(capsys, "runs", "show", "2")
    assert code == 0
    assert "job_id: '777'" in out and "partition: skx" in out and "restarted after node failure" in out
    assert "particles/speed_of_light: [50.0, 100.0]" in out and "(exists)" in out

    code, _, err = run(capsys, "runs", "set-status", "2", "finished")
    assert code == 2 and "invalid status" in err

    code, out, _ = run(capsys, "runs", "check")
    assert code == 0 and "0 error(s)" in out

    inp = ws["data"] / "run0001" / "athinput.kh_org"
    inp.write_text(inp.read_text().replace("tlim       = 600.0", "tlim       = 900.0"))
    code, out, _ = run(capsys, "runs", "check")
    assert code == 1 and "changed after registration" in out and "1 error(s)" in out
    code, out, _ = run(capsys, "runs", "rehash", "1", "--note", "longer run")
    assert code == 0 and "tlim: 600.0 -> 900.0" in out
    assert run(capsys, "runs", "check")[0] == 0

    (ws["data"] / "run0005").mkdir()
    code, out, _ = run(capsys, "runs", "check")
    assert code == 0 and "WARNING run 5" in out and "1 warning(s)" in out


def test_new_never_reuses_numbers_of_existing_directories(capsys, ws):
    (ws["data"] / "run0001").mkdir()
    (ws["data"] / "run7").mkdir()  # legacy spelling
    code, out, err = run(capsys, "runs", "new", "--purpose", "x", "--template", ws["template"])
    assert code == 0, err
    assert "created run 8" in out and (ws["data"] / "run0008").is_dir()
    assert list((ws["data"] / "run0001").iterdir()) == [] and list((ws["data"] / "run7").iterdir()) == []
    code, _, err = run(capsys, "runs", "new", "--purpose", "x", "--template", ws["template"], "--from", "1")
    assert code == 2  # mutually exclusive
    code, _, err = run(capsys, "runs", "new", "--template", ws["template"])
    assert code == 2  # purpose required


def test_explicit_registry_option(capsys, ws, monkeypatch):
    other = ws["tmp"] / "elsewhere.yaml"
    monkeypatch.delenv("PARTICLE_ACCEL_REGISTRY")
    # a missing (e.g. mistyped) registry is never created implicitly
    code, out, err = run(capsys, "runs", "new", "--purpose", "p", "--template", ws["template"], "--registry", other)
    assert code == 2 and "does not exist" in err and "shearpic runs init" in err
    assert not other.exists() and not (ws["data"] / "run0001").exists()
    code, out, _ = run(capsys, "runs", "list", "--registry", other)
    assert code == 0 and "does not exist" in out and not other.exists()
    assert run(capsys, "runs", "note", "1", "x", "--registry", other)[0] == 2

    code, out, err = run(capsys, "runs", "init", "--registry", other)
    assert code == 0 and f"created empty registry {other}" in out
    assert load_registry(other).runs == []
    code, out, err = run(capsys, "runs", "new", "--purpose", "p", "--template", ws["template"], "--registry", other)
    assert code == 0, err
    assert other.exists() and load_registry(other).get_run(1).purpose == "p"
    before = other.read_bytes()
    code, out, _ = run(capsys, "runs", "init", "--registry", other)
    assert code == 0 and "already exists (1 run(s)); nothing changed" in out and other.read_bytes() == before
    code, out, _ = run(capsys, "--registry", other, "runs", "list")
    assert code == 0 and " p" in out


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_lock_unlock_and_adopt(capsys, ws):
    d = ws["data"] / "run0010"
    d.mkdir()
    (d / "athinput.kh_org").write_text(TEMPLATE)
    (d / "kh.cli.hst").write_text("# [1]=time [2]=dt\n0.0 0.1\n1.5 0.1\n")
    _age(d)
    code, out, err = run(capsys, "runs", "adopt", d, "--purpose", "legacy run", "--tag", "old")
    assert code == 0, err
    assert "as run 10 [completed]" in out and sorted(p.name for p in d.iterdir()) == ["athinput.kh_org", "kh.cli.hst"]
    code, _, err = run(capsys, "runs", "adopt", d, "--purpose", "again")
    assert code == 2 and "already registered" in err
    code, _, err = run(capsys, "runs", "adopt", ws["tmp"] / "missing", "--purpose", "x")
    assert code == 2 and "not a directory" in err

    try:
        code, out, _ = run(capsys, "runs", "lock", "10")
        assert code == 0 and "locked" in out
        assert not os.access(d / "kh.cli.hst", os.W_OK)
        code, out, _ = run(capsys, "runs", "list")
        assert "L legacy run" in out
    finally:
        code, out, _ = run(capsys, "runs", "unlock", "10")
    assert code == 0 and os.access(d / "kh.cli.hst", os.W_OK)

    # a file written minutes ago: Athena++ may still be running
    (d / "kh.cli.out2.00003.athdf").write_bytes(b"")
    code, _, err = run(capsys, "runs", "lock", "10")
    assert code == 2 and "minute(s) ago" in err and "--force" in err
    assert os.access(d / "kh.cli.hst", os.W_OK)
    try:
        code, out, _ = run(capsys, "runs", "lock", "10", "--force")
        assert code == 0 and not os.access(d / "kh.cli.hst", os.W_OK)
    finally:
        assert run(capsys, "runs", "unlock", "10")[0] == 0

    fresh = ws["data"] / "run0011"
    fresh.mkdir()
    (fresh / "athinput.kh_org").write_text(TEMPLATE)
    (fresh / "kh.cli.hst").write_text("# [1]=time\n0.0\n")
    code, out, err = run(capsys, "runs", "adopt", fresh, "--purpose", "live")
    assert code == 0 and "as run 11 [running]" in out and "warning:" in err and "--status completed" in err
    code, _, err = run(capsys, "runs", "lock", "11")
    assert code == 2 and "run 11 is running" in err


def test_experiments_cli(capsys, ws):
    for purpose in ("c=50", "c=100"):
        src = ("--template", ws["template"]) if purpose == "c=50" else ("--from", "1")
        assert run(capsys, "runs", "new", "--purpose", purpose, *src)[0] == 0
    code, out, _ = run(capsys, "exp", "list")
    assert code == 0 and "no experiments" in out
    code, out, err = run(capsys, "exp", "add", "cscan", "--description", "speed of light scan",
                         "--question", "P ~ c^-2?", "--run", "1:$c=50$:C0:-", "--run", "2:$c=100$:tab:red:--",
                         "--figure", "outputs/cscan.pdf")
    assert code == 0, err
    code, out, _ = run(capsys, "exp", "show", "cscan")
    assert code == 0
    assert "question   : P ~ c^-2?" in out and "tab:red" in out and "$c=100$" in out and "outputs/cscan.pdf" in out
    code, out, _ = run(capsys, "exp", "list")
    assert "cscan" in out and "1,2" in out

    code, _, err = run(capsys, "exp", "add", "ghost", "--description", "d", "--run", "99:ghost")
    assert code == 2 and "run 99 is not registered" in err
    code, _, err = run(capsys, "exp", "add", "cscan", "--description", "d", "--run", "1:x")
    assert code == 2 and "already exists" in err
    assert run(capsys, "exp", "add", "cscan", "--description", "d", "--run", "1:x", "--replace")[0] == 0
    code, _, err = run(capsys, "exp", "show", "nope")
    assert code == 2 and "no experiment 'nope'" in err

    text = ws["reg"].read_text()
    ws["reg"].write_text(text.replace("{run: 1,", "{run: 42,"))
    code, _, err = run(capsys, "exp", "show", "cscan")
    assert code == 2 and "unregistered run(s) 42" in err
    code, out, _ = run(capsys, "runs", "check")
    assert code == 1 and "references unknown run 42" in out


# ---------------------------------------------------------------------- info
def _synthetic_run(root: Path) -> Path:
    d = root / "run0005"
    d.mkdir()
    (d / "athinput.kh_org").write_text(TEMPLATE)
    (d / "kh.cli.hst").write_text("# Athena++ history data\n# [1]=time     [2]=dt\n"
                                  "  0.00000e+00  1.0e-02\n  5.00000e-01  1.0e-02\n  1.50000e+02  1.0e-02\n")
    for n in range(3):
        (d / f"kh.cli.out2.{n:05d}.athdf").write_bytes(b"")
    (d / "high_energy_particles").mkdir()
    (d / "high_energy_particles" / "trajectory_initmbid_3_pid_17.tab").write_text("")
    for gid in range(2):
        (d / f"kh.cli.block{gid}.out4.00001.par.bin").write_bytes(b"")
    (d / "kh.cli.00002.rst").write_bytes(b"")
    return d


def test_info_synthetic_run_by_path_and_id(capsys, ws):
    d = _age(_synthetic_run(ws["data"]))
    code, out, err = run(capsys, "info", d)
    assert code == 0, err
    assert f"run directory: {d}" in out and "registered   : no" in out
    assert "m_cr" in out and "q_mc : 200.0" in out
    assert "athdf out2   : 3 snapshots (#00000-#00002), t = 0-100" in out
    assert "history      : kh.cli.hst (last t = 150)" in out
    assert "trajectories : 1 file(s)" in out
    assert "particles    : 2 file(s) from 1 output(s)" in out
    assert "restarts     : 1 file(s)" in out

    assert run(capsys, "runs", "adopt", d, "--purpose", "synthetic")[0] == 0
    code, out, _ = run(capsys, "info", "5")
    assert code == 0 and "registered   : run 5 [completed] synthetic" in out and "3 snapshots" in out
    code, out, _ = run(capsys, "info", "run0005")
    assert code == 0 and "registered   : run 5" in out

    code, _, err = run(capsys, "info", "404")
    assert code == 2 and err.startswith("error: run 404 not found")


def test_runs_move_and_missing_directory(capsys, ws):
    assert run(capsys, "runs", "new", "--purpose", "fiducial", "--template", ws["template"])[0] == 0
    d = ws["data"] / "run0001"
    moved = ws["tmp"] / "backup" / "run0001"
    moved.parent.mkdir()
    d.rename(moved)
    (ws["data"] / "run1").mkdir()  # unrelated legacy directory with the same number
    code, out, _ = run(capsys, "runs", "show", "1")
    assert code == 0 and f"# resolved run directory: {d} (MISSING" in out
    code, out, _ = run(capsys, "runs", "check")
    assert code == 1 and "ERROR   run 1: run directory" in out and "runs move 1" in out
    for argv in (("runs", "set-status", "1", "submitted"), ("runs", "rehash", "1", "--note", "x"),
                 ("runs", "lock", "1", "--force"), ("runs", "new", "--purpose", "c", "--from", "1")):
        code, _, err = run(capsys, *argv)
        assert code == 2 and "runs move 1 NEW_PATH" in err, argv
    code, _, err = run(capsys, "info", "1")
    assert code == 2 and "does not exist" in err

    code, _, err = run(capsys, "runs", "move", "1", ws["data"] / "run1")
    assert code == 2 and "is missing from" in err
    code, out, err = run(capsys, "runs", "move", "1", moved, "--note", "disk full")
    assert code == 0 and f"run 1: data_path {d} -> {moved}" in out
    code, out, _ = run(capsys, "runs", "show", "1")
    assert "moved: data_path" in out and "disk full" in out and "(exists)" in out
    assert run(capsys, "runs", "set-status", "1", "submitted")[0] == 0


def test_runs_new_from_drifted_parent(capsys, ws):
    assert run(capsys, "runs", "new", "--purpose", "fiducial", "--template", ws["template"])[0] == 0
    inp = ws["data"] / "run0001" / "athinput.kh_org"
    inp.write_text(inp.read_text().replace("M_A = 10", "M_A = 12"))
    code, _, err = run(capsys, "runs", "new", "--purpose", "clone", "--from", "1", "--set", "problem/tau=1")
    assert code == 2 and "changed after registration" in err and "rehash 1" in err
    assert not (ws["data"] / "run0002").exists()
    assert run(capsys, "runs", "rehash", "1", "--note", "M_A=12 intended")[0] == 0
    code, out, err = run(capsys, "runs", "new", "--purpose", "clone", "--from", "1")
    assert code == 0, err
    assert "parent_athinput_sha256:" in run(capsys, "runs", "show", "2")[1]


_CLI = "import sys; from shearpic.cli import main; sys.exit(main(sys.argv[1:]))"


def test_parallel_set_status_from_job_array(capsys, ws):
    """Eight job-array tasks mark their own run submitted at the same time: no update is lost."""
    assert run(capsys, "runs", "new", "--purpose", "base", "--template", ws["template"])[0] == 0
    for i in range(2, 9):
        assert run(capsys, "runs", "new", "--purpose", f"member {i}", "--from", "1")[0] == 0
    import shearpic

    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(shearpic.__file__).parents[1]),
                                                       os.environ.get("PYTHONPATH", "")]))
    procs = [subprocess.Popen([sys.executable, "-c", _CLI, "runs", "set-status", str(i), "submitted",
                               "--job-id", f"1000_{i}", "--registry", str(ws["reg"])],
                              env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for i in range(1, 9)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    reg = load_registry(ws["reg"])
    assert [r.status for r in reg.runs] == ["submitted"] * 8
    assert [r.slurm["job_id"] for r in reg.runs] == [f"1000_{i}" for i in range(1, 9)]
    assert ws["reg"].read_text().startswith("# test registry\n")


@pytest.mark.data
def test_info_run423(capsys, run423, tmp_path, monkeypatch):
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(tmp_path / "runs.yaml"))
    code, out, err = run(capsys, "info", run423)
    assert code == 0, err
    assert "m_cr" in out and "2.94137" in out
    assert "13 snapshots" in out and "t = 0-600" in out
    assert not (tmp_path / "runs.yaml").exists()  # info never writes the registry


def test_info_reports_mixed_particle_formats(capsys, ws):
    d = ws["tmp"] / "run0042"
    d.mkdir()
    (d / "athinput.kh_org").write_text(TEMPLATE)
    for ext in ("tab", "bin"):
        (d / f"kh.test.block0.out4.00001.par.{ext}").write_text("")
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the library warning must not leak into the CLI
        code, out, err = run(capsys, "info", str(d))
    assert code == 0, err
    assert "output(s) 00001 exist in both .par.tab and .par.bin" in out
