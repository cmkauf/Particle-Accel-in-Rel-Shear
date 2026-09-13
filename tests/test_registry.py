"""Run registry (shearpic.registry): new/clone/adopt, status, lock, experiments and check().

Everything happens in tmp directories; the repository's runs.yaml must never change.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import stat
import subprocess
import sys
import time
import warnings
from pathlib import Path

import pytest

from shearpic import env as envmod
from shearpic import registry as regmod
from shearpic.io.athinput import parse_athinput, read_athinput
from shearpic.registry import (
    STATUSES,
    Registry,
    RegistryError,
    diff_athinput,
    extract_params,
    init_registry,
    load_registry,
    parse_overrides,
    registry_lock,
    to_run_id,
    registry_path,
)

TEMPLATE = """\
<comment>
problem   = Driven KH instability
reference =
configure = --prob kh_driven -b --eos isothermal --nghost 4 --p charged -mpi -hdf5

<job>
problem_id = kh.test  # problem ID: base name of output files

<output1>
file_type  = hst        # History data dump
dt         = 0.5        # time increment between outputs

<output2>
file_type  = hdf5       # Binary data dump
variable   = prim       # variables to be output
dt         = 50.0       # time increment between outputs

<output3>
file_type  = rst        # restart dump
dt         = 100.0      # time increment between outputs

<time>
cfl_number = 0.4        # The Courant, Friedrichs, & Lewy (CFL) Number
nlim       = -1         # cycle limit
tlim       = 600.0      # time limit

<mesh>
nx1        = 32         # Number of zones in X1-direction
x1min      = -1.5707963267948966
x1max      = 1.5707963267948966
ix1_bc     = periodic
ox1_bc     = periodic
nx2        = 64
x2min      = -6.283185307179586
x2max      = 6.283185307179586
ix2_bc     = periodic
ox2_bc     = periodic
nx3        = 1
x3min      = -0.5
x3max      = 0.5

<meshblock>
nx1        = 16
nx2        = 16
nx3        = 1

<hydro>
iso_sound_speed = 5              # sound speed

<particles>
backreaction = true     # turn on/off the back reaction of the gas drag
charge_over_mass_over_c = 200.0     # charge of each particles
speed_of_light = 50.0   # speed of light which only used in particle module

<analysis>
dt    = 10.0     # must not be confused with output dt

<problem>
iprob = 0         # 0: tanh profile; 1: sin profile
shear_strength = 1.0  # Shear velocity in unit of U_0
y1 = -3.141592653589793
y2 = 3.141592653589793
M_A = 10          # Alfven Mach number
tau = 0.5         # relaxation time
nu_iso = 0.0      # isotropic viscosity
eta_ohm = 0.0     # Ohmic resistivity
npx1 = 64
npx2 = 128
npx3 = 1
vp_par = 50.0     # reduced momentum p/m
cr_mass = 0.0005  # cosmic ray mass density ratio
"""

REGISTRY_TEXT = """\
# Run registry for tests.
# This header comment must survive CLI edits.
schema_version: 1
run_dir_format: run{id:04d}
runs: []
experiments: {}
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
    """Workspace: tmp registry + data root + template athinput."""
    reg_file = tmp_path / "runs.yaml"
    reg_file.write_text(REGISTRY_TEXT)
    data = tmp_path / "data"
    data.mkdir()
    tdir = tmp_path / "templates"
    tdir.mkdir()
    template = tdir / "athinput.kh_org"
    template.write_text(TEMPLATE)
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(reg_file))
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "local")

    class W:
        pass

    w = W()
    w.reg_file, w.data, w.template, w.tmp = reg_file, data, template, tmp_path
    return w


def _new(reg: Registry, ws, purpose="fiducial", **kw):
    if "parent" not in kw:
        kw.setdefault("template", ws.template)
    return reg.new_run(purpose, **kw)


def _age(root: Path, hours: float = 2.0) -> Path:
    """Backdate every file and directory below ``root`` (a finished simulation)."""
    t = time.time() - hours * 3600
    for dirpath, dirnames, filenames in os.walk(root):
        for name in dirnames + filenames:
            os.utime(os.path.join(dirpath, name), (t, t))
    os.utime(root, (t, t))
    return root


# ------------------------------------------------------------------ helpers
def test_registry_path_precedence(tmp_path, monkeypatch):
    monkeypatch.delenv("PARTICLE_ACCEL_REGISTRY", raising=False)
    monkeypatch.setattr(envmod, "repo_root", lambda: tmp_path / "checkout")
    assert registry_path() == tmp_path / "checkout" / "runs.yaml"
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(tmp_path / "env.yaml"))
    assert registry_path() == tmp_path / "env.yaml"
    assert registry_path(tmp_path / "explicit.yaml") == tmp_path / "explicit.yaml"


def test_registry_path_without_checkout_is_an_error(tmp_path, monkeypatch):
    # e.g. after a non-editable `pip install .`: no silent runs.yaml inside the venv
    monkeypatch.delenv("PARTICLE_ACCEL_REGISTRY", raising=False)
    monkeypatch.setattr(envmod, "repo_root", lambda: None)
    with pytest.raises(RegistryError, match="PARTICLE_ACCEL_REGISTRY or pass --registry"):
        registry_path()
    with pytest.raises(RegistryError, match="cannot locate"):
        load_registry()
    assert registry_path(tmp_path / "x.yaml") == tmp_path / "x.yaml"


def test_parse_helpers():
    assert to_run_id(7) == to_run_id("7") == to_run_id("run0007") == to_run_id("run7") == 7
    with pytest.raises(ValueError):
        to_run_id("myrun7")
    assert parse_overrides({"problem/M_A": "20", "particles/backreaction": "false"}) == {
        "problem": {"M_A": 20}, "particles": {"backreaction": False}}
    assert parse_overrides(["<problem>/tau=0.25"]) == {"problem": {"tau": 0.25}}
    for bad in (["problem.M_A=3"], ["problem/M_A"], {"M_A": 3}):
        with pytest.raises(ValueError):
            parse_overrides(bad)


def test_extract_params_and_diff():
    inp = parse_athinput(TEMPLATE)
    p = extract_params(inp)
    assert p["c"] == 50.0 and p["q_mc"] == 200.0 and p["M_A"] == 10 and p["nx2"] == 64
    assert p["backreaction"] is True and p["cs"] == 5 and p["tlim"] == 600.0
    assert "pres" not in p  # absent keys are skipped, not defaulted
    new = parse_athinput(TEMPLATE.replace("M_A = 10 ", "M_A = 20 ").replace("nu_iso = 0.0", "nu_iso = 0")
                         + "extra = 1\n")
    # 0.0 -> 0 is a reformat, not a change; added keys appear with None on the old side
    assert diff_athinput(inp, new) == {"problem/M_A": [10, 20], "problem/extra": [None, 1]}


# ----------------------------------------------------------------- new_run
def test_new_run_from_template(ws):
    reg = load_registry()
    assert reg.path == ws.reg_file and reg.next_id() == 1
    template_bytes = ws.template.read_bytes()
    rec = _new(reg, ws, tags=["fiducial"], pgen_source="kh_driven.cpp", pgen_commit="abc123", author="Student")

    run_dir = ws.data / "run0001"
    assert rec.id == 1 and rec.status == "planned" and Path(rec.data_path) == run_dir
    assert (run_dir / "athinput.kh_org").read_bytes() == template_bytes  # untouched copy
    assert ws.template.read_bytes() == template_bytes                     # template never modified
    assert (run_dir / "run_info.yaml").exists()
    assert "purpose: fiducial" in (run_dir / "run_info.yaml").read_text()
    assert rec.athinput_sha256 == read_athinput(run_dir / "athinput.kh_org").sha256
    assert rec.params["c"] == 50.0 and rec.problem_id == "kh.test" and rec.machine == "local"
    assert rec.pgen == {"source": "kh_driven.cpp", "git_commit": "abc123"} and rec.author == "Student"
    assert rec.template["changes"] == {} and rec.parent is None

    again = load_registry()
    got = again.get_run(1)
    assert got.purpose == "fiducial" and got.tags == ["fiducial"] and again.next_id() == 2
    assert got.run_dir() == run_dir
    cfg = got.config()
    assert cfg.c == 50.0 and cfg.nx == (32, 64, 1) and cfg.run_id == 1
    assert again.check() == []


def test_clone_with_overrides_records_changes(ws):
    reg = load_registry()
    _new(reg, ws)
    rec = reg.new_run("c=100 at fixed q/mc", parent="run0001",
                      overrides={"particles/speed_of_light": "100", "problem/M_A": 20})
    assert rec.id == 2 and rec.parent == 1
    assert rec.changes_from_parent == {"particles/speed_of_light": [50.0, 100.0], "problem/M_A": [10, 20]}
    assert rec.parent_athinput_sha256 == load_registry().get_run(1).athinput_sha256
    inp = read_athinput(ws.data / "run0002" / "athinput.kh_org")
    assert inp.get("particles", "speed_of_light") == 100 and inp.get("problem", "M_A") == 20
    assert inp.get("analysis", "dt") == 10.0 and inp.get("output2", "dt") == 50.0
    assert load_registry().get_run(2).params["c"] == 100


def test_integer_override_of_float_key_stays_float(ws):
    reg = load_registry()
    _new(reg, ws)
    rec = reg.new_run("c=100", parent=1, overrides=["particles/speed_of_light=100", "problem/M_A=20",
                                                    "problem/tau=1", "particles/backreaction=false"])
    c_old, c_new = rec.changes_from_parent["particles/speed_of_light"]
    assert isinstance(c_new, float) and c_new == 100.0
    assert type(rec.changes_from_parent["problem/tau"][1]) is float        # tau = 0.5 in the source
    assert type(rec.changes_from_parent["problem/M_A"][1]) is int          # M_A = 10 was an int: untouched
    assert rec.changes_from_parent["particles/backreaction"] == [True, False]
    text = (ws.data / "run0002" / "athinput.kh_org").read_text()
    assert re.search(r"^speed_of_light\s*=\s*100\.0\b", text, re.M) and re.search(r"^tau\s*=\s*1\.0\b", text, re.M)
    written = read_athinput(ws.data / "run0002" / "athinput.kh_org")
    assert type(written.get("particles", "speed_of_light")) is float and type(written.get("problem", "tau")) is float
    got = load_registry().get_run(2)
    assert type(got.params["c"]) is float and "speed_of_light: [50.0, 100.0]" in ws.reg_file.read_text()
    dry = reg.new_run("dry", template=ws.template, overrides={"particles/speed_of_light": 100}, dry_run=True)
    assert type(dry.params["c"]) is float


def test_unknown_override_key_warns(ws):
    reg = load_registry()
    with pytest.warns(UserWarning, match="shear_strenght"):
        rec = _new(reg, ws, overrides={"problem/shear_strenght": 2.0})
    assert rec.template["changes"] == {"problem/shear_strenght": [None, 2.0]}


def test_new_run_skips_ids_of_unregistered_directories(ws):
    existing = ws.data / "run0001"
    existing.mkdir()
    (existing / "precious.dat").write_text("do not touch")
    (ws.data / "run424").mkdir()                  # legacy, different zero padding
    other_root = ws.tmp / "other_root"
    (other_root / "run00500").mkdir(parents=True)
    reg = load_registry()
    assert reg.next_id() == 425
    assert reg.next_id(extra_roots=[other_root]) == 501
    rec = _new(reg, ws)
    assert rec.id == 425 and (ws.data / "run0425" / "athinput.kh_org").exists()
    assert sorted(p.name for p in existing.iterdir()) == ["precious.dat"]
    rec = _new(reg, ws, data_root=other_root)     # the target root counts too
    assert rec.id == 501 and (other_root / "run0501").is_dir()
    assert not (ws.data / "run0501").exists()


def test_new_run_refuses_existing_directory_with_other_padding(ws, monkeypatch):
    legacy = ws.data / "run1"
    legacy.mkdir()
    (legacy / "precious.dat").write_text("do not touch")
    before = ws.reg_file.read_bytes()
    reg = load_registry()
    monkeypatch.setattr(Registry, "next_id", lambda self, *a, **k: 1)  # e.g. created by someone else meanwhile
    with pytest.raises(FileExistsError, match="adopt") as err:
        _new(reg, ws)
    assert str(legacy) in str(err.value)
    assert sorted(p.name for p in ws.data.iterdir()) == ["run1"]
    assert sorted(p.name for p in legacy.iterdir()) == ["precious.dat"]
    assert ws.reg_file.read_bytes() == before and load_registry().runs == []


def test_new_run_argument_validation(ws):
    reg = load_registry()
    with pytest.raises(ValueError, match="purpose"):
        reg.new_run("  ", template=ws.template)
    with pytest.raises(ValueError, match="exactly one"):
        reg.new_run("x")
    _new(reg, ws)
    with pytest.raises(ValueError, match="exactly one"):
        reg.new_run("x", template=ws.template, parent=1)
    with pytest.raises(KeyError, match="run 9 is not registered"):
        reg.new_run("x", parent=9)
    with pytest.raises(FileNotFoundError):
        reg.new_run("x", template=ws.tmp / "nope")


def test_clone_refuses_drifted_parent(ws):
    reg = load_registry()
    _new(reg, ws)
    inp = ws.data / "run0001" / "athinput.kh_org"
    inp.write_text(inp.read_text().replace("M_A = 10 ", "M_A = 99 "))
    before = ws.reg_file.read_bytes()
    for dry in (True, False):
        with pytest.raises(ValueError, match="changed after registration") as err:
            reg.new_run("clone", parent=1, overrides={"problem/tau": 1.0}, dry_run=dry)
        assert "rehash 1" in str(err.value) and "--template" in str(err.value)
    assert ws.reg_file.read_bytes() == before and not (ws.data / "run0002").exists()
    reg.rehash(1, "M_A=99 intended")
    rec = reg.new_run("clone", parent=1)
    assert rec.id == 2 and rec.parent_athinput_sha256 == read_athinput(inp).sha256
    assert read_athinput(ws.data / "run0002" / "athinput.kh_org").get("problem", "M_A") == 99


def test_dry_run_writes_nothing(ws):
    before = ws.reg_file.read_bytes()
    reg = load_registry()
    rec = _new(reg, ws, data_root=ws.tmp / "fresh_root", dry_run=True, overrides={"problem/M_A": 3})
    assert rec.id == 1 and rec.template["changes"] == {"problem/M_A": [10, 3]}
    assert not (ws.tmp / "fresh_root").exists()
    assert ws.reg_file.read_bytes() == before and reg.runs == []


def test_explicit_data_root(ws):
    reg = load_registry()
    rec = _new(reg, ws, data_root=ws.tmp / "other_root")
    assert (ws.tmp / "other_root" / "run0001" / "athinput.kh_org").exists()
    assert reg.run_dir(rec.id) == ws.tmp / "other_root" / "run0001"


# ------------------------------------------------------------------- adopt
def _fake_run(root: Path, name: str, with_hst: bool = True) -> Path:
    d = root / name
    d.mkdir(parents=True)
    (d / "athinput.kh_org").write_text(TEMPLATE)
    if with_hst:
        (d / "kh.test.hst").write_text("# Athena++ history data\n# [1]=time     [2]=dt\n0.0 0.1\n0.5 0.1\n")
    return d


def test_adopt_run_writes_nothing_into_directory(ws):
    d = _age(_fake_run(ws.data, "run0042"))
    listing = sorted(p.name for p in d.iterdir())
    reg = load_registry()
    rec = reg.adopt_run(d, "legacy fiducial run")
    assert rec.id == 42 and rec.status == "completed" and rec.params["q_mc"] == 200.0
    assert sorted(p.name for p in d.iterdir()) == listing
    assert "adopted" in rec.notes[0]["text"]
    assert reg.next_id() == 43
    with pytest.raises(ValueError, match="already registered"):
        reg.adopt_run(d, "again")
    with pytest.raises(ValueError, match="already registered"):
        reg.adopt_run(d, "again", run_id=5)
    other = _age(_fake_run(ws.tmp / "elsewhere", "my_test", with_hst=False))
    rec2 = reg.adopt_run(other, "oddly named directory")
    assert rec2.id == 43 and rec2.status == "planned"
    assert load_registry().check() == []


def test_adopt_recently_modified_run_is_running(ws):
    d = _age(_fake_run(ws.data, "run0042"))
    (d / "kh.test.out2.00007.athdf").write_bytes(b"")  # written just now: Athena++ is still going
    reg = load_registry()
    with pytest.warns(UserWarning, match="minute\\(s\\) ago.*running"):
        rec = reg.adopt_run(d, "live run")
    assert rec.status == "running"
    d2 = _fake_run(ws.data, "run0043")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert reg.adopt_run(d2, "explicit", status="completed").status == "completed"


# ------------------------------------------------------------------ status
def test_set_status_notes_and_transitions(ws):
    reg = load_registry()
    _new(reg, ws)
    rec = reg.set_status(1, "submitted", note="first try", job_id=1234567, partition="skx", nodes=4)
    assert rec.status == "submitted"
    assert rec.slurm == {"job_id": "1234567", "partition": "skx", "nodes": 4}
    assert rec.notes[-1]["text"] == "status planned -> submitted (job 1234567): first try"
    assert len(rec.notes[-1]["date"]) == 10

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        reg.set_status(1, "running")
    (ws.data / "run0001" / "kh.test.hst").write_text("# [1]=time\n0.0\n")
    reg.set_status(1, "COMPLETED")
    with pytest.warns(UserWarning, match="unusual status change completed -> running"):
        reg.set_status(1, "running")
    with pytest.raises(ValueError, match="invalid status"):
        reg.set_status(1, "done")
    reg.add_note(1, "restarted from rst 4")
    got = load_registry().get_run(1)
    assert got.status == "running" and got.last_note == "restarted from rst 4"
    assert [n["text"].split(":")[0] for n in got.notes[:3]] == [
        "status planned -> submitted (job 1234567)", "status submitted -> running", "status running -> completed"]


def test_superseded_by(ws):
    reg = load_registry()
    _new(reg, ws)
    _new(reg, ws, purpose="fixed version")
    with pytest.raises(KeyError):
        reg.set_status(1, "superseded", superseded_by=7)
    rec = reg.set_status(1, "superseded", superseded_by=2)
    assert rec.superseded_by == 2 and rec.status == "superseded"


def test_get_run_error_is_helpful(ws):
    reg = load_registry()
    _new(reg, ws)
    with pytest.raises(KeyError) as info:
        reg.get_run(5)
    assert "run 5 is not registered" in info.value.args[0] and "1" in info.value.args[0]
    assert reg.run(1).id == 1 and reg.has_run("run0001") and not reg.has_run(5)
    assert set(STATUSES) >= {"planned", "completed"}


# -------------------------------------------------------------------- lock
@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_lock_and_unlock(ws):
    reg = load_registry()
    _new(reg, ws)
    d = ws.data / "run0001"
    (d / "sub").mkdir()
    (d / "sub" / "data.bin").write_bytes(b"123")
    with pytest.raises(ValueError, match="planned"):
        reg.lock_run(1)
    reg.set_status(1, "running")
    with pytest.raises(ValueError, match="still needs to write"):
        reg.lock_run(1)
    reg.set_status(1, "failed")
    with pytest.raises(ValueError, match="minute\\(s\\) ago") as err:  # files were written seconds ago
        reg.lock_run(1)
    assert "0 minute(s) ago" in str(err.value) and "force" in str(err.value)
    assert not load_registry().get_run(1).locked
    _age(d)
    try:
        n = reg.lock_run(1)
        assert n >= 4
        for p in (d, d / "sub", d / "athinput.kh_org", d / "sub" / "data.bin"):
            assert not stat.S_IMODE(p.stat().st_mode) & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH), p
        with pytest.raises(PermissionError):
            (d / "new_file").write_text("x")
        rec = load_registry().get_run(1)
        assert rec.locked and "locked" in rec.last_note
        assert load_registry().check() == []
    finally:
        reg.unlock_run(1)
    assert os.access(d / "sub" / "data.bin", os.W_OK)
    (d / "new_file").write_text("x")
    assert not load_registry().get_run(1).locked
    try:  # force locks a live directory
        with pytest.warns(UserWarning, match="unusual status change failed -> running"):
            reg.set_status(1, "running")
        reg.lock_run(1, force=True)
        assert load_registry().get_run(1).locked is True
    finally:
        reg.unlock_run(1)


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores permissions")
def test_lock_continues_after_chmod_failure(ws, monkeypatch):
    reg = load_registry()
    _new(reg, ws)
    d = ws.data / "run0001"
    for i in range(5):
        (d / f"f{i}.bin").write_bytes(b"1")
    reg.set_status(1, "failed")
    _age(d)
    real_chmod = os.chmod
    bad = {str(d / "f1.bin"), str(d / "f3.bin")}

    def flaky_chmod(path, mode, *a, **k):
        if str(path) in bad:
            raise PermissionError(1, "Operation not permitted", str(path))
        return real_chmod(path, mode, *a, **k)

    monkeypatch.setattr(regmod.os, "chmod", flaky_chmod)
    try:
        with pytest.warns(UserWarning, match="2 path\\(s\\) could not be made read-only"):
            n = reg.lock_run(1)
        assert n >= 5  # everything else was still locked
        assert not os.access(d / "f4.bin", os.W_OK) and not os.access(d, os.W_OK)
        assert os.access(d / "f1.bin", os.W_OK)
        rec = load_registry().get_run(1)
        assert rec.locked == "partial" and "f1.bin" in rec.last_note and "2 path(s)" in rec.last_note
        assert any("partially locked" in p.message for p in load_registry().check())
    finally:
        monkeypatch.setattr(regmod.os, "chmod", real_chmod)
        reg.unlock_run(1)
    assert load_registry().get_run(1).locked is False


def test_chmod_tree_collects_walk_errors(tmp_path, monkeypatch):
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "x").write_text("1")
    real_walk = os.walk

    def walk(root, onerror=None, **kw):
        onerror(PermissionError(13, "Permission denied", str(tmp_path / "hidden")))
        yield from real_walk(root, onerror=onerror, **kw)

    monkeypatch.setattr(regmod.os, "walk", walk)
    n, failures = regmod._chmod_tree(tmp_path, add=stat.S_IWUSR)
    assert failures == [(str(tmp_path / "hidden"), "Permission denied")]


# -------------------------------------------------------------------- yaml
def test_comments_survive_edits(ws):
    reg = load_registry()
    _new(reg, ws)
    text = ws.reg_file.read_text().replace("purpose: fiducial", "purpose: fiducial  # reproduces Fig. 3")
    ws.reg_file.write_text(text)
    reg = load_registry()
    reg.set_status(1, "submitted", job_id="99")
    _new(reg, ws, purpose="second")
    reg.add_experiment("scan", "a scan", [(1, "fid"), (2, "second")])
    out = ws.reg_file.read_text()
    assert out.startswith("# Run registry for tests.\n# This header comment must survive CLI edits.\n")
    assert "purpose: fiducial  # reproduces Fig. 3" in out
    assert "job_id: '99'" in out
    rec = load_registry().get_run(1)
    assert rec.purpose == "fiducial" and rec.slurm["job_id"] == "99"


def test_stale_registry_objects_apply_changes_on_top(ws):
    a = load_registry()
    b = load_registry()
    _new(a, ws)
    rec = _new(b, ws, data_root=ws.tmp / "root_b")   # b re-reads the file inside the lock
    assert rec.id == 2 and (ws.tmp / "root_b" / "run0002").is_dir()
    a.add_note(1, "from a")
    b.add_note(1, "from b")
    got = load_registry()
    assert [r.id for r in got.runs] == [1, 2]
    assert [n["text"] for n in got.get_run(1).notes] == ["from a", "from b"]


def test_unsaved_changes_conflict_is_detected(ws):
    _new(load_registry(), ws)
    a = load_registry()
    b = load_registry()
    b.add_note(1, "unsaved in b", save=False)
    a.add_note(1, "saved by a")
    with pytest.raises(RegistryError, match="modified by someone else"):
        b.save()
    with pytest.raises(RegistryError, match="unsaved changes"):
        b.add_note(1, "more")
    notes = [n["text"] for n in load_registry().get_run(1).notes]
    assert notes == ["saved by a"]


def test_transaction_rolls_back_on_error(ws):
    reg = load_registry()
    _new(reg, ws)
    before = ws.reg_file.read_bytes()
    with pytest.raises(KeyError):
        with reg.transaction():
            reg.add_note(1, "first")
            reg.add_note(99, "unknown run")
    assert ws.reg_file.read_bytes() == before
    assert load_registry().get_run(1).notes == [] and reg.get_run(1).notes == []
    with reg.transaction():
        reg.add_note(1, "a")
        reg.set_status(1, "failed")
        assert load_registry().get_run(1).notes == []  # nothing is written before the block ends
    assert [n["text"] for n in load_registry().get_run(1).notes] == ["a", "status planned -> failed"]
    mtime = ws.reg_file.stat().st_mtime_ns
    with reg.transaction():
        reg.get_run(1)  # read-only blocks do not rewrite the file
    assert ws.reg_file.stat().st_mtime_ns == mtime


def test_registry_lock_is_exclusive_and_reentrant(ws):
    import threading

    order = []
    with registry_lock(ws.reg_file) as lock:
        assert lock == Path(str(ws.reg_file) + ".lock") and lock.exists()
        with registry_lock(ws.reg_file):  # re-entrant in the same thread
            pass

        def other():
            with registry_lock(ws.reg_file):
                order.append("other")

        t = threading.Thread(target=other)
        t.start()
        time.sleep(0.2)
        order.append("main")
    t.join(5)
    assert order == ["main", "other"]
    with registry_lock(ws.reg_file):
        def impatient():
            try:
                with registry_lock(ws.reg_file, timeout=0.1):
                    order.append("got it")
            except RegistryError as err:
                order.append(str(err))

        t = threading.Thread(target=impatient)
        t.start()
        t.join(5)
    assert "could not lock the registry" in order[-1]


def test_exclusive_file_lock_fallback(tmp_path):
    lock = tmp_path / "runs.yaml.lock"
    release = regmod._acquire_exclusive_file(lock, timeout=1)
    assert lock.exists() and "pid" in lock.read_text()
    with pytest.raises(RegistryError, match="could not lock"):
        regmod._acquire_exclusive_file(lock, timeout=0.1)
    release()
    assert not lock.exists()


_NOTE_WORKER = """
import sys, time
from shearpic.registry import load_registry
path, worker, start = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
while time.time() < start:
    time.sleep(0.001)
for k in range(2):
    load_registry(path).add_note(1, f"worker {worker} note {k}")
"""


def test_concurrent_processes_lose_no_updates(ws):
    """8 processes (like SLURM job-array tasks) append 2 notes each at the same moment."""
    reg = load_registry()
    _new(reg, ws)
    env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(Path(regmod.__file__).parents[1]),
                                                       os.environ.get("PYTHONPATH", "")]))
    start = time.time() + 1.5
    procs = [subprocess.Popen([sys.executable, "-c", _NOTE_WORKER, str(ws.reg_file), str(i), str(start)], env=env,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) for i in range(8)]
    for p in procs:
        out, err = p.communicate(timeout=120)
        assert p.returncode == 0, err
    notes = [n["text"] for n in load_registry().get_run(1).notes]
    assert sorted(notes) == sorted(f"worker {i} note {k}" for i in range(8) for k in range(2))
    assert load_registry().check() == []


def test_missing_registry_file_is_read_only(tmp_path, ws):
    path = tmp_path / "sub" / "new_registry.yaml"
    reg = load_registry(path)
    assert reg.runs == [] and not reg.exists and not path.exists()
    with pytest.raises(RegistryError, match="runs init"):
        _new(reg, ws)
    assert not (ws.data / "run0001").exists()  # nothing was created
    rec = _new(reg, ws, dry_run=True)  # dry runs need no file
    assert rec.id == 1
    with pytest.raises(RegistryError, match="does not exist"):
        reg.save()
    assert not path.exists()

    reg = load_registry(path, create=True)
    assert path.exists() and reg.exists and reg.runs == []
    _new(reg, ws)
    assert load_registry(path).get_run(1).purpose == "fiducial"
    with pytest.raises(FileExistsError):
        init_registry(path)
    fresh = init_registry(tmp_path / "other.yaml")
    assert fresh.path.read_text().startswith("# Run registry") and fresh.runs == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX permission bits")
def test_new_registry_file_respects_umask(tmp_path):
    old = os.umask(0o022)
    try:
        init_registry(tmp_path / "shared.yaml")
        assert stat.S_IMODE((tmp_path / "shared.yaml").stat().st_mode) == 0o644
        os.umask(0o002)
        regmod._atomic_write(tmp_path / "group.yaml", "x: 1\n")
        assert stat.S_IMODE((tmp_path / "group.yaml").stat().st_mode) == 0o664
        os.chmod(tmp_path / "group.yaml", 0o600)  # an existing file keeps its mode
        regmod._atomic_write(tmp_path / "group.yaml", "x: 2\n")
        assert stat.S_IMODE((tmp_path / "group.yaml").stat().st_mode) == 0o600
    finally:
        os.umask(old)


def test_malformed_registry(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("schema_version: 2\nruns: []\n")
    with pytest.raises(RegistryError, match="schema_version"):
        load_registry(bad)
    bad.write_text("runs: {a: 1}\n")
    with pytest.raises(RegistryError, match="list"):
        load_registry(bad)


# ------------------------------------------------------------- experiments
def test_experiments(ws):
    reg = load_registry()
    _new(reg, ws)
    _new(reg, ws, purpose="c=100")
    e = reg.add_experiment("c_scan", "speed of light scan", [
        {"run": 1, "label": "$c=50$", "color": "C0", "linestyle": "-"}, (2, "$c=100$", "tab:red")],
        question="Does heating scale as c^-2?", figures=["outputs/c_scan.pdf"])
    assert e.run_ids == [1, 2] and len(e) == 2 and e.figures == ("outputs/c_scan.pdf",)
    first, second = list(load_registry().experiment("c_scan"))
    assert first.style() == {"label": "$c=50$", "color": "C0", "linestyle": "-"}
    assert second.style() == {"label": "$c=100$", "color": "tab:red"}
    assert first.run_dir == ws.data / "run0001" and first.config().c == 50.0

    with pytest.raises(KeyError, match="not registered"):
        reg.add_experiment("dangling", "refers to nothing", [(99, "ghost")])
    with pytest.raises(ValueError, match="label"):
        reg.add_experiment("nolabel", "missing label", [{"run": 1}])
    with pytest.raises(ValueError, match="already exists"):
        reg.add_experiment("c_scan", "dup", [(1, "a")])
    with pytest.raises(KeyError, match="no experiment"):
        reg.experiment("nope")
    assert "dangling" not in load_registry().experiment_names

    text = ws.reg_file.read_text()
    assert text.count("{run: 2,") == 1
    ws.reg_file.write_text(text.replace("{run: 2,", "{run: 7,"))  # hand edit: dangling reference
    reg = load_registry()
    with pytest.raises(KeyError, match="unregistered run"):
        reg.experiment("c_scan")
    msgs = [p.message for p in reg.check() if p.level == "error"]
    assert any("references unknown run 7" in m for m in msgs)


# ------------------------------------------------------------------- check
def test_check_detects_hash_drift_and_rehash(ws):
    reg = load_registry()
    _new(reg, ws)
    inp = ws.data / "run0001" / "athinput.kh_org"
    inp.write_text(inp.read_text().replace("M_A = 10 ", "M_A = 30 "))
    problems = load_registry().check()
    assert [p.level for p in problems] == ["error"]
    assert problems[0].run_id == 1 and "changed after registration" in problems[0].message
    assert "ERROR" in str(problems[0])

    reg = load_registry()
    with pytest.raises(ValueError):
        reg.rehash(1, "")
    changes = reg.rehash(1, "M_A=30 was intended")
    assert changes == {"M_A": [10, 30]}
    assert load_registry().check() == []
    assert "M_A: 10 -> 30" in load_registry().get_run(1).last_note


def test_check_unregistered_and_missing_dirs(ws):
    reg = load_registry()
    _new(reg, ws)
    _fake_run(ws.data, "run0077")
    (ws.data / "not_a_run").mkdir()
    problems = reg.check()
    assert [(p.level, p.run_id) for p in problems] == [("warning", 77)]
    assert "not registered" in problems[0].message

    shutil.rmtree(ws.data / "run0001")
    problems = load_registry().check()
    assert ("error", 1) in [(p.level, p.run_id) for p in problems]
    assert any("does not exist" in p.message for p in problems)


# --------------------------------------------------- run directory resolution
def test_run_dir_never_guesses_other_padding(ws):
    reg = load_registry()
    _new(reg, ws)
    reg.set_status(1, "failed")
    moved = ws.tmp / "backup" / "run0001"
    moved.parent.mkdir()
    (ws.data / "run0001").rename(moved)
    legacy = _age(_fake_run(ws.data, "run1"))           # an unrelated legacy run with the same number
    legacy_inp = legacy / "athinput.kh_org"
    legacy_inp.write_text(legacy_inp.read_text().replace("M_A = 10 ", "M_A = 77 "))
    before = legacy_inp.read_bytes()

    reg = load_registry()
    assert reg.run_dir(1) == ws.data / "run0001"         # the recorded path, not run1
    probs = reg.check()
    assert any(p.level == "error" and p.run_id == 1 and "does not exist" in p.message for p in probs)
    for call in (lambda: reg.rehash(1, "x"), lambda: reg.lock_run(1, force=True), lambda: reg.unlock_run(1),
                 lambda: reg.set_status(1, "planned"), lambda: reg.new_run("clone", parent=1)):
        with pytest.raises(FileNotFoundError, match="runs move 1"):
            call()
    with pytest.raises(FileNotFoundError):
        reg.config(1)
    assert legacy_inp.read_bytes() == before and load_registry().get_run(1).params["M_A"] == 10
    assert os.access(legacy, os.W_OK)

    with pytest.raises(ValueError, match="differs from the registered one"):
        reg.move_run(1, legacy)                          # wrong directory is refused
    rec = reg.move_run(1, moved, note="copied to backup disk")
    assert rec.data_path == str(moved) and "moved" in rec.last_note and "backup disk" in rec.last_note
    assert load_registry().run_dir(1) == moved
    assert reg.set_status(1, "planned").status == "planned"
    assert _new(reg, ws).id == 2
    with pytest.raises(ValueError, match="already the directory of run 1"):
        reg.move_run(2, moved)


def test_move_run_force_and_unverifiable(ws):
    reg = load_registry()
    other = _age(_fake_run(ws.tmp / "x", "no_input", with_hst=False))
    (other / "athinput.kh_org").unlink()
    with pytest.warns(UserWarning, match="no athinput"):
        reg.adopt_run(other, "no input file")
    target = ws.tmp / "y"
    target.mkdir()
    with pytest.raises(ValueError, match="no athinput hash"):
        reg.move_run(1, target)
    with pytest.warns(UserWarning, match="recording"):
        rec = reg.move_run(1, target, force=True)
    assert rec.data_path == str(target) and "unverified" in rec.last_note
    with pytest.raises(FileNotFoundError, match="not a directory"):
        reg.move_run(1, ws.tmp / "nowhere")


def test_same_name_under_moved_data_root(ws, monkeypatch):
    reg = load_registry()
    _new(reg, ws)
    new_root = ws.tmp / "scratch_runs"
    ws.data.rename(new_root)                              # e.g. registered on a laptop, used on the cluster
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(new_root))
    reg = load_registry()
    d = new_root / "run0001"
    assert reg.run_dir(1) == d and reg.config(1).c == 50.0
    probs = reg.check()
    assert [(p.level, p.run_id) for p in probs] == [("warning", 1)]
    assert "recorded directory" in probs[0].message and "runs move 1" in probs[0].message
    with pytest.warns(UserWarning, match="using"):
        reg.set_status(1, "submitted")                    # same name and the registered athinput: accepted
    with pytest.raises(FileNotFoundError, match="confirm it with `shearpic runs move 1"):
        reg.rehash(1, "never from a guessed directory")
    inp = d / "athinput.kh_org"
    inp.write_text(inp.read_text().replace("M_A = 10 ", "M_A = 11 "))
    with pytest.raises(ValueError, match="may be a different run"):
        reg.set_status(1, "running")
    with pytest.warns(UserWarning, match="unusual status change"):
        assert reg.set_status(1, "archived").status == "archived"   # archiving never needs the directory
    with pytest.warns(UserWarning, match="differs from the registered one"):
        reg.move_run(1, d, force=True)
    assert [p.level for p in load_registry().check()] == ["error"]  # only the hash drift is left


def test_check_completed_without_hst(ws):
    reg = load_registry()
    _new(reg, ws)
    with pytest.warns(UserWarning, match="no \\*.hst"):
        reg.set_status(1, "completed")
    errors = [p for p in reg.check() if p.level == "error"]
    assert len(errors) == 1 and "no *.hst" in errors[0].message
    (ws.data / "run0001" / "kh.test.hst").write_text("# [1]=time\n0.0\n")
    assert reg.check() == []


def test_check_hand_edited_problems(ws):
    ws.reg_file.write_text("""\
schema_version: 1
run_dir_format: run{id:04d}
runs:
  - id: 1
    purpose: ''
    status: done
    data_path: run0001
    parent: 12
  - id: 1
    purpose: duplicate
    status: planned
  - id: 2
    purpose: stale job
    status: running
    updated: '2020-01-01T00:00:00+00:00'
    superseded_by: 44
  - purpose: no id
experiments:
  e1:
    description: d
    runs:
      - {run: 1}
""")
    (ws.data / "run0001").mkdir()
    (ws.data / "run0002").mkdir()
    reg = load_registry()
    probs = reg.check()
    errors = {(p.run_id, p.message.split(" (")[0]) for p in probs if p.level == "error"}
    assert reg.run_dir(2) == ws.data / "run0002"  # no data_path: exactly run_dir_format
    warns = [p for p in probs if p.level == "warning"]
    text = "\n".join(map(str, probs))
    assert (1, "id 1 is used by 2 records") in errors
    assert (1, "empty purpose") in errors
    assert (1, "invalid status 'done'") in errors
    assert (1, "parent references unknown run 12") in errors
    assert (2, "superseded_by references unknown run 44") in errors
    assert "non-integer id" in text
    assert "entry for run 1 has no label" in text
    assert any("not updated for" in p.message and p.run_id == 2 for p in warns)
    # the relative data_path resolves against the data root
    assert reg.run_dir(1) == ws.data / "run0001"


@pytest.mark.data
def test_adopt_run423_read_only(run423, tmp_path):
    before = {p.name: p.stat().st_mtime_ns for p in run423.iterdir()}
    reg = load_registry(tmp_path / "runs.yaml", create=True)
    rec = reg.adopt_run(run423, "fiducial run of arXiv:2512.12720")
    assert rec.id == 423 and rec.status == "completed"
    assert rec.params["c"] == 50.0 and rec.params["q_mc"] == 200.0 and rec.problem_id == "org.stir.feedback"
    assert not (run423 / "run_info.yaml").exists()
    assert {p.name: p.stat().st_mtime_ns for p in run423.iterdir()} == before
    reg2 = load_registry(tmp_path / "runs.yaml")
    assert reg2.run_dir(423) == run423
    assert abs(reg2.config(423).m_cr - 2.94137e-8) < 1e-12
    assert [p for p in reg2.check(data_roots=[]) if p.level == "error"] == []


# ------------------------------------------------- CRLF hashes, stale copies, lock files
def test_crlf_athinput_hashes_consistently(ws):
    """Recorded hash (AthInput.sha256) and the check()/move hash must agree for CRLF files."""
    d = ws.data / "run0043"
    d.mkdir()
    (d / "athinput.kh_org").write_bytes(TEMPLATE.replace("\n", "\r\n").encode())
    (d / "kh.test.hst").write_text("# Athena++ history data\n# [1]=time     [2]=dt\n0.0 0.1\n")
    _age(d)
    reg = load_registry()
    reg.adopt_run(d, "CRLF input")
    assert regmod._athinput_sha(d / "athinput.kh_org") == read_athinput(d / "athinput.kh_org").sha256
    assert not [p for p in load_registry().check() if p.level == "error"]
    copy = ws.tmp / "moved" / "run0043"
    shutil.copytree(d, copy)
    load_registry().move_run(43, copy)  # identical copy: accepted without --force


def test_check_explains_stale_copy_of_registered_run(ws):
    reg = load_registry()
    rec = _new(reg, ws)
    stale = ws.data / f"run{rec.id}"  # same number, different zero padding
    _fake_run(ws.data, stale.name)
    msgs = [p.message for p in load_registry().check() if p.run_id == rec.id and p.level == "warning"]
    assert any("has the number of registered run" in m for m in msgs), msgs
    assert not any("runs adopt" in m for m in msgs)


def test_exclusive_fallback_never_uses_the_flock_file_name(ws, monkeypatch):
    lock = ws.reg_file.with_name(ws.reg_file.name + ".lock")
    lock.write_text("")  # left behind by a POSIX host using flock
    monkeypatch.setattr(regmod, "fcntl", None)
    with registry_lock(ws.reg_file, timeout=1.0) as held:
        assert held == Path(os.path.abspath(lock))
        assert lock.with_name(lock.name + ".excl").exists()
    assert not lock.with_name(lock.name + ".excl").exists()
