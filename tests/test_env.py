"""Unit tests for shearpic.env: machine detection, CPU/memory limits, data/output roots, run lookup.

``detect_environment`` is cached on environment variables, so tests that patch the host name
clear ``_detect_cached``; the autouse fixture also clears it around every test.  Tests of the
default roots patch :func:`repo_root` and never use the real checkout.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from shearpic import env as envmod
from shearpic.env import Environment, detect_environment, output_dir, repo_root, resolve_run, run_output_dir

GiB = 2**30


@pytest.fixture(autouse=True)
def _fresh_detection(monkeypatch, tmp_path):
    for key in ("SCRATCH", "PYTHON_CPU_COUNT", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(envmod, "_have_sbatch", lambda: False)  # tests must not depend on the host's PATH
    envmod._detect_cached.cache_clear()
    yield
    envmod._detect_cached.cache_clear()


def _set_host(monkeypatch, name: str) -> None:
    monkeypatch.setattr(envmod.socket, "gethostname", lambda: name)
    monkeypatch.setattr(envmod.socket, "getfqdn", lambda *a: name)
    envmod._detect_cached.cache_clear()


def _no_queue_map(monkeypatch):
    real_open = open

    def fake_open(path, *a, **k):
        if str(path) == "/usr/local/etc/queue.map":
            raise OSError("not a TACC node")
        return real_open(path, *a, **k)

    monkeypatch.setattr("builtins.open", fake_open)


def _fake_repo(monkeypatch, root: Path | None) -> None:
    monkeypatch.setattr(envmod, "repo_root", lambda: root)
    envmod._detect_cached.cache_clear()


# ------------------------------------------------------------------- machine
def test_local_default(monkeypatch, tmp_path):
    _set_host(monkeypatch, "my-laptop.local")
    _no_queue_map(monkeypatch)
    monkeypatch.delenv("PARTICLE_ACCEL_DATA", raising=False)
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "out"))
    _fake_repo(monkeypatch, tmp_path / "checkout")
    e = detect_environment()
    assert isinstance(e, Environment)
    assert e.machine == "local" and not e.is_hpc
    assert not e.in_job and not e.is_login_node and e.job_id is None
    assert e.n_cpus >= 1 and e.max_workers() == e.n_cpus
    assert e.data_roots == (tmp_path / "checkout" / "data",) and e.data_root == tmp_path / "checkout" / "data"
    assert e.output_root == tmp_path / "out"
    monkeypatch.delenv("PARTICLE_ACCEL_OUTPUT")
    assert detect_environment().output_root == tmp_path / "checkout" / "outputs"
    assert "machine" in e.describe() and "cpus usable" in e.describe()


def test_repo_root_is_verified(monkeypatch, tmp_path):
    real = repo_root()
    assert real is not None and (real / "pyproject.toml").is_file() and (real / "src" / "shearpic").is_dir()

    # a non-editable install: .../site-packages/shearpic/env.py -> parents[2] is lib/python3.x
    site = tmp_path / "venv" / "lib" / "python3.12" / "site-packages" / "shearpic"
    site.mkdir(parents=True)
    monkeypatch.setattr(envmod, "__file__", str(site / "env.py"))
    assert repo_root() is None

    checkout = tmp_path / "checkout"
    (checkout / "src" / "shearpic").mkdir(parents=True)
    monkeypatch.setattr(envmod, "__file__", str(checkout / "src" / "shearpic" / "env.py"))
    (checkout / "pyproject.toml").write_text('[project]\nname = "someotherpackage"\n')
    assert repo_root() is None
    (checkout / "pyproject.toml").write_text('[project]\nname = "shearpic"\nversion = "0.1.0"\n')
    assert repo_root() == checkout.resolve()
    (checkout / "src" / "shearpic").rmdir()
    assert repo_root() is None


def test_no_checkout_falls_back_to_cwd_with_note(monkeypatch, tmp_path):
    _set_host(monkeypatch, "my-laptop.local")
    _no_queue_map(monkeypatch)
    for key in ("PARTICLE_ACCEL_DATA", "PARTICLE_ACCEL_OUTPUT"):
        monkeypatch.delenv(key, raising=False)
    _fake_repo(monkeypatch, None)
    monkeypatch.chdir(tmp_path)
    e = detect_environment()
    assert e.data_roots == (tmp_path / "data",) and e.output_root == tmp_path / "outputs"
    assert any("no shearpic repository checkout" in n and "PARTICLE_ACCEL_DATA" in n for n in e.notes)
    # explicit settings need no note
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(tmp_path / "d"))
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "o"))
    assert not any("checkout" in n for n in detect_environment().notes)


def test_machine_override(monkeypatch):
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "Frontera")
    assert detect_environment().machine == "frontera"


def test_tacc_system(monkeypatch):
    _set_host(monkeypatch, "c123-456")
    monkeypatch.setenv("TACC_SYSTEM", "stampede3")
    assert detect_environment().machine == "stampede3"


def test_hostname_stampede3(monkeypatch):
    _set_host(monkeypatch, "c301-001.stampede3.tacc.utexas.edu")
    assert detect_environment().machine == "stampede3"


def test_other_slurm_cluster(monkeypatch):
    _set_host(monkeypatch, "gl3001.arc-ts.umich.edu")
    _no_queue_map(monkeypatch)
    monkeypatch.setenv("SLURM_CLUSTER_NAME", "greatlakes")
    monkeypatch.setenv("SLURM_JOB_ID", "123456")
    monkeypatch.setenv("SLURM_JOB_PARTITION", "standard")
    monkeypatch.setenv("SLURM_JOB_NUM_NODES", "2")
    e = detect_environment()
    assert e.machine == "slurm" and e.is_hpc
    assert e.in_job and e.job_id == "123456" and e.partition == "standard" and e.n_nodes == 2
    assert not e.is_login_node


def test_slurm_login_node_detected_via_sbatch(monkeypatch):
    # login nodes of other SLURM clusters have no SLURM_* variables and names like gl-login1
    _set_host(monkeypatch, "gl-login1.arc-ts.umich.edu")
    _no_queue_map(monkeypatch)
    e = detect_environment()
    assert e.machine == "local" and not e.is_login_node  # no sbatch: an ordinary computer
    monkeypatch.setattr(envmod, "_have_sbatch", lambda: True)
    envmod._detect_cached.cache_clear()
    e = detect_environment()
    assert e.machine == "slurm" and e.is_login_node and e.max_workers() == 1


@pytest.mark.parametrize("host, login", [
    ("login1", True), ("login2.stampede3.tacc.utexas.edu", True), ("gl-login1.arc-ts.umich.edu", True),
    ("LOGIN3", True), ("frontera_login4", True), ("login", True),
    ("c449-001", False), ("c449-001.stampede3.tacc.utexas.edu", False), ("loginnode7", False),
    ("mylogin1", False), ("blogin-x", False),
])
def test_login_host_pattern(host, login):
    assert envmod._is_login_host(host) is login


def test_login_node(monkeypatch):
    _set_host(monkeypatch, "login1.stampede3.tacc.utexas.edu")
    e = detect_environment()
    assert e.machine == "stampede3"
    assert e.is_login_node and not e.in_job
    assert e.max_workers() == 1
    assert any("idev" in n for n in e.notes)
    with pytest.warns(UserWarning, match="login node"):
        envmod.warn_if_login_node("a test")


def test_stampede3_compute_node_is_not_login(monkeypatch):
    _set_host(monkeypatch, "c449-001.stampede3.tacc.utexas.edu")
    monkeypatch.setenv("SLURM_JOB_ID", "7")
    e = detect_environment()
    assert e.machine == "stampede3" and e.in_job and not e.is_login_node
    monkeypatch.delenv("SLURM_JOB_ID")
    assert not detect_environment().is_login_node  # the name has no login component


def test_compute_node_in_job_is_not_login(monkeypatch):
    _set_host(monkeypatch, "login1.stampede3.tacc.utexas.edu")  # e.g. idev keeps a login-like name
    monkeypatch.setenv("SLURM_JOB_ID", "42")
    e = detect_environment()
    assert e.in_job and not e.is_login_node and e.max_workers() == e.n_cpus


def test_local_login_named_host_is_not_login_node(monkeypatch):
    _set_host(monkeypatch, "login-laptop")
    _no_queue_map(monkeypatch)
    assert not detect_environment().is_login_node


def test_detection_cache_follows_env_vars(monkeypatch):
    first = detect_environment()
    assert detect_environment() is first
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "stampede3")
    second = detect_environment()
    assert second is not first and second.machine == "stampede3"


def test_public_names_exported():
    assert "warn_if_login_node" in envmod.__all__ and "run_output_dir" in envmod.__all__
    for name in envmod.__all__:
        assert hasattr(envmod, name)


# ---------------------------------------------------------------------- cpus
def test_slurm_cpus_on_node_caps_cpus(monkeypatch):
    uncapped = detect_environment().n_cpus
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "1")
    assert detect_environment().n_cpus == 1
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "100000")
    assert detect_environment().n_cpus == uncapped  # a cap never raises the count
    monkeypatch.setenv("PYTHON_CPU_COUNT", "3")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "2")
    assert detect_environment().n_cpus == 2
    monkeypatch.delenv("SLURM_CPUS_ON_NODE")
    assert detect_environment().n_cpus == 3


# -------------------------------------------------------------------- memory
def test_memory_detected():
    mem = detect_environment().mem_bytes
    assert mem is None or mem > 2**26


@pytest.fixture
def fake_mem(monkeypatch, tmp_path):
    """Point the memory probes at fake /proc and /sys/fs/cgroup trees; no sysctl."""
    proc = tmp_path / "proc_self_cgroup"
    cg = tmp_path / "cgroup"
    cg.mkdir()
    meminfo = tmp_path / "meminfo"
    monkeypatch.setattr(envmod, "_PROC_CGROUP", proc)
    monkeypatch.setattr(envmod, "_CGROUP_ROOT", cg)
    monkeypatch.setattr(envmod, "_MEMINFO", meminfo)
    monkeypatch.setattr(envmod, "_sysctl_memsize", lambda: 64 * GiB)

    def set_avail(nbytes):
        meminfo.write_text(f"MemTotal:       999999999 kB\nMemFree:  1 kB\nMemAvailable:   {nbytes // 1024} kB\n")

    return proc, cg, set_avail


def test_memory_slurm_allocation_wins(monkeypatch, fake_mem):
    proc, cg, set_avail = fake_mem
    set_avail(180 * GiB)  # a 192 GB skx node, mostly free
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "4096")  # --mem=4G
    assert envmod._memory_bytes() == 4 * GiB
    monkeypatch.delenv("SLURM_MEM_PER_NODE")
    monkeypatch.setenv("SLURM_MEM_PER_CPU", "1000")
    monkeypatch.setenv("SLURM_CPUS_ON_NODE", "3")
    assert envmod._memory_bytes() == 3000 * 2**20
    set_avail(1 * GiB)  # other jobs use the node's memory: the smaller value wins
    assert envmod._memory_bytes() == 1 * GiB
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "0")  # --mem=0: whole node, no explicit limit
    monkeypatch.delenv("SLURM_MEM_PER_CPU")
    assert envmod._memory_bytes() == 1 * GiB
    assert envmod._slurm_size_bytes("2G") == 2 * GiB and envmod._slurm_size_bytes("junk") is None


def test_memory_slurm_vars_invalidate_cache(monkeypatch, fake_mem):
    _, _, set_avail = fake_mem
    set_avail(100 * GiB)
    assert detect_environment().mem_bytes == 100 * GiB
    monkeypatch.setenv("SLURM_MEM_PER_NODE", "2048")
    assert detect_environment().mem_bytes == 2 * GiB


def test_memory_cgroup_v2_own_path(fake_mem):
    proc, cg, set_avail = fake_mem
    set_avail(50 * GiB)
    job = cg / "system.slice" / "slurmstepd.scope" / "job_99"
    task = job / "step_0" / "user" / "task_0"
    task.mkdir(parents=True)
    (task / "memory.max").write_text("max\n")
    (job / "memory.max").write_text(f"{8 * GiB}\n")
    (cg / "memory.max").write_text(f"{40 * GiB}\n")  # root/container limit is only the fallback
    proc.write_text("0::/system.slice/slurmstepd.scope/job_99/step_0/user/task_0\n")
    assert envmod._own_cgroup_limit() == 8 * GiB
    assert envmod._memory_bytes() == 8 * GiB
    (job / "memory.max").write_text("max\n")
    assert envmod._memory_bytes() == 40 * GiB  # root cgroup limit, below MemAvailable
    (cg / "memory.max").unlink()
    assert envmod._memory_bytes() == 50 * GiB  # no limit anywhere -> MemAvailable


def test_memory_cgroup_v1(fake_mem):
    proc, cg, set_avail = fake_mem
    set_avail(50 * GiB)
    d = cg / "memory" / "slurm" / "uid_1" / "job_5"
    d.mkdir(parents=True)
    (d / "memory.limit_in_bytes").write_text(f"{6 * GiB}\n")
    (cg / "memory" / "memory.limit_in_bytes").write_text("9223372036854771712\n")  # v1 "unlimited"
    proc.write_text("12:pids:/slurm/uid_1/job_5\n11:cpuset:/slurm\n4:memory:/slurm/uid_1/job_5\n")
    assert envmod._memory_bytes() == 6 * GiB
    proc.write_text("4:cpuacct,memory:/slurm/uid_1/job_5\n")  # combined controllers
    assert envmod._memory_bytes() == 6 * GiB


def test_memory_macos_fallback(fake_mem):
    # no /proc at all (fake paths do not exist): sysctl hw.memsize
    assert envmod._memory_bytes() == 64 * GiB


# -------------------------------------------------------------------- roots
def test_data_roots_from_env_multiple(monkeypatch, tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", f"{a}{os.pathsep}{b}{os.pathsep}")
    e = detect_environment()
    assert e.data_roots == (a, b) and e.data_root == a


def test_data_root_user_expansion(monkeypatch):
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", "~/sims")
    assert detect_environment().data_root == Path.home() / "sims"


def test_scratch_defaults_on_stampede3(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "stampede3")
    monkeypatch.setenv("SCRATCH", str(tmp_path / "scratch"))
    monkeypatch.delenv("PARTICLE_ACCEL_DATA", raising=False)
    monkeypatch.delenv("PARTICLE_ACCEL_OUTPUT", raising=False)
    e = detect_environment()
    assert e.data_roots == (tmp_path / "scratch" / "particle-accel" / "runs",)
    assert e.output_root == tmp_path / "scratch" / "particle-accel" / "outputs"
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(tmp_path / "mine"))
    assert detect_environment().data_roots == (tmp_path / "mine",)  # explicit setting wins


def test_hpc_without_scratch_notes(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "stampede3")
    monkeypatch.delenv("PARTICLE_ACCEL_DATA", raising=False)
    _fake_repo(monkeypatch, tmp_path / "checkout")
    e = detect_environment()
    assert e.data_roots == (tmp_path / "checkout" / "data",)
    assert any("SCRATCH" in n for n in e.notes)


# ---------------------------------------------------------------- resolve_run
@pytest.fixture
def roots(tmp_path):
    r1, r2 = tmp_path / "root1", tmp_path / "root2"
    (r1 / "run0423").mkdir(parents=True)
    (r1 / "run7").mkdir()
    (r1 / "run12.txt").touch()  # files and non-matching names are ignored
    (r2 / "run404").mkdir(parents=True)
    (r2 / "run0012").mkdir()
    return r1, r2


@pytest.mark.parametrize("spec", [423, "423", "run423", "run0423", "run000423"])
def test_resolve_run_padding_variants(roots, spec):
    assert resolve_run(spec, roots=list(roots)) == roots[0] / "run0423"


def test_resolve_run_other_roots_and_env(monkeypatch, roots):
    r1, r2 = roots
    assert resolve_run(7, roots=[r1, r2]) == r1 / "run7"
    assert resolve_run("run0007", roots=r1) == r1 / "run7"
    assert resolve_run(404, roots=[r1, r2]) == r2 / "run404"
    assert resolve_run(12, roots=f"{r1}{os.pathsep}{r2}") == r2 / "run0012"
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", f"{r1}{os.pathsep}{r2}")
    assert resolve_run("run404") == r2 / "run404"


def test_resolve_run_ambiguous_padding_raises(roots):
    r1, r2 = roots
    (r2 / "run423").mkdir()  # run0423 in root1 and run423 in root2
    with pytest.raises(ValueError, match="ambiguous") as err:
        resolve_run(423, roots=[r1, r2])
    assert str(r1 / "run0423") in str(err.value) and str(r2 / "run423") in str(err.value)
    (r1 / "run07").mkdir()  # run7 and run07 in the same root
    with pytest.raises(ValueError, match="2 directories"):
        resolve_run("run7", roots=[r1])
    # the same root listed twice is not ambiguous
    assert resolve_run(404, roots=[r2, r2]) == r2 / "run404"


def test_resolve_run_paths(roots, tmp_path):
    d = roots[0] / "run7"
    assert resolve_run(d) == d
    assert resolve_run(str(d)) == d
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_run(tmp_path / "nowhere")


def test_resolve_run_relative_names_and_forward_slashes(monkeypatch, tmp_path, roots):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "mytest").mkdir()
    (tmp_path / "sub" / "deeper").mkdir(parents=True)
    assert resolve_run("mytest") == Path("mytest")                  # bare directory name in the cwd
    assert resolve_run("sub/deeper") == Path("sub/deeper")          # forward slash (also on Windows)
    assert resolve_run("./mytest") == Path("mytest")
    with pytest.raises(FileNotFoundError, match="does not exist"):
        resolve_run("sub/missing")
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    assert resolve_run("~/mytest") == tmp_path / "mytest"
    # a directory named like a run id in the cwd does not shadow the data roots
    (tmp_path / "run0423").mkdir()
    assert resolve_run("run0423", roots=list(roots)) == roots[0] / "run0423"
    assert envmod._looks_like_path("a/b") and not envmod._looks_like_path("423")
    monkeypatch.setattr(envmod.os, "altsep", "\\")
    assert envmod._looks_like_path("a\\b")


def test_resolve_run_not_found_lists_roots(roots, tmp_path):
    missing_root = tmp_path / "missing"
    with pytest.raises(FileNotFoundError) as err:
        resolve_run(999, roots=[*roots, missing_root])
    msg = str(err.value)
    assert "999" in msg and str(roots[0]) in msg and str(roots[1]) in msg and str(missing_root) in msg
    assert "PARTICLE_ACCEL_DATA" in msg
    with pytest.raises(ValueError, match="cannot interpret"):
        resolve_run("simulation-423", roots=list(roots))


# ---------------------------------------------------------------- output_dir
def test_output_dir_creation(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "out"))
    p = output_dir("run0423", "spectra", 5)
    assert p == tmp_path / "out" / "run0423" / "spectra" / "5" and p.is_dir()
    assert output_dir("run0423", "spectra", 5) == p  # idempotent
    q = output_dir("later", create=False)
    assert q == tmp_path / "out" / "later" and not q.exists()


def test_run_output_dir_normalises_run_names(monkeypatch, tmp_path):
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "out"))
    expected = tmp_path / "out" / "run0423" / "spectra"
    for spec in (423, "423", "run423", "run0423", "run000423", Path("/data/results/run423"),
                 "/scratch/runs/run0423/", tmp_path / "run423"):
        assert run_output_dir(spec, "spectra", create=False) == expected, spec
    p = run_output_dir("run423", "spectra")
    assert p == expected and p.is_dir()
    assert run_output_dir(Path("/x/my_test"), create=False) == tmp_path / "out" / "my_test"
    assert run_output_dir("myrun423", create=False) == tmp_path / "out" / "myrun423"  # anchored match only
    assert run_output_dir(12345, create=False).name == "run12345"
    with pytest.raises(ValueError):
        run_output_dir(True)
