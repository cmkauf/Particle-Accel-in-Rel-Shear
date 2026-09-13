"""Unit tests for shearpic.parallel.parallel_map and its worker-count logic."""

from __future__ import annotations

import dataclasses
import math
import os
import pickle
import sys
import threading
import time
import types
import warnings

import pytest

from shearpic import parallel as par
from shearpic.env import Environment, detect_environment
from shearpic.parallel import (TaskError, in_parallel_worker, mpi_world_size, parallel_map, parse_bytes,
                               resolve_workers)

BACKENDS = ["serial", "thread", "process"]

# loky's fork() DeprecationWarning is filtered in parallel_map and in pyproject.toml


def square(x):
    return x * x


def fail_on_three(x):
    if x == 3:
        raise ValueError(f"bad item {x}")
    return x + 100


def worker_env(_):
    import numpy as np  # noqa: F401  (loads the BLAS library inside the worker)
    from threadpoolctl import threadpool_info

    blas_threads = [info["num_threads"] for info in threadpool_info()]
    return os.environ.get("OMP_NUM_THREADS"), os.environ.get("MPLBACKEND"), os.getpid(), blas_threads


def nested_probe(_):
    """Runs in an outer worker: what does a nested parallel_map decide?"""
    inner_pids = parallel_map(lambda _: os.getpid(), range(4))  # backend='auto', n_workers='auto'
    return {
        "in_worker": in_parallel_worker(),
        "auto_workers": resolve_workers(100),
        "auto_backend": par._choose_backend("auto", resolve_workers(100), 100),
        "explicit_workers": resolve_workers(100, n_workers=3),
        "explicit_backend": par._choose_backend("thread", 3, 100),
        "inner_ran_here": set(inner_pids) == {os.getpid()},
        "inner_threads_ok": parallel_map(square, range(6), backend="thread", n_workers=2) == [i * i for i in range(6)],
    }


def sleep_or_fail(x):
    if x == 0:
        raise ValueError("first item fails at once")
    time.sleep(2.0)
    return x


class NeedsTwoArgs(Exception):
    """Pickles, but cannot be unpickled: BaseException.__reduce__ calls cls(*self.args) with one arg."""

    def __init__(self, a, b):
        super().__init__(f"{a} and {b}")


class HoldsLock(Exception):
    """Cannot be pickled at all (a lock in the instance __dict__)."""

    def __init__(self, msg):
        super().__init__(msg)
        self.lock = threading.Lock()


def raise_custom(x):
    if x == 1:
        raise NeedsTwoArgs("left", "right")
    if x == 2:
        raise HoldsLock("has a lock")
    if x == 3:
        raise ValueError("plain")
    return x


def fake_env(**changes) -> Environment:
    base = Environment(machine="local", hostname="test", in_job=False, is_login_node=False, job_id=None,
                       partition=None, n_nodes=1, n_cpus=8, mem_bytes=10 * 2**30,
                       data_roots=(detect_environment().data_root,), output_root=detect_environment().output_root)
    return dataclasses.replace(base, **changes)


# --------------------------------------------------------------------- results
@pytest.mark.parametrize("backend", BACKENDS)
def test_backends_ordered_identical_results(backend):
    items = list(range(23))
    offset = 7
    closure = lambda x: (x - offset) ** 3  # noqa: E731  (notebook-style lambda capturing a variable)
    assert parallel_map(square, items, backend=backend, n_workers=2) == [x * x for x in items]
    assert parallel_map(closure, items, backend=backend, n_workers=2) == [(x - offset) ** 3 for x in items]
    assert parallel_map(lambda s: s.upper(), iter(["a", "b", "c"]), backend=backend, n_workers=2) == ["A", "B", "C"]


def test_all_backends_agree_on_non_trivial_work():
    def work(n):
        return sum(math.sin(k * n) for k in range(200))

    items = [0.1 * i for i in range(40)]
    results = [parallel_map(work, items, backend=b, n_workers=3) for b in BACKENDS]
    assert results[0] == results[1] == results[2]


def test_empty_items():
    for b in BACKENDS:
        assert parallel_map(square, [], backend=b) == []


def test_auto_backend_serial_for_one_worker():
    assert parallel_map(square, [1, 2, 3], n_workers=1) == [1, 4, 9]
    assert parallel_map(square, [5]) == [25]


def test_unknown_backend_and_on_error():
    with pytest.raises(ValueError, match="unknown backend"):
        parallel_map(square, [1, 2], backend="dask", n_workers=2)
    with pytest.raises(ValueError, match="on_error"):
        parallel_map(square, [1, 2], on_error="ignore")


# ---------------------------------------------------------------------- errors
@pytest.mark.parametrize("backend", BACKENDS)
def test_on_error_raise(backend):
    with pytest.raises(RuntimeError, match=r"item 3 \(3\) failed") as err:
        parallel_map(fail_on_three, range(6), backend=backend, n_workers=2)
    assert "bad item 3" in str(err.value)


@pytest.mark.parametrize("backend", BACKENDS)
def test_on_error_collect(backend):
    out = parallel_map(fail_on_three, range(6), backend=backend, n_workers=2, on_error="collect")
    assert out[:3] == [100, 101, 102] and out[4:] == [104, 105]
    err = out[3]
    assert isinstance(err, TaskError) and not err
    assert err.index == 3 and err.item == "3"
    assert isinstance(err.error, ValueError) and "bad item 3" in str(err.error)
    assert "ValueError" in err.traceback
    assert [r for r in out if r] == [100, 101, 102, 104, 105]


# -------------------------------------------------------------------- chunking
@pytest.mark.parametrize("backend", ["thread", "process"])
def test_chunking_many_small_items(backend):
    n = 5000  # > 64 * workers -> automatic batching
    out = parallel_map(square, range(n), backend=backend, n_workers=2)
    assert out == [i * i for i in range(n)]
    out = parallel_map(square, range(101), backend=backend, n_workers=2, chunksize=7)
    assert out == [i * i for i in range(101)]


def test_chunked_errors_collected_per_item():
    out = parallel_map(fail_on_three, range(300), backend="thread", n_workers=2, chunksize=50, on_error="collect")
    assert isinstance(out[3], TaskError) and out[4] == 104 and len(out) == 300


def test_progress_bar_runs(capsys):
    assert parallel_map(square, range(5), backend="thread", n_workers=2, progress=True, desc="sq") == [0, 1, 4, 9, 16]
    assert parallel_map(square, range(5), backend="serial", progress=True) == [0, 1, 4, 9, 16]


# ------------------------------------------------------------- process workers
def test_process_workers_single_threaded_blas(monkeypatch):
    monkeypatch.setenv("OMP_NUM_THREADS", "8")  # the parent's setting must not leak into workers
    out = parallel_map(worker_env, range(8), backend="process", n_workers=2)
    assert all(omp == "1" for omp, _, _, _ in out)
    assert all(mpl == "Agg" for _, mpl, _, _ in out)
    assert all(pid != os.getpid() for _, _, pid, _ in out)
    assert all(n == 1 for *_, threads in out for n in threads)


# -------------------------------------------------------------- worker counts
def test_parse_bytes():
    assert parse_bytes(None) is None
    assert parse_bytes(1000) == 1000
    assert parse_bytes(2.5e9) == 2_500_000_000
    assert parse_bytes("2GB") == 2 * 2**30
    assert parse_bytes("512MiB") == 512 * 2**20
    assert parse_bytes("1.5g") == int(1.5 * 2**30)
    assert parse_bytes(" 4 kB ") == 4 * 2**10
    assert parse_bytes("123") == 123
    assert parse_bytes("1T") == 2**40
    with pytest.raises(ValueError):
        parse_bytes("lots")


def test_mem_per_task_caps_workers(monkeypatch):
    monkeypatch.setattr(par, "detect_environment", lambda: fake_env(n_cpus=8, mem_bytes=10 * 2**30))
    assert resolve_workers(100) == 8
    assert resolve_workers(100, mem_per_task="4GB") == 2        # 0.8 * 10 GiB // 4 GiB
    assert resolve_workers(100, mem_per_task="100GB") == 1      # never below one
    assert resolve_workers(3, mem_per_task="1GB") == 3          # never more than the items
    assert resolve_workers(100, n_workers=16, mem_per_task="2GB") == 4
    assert resolve_workers(0) == 1
    monkeypatch.setattr(par, "detect_environment", lambda: fake_env(n_cpus=8, mem_bytes=None))
    assert resolve_workers(100, mem_per_task="100GB") == 8      # unknown memory: no cap


def test_login_node_forces_one_worker(monkeypatch):
    monkeypatch.setattr(par, "detect_environment", lambda: fake_env(machine="stampede3", is_login_node=True))
    assert resolve_workers(50) == 1
    # 'auto' then runs serially in this process
    out = parallel_map(lambda _: os.getpid(), range(4))
    assert out == [os.getpid()] * 4
    with pytest.warns(UserWarning, match="login node"):
        resolve_workers(50, n_workers=4)


def test_mpi_world_size_from_env(monkeypatch):
    assert mpi_world_size() == 1
    monkeypatch.setenv("OMPI_COMM_WORLD_SIZE", "8")
    assert mpi_world_size() == 8
    monkeypatch.setenv("PMI_SIZE", "4")  # checked first (Intel MPI / ibrun)
    assert mpi_world_size() == 4
    monkeypatch.delenv("PMI_SIZE")
    monkeypatch.delenv("OMPI_COMM_WORLD_SIZE")
    monkeypatch.setenv("PMIX_SIZE", "2")
    assert mpi_world_size() == 2
    monkeypatch.setenv("PMIX_SIZE", "not-a-number")
    assert mpi_world_size() == 1


def test_auto_selects_mpi_under_launcher(monkeypatch):
    try:
        import mpi4py  # noqa: F401
    except ImportError:
        monkeypatch.setenv("PMI_SIZE", "2")
        with pytest.raises(ImportError, match="mpi4py"):
            parallel_map(square, [1, 2, 3])
    else:  # pragma: no cover - depends on the machine
        pytest.skip("mpi4py installed; launching real MPI is out of scope for unit tests")


# ------------------------------------------------------------ nested parallel_map
@pytest.mark.parametrize("outer", ["thread", "process"])
def test_nested_parallel_map_runs_serially_inside_workers(outer):
    assert not in_parallel_worker()
    out = parallel_map(nested_probe, range(2), backend=outer, n_workers=2)
    for probe in out:
        assert probe["in_worker"]
        assert probe["auto_workers"] == 1 and probe["auto_backend"] == "serial"
        assert probe["inner_ran_here"]  # the nested 'auto' map did not start processes
        # explicit arguments are still honoured
        assert probe["explicit_workers"] == 3 and probe["explicit_backend"] == "thread"
        assert probe["inner_threads_ok"]
    assert not in_parallel_worker()  # the flag does not leak into the caller


def test_serial_map_does_not_mark_worker(monkeypatch):
    monkeypatch.setattr(par, "detect_environment", lambda: fake_env(n_cpus=8))
    # a serial outer loop runs in the calling process: nested maps may still use the CPUs
    assert parallel_map(lambda _: (in_parallel_worker(), resolve_workers(100)), range(2), backend="serial") == [
        (False, 8), (False, 8)]
    monkeypatch.setenv(par.WORKER_ENV_VAR, "1")  # what loky workers see
    assert in_parallel_worker() and resolve_workers(100) == 1 and par._choose_backend("auto", 8, 100) == "serial"
    assert par._THREAD_ENV[par.WORKER_ENV_VAR] == "1"


# ------------------------------------------------------- process pool life cycle
def _time_fast_map():
    t0 = time.perf_counter()
    assert parallel_map(square, range(4), backend="process", n_workers=2) == [0, 1, 4, 9]
    return time.perf_counter() - t0


def test_failed_process_map_does_not_block_the_next_call():
    parallel_map(square, range(4), backend="process", n_workers=2)  # warm pool
    t0 = time.perf_counter()
    with pytest.raises(RuntimeError, match="first item fails"):
        parallel_map(sleep_or_fail, range(12), backend="process", n_workers=2)
    assert time.perf_counter() - t0 < 5.0
    # the queued 2 s tasks of the failed map must not delay this one
    assert _time_fast_map() < 5.0


def test_interrupted_process_map_does_not_block_the_next_call(monkeypatch):
    parallel_map(square, range(4), backend="process", n_workers=2)

    def interrupted(futures):
        time.sleep(0.2)  # let the workers pick up (uncancellable) tasks
        raise KeyboardInterrupt

    monkeypatch.setattr(par, "as_completed", interrupted)
    with pytest.raises(KeyboardInterrupt):
        parallel_map(sleep_or_fail, range(1, 13), backend="process", n_workers=2)
    monkeypatch.undo()
    assert _time_fast_map() < 5.0


def test_process_pool_size_independent_of_item_count(monkeypatch):
    import joblib.externals.loky as loky

    monkeypatch.setattr(par, "detect_environment", lambda: fake_env(n_cpus=3, mem_bytes=None))
    sizes = []
    real = loky.get_reusable_executor

    def recording(*args, **kwargs):
        sizes.append(kwargs["max_workers"])
        return real(*args, **kwargs)

    monkeypatch.setattr(loky, "get_reusable_executor", recording)
    for n in (2, 3, 7):
        assert parallel_map(square, range(n), backend="process") == [i * i for i in range(n)]
    assert sizes == [3, 3, 3]  # a changing item count does not resize the pool
    sizes.clear()
    assert parallel_map(square, range(7), backend="process", mem_per_task=1, n_workers=2) == [i * i for i in range(7)]
    assert sizes == [2]


# -------------------------------------------------------- unpicklable exceptions
def test_portable_error():
    err = ValueError("fine")
    assert par._portable_error(err) is err
    for bad, text in ((NeedsTwoArgs("left", "right"), "NeedsTwoArgs: left and right"),
                      (HoldsLock("has a lock"), "HoldsLock: has a lock")):
        sub = par._portable_error(bad)
        assert type(sub) is RuntimeError and str(sub) == text
        pickle.loads(pickle.dumps(sub))


@pytest.mark.parametrize("backend", BACKENDS)
def test_collect_unpicklable_exceptions(backend):
    out = parallel_map(raise_custom, range(5), backend=backend, n_workers=2, on_error="collect")
    assert out[0] == 0 and out[4] == 4
    assert all(isinstance(out[i], TaskError) for i in (1, 2, 3))
    assert "NeedsTwoArgs" in out[1].traceback and "left and right" in out[1].traceback
    assert "HoldsLock" in out[2].traceback
    assert isinstance(out[3].error, ValueError)  # picklable exceptions are kept as they are
    if backend == "process":
        assert type(out[1].error) is RuntimeError and str(out[1].error) == "NeedsTwoArgs: left and right"
        assert type(out[2].error) is RuntimeError and str(out[2].error) == "HoldsLock: has a lock"
    else:  # same process: the original objects are kept
        assert isinstance(out[1].error, NeedsTwoArgs) and isinstance(out[2].error, HoldsLock)


# ------------------------------------------------------------------ fork warning
def test_process_backend_silences_loky_fork_warning():
    from joblib.externals.loky import get_reusable_executor

    # force loky to fork fresh workers while this process is certainly multi-threaded
    get_reusable_executor(max_workers=2).shutdown(wait=True, kill_workers=True)
    release = threading.Event()
    helper = threading.Thread(target=release.wait, daemon=True)
    helper.start()
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            assert parallel_map(square, range(6), backend="process", n_workers=2) == [i * i for i in range(6)]
    finally:
        release.set()
    assert not [w for w in caught if "multi-threaded" in str(w.message)]


# ----------------------------------------------------------- MPI with a fake mpi4py
class FakeCommWorld:
    """``MPI.COMM_WORLD`` for ranks simulated by threads.

    ``gather`` pickles the payload (like mpi4py) and waits on a barrier: if one rank never
    reaches the collective, the others fail with BrokenBarrierError instead of hanging.
    """

    def __init__(self, size: int, timeout: float = 5.0):
        self.size = size
        self._rank = threading.local()
        self._barrier = threading.Barrier(size, timeout=timeout)
        self._slots = [None] * size
        self.gather_calls = 0

    def Get_rank(self):  # noqa: N802  (mpi4py spelling)
        return self._rank.value

    def Get_size(self):  # noqa: N802
        return self.size

    def gather(self, obj, root=0):
        rank = self._rank.value
        self._slots[rank] = pickle.loads(pickle.dumps(obj))
        self.gather_calls += 1
        self._barrier.wait()
        return list(self._slots) if rank == root else None


def run_fake_mpi(monkeypatch, size, call):
    """Run ``call()`` once per simulated rank; returns [('ok', value) | ('error', exc)] by rank."""
    world = FakeCommWorld(size)
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
    return outcomes, world


def test_mpi_results_gathered_on_rank_zero(monkeypatch):
    outcomes, world = run_fake_mpi(monkeypatch, 3, lambda: parallel_map(square, range(10), backend="mpi"))
    assert outcomes == [("ok", [i * i for i in range(10)]), ("ok", None), ("ok", None)]
    assert world.gather_calls == 3


@pytest.mark.parametrize("bad", [3, 4])  # a failure on rank 0 or on rank 1
def test_mpi_on_error_raise_reaches_gather_on_every_rank(monkeypatch, bad):
    def fail(x):
        if x == bad:
            raise ValueError(f"bad item {x}")
        return x

    outcomes, world = run_fake_mpi(monkeypatch, 3, lambda: parallel_map(fail, range(9), backend="mpi"))
    status, err = outcomes[0]
    assert status == "error" and isinstance(err, RuntimeError)  # rank 0 raises after the gather
    assert f"item {bad} ({bad}) failed" in str(err) and f"bad item {bad}" in str(err)
    assert outcomes[1:] == [("ok", None), ("ok", None)]  # no rank is left waiting in the gather
    assert world.gather_calls == 3


def test_mpi_raise_reports_lowest_failing_index(monkeypatch):
    def fail_many(x):
        if x in (5, 7):  # rank 1 (1, 4, 7) and rank 2 (2, 5, 8)
            raise KeyError(x)
        return x

    outcomes, _ = run_fake_mpi(monkeypatch, 3, lambda: parallel_map(fail_many, range(9), backend="mpi"))
    assert outcomes[0][0] == "error" and "item 5 (5) failed" in str(outcomes[0][1])


def test_mpi_collect_with_unpicklable_error(monkeypatch):
    outcomes, _ = run_fake_mpi(monkeypatch, 2, lambda: parallel_map(raise_custom, range(5), backend="mpi",
                                                                      on_error="collect"))
    status, out = outcomes[0]
    assert status == "ok" and outcomes[1] == ("ok", None)
    assert out[0] == 0 and out[4] == 4
    assert str(out[1].error) == "NeedsTwoArgs: left and right" and str(out[2].error) == "HoldsLock: has a lock"
    assert isinstance(out[3].error, ValueError)


def test_nested_map_inside_mpi_rank_is_serial(monkeypatch):
    monkeypatch.setenv("PMI_SIZE", "2")  # 'auto' selects MPI at the top level only

    def rank_task(x):
        inner = parallel_map(square, range(x + 2))  # would call gather again if it chose MPI
        return x, inner, in_parallel_worker(), resolve_workers(100)

    outcomes, world = run_fake_mpi(monkeypatch, 2, lambda: parallel_map(rank_task, range(4)))
    status, out = outcomes[0]
    assert status == "ok" and outcomes[1] == ("ok", None)
    assert [r[0] for r in out] == [0, 1, 2, 3]
    assert all(r[1] == [i * i for i in range(r[0] + 2)] and r[2] and r[3] == 1 for r in out)
    assert world.gather_calls == 2


def sleep_then_mark(arg):
    i, directory = arg
    if i == 0:
        raise ValueError("first item fails at once")
    time.sleep(0.5)
    with open(os.path.join(directory, f"marker{i}"), "w"):
        pass
    return i


def test_failed_process_map_kills_queued_tasks(tmp_path):
    """After the first failure no queued task may keep running in the old workers."""
    parallel_map(square, range(4), backend="process", n_workers=2)  # warm pool
    with pytest.raises(RuntimeError, match="first item fails"):
        parallel_map(sleep_then_mark, [(i, str(tmp_path)) for i in range(40)], backend="process", n_workers=2,
                     chunksize=1)
    time.sleep(3.0)
    assert len(list(tmp_path.glob("marker*"))) <= 2  # at most the tasks already running when killed
