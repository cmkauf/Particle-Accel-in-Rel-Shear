"""``parallel_map``: embarrassingly parallel analysis on a laptop, a compute node or over MPI::

    results = parallel_map(analyse_snapshot, snapshot_paths, mem_per_task="2GB")

Worker functions should take paths and a RunConfig and return small reduced results.
"""

from __future__ import annotations

import contextlib
import contextvars
import math
import os
import pickle
import re
import sys
import traceback
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Sequence

from .env import detect_environment

__all__ = ["parallel_map", "TaskError", "resolve_workers", "mpi_world_size", "parse_bytes",
           "in_parallel_worker"]

#: Set to "1" in loky worker processes to mark nested calls.
WORKER_ENV_VAR = "SHEARPIC_IN_WORKER"

_THREAD_ENV = {
    "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1", "VECLIB_MAXIMUM_THREADS": "1", "MPLBACKEND": "Agg",
    WORKER_ENV_VAR: "1",
}

# True while a worker runs a task; a ContextVar keeps thread workers independent
_IN_WORKER: contextvars.ContextVar[bool] = contextvars.ContextVar("shearpic_in_worker", default=False)

# Python >= 3.12 warns about fork() in a threaded parent, which is harmless for loky because
# its children exec a fresh interpreter immediately
_FORK_WARNING = r"This process .* is multi-threaded, use of fork\(\) may lead to deadlocks"


@dataclass
class TaskError:
    """Falsy placeholder result for a failed item with ``on_error='collect'``.

    With the process and MPI backends an exception that cannot be pickled is replaced by
    ``RuntimeError('<Type>: <message>')``; ``traceback`` always holds the original text.
    """

    index: int
    item: str
    error: BaseException
    traceback: str

    def __bool__(self) -> bool:  # so that [r for r in results if r] drops failures
        return False


def parse_bytes(value: int | float | str | None) -> int | None:
    """Bytes from a number or a size string such as '2GB' or '512MiB' in binary units; None stays None."""
    if value is None or isinstance(value, (int, float)):
        return None if value is None else int(value)
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMgGtT]?)i?[bB]?\s*", value)
    if not m:
        raise ValueError(f"cannot parse memory size {value!r}")
    scale = {"": 1, "k": 2**10, "m": 2**20, "g": 2**30, "t": 2**40}[m.group(2).lower()]
    return int(float(m.group(1)) * scale)


def mpi_world_size() -> int:
    """MPI size from launcher environment variables (no mpi4py import needed)."""
    for key in ("PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "PMIX_SIZE"):
        v = os.environ.get(key, "")
        if v.isdigit():
            return int(v)
    return 1


def in_parallel_worker() -> bool:
    """True inside a task run by a thread, process or MPI worker of :func:`parallel_map`.

    Detected from a context variable, and in loky worker processes from ``SHEARPIC_IN_WORKER=1``.
    """
    return _IN_WORKER.get() or os.environ.get(WORKER_ENV_VAR) == "1"


def resolve_workers(n_items: int, n_workers: int | str = "auto", mem_per_task: int | str | None = None) -> int:
    """Number of workers for n_items tasks.

    'auto' means all usable CPUs, but 1 on login nodes and inside another parallel_map worker;
    an integer is used as given.  Either is capped by 0.8 * memory / mem_per_task and n_items.
    """
    env = detect_environment()
    if n_workers == "auto":
        n = 1 if in_parallel_worker() else env.max_workers()
    else:
        n = int(n_workers)
        if env.is_login_node and n > 1:
            warnings.warn("more than one worker requested on a login node; use idev or sbatch", stacklevel=3)
    mem = parse_bytes(mem_per_task)
    if mem and env.mem_bytes:
        n = min(n, max(1, int(0.8 * env.mem_bytes // mem)))
    return max(1, min(n, max(1, n_items)))


def _choose_backend(backend: str, workers: int, n_items: int) -> str:
    """Resolve ``backend='auto'`` (other names are returned unchanged)."""
    if backend != "auto":
        return backend
    if in_parallel_worker():
        return "serial"  # nested call: the outer map already uses the CPUs / MPI ranks
    if mpi_world_size() > 1:
        return "mpi"
    if workers > 1 and n_items > 1:
        return "process"
    return "serial"


@contextlib.contextmanager
def _quiet_loky_fork():
    """Ignore only the DeprecationWarning about fork() in a threaded process."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=_FORK_WARNING, category=DeprecationWarning)
        yield


def _worker_init() -> None:
    try:
        from threadpoolctl import threadpool_limits

        threadpool_limits(1)
    except Exception:  # pragma: no cover
        pass


def _portable_error(err: BaseException) -> BaseException:
    """``err`` if it survives a pickle round trip, else a RuntimeError with its type and message."""
    try:
        pickle.loads(pickle.dumps(err))
        return err
    except Exception:  # noqa: BLE001
        try:
            message = str(err)
        except Exception:  # noqa: BLE001  pragma: no cover
            message = "<unprintable exception>"
        return RuntimeError(f"{type(err).__name__}: {message}")


def _run_chunk(fn: Callable, chunk: Sequence[tuple[int, Any]], collect: bool, in_worker: bool = False,
               portable: bool = False) -> list[tuple[int, Any]]:
    """Run fn on (index, item) pairs, optionally flagged as a worker and with picklable collected errors."""
    token = _IN_WORKER.set(True) if in_worker else None
    try:
        out = []
        for i, item in chunk:
            try:
                out.append((i, fn(item)))
            except Exception as err:  # noqa: BLE001
                if not collect:
                    raise RuntimeError(f"parallel_map item {i} ({_short(item)}) failed: {err!r}") from err
                tb = traceback.format_exc()
                out.append((i, TaskError(i, _short(item), _portable_error(err) if portable else err, tb)))
        return out
    finally:
        if token is not None:
            _IN_WORKER.reset(token)


def _short(item: Any, limit: int = 120) -> str:
    s = repr(item)
    return s if len(s) <= limit else s[: limit - 3] + "..."


def _progress(total: int, enabled: bool, desc: str | None):
    if not enabled:
        return None
    from tqdm.auto import tqdm

    return tqdm(total=total, desc=desc)


def parallel_map(
    fn: Callable[[Any], Any],
    items: Iterable[Any],
    *,
    backend: str = "auto",
    n_workers: int | str = "auto",
    mem_per_task: int | str | None = None,
    chunksize: int | str = "auto",
    progress: bool = False,
    desc: str | None = None,
    on_error: str = "raise",
) -> list[Any] | None:
    """Apply fn to every item and return the results in input order.

    Parameters
    ----------
    backend : 'serial', 'thread', 'process' (loky, one BLAS thread per worker), 'mpi', or 'auto', which
        picks mpi under a multi-rank launch, process for more than one worker and serial otherwise.
    n_workers : number of workers or 'auto', see :func:`resolve_workers`.
    mem_per_task : peak memory of one call in bytes or as a string like '2GB'; caps the workers.
    chunksize : items per task sent to a worker; 'auto' batches very many small items.
    progress, desc : show a tqdm progress bar with this label.
    on_error : 'raise' stops at the first failure, 'collect' returns a TaskError for failed items.

    Under MPI only rank 0 gets the results and the other ranks get None.  With on_error='raise'
    each rank runs until its first failure and the results are gathered before rank 0 raises
    for the lowest failing index, so no rank is left waiting in a collective.
    """
    if on_error not in ("raise", "collect"):
        raise ValueError("on_error must be 'raise' or 'collect'")
    items = list(items)
    n = len(items)
    if n == 0:
        return []
    collect = on_error == "collect"
    # the pool size ignores n so that calls with different item counts reuse the loky executor
    pool = resolve_workers(sys.maxsize, n_workers, mem_per_task)
    workers = min(pool, n)

    backend = _choose_backend(backend, workers, n)
    if backend == "mpi":
        return _mpi_map(fn, items, on_error, progress, desc)
    if backend == "serial" or workers == 1:
        results: list[Any] = [None] * n
        bar = _progress(n, progress, desc)
        for i, item in enumerate(items):
            (_, res), = _run_chunk(fn, [(i, item)], collect)
            results[i] = res
            if bar is not None:
                bar.update()
        if bar is not None:
            bar.close()
        return results

    if chunksize == "auto":
        chunksize = 1 if n <= 64 * workers else math.ceil(n / (16 * workers))
    indexed = list(enumerate(items))
    chunks = [indexed[k:k + int(chunksize)] for k in range(0, n, int(chunksize))]

    if backend not in ("thread", "process"):
        raise ValueError(f"unknown backend {backend!r}")

    results = [None] * n
    bar = _progress(n, progress, desc)
    futures = []
    executor = None
    # catch_warnings changes global state, so only the process backend, where loky forks, uses it
    with _quiet_loky_fork() if backend == "process" else contextlib.nullcontext():
        try:
            if backend == "thread":
                executor = ThreadPoolExecutor(max_workers=workers)
            else:
                from joblib.externals.loky import get_reusable_executor

                # do not wait for queued tasks of an executor left shut down by a previous call
                executor = get_reusable_executor(max_workers=pool, initializer=_worker_init, env=_THREAD_ENV,
                                                 kill_workers=True)
            portable = backend == "process"
            futures = [executor.submit(_run_chunk, fn, chunk, collect, True, portable) for chunk in chunks]
            for fut in as_completed(futures):
                done = fut.result()
                for i, res in done:
                    results[i] = res
                if bar is not None:
                    bar.update(len(done))
        except BaseException:
            if backend == "process" and executor is not None:
                # Queued loky tasks cannot be cancelled, so the workers are killed.  Cancelling
                # the futures first would crash loky's manager thread before it kills them.
                executor.shutdown(wait=False, kill_workers=True)
            else:
                for f in futures:
                    f.cancel()
            raise
        finally:
            if bar is not None:
                bar.close()
            if backend == "thread" and executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
    return results


def _mpi_map(fn, items, on_error, progress, desc):
    """Run one shard of items per MPI rank and gather the results on rank 0.

    Failures are collected on every rank so that all ranks reach comm.gather; with
    on_error='raise', rank 0 raises after the gather.
    """
    try:
        from mpi4py import MPI
    except ImportError as err:  # pragma: no cover - depends on the machine
        raise ImportError("backend='mpi' needs mpi4py (pip install 'shearpic[mpi]')") from err
    comm = MPI.COMM_WORLD
    rank, size = comm.Get_rank(), comm.Get_size()
    raise_errors = on_error == "raise"
    mine = [(i, items[i]) for i in range(rank, len(items), size)]
    local = []
    for k, pair in enumerate(mine):
        done = _run_chunk(fn, [pair], collect=True, in_worker=True, portable=True)
        local.extend(done)
        if progress and rank == 0:
            print(f"[rank 0] {desc or 'parallel_map'}: {k + 1}/{len(mine)} local items", flush=True)
        if raise_errors and isinstance(done[-1][1], TaskError):
            break  # this rank's lowest failing index is known; skip the rest of its shard
    gathered = comm.gather(local, root=0)
    if rank != 0:
        return None
    results = [None] * len(items)
    for part in gathered:
        for i, res in part:
            results[i] = res
    if raise_errors:
        failed = [(i, r) for part in gathered for i, r in part if isinstance(r, TaskError)]
        if failed:
            _, first = min(failed, key=lambda pair: pair[0])
            raise RuntimeError(f"parallel_map item {first.index} ({first.item}) failed: {first.error!r}\n"
                               f"worker traceback:\n{first.traceback}") from first.error
    return results
