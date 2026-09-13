"""Machine detection, usable CPUs and memory, and the data and output roots.

Run directories are read from the data roots and never written; products go under the output
root.  The defaults are ``$SCRATCH/particle-accel/runs`` and ``.../outputs`` on a cluster, else
``data/`` and ``outputs/`` in the repository checkout or the working directory, overridden by
``PARTICLE_ACCEL_DATA`` (roots separated by ``os.pathsep``) and ``PARTICLE_ACCEL_OUTPUT``.
"""

from __future__ import annotations

import functools
import os
import re
import shutil
import socket
import subprocess
import warnings
from dataclasses import dataclass, field
from pathlib import Path

__all__ = [
    "Environment",
    "detect_environment",
    "resolve_run",
    "output_dir",
    "run_output_dir",
    "repo_root",
    "warn_if_login_node",
]

PROJECT_DIRNAME = "particle-accel"
PACKAGE_NAME = "shearpic"
#: Directory name of a run under the output root, as in the registry.
RUN_NAME_FORMAT = "run{id:04d}"

_PYPROJECT_NAME_RE = re.compile(r"""^\s*name\s*=\s*["']shearpic["']\s*(#.*)?$""", re.M)


def repo_root() -> Path | None:
    """The repository checkout containing this package, or None for an installed package.

    The directory two levels above ``src/shearpic`` is accepted only if it has that directory
    and a ``pyproject.toml`` declaring ``name = "shearpic"``.
    """
    root = Path(__file__).resolve().parents[2]
    try:
        text = (root / "pyproject.toml").read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    if not _PYPROJECT_NAME_RE.search(text) or not (root / "src" / PACKAGE_NAME).is_dir():
        return None
    return root


@dataclass(frozen=True)
class Environment:
    """Machine, resources and paths of the current process, from :func:`detect_environment`."""

    machine: str                      # 'stampede3' | 'slurm' | 'local'
    hostname: str
    in_job: bool                      # inside a SLURM allocation
    is_login_node: bool
    job_id: str | None
    partition: str | None
    n_nodes: int
    n_cpus: int                       # CPUs this process may use on this node
    mem_bytes: int | None             # memory this process may use (job/cgroup limit or available)
    data_roots: tuple[Path, ...]
    output_root: Path
    notes: tuple[str, ...] = field(default=())

    @property
    def is_hpc(self) -> bool:
        return self.machine != "local"

    @property
    def data_root(self) -> Path:
        return self.data_roots[0]

    def max_workers(self) -> int:
        """Workers allowed for heavy work: 1 on login nodes, where TACC policy forbids it, else n_cpus."""
        return 1 if self.is_login_node else self.n_cpus

    def describe(self) -> str:
        mem = f"{self.mem_bytes / 2**30:.1f} GiB" if self.mem_bytes else "unknown"
        lines = [
            f"machine      : {self.machine}",
            f"hostname     : {self.hostname}",
            f"login node   : {self.is_login_node}",
            f"SLURM job    : {self.job_id or '-'} (partition {self.partition or '-'}, {self.n_nodes} node(s))",
            f"cpus usable  : {self.n_cpus}",
            f"memory       : {mem}",
            f"data roots   : {', '.join(map(str, self.data_roots))}",
            f"output root  : {self.output_root}",
        ]
        lines += [f"note         : {n}" for n in self.notes]
        return "\n".join(lines)


# ------------------------------------------------------------------ detection
def _hostnames() -> list[str]:
    names = {socket.gethostname()}
    try:
        names.add(socket.getfqdn())
    except OSError:  # pragma: no cover
        pass
    return [n.lower() for n in names if n]


def _detect_machine() -> str:
    override = os.environ.get("PARTICLE_ACCEL_MACHINE")
    if override:
        return override.lower()
    if os.environ.get("TACC_SYSTEM", "").lower() == "stampede3":
        return "stampede3"
    if any("stampede3" in h for h in _hostnames()):
        return "stampede3"
    try:
        with open("/usr/local/etc/queue.map") as fh:
            if "stampede3" in fh.readline().lower():
                return "stampede3"
    except OSError:
        pass
    if os.environ.get("SLURM_CLUSTER_NAME") or os.environ.get("SLURM_JOB_ID"):
        return "slurm"
    if _have_sbatch():  # SLURM login nodes have no SLURM_* variables, but they have sbatch
        return "slurm"
    return "local"


def _have_sbatch() -> bool:
    return shutil.which("sbatch") is not None


_LOGIN_RE = re.compile(r"(^|[-_.])login\d*([-_.]|$)")


def _is_login_host(hostname: str) -> bool:
    """``login1``, ``login2.stampede3.tacc.utexas.edu``, ``gl-login1`` -> True; ``c449-001`` -> False."""
    return bool(_LOGIN_RE.search(hostname.lower()))


def _cpu_count() -> int:
    """CPUs this process may use, capped by SLURM_CPUS_ON_NODE."""
    if os.environ.get("PYTHON_CPU_COUNT", "").isdigit():
        n = int(os.environ["PYTHON_CPU_COUNT"])
    elif hasattr(os, "process_cpu_count"):  # Python >= 3.13
        n = os.process_cpu_count() or 1
    elif hasattr(os, "sched_getaffinity"):  # Linux
        n = len(os.sched_getaffinity(0))
    else:  # macOS
        n = os.cpu_count() or 1
    slurm = os.environ.get("SLURM_CPUS_ON_NODE", "")
    if slurm.isdigit():
        n = min(n, int(slurm))
    return max(1, n)


# ------------------------------------------------------------------- memory
#: Files read by :func:`_memory_bytes`, kept as module attributes so that tests can replace them.
_PROC_CGROUP = Path("/proc/self/cgroup")
_CGROUP_ROOT = Path("/sys/fs/cgroup")
_MEMINFO = Path("/proc/meminfo")
_NO_LIMIT = 1 << 60  # cgroup v1 reports no limit as a number just below 2**63

_SIZE_RE = re.compile(r"^\s*(\d+)\s*([KMGT]?)B?\s*$", re.I)


def _slurm_size_bytes(text: str | None) -> int | None:
    """Bytes of a SLURM memory value in MB, or with a K/M/G/T suffix; None for 0 or unparsable text."""
    if not text:
        return None
    m = _SIZE_RE.match(text)
    if not m or int(m.group(1)) == 0:  # --mem=0 means "all memory of the node"
        return None
    scale = {"": 2**20, "K": 2**10, "M": 2**20, "G": 2**30, "T": 2**40}[m.group(2).upper()]
    return int(m.group(1)) * scale


def _slurm_memory_bytes() -> int | None:
    """Memory of this SLURM job on this node, SLURM_MEM_PER_NODE or SLURM_MEM_PER_CPU * SLURM_CPUS_ON_NODE."""
    per_node = _slurm_size_bytes(os.environ.get("SLURM_MEM_PER_NODE"))
    if per_node is not None:
        return per_node
    per_cpu = _slurm_size_bytes(os.environ.get("SLURM_MEM_PER_CPU"))
    cpus = os.environ.get("SLURM_CPUS_ON_NODE", "")
    if per_cpu is not None and cpus.isdigit() and int(cpus) > 0:
        return per_cpu * int(cpus)
    return None


def _read_limit(path: Path) -> int | None:
    try:
        text = path.read_text().strip()
    except OSError:
        return None
    if text.isdigit() and 0 < int(text) < _NO_LIMIT:
        return int(text)
    return None  # "max" (cgroup v2) or the v1 "unlimited" sentinel


def _own_cgroup_limit(proc_cgroup: Path | None = None, cgroup_root: Path | None = None) -> int | None:
    """Smallest memory limit of this process's cgroup and its ancestors, or None.

    The limit is ``<root>/<path>/memory.max`` for cgroup v2 (a ``0::/path`` line in
    ``/proc/self/cgroup``) and ``<root>/memory/<path>/memory.limit_in_bytes`` for cgroup v1.
    Ancestors are checked because SLURM sets the job level and leaves ``max`` on the task level.
    """
    proc_cgroup = _PROC_CGROUP if proc_cgroup is None else Path(proc_cgroup)
    cgroup_root = _CGROUP_ROOT if cgroup_root is None else Path(cgroup_root)
    try:
        lines = proc_cgroup.read_text().splitlines()
    except OSError:
        return None
    candidates: list[tuple[Path, str]] = []
    for line in lines:
        parts = line.strip().split(":", 2)
        if len(parts) != 3:
            continue
        hierarchy, controllers, rel = parts
        rel = rel.strip().lstrip("/")
        if hierarchy == "0" and controllers == "":
            candidates.append((cgroup_root / rel, "memory.max"))
        elif "memory" in controllers.split(","):
            candidates.append((cgroup_root / "memory" / rel, "memory.limit_in_bytes"))
    limits = []
    for base, fname in candidates:
        d = base
        while True:
            if d == cgroup_root or d == cgroup_root / "memory":
                break  # the root cgroup is read by _root_cgroup_limit
            lim = _read_limit(d / fname)
            if lim is not None:
                limits.append(lim)
            if d.parent == d:
                break
            d = d.parent
    return min(limits) if limits else None


def _root_cgroup_limit(cgroup_root: Path | None = None) -> int | None:
    """Memory limit at the top of the visible cgroup tree, which inside a container is the container's."""
    cgroup_root = _CGROUP_ROOT if cgroup_root is None else Path(cgroup_root)
    for path in (cgroup_root / "memory.max", cgroup_root / "memory" / "memory.limit_in_bytes"):
        lim = _read_limit(path)
        if lim is not None:
            return lim
    return None


def _mem_available(meminfo: Path | None = None) -> int | None:
    meminfo = _MEMINFO if meminfo is None else Path(meminfo)
    try:
        with open(meminfo) as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) * 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def _sysctl_memsize() -> int | None:
    try:  # macOS
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=2)
        if out.returncode == 0 and out.stdout.strip().isdigit():
            return int(out.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return None


def _memory_bytes() -> int | None:
    """Bytes of memory this process may use on this node.

    The limit is the SLURM allocation, else this process's cgroup limit, else the root cgroup
    limit.  The smaller of the limit and MemAvailable is returned, since other jobs may already
    use part of the node.  Without either, the total RAM from ``sysctl hw.memsize`` (macOS).
    """
    limit = _slurm_memory_bytes()
    if limit is None:
        limit = _own_cgroup_limit()
    if limit is None:
        limit = _root_cgroup_limit()
    avail = _mem_available()
    if limit is not None and avail is not None:
        return min(limit, avail)
    if limit is not None:
        return limit
    if avail is not None:
        return avail
    return _sysctl_memsize()


def _split_roots(value: str) -> tuple[Path, ...]:
    return tuple(Path(os.path.expandvars(os.path.expanduser(p))) for p in value.split(os.pathsep) if p)


@functools.lru_cache(maxsize=None)
def _detect_cached(env_key: tuple) -> Environment:
    machine = _detect_machine()
    hostname = socket.gethostname()
    job_id = os.environ.get("SLURM_JOB_ID")
    in_job = job_id is not None
    is_login = (not in_job) and machine != "local" and _is_login_host(hostname)
    notes = []
    if is_login:
        notes.append("login node: heavy analysis is not allowed here -- start `idev -p skx-dev` or submit a job")

    scratch = os.environ.get("SCRATCH")
    repo = repo_root()
    base = repo if repo is not None else Path.cwd()
    fallback_used = False
    if os.environ.get("PARTICLE_ACCEL_DATA"):
        data_roots = _split_roots(os.environ["PARTICLE_ACCEL_DATA"])
    elif machine != "local" and scratch:
        data_roots = (Path(scratch) / PROJECT_DIRNAME / "runs",)
    else:
        data_roots = (base / "data",)
        fallback_used = True
    if os.environ.get("PARTICLE_ACCEL_OUTPUT"):
        output_root = _split_roots(os.environ["PARTICLE_ACCEL_OUTPUT"])[0]
    elif machine != "local" and scratch:
        output_root = Path(scratch) / PROJECT_DIRNAME / "outputs"
    else:
        output_root = base / "outputs"
        fallback_used = True
    if machine != "local" and not scratch and not os.environ.get("PARTICLE_ACCEL_DATA"):
        notes.append("$SCRATCH is not set; using repository-relative data/ and outputs/")
    if repo is None and fallback_used:
        notes.append(f"no shearpic repository checkout found (installed package?); data/ and outputs/ default to "
                     f"the current directory {base} -- set PARTICLE_ACCEL_DATA and PARTICLE_ACCEL_OUTPUT")

    nodes = os.environ.get("SLURM_JOB_NUM_NODES", "1")
    return Environment(
        machine=machine,
        hostname=hostname,
        in_job=in_job,
        is_login_node=is_login,
        job_id=job_id,
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        n_nodes=int(nodes) if nodes.isdigit() else 1,
        n_cpus=_cpu_count(),
        mem_bytes=_memory_bytes(),
        data_roots=data_roots,
        output_root=output_root,
        notes=tuple(notes),
    )


_ENV_KEYS = ("PARTICLE_ACCEL_MACHINE", "PARTICLE_ACCEL_DATA", "PARTICLE_ACCEL_OUTPUT", "TACC_SYSTEM", "SCRATCH",
             "SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_JOB_PARTITION", "SLURM_JOB_NUM_NODES",
             "SLURM_CLUSTER_NAME", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU", "PYTHON_CPU_COUNT", "PATH")


def detect_environment() -> Environment:
    """Detect the current environment, cached until a relevant variable or the working directory changes.

    The machine is ``PARTICLE_ACCEL_MACHINE`` if set, else ``'stampede3'`` when ``TACC_SYSTEM``, the host
    name or ``/usr/local/etc/queue.map`` says so, ``'slurm'`` on other clusters with SLURM variables or
    ``sbatch``, and ``'local'`` otherwise.  A login node is a cluster host outside a job whose name has
    a ``login`` component.
    """
    try:
        cwd = os.getcwd()  # the default paths of an installed package depend on it
    except OSError:  # pragma: no cover - deleted working directory
        cwd = None
    return _detect_cached(tuple(os.environ.get(k) for k in _ENV_KEYS) + (cwd,))


# ---------------------------------------------------------------------- paths
_RUN_RE = re.compile(r"^run0*(\d+)$")


def _looks_like_path(run: str) -> bool:
    """True if the string is meant as a directory path rather than a run number or name."""
    if os.sep in run or (os.altsep and os.altsep in run) or "/" in run:
        return True
    if run.startswith((".", "~")):
        return True
    if _RUN_RE.match(run) or run.isdigit():
        return False  # '423' / 'run0423' are run ids even if such a folder exists in the cwd
    return Path(run).expanduser().is_dir()


def resolve_run(run: int | str | Path, roots: Path | str | list | None = None) -> Path:
    """Directory of a run given its number (423), name ('run0423') or path.

    A Path, or a string that contains a path separator, starts with ``.`` or ``~`` or names an
    existing directory other than a run name, is used as a path.  Otherwise the roots, by
    default ``detect_environment().data_roots``, are searched for ``run<N>`` with any zero
    padding; roots may be a path, an ``os.pathsep``-separated string or a list of paths.

    Raises FileNotFoundError when nothing is found, and ValueError when run is not
    interpretable or several directories have the same run number.
    """
    if isinstance(run, Path) or (isinstance(run, str) and _looks_like_path(run)):
        p = Path(run).expanduser()
        if p.is_dir():
            return p
        raise FileNotFoundError(f"run directory {p} does not exist")
    if isinstance(run, str):
        m = _RUN_RE.match(run)
        if not m and not run.isdigit():
            raise ValueError(f"cannot interpret run {run!r}; use 423, 'run0423' or a path")
        run = int(m.group(1)) if m else int(run)
    if roots is None:
        search = detect_environment().data_roots
    elif isinstance(roots, (str, Path)):
        search = _split_roots(str(roots))
    else:
        search = tuple(Path(r) for r in roots)
    matches: list[Path] = []
    seen: set[Path] = set()
    for root in search:
        if not root.is_dir():
            continue
        for cand in sorted(root.glob("run*")):
            m = _RUN_RE.match(cand.name)
            if m and int(m.group(1)) == run and cand.is_dir():
                key = cand.resolve()
                if key not in seen:  # a root listed twice or reached through a symlink is not ambiguous
                    seen.add(key)
                    matches.append(cand)
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(
            f"run {run} is ambiguous: {len(matches)} directories match ({', '.join(map(str, matches))}). "
            "Pass the intended directory as a path."
        )
    raise FileNotFoundError(
        f"run {run} not found in {[str(r) for r in search]}. Set PARTICLE_ACCEL_DATA to the directory "
        "that contains your run folders."
    )


def output_dir(*parts: str | int, create: bool = True) -> Path:
    """A directory under the output root, e.g. ``output_dir('run0423', 'spectra')``."""
    p = detect_environment().output_root.joinpath(*map(str, parts))
    if create:
        p.mkdir(parents=True, exist_ok=True)
    return p


def run_output_dir(run_dir_or_id: int | str | Path, *parts: str | int, create: bool = True) -> Path:
    """Product directory of one run, ``<output_root>/run0423/<parts...>``.

    Ids, ``run<N>`` names and run directory paths are normalised to ``run{id:04d}``, so
    ``.../run423`` and ``.../run0423`` share a directory; other names are used as they are.
    """
    if isinstance(run_dir_or_id, bool):
        raise ValueError(f"not a run: {run_dir_or_id!r}")
    if isinstance(run_dir_or_id, int):
        name = RUN_NAME_FORMAT.format(id=run_dir_or_id)
    else:
        text = str(run_dir_or_id).strip()
        if text.isdigit():
            name = RUN_NAME_FORMAT.format(id=int(text))
        else:
            base = Path(text).expanduser().name
            if not base:
                raise ValueError(f"cannot derive a run name from {run_dir_or_id!r}")
            m = _RUN_RE.match(base)
            name = RUN_NAME_FORMAT.format(id=int(m.group(1))) if m else base
    return output_dir(name, *parts, create=create)


def warn_if_login_node(what: str = "this analysis") -> None:
    """Warn when called on a login node, where heavy work should move to idev or sbatch."""
    env = detect_environment()
    if env.is_login_node:
        warnings.warn(f"running {what} on a login node ({env.hostname}); use idev or sbatch instead", stacklevel=2)
