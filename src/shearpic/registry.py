"""Run registry: a ``runs.yaml`` file recording the purpose, status and input of every run.

Each run record holds the SHA-256 of its athinput and a snapshot of key parameters, and a
run cloned from another stores its input differences from the parent.  An experiment is a
named group of runs with plot labels and styles.  Every change re-reads and saves the file
under a lock on ``runs.yaml.lock``, so concurrent commands such as SLURM array tasks do not
overwrite each other; create the file once with ``shearpic runs init``::

    reg = load_registry()
    rec = reg.new_run("c=100 at fixed q/mc", parent=1, overrides={"particles/speed_of_light": 100})
    reg.set_status(rec.id, "submitted", job_id="1234567")
    for entry in reg.experiment("speed_of_light"):
        ax.plot(t, y, **entry.style())
"""

from __future__ import annotations

import contextlib
import errno
import hashlib
import io
import os
import random
import re
import socket
import stat
import subprocess
import tempfile
import threading
import time
import warnings
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap, CommentedSeq
from ruamel.yaml.representer import RoundTripRepresenter

from .io.athinput import AthInput, parse_value, read_athinput, write_athinput

try:  # POSIX
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

if TYPE_CHECKING:  # pragma: no cover
    from .config import RunConfig

__all__ = [
    "STATUSES",
    "TRANSITIONS",
    "PARAM_KEYS",
    "REGISTRY_ENV",
    "RUN_INFO_NAME",
    "RECENT_SECONDS",
    "RegistryError",
    "RunRecord",
    "Problem",
    "ExperimentRun",
    "Experiment",
    "Registry",
    "load_registry",
    "init_registry",
    "registry_path",
    "registry_lock",
    "extract_params",
    "diff_athinput",
    "parse_overrides",
    "to_run_id",
    "new_run",
    "adopt_run",
    "set_status",
    "lock_run",
    "unlock_run",
    "move_run",
    "check",
]

STATUSES: tuple[str, ...] = ("planned", "submitted", "running", "completed", "failed", "superseded", "archived")

#: Expected status changes; any other change is allowed but warns.
TRANSITIONS: dict[str, frozenset[str]] = {
    "planned": frozenset({"submitted", "running", "failed", "superseded", "archived"}),
    "submitted": frozenset({"running", "completed", "failed", "planned", "superseded"}),
    "running": frozenset({"completed", "failed", "submitted", "superseded"}),
    "completed": frozenset({"superseded", "archived"}),
    "failed": frozenset({"planned", "submitted", "superseded", "archived"}),
    "superseded": frozenset({"archived"}),
    "archived": frozenset(),
}

#: Parameters copied into ``params``: name -> (athinput block, key).
PARAM_KEYS: dict[str, tuple[str, str]] = {
    "c": ("particles", "speed_of_light"),
    "q_mc": ("particles", "charge_over_mass_over_c"),
    "M_A": ("problem", "M_A"),
    "shear_strength": ("problem", "shear_strength"),
    "vp_par": ("problem", "vp_par"),
    "cr_mass": ("problem", "cr_mass"),
    "tau": ("problem", "tau"),
    "nu_iso": ("problem", "nu_iso"),
    "eta_ohm": ("problem", "eta_ohm"),
    "iprob": ("problem", "iprob"),
    "y1": ("problem", "y1"),
    "y2": ("problem", "y2"),
    **{f"nx{d}": ("mesh", f"nx{d}") for d in (1, 2, 3)},
    **{f"x{d}{m}": ("mesh", f"x{d}{m}") for d in (1, 2, 3) for m in ("min", "max")},
    **{f"npx{d}": ("problem", f"npx{d}") for d in (1, 2, 3)},
    "tlim": ("time", "tlim"),
    "backreaction": ("particles", "backreaction"),
    "cs": ("hydro", "iso_sound_speed"),
}

REGISTRY_ENV = "PARTICLE_ACCEL_REGISTRY"
RUN_INFO_NAME = "run_info.yaml"
STALE_DAYS = 14
#: A run directory with a file modified less than this many seconds ago may still be in use by Athena++.
RECENT_SECONDS = 3600.0
#: Seconds to wait for the registry lock.
LOCK_TIMEOUT = 120.0

_RUN_NAME_RE = re.compile(r"^run0*(\d+)$")
_DEFAULT_TEXT = """\
# Run registry for the Particle-Accel-in-Rel-Shear project.
# Edit with the CLI (`shearpic runs --help`); free-form edits to purpose, notes, tags and
# experiments are fine, but never edit ids, hashes or params by hand.
schema_version: 1
run_dir_format: run{id:04d}
runs: []
experiments: {}
"""


class RegistryError(RuntimeError):
    """The registry file is malformed, missing, locked, or was changed on disk by someone else."""


# ============================================================== YAML plumbing
class _Representer(RoundTripRepresenter):
    """Round-trip representer that writes ``None`` as an explicit ``null``."""


_Representer.add_representer(  # registered on the subclass so ruamel's global tables stay untouched
    type(None), lambda self, data: self.represent_scalar("tag:yaml.org,2002:null", "null")
)


def _yaml() -> YAML:
    y = YAML(typ="rt")
    y.Representer = _Representer
    y.preserve_quotes = True
    y.indent(mapping=2, sequence=4, offset=2)
    y.width = 120
    return y


def _to_yaml(obj: Any) -> Any:
    """Convert plain Python to ruamel nodes, with flow style for lists of scalars."""
    if isinstance(obj, Mapping):
        m = CommentedMap()
        for k, v in obj.items():
            m[k] = _to_yaml(v)
        return m
    if isinstance(obj, (list, tuple)):
        s = CommentedSeq(_to_yaml(v) for v in obj)
        if all(not isinstance(v, (Mapping, list, tuple)) for v in obj):
            s.fa.set_flow_style()
        else:
            s.fa.set_block_style()
        return s
    if isinstance(obj, Path):
        return str(obj)
    return obj


def _plain(obj: Any) -> Any:
    """Convert ruamel nodes to plain Python; dates become ISO strings."""
    import datetime as _dt

    if isinstance(obj, Mapping):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, str):
        return str(obj)
    return obj


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _athinput_sha(path: Path) -> str:
    """SHA-256 of an athinput file, identical to :attr:`AthInput.sha256`."""
    return read_athinput(Path(path)).sha256


def _current_umask() -> int:
    mask = os.umask(0)
    os.umask(mask)
    return mask


def _atomic_write(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file and ``os.replace``.

    An existing file keeps its permission bits; a new file gets ``0o666 & ~umask`` as with
    ``open()``, rather than the ``0600`` of ``mkstemp``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        if path.exists():
            os.chmod(tmp, stat.S_IMODE(path.stat().st_mode))
        else:
            os.chmod(tmp, 0o666 & ~_current_umask())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _now() -> str:
    import datetime as _dt

    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _today() -> str:
    import datetime as _dt

    return _dt.date.today().isoformat()


def _default_author() -> str:
    """``git config user.name`` in the repository, else ``$USER``, else ``'unknown'``."""
    from .env import repo_root

    try:
        out = subprocess.run(["git", "config", "user.name"], capture_output=True, text=True, timeout=3,
                             cwd=repo_root())
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    for key in ("USER", "USERNAME", "LOGNAME"):
        if os.environ.get(key):
            return os.environ[key]
    return "unknown"


def _format_ids(ids: Iterable[int]) -> str:
    """``[1, 2, 3, 5]`` -> ``'1-3, 5'``."""
    ids = sorted(set(i for i in ids if isinstance(i, int)))
    if not ids:
        return "none"
    parts, start, prev = [], ids[0], ids[0]
    for i in ids[1:] + [None]:
        if i is not None and i == prev + 1:
            prev = i
            continue
        parts.append(str(start) if start == prev else f"{start}-{prev}")
        if i is not None:
            start = prev = i
    return ", ".join(parts)


def _same_path(a: str | Path, b: str | Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return os.path.abspath(a) == os.path.abspath(b)


def _run_dirs(root: Path) -> Iterator[tuple[int, Path]]:
    """``(id, path)`` of every ``run<N>`` directory, with any zero padding, directly under ``root``."""
    try:
        entries = sorted(Path(root).iterdir())
    except OSError:
        return
    for cand in entries:
        m = _RUN_NAME_RE.match(cand.name)
        if m and cand.is_dir():
            yield int(m.group(1)), cand


def _recent_file(root: Path, seconds: float = RECENT_SECONDS) -> tuple[float, Path] | None:
    """``(age in seconds, path)`` of the first file below ``root`` modified less than ``seconds`` ago, else None.

    Symlinks are not followed and unreadable directories are skipped.
    """
    now = time.time()
    for dirpath, _dirnames, filenames in os.walk(root, onerror=lambda err: None):
        for name in filenames:
            p = os.path.join(dirpath, name)
            try:
                age = now - os.lstat(p).st_mtime
            except OSError:
                continue
            if age < seconds:
                return max(age, 0.0), Path(p)
    return None


# ================================================================== locking
_LOCK_STATE = threading.local()
_UNSUPPORTED_LOCK_ERRNOS = {getattr(errno, n) for n in ("ENOLCK", "ENOSYS", "EOPNOTSUPP", "ENOTSUP") if hasattr(errno, n)}


def _lock_timeout_error(lock: Path, timeout: float, holder: str = "") -> RegistryError:
    return RegistryError(
        f"could not lock the registry within {timeout:g} s ({lock} is held by another shearpic process{holder}). "
        "Try again; if no other shearpic command is running, the lock is stale"
        + ("" if fcntl is not None else f" -- delete {lock}") + "."
    )


def _acquire_exclusive_file(lock: Path, timeout: float) -> Any:
    """Lock by creating ``lock`` with ``O_EXCL``, retrying until ``timeout``; returns a release function."""
    deadline = time.monotonic() + timeout
    while True:
        try:
            fd = os.open(lock, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o666)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                try:
                    holder = f": {lock.read_text().strip()}"
                except OSError:
                    holder = ""
                raise _lock_timeout_error(lock, timeout, holder) from None
            time.sleep(0.02 + 0.03 * random.random())
    with os.fdopen(fd, "w") as fh:
        fh.write(f"pid {os.getpid()} on {socket.gethostname()}\n")

    def release() -> None:
        try:
            os.unlink(lock)
        except OSError:
            pass

    return release


def _acquire_flock(lock: Path, timeout: float) -> Any:
    """Lock ``lock`` with ``fcntl.flock``, retrying until ``timeout``, or with ``O_EXCL`` where flock is unsupported."""
    deadline = time.monotonic() + timeout
    fd = os.open(lock, os.O_RDWR | os.O_CREAT, 0o666)
    unsupported = False
    try:
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as err:
                if err.errno in (errno.EAGAIN, errno.EWOULDBLOCK, errno.EACCES):
                    if time.monotonic() >= deadline:
                        raise _lock_timeout_error(lock, timeout) from None
                    time.sleep(0.01 + 0.02 * random.random())
                elif err.errno in _UNSUPPORTED_LOCK_ERRNOS:  # Lustre or NFS mounts without flock support
                    unsupported = True
                    break
                else:
                    raise
    except BaseException:
        os.close(fd)
        raise
    if unsupported:
        os.close(fd)
        return _acquire_exclusive_file(lock.with_name(lock.name + ".excl"), max(deadline - time.monotonic(), 0.0))

    def release() -> None:
        try:
            fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    return release


@contextlib.contextmanager
def registry_lock(path: str | Path, timeout: float = LOCK_TIMEOUT) -> Iterator[Path]:
    """Hold an exclusive lock on ``<path>.lock`` and yield the lock file path.

    Uses ``fcntl.flock`` where available, otherwise an ``O_EXCL`` lock file.  The lock is
    re-entrant within a thread; other threads and processes wait up to ``timeout`` seconds
    before :class:`RegistryError` is raised.
    """
    reg = Path(path).expanduser()
    lock = Path(os.path.abspath(reg.with_name(reg.name + ".lock")))
    held = getattr(_LOCK_STATE, "held", None)
    if held is None:
        held = _LOCK_STATE.held = {}
    key = str(lock)
    if held.get(key):
        held[key] += 1
        try:
            yield lock
        finally:
            held[key] -= 1
        return
    if not lock.parent.is_dir():
        raise RegistryError(f"cannot lock {reg}: directory {lock.parent} does not exist")
    # The O_EXCL file has its own name because flock hosts leave <path>.lock in place, which
    # would permanently block a host without fcntl sharing the directory.
    release = (_acquire_flock(lock, timeout) if fcntl is not None
               else _acquire_exclusive_file(lock.with_name(lock.name + ".excl"), timeout))
    held[key] = 1
    try:
        yield lock
    finally:
        held.pop(key, None)
        release()


# ============================================================ pure helpers
def registry_path(path: str | Path | None = None) -> Path:
    """Registry file: ``path``, else ``$PARTICLE_ACCEL_REGISTRY``, else ``<repo>/runs.yaml``.

    Raises :class:`RegistryError` if neither is set and shearpic is not running from its git checkout.
    """
    if path is not None:
        return Path(path).expanduser()
    if os.environ.get(REGISTRY_ENV):
        return Path(os.environ[REGISTRY_ENV]).expanduser()
    from .env import repo_root

    root = repo_root()
    if root is None:
        raise RegistryError(
            "cannot locate the repository runs.yaml (shearpic is not running from its git checkout); "
            f"set {REGISTRY_ENV} or pass --registry"
        )
    return root / "runs.yaml"


def to_run_id(value: int | str) -> int:
    """Run id from ``7``, ``'7'``, ``'run7'`` or ``'run0007'``; raises ``ValueError`` otherwise."""
    if isinstance(value, bool):
        raise ValueError(f"not a run id: {value!r}")
    if isinstance(value, int):
        return value
    s = str(value).strip()
    if s.isdigit():
        return int(s)
    m = _RUN_NAME_RE.match(s)
    if m:
        return int(m.group(1))
    raise ValueError(f"not a run id: {value!r} (use 7 or run0007)")


def extract_params(inp: AthInput) -> dict[str, Any]:
    """Parsed athinput values of the parameters in :data:`PARAM_KEYS`.

    Keys absent from the file are skipped rather than filled with Athena++ defaults.
    """
    out = {}
    for name, (block, key) in PARAM_KEYS.items():
        if key in inp.blocks.get(block, {}):
            out[name] = inp.blocks[block][key]
    return out


def diff_athinput(old: AthInput, new: AthInput) -> dict[str, list[Any]]:
    """Values that differ between two athinputs, as ``{"block/key": [old, new]}``.

    Parsed values are compared, so ``1`` equals ``1.0``; a key missing on one side is None there.
    """
    changes: dict[str, list[Any]] = {}
    blocks = list(new.blocks) + [b for b in old.blocks if b not in new.blocks]
    for block in blocks:
        o, n = old.blocks.get(block, {}), new.blocks.get(block, {})
        for key in list(n) + [k for k in o if k not in n]:
            ov, nv = o.get(key), n.get(key)
            if key not in o or key not in n or ov != nv or isinstance(ov, bool) != isinstance(nv, bool):
                changes[f"{block}/{key}"] = [ov, nv]
    return changes


def parse_overrides(overrides: Mapping[str, Any] | Iterable[str] | None) -> dict[str, dict[str, Any]]:
    """Convert ``{"block/key": value}`` or ``["block/key=value", ...]`` to ``{block: {key: value}}``.

    String values are parsed with :func:`shearpic.io.athinput.parse_value`.
    """
    if overrides is None:
        return {}
    if isinstance(overrides, Mapping):
        items = list(overrides.items())
    else:
        items = []
        for text in overrides:
            if "=" not in text:
                raise ValueError(f"override {text!r} must look like block/key=value")
            k, v = text.split("=", 1)
            items.append((k.strip(), v))
    out: dict[str, dict[str, Any]] = {}
    for path, value in items:
        parts = str(path).strip().strip("/").split("/")
        if len(parts) != 2 or not all(parts):
            raise ValueError(f"override key {path!r} must look like block/key (e.g. problem/shear_strength)")
        block, key = (p.strip().strip("<>") for p in parts)
        out.setdefault(block, {})[key] = parse_value(value) if isinstance(value, str) else value
    return out


# ================================================================ data model
@dataclass
class RunRecord:
    """One entry of ``runs:``, read-only; change it through :class:`Registry`.

    ``data_path`` is the absolute run directory, ``pgen`` is ``{source, git_commit}`` of the
    problem generator, ``athinput_sha256`` is the hash of ``athinput_file`` at registration,
    ``template`` is ``{path, sha256, changes}`` for runs made from a template file, ``params``
    holds the :data:`PARAM_KEYS` values, ``changes_from_parent`` is ``{"block/key": [old, new]}``
    relative to ``parent``, ``locked`` is True, False or ``'partial'``, and ``notes`` is a
    list of ``{date, text}``, oldest first.
    """

    id: int
    purpose: str = ""
    status: str = "planned"
    created: str | None = None
    updated: str | None = None
    author: str | None = None
    machine: str | None = None
    data_path: str | None = None
    problem_id: str | None = None
    pgen: dict = field(default_factory=lambda: {"source": None, "git_commit": None})
    athinput_file: str | None = None
    athinput_sha256: str | None = None
    template: dict | None = None
    params: dict = field(default_factory=dict)
    parent: int | None = None
    parent_athinput_sha256: str | None = None
    changes_from_parent: dict | None = None
    slurm: dict = field(default_factory=lambda: {"job_id": None, "partition": None, "nodes": None})
    superseded_by: int | None = None
    locked: bool | str = False
    tags: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    registry: "Registry | None" = field(default=None, repr=False, compare=False)

    _FIELDS = ("id", "purpose", "status", "created", "updated", "author", "machine", "data_path", "problem_id",
               "pgen", "athinput_file", "athinput_sha256", "template", "params", "parent", "parent_athinput_sha256",
               "changes_from_parent", "slurm", "superseded_by", "locked", "tags", "notes")

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, Any], registry: "Registry | None" = None) -> "RunRecord":
        d = _plain(mapping)
        kw = {k: d[k] for k in cls._FIELDS if k in d}
        rec = cls(**kw, registry=registry)
        defaults = cls(id=None)
        for name in ("pgen", "params", "slurm", "tags", "notes", "status", "purpose"):
            if getattr(rec, name) is None:  # a hand-edited key left without a value
                setattr(rec, name, getattr(defaults, name))
        rec.locked = "partial" if rec.locked == "partial" else bool(rec.locked)
        return rec

    def to_dict(self) -> dict[str, Any]:
        """Plain dict in the on-disk key order, without the registry reference."""
        return {k: getattr(self, k) for k in self._FIELDS}

    # ----------------------------------------------------------- convenience
    def _reg(self) -> "Registry":
        if self.registry is None:
            raise ValueError(f"run {self.id} is not attached to a registry (dry run?)")
        return self.registry

    def run_dir(self) -> Path:
        """Run directory as resolved by :meth:`Registry.run_dir`."""
        return self._reg().run_dir(self.id)

    def config(self, **kw) -> "RunConfig":
        """:class:`shearpic.config.RunConfig` of this run; see :meth:`Registry.config`."""
        return self._reg().config(self.id, **kw)

    @property
    def last_note(self) -> str | None:
        return self.notes[-1].get("text") if self.notes else None


@dataclass(frozen=True)
class Problem:
    """One finding of :meth:`Registry.check`."""

    level: str            # 'error' or 'warning'
    run_id: int | None    # None for registry-wide problems
    message: str

    def __str__(self) -> str:
        where = f"run {self.run_id}" if self.run_id is not None else "registry"
        return f"{self.level.upper():<7} {where}: {self.message}"


@dataclass(frozen=True)
class ExperimentRun:
    """A run inside an experiment, with its plot style and resolved record."""

    run: int
    label: str
    color: str | None
    linestyle: str | None
    record: RunRecord = field(repr=False)

    @property
    def run_dir(self) -> Path:
        return self.record.run_dir()

    def config(self, **kw) -> "RunConfig":
        return self.record.config(**kw)

    def style(self) -> dict[str, str]:
        """Matplotlib keyword arguments ``label``, ``color`` and ``linestyle``, omitting unset ones."""
        s = {"label": self.label, "color": self.color, "linestyle": self.linestyle}
        return {k: v for k, v in s.items() if v is not None}


@dataclass(frozen=True)
class Experiment:
    """A named group of runs answering one question; iterating yields :class:`ExperimentRun`."""

    name: str
    description: str
    question: str | None
    runs: tuple[ExperimentRun, ...]
    figures: tuple[str, ...] = ()

    def __iter__(self) -> Iterator[ExperimentRun]:
        return iter(self.runs)

    def __len__(self) -> int:
        return len(self.runs)

    @property
    def run_ids(self) -> list[int]:
        return [e.run for e in self.runs]


@dataclass
class _NewRunPlan:
    rec: RunRecord
    run_dir: Path
    name: str
    inp: AthInput


# ================================================================== registry
class Registry:
    """In-memory view of ``runs.yaml``, created with :func:`load_registry`.

    Mutating methods run as one :meth:`transaction` and save immediately; with ``save=False``
    they only change the in-memory copy.
    """

    def __init__(self, path: Path, doc: CommentedMap, loaded_sha: str | None):
        self.path = Path(path)
        self._doc = doc
        self._loaded_sha = loaded_sha
        self._yaml = _yaml()
        self._txn_depth = 0
        self._dirty = False  # unsaved in-memory changes

    # ------------------------------------------------------------- load/save
    @staticmethod
    def _read(p: Path) -> tuple[CommentedMap, str | None]:
        y = _yaml()
        if p.exists():
            raw = p.read_bytes()
            sha = _sha256_bytes(raw)
            try:
                doc = y.load(raw.decode("utf-8"))
            except Exception as err:  # ruamel raises many parser error types
                raise RegistryError(f"{p} is not valid YAML: {err}") from None
        else:
            sha, doc = None, None
        if doc is None:
            doc = y.load(_DEFAULT_TEXT)
        if not isinstance(doc, CommentedMap):
            raise RegistryError(f"{p}: top level must be a mapping with schema_version, runs, experiments")
        version = doc.get("schema_version", 1)
        if version != 1:
            raise RegistryError(f"{p}: unsupported schema_version {version!r} (this shearpic understands 1)")
        for key, default in (("schema_version", 1), ("run_dir_format", "run{id:04d}")):
            if key not in doc:
                doc[key] = default
        if doc.get("runs") is None:
            doc["runs"] = CommentedSeq()
        if doc.get("experiments") is None:
            doc["experiments"] = CommentedMap()
        if not isinstance(doc["runs"], list):
            raise RegistryError(f"{p}: 'runs' must be a list")
        if not isinstance(doc["experiments"], Mapping):
            raise RegistryError(f"{p}: 'experiments' must be a mapping")
        return doc, sha

    @classmethod
    def load(cls, path: str | Path | None = None, create: bool = False) -> "Registry":
        """Load the registry file given by :func:`registry_path`.

        A missing file loads as an empty registry that cannot be saved, so a mistyped path
        does not start a new registry; ``create=True`` writes an empty file first.
        """
        p = registry_path(path)
        if create and not p.exists():
            p.parent.mkdir(parents=True, exist_ok=True)
            with registry_lock(p):
                if not p.exists():
                    _atomic_write(p, _DEFAULT_TEXT)
        doc, sha = cls._read(p)
        return cls(p, doc, sha)

    @property
    def exists(self) -> bool:
        """Whether the registry file exists on disk."""
        return self.path.exists()

    def _missing_error(self) -> RegistryError:
        if self._loaded_sha is not None:
            return RegistryError(f"{self.path} was deleted since it was loaded; nothing was saved")
        return RegistryError(
            f"registry file {self.path} does not exist; nothing was saved. Create it with "
            f"`shearpic runs init --registry {self.path}` (or load_registry(path, create=True)); if you meant an "
            f"existing registry, check --registry / ${REGISTRY_ENV} for a typo."
        )

    def _reload(self) -> None:
        self._doc, self._loaded_sha = self._read(self.path)
        self._dirty = False

    def save(self) -> None:
        """Write the file atomically under the registry lock.

        Raises :class:`RegistryError` if the file is missing or changed on disk since it was loaded.
        """
        if not self.path.exists():
            raise self._missing_error()
        with registry_lock(self.path):
            self._save_locked()

    def _save_locked(self) -> None:
        if not self.path.exists():
            raise self._missing_error()
        current = _sha256_bytes(self.path.read_bytes())
        if current != self._loaded_sha:
            raise RegistryError(
                f"{self.path} was modified by someone else since it was loaded; nothing was saved. "
                "Re-run the command (reload the registry) to apply your change on top of theirs."
            )
        if isinstance(self._doc["runs"], CommentedSeq) and self._doc["runs"]:
            self._doc["runs"].fa.set_block_style()
        if isinstance(self._doc["experiments"], CommentedMap) and self._doc["experiments"]:
            self._doc["experiments"].fa.set_block_style()
        text = self.dumps()
        self._yaml.load(text)  # make sure the text parses before replacing the file
        _atomic_write(self.path, text)
        self._loaded_sha = _sha256_bytes(text.encode("utf-8"))
        self._dirty = False

    @contextlib.contextmanager
    def transaction(self) -> Iterator["Registry"]:
        """Lock and re-read the file, run the block, and save if anything changed.

        Mutating methods called inside the block join the transaction, and transactions may
        be nested.  If the block raises, the in-memory state is restored and nothing is
        written.  Raises :class:`RegistryError` if the file is missing, or if it changed on
        disk while this object held unsaved ``save=False`` changes.
        """
        if self._txn_depth:
            self._txn_depth += 1
            try:
                yield self
            finally:
                self._txn_depth -= 1
            return
        if not self.path.exists():
            raise self._missing_error()
        with registry_lock(self.path):
            if not self.path.exists():
                raise self._missing_error()
            if _sha256_bytes(self.path.read_bytes()) != self._loaded_sha:
                if self._dirty:
                    raise RegistryError(
                        f"{self.path} was modified by someone else while this registry had unsaved changes "
                        "(save=False); nothing was saved. Reload the registry and apply the changes again."
                    )
                self._reload()
            snapshot, dirty_before = self.dumps(), self._dirty
            self._txn_depth = 1
            try:
                yield self
                if self._dirty:
                    self._save_locked()
            except BaseException:
                self._doc, self._dirty = self._yaml.load(snapshot), dirty_before
                raise
            finally:
                self._txn_depth = 0

    @contextlib.contextmanager
    def _mutation(self, save: bool) -> Iterator[None]:
        """Context of one change: a transaction when saving or already in one, else in memory only."""
        if save or self._txn_depth:
            with self.transaction():
                yield
                self._dirty = True
        else:
            yield
            self._dirty = True

    def dumps(self, node: Any = None) -> str:
        """YAML text of the whole registry, or of one node such as a run mapping."""
        buf = io.StringIO()
        self._yaml.dump(self._doc if node is None else node, buf)
        return buf.getvalue()

    # ---------------------------------------------------------------- access
    @property
    def run_dir_format(self) -> str:
        return str(self._doc.get("run_dir_format", "run{id:04d}"))

    def _run_maps(self) -> list[CommentedMap]:
        return [m for m in self._doc["runs"] if isinstance(m, Mapping)]

    def _find_map(self, run_id: int | str) -> CommentedMap:
        rid = to_run_id(run_id)
        for m in self._run_maps():
            if m.get("id") == rid:
                return m
        raise KeyError(
            f"run {rid} is not registered in {self.path} (registered: {_format_ids(self.ids)}). "
            "See `shearpic runs list`; register an existing directory with `shearpic runs adopt PATH --purpose ...`."
        )

    @property
    def runs(self) -> list[RunRecord]:
        """All run records, in file order."""
        return [RunRecord.from_mapping(m, self) for m in self._run_maps()]

    @property
    def ids(self) -> list[int]:
        return [m.get("id") for m in self._run_maps() if isinstance(m.get("id"), int)]

    def has_run(self, run_id: int | str) -> bool:
        try:
            self._find_map(run_id)
        except (KeyError, ValueError):
            return False
        return True

    def get_run(self, run_id: int | str) -> RunRecord:
        """Record of run ``run_id``; ``KeyError`` if it is not registered."""
        return RunRecord.from_mapping(self._find_map(run_id), self)

    run = get_run

    def next_id(self, data_roots: Sequence[str | Path] | None = None, extra_roots: Iterable[str | Path] = ()) -> int:
        """One more than the largest id that is registered or used by a ``run<N>`` directory.

        Directories are searched under the data roots and ``extra_roots``, so a new run never
        takes the number of an unregistered directory.
        """
        ids = set(self.ids)
        for root in (*self._data_roots(data_roots), *map(Path, extra_roots)):
            ids.update(rid for rid, _ in _run_dirs(Path(root).expanduser()))
        return max(ids, default=0) + 1

    def run_record_yaml(self, run_id: int | str) -> str:
        return self.dumps(self._find_map(run_id))

    @staticmethod
    def _data_roots(data_roots: Sequence[str | Path] | None = None) -> tuple[Path, ...]:
        if data_roots is not None:
            return tuple(Path(r) for r in data_roots)
        from .env import detect_environment

        return detect_environment().data_roots

    @staticmethod
    def _recorded_dir(m: Mapping[str, Any]) -> Path | None:
        """The recorded ``data_path`` if it is absolute, else None."""
        if not m.get("data_path"):
            return None
        p = Path(str(m["data_path"])).expanduser()
        return p if p.is_absolute() else None

    def _resolve_dir(self, m: Mapping[str, Any], roots: Sequence[Path]) -> Path:
        rid = m.get("id")
        stored = Path(str(m["data_path"])).expanduser() if m.get("data_path") else None
        cands: list[Path] = []
        if stored is not None:
            if stored.is_absolute():
                cands.append(stored)
                cands += [Path(r) / stored.name for r in roots]
            else:
                cands += [Path(r) / stored for r in roots]
        else:
            cands += [Path(r) / self.run_dir_format.format(id=rid) for r in roots]
        for c in cands:
            if c.is_dir():
                return c
        if cands:
            return cands[0]
        return stored if stored is not None else Path(self.run_dir_format.format(id=rid))

    def run_dir(self, run_id: int | str, data_roots: Sequence[str | Path] | None = None) -> Path:
        """Directory of a run.

        Candidates, in order, are the recorded absolute ``data_path`` and a directory of the
        same name under each data root, since data roots differ between machines.  A relative
        ``data_path``, or ``run_dir_format`` when none is recorded, is looked up under each
        root.  Names must match exactly.  Returns the first existing candidate, else the first one.
        """
        return self._resolve_dir(self._find_map(run_id), self._data_roots(data_roots))

    def _checked_run_dir(self, m: Mapping[str, Any], action: str, *, verify_hash: bool = True,
                         strict: bool = False) -> Path:
        """Run directory for an operation that changes something; raises instead of guessing.

        A same-name directory under a data root stands in for a missing recorded ``data_path``
        only if it holds the registered athinput (when ``verify_hash``) and never when ``strict``.
        """
        rid = m.get("id")
        roots = self._data_roots()
        d = self._resolve_dir(m, roots)
        recorded = self._recorded_dir(m)
        move_hint = f"If the run directory moved, record the new location with `shearpic runs move {rid} NEW_PATH`."
        if not d.is_dir():
            where = f"recorded directory {m.get('data_path')}" if m.get("data_path") else f"directory {d}"
            name = Path(str(m["data_path"])).name if m.get("data_path") else d.name
            raise FileNotFoundError(
                f"run {rid}: cannot {action}: {where} does not exist (nor does {name} under the data roots "
                f"{', '.join(map(str, roots)) or 'none'}). {move_hint}"
            )
        if recorded is None or _same_path(d, recorded):
            return d
        if strict:
            sha, name = m.get("athinput_sha256"), m.get("athinput_file")
            same_input = bool(sha and name and (d / str(name)).is_file() and _athinput_sha(d / str(name)) == sha)
            force = "" if same_input else " --force"
            raise FileNotFoundError(
                f"run {rid}: cannot {action}: the recorded directory {recorded} does not exist; a directory with the "
                f"same name exists at {d}. If that is this run, confirm it with `shearpic runs move {rid} {d}{force}` "
                "first."
            )
        if verify_hash:
            sha, name = m.get("athinput_sha256"), m.get("athinput_file")
            if not (sha and name and (d / str(name)).is_file() and _athinput_sha(d / str(name)) == sha):
                raise ValueError(
                    f"run {rid}: cannot {action}: the recorded directory {recorded} does not exist, and {d} (same name "
                    "under a data root) does not contain the registered athinput, so it may be a different run. "
                    f"If it is this run, record it with `shearpic runs move {rid} {d} --force`."
                )
        warnings.warn(f"run {rid}: recorded directory {recorded} does not exist; using {d} (same name under a data "
                      f"root). Record the new location with `shearpic runs move {rid} {d}`.", stacklevel=3)
        return d

    def find_run_by_path(self, path: str | Path) -> RunRecord | None:
        """The record whose run directory is ``path`` after resolving symlinks, else None."""
        target = Path(path).expanduser().resolve()
        roots = self._data_roots()
        for m in self._run_maps():
            if isinstance(m.get("id"), int) and self._resolve_dir(m, roots).resolve() == target:
                return RunRecord.from_mapping(m, self)
        return None

    def athinput_path(self, run_id: int | str) -> Path | None:
        m = self._find_map(run_id)
        d = self.run_dir(run_id)
        if m.get("athinput_file"):
            return d / str(m["athinput_file"])
        found = sorted(d.glob("athinput.*")) if d.is_dir() else []
        return found[0] if found else None

    def config(self, run_id: int | str, **kw) -> "RunConfig":
        """:class:`shearpic.config.RunConfig` of a registered run; ``kw`` go to ``RunConfig.from_athinput``."""
        from .config import RunConfig

        d = self.run_dir(run_id)
        if not d.is_dir():
            raise FileNotFoundError(f"run {run_id}: directory {d} does not exist")
        inp = self.athinput_path(run_id)
        if inp is None or not inp.exists():
            raise FileNotFoundError(f"run {run_id}: no athinput file in {d}")
        return RunConfig.from_athinput(inp, run_dir=d, **kw)

    # ----------------------------------------------------------- experiments
    @property
    def experiment_names(self) -> list[str]:
        return [str(k) for k in self._doc["experiments"]]

    def experiment_data(self, name: str) -> dict[str, Any]:
        """The stored experiment as plain Python, without resolving or validating its runs."""
        exps = self._doc["experiments"]
        if name not in exps:
            have = ", ".join(self.experiment_names) or "none"
            raise KeyError(f"no experiment {name!r} in {self.path} (have: {have})")
        data = _plain(exps[name])
        return data if isinstance(data, dict) else {}

    def experiment(self, name: str) -> Experiment:
        """Experiment ``name`` with its runs resolved; ``KeyError`` if it or any of its runs is not registered."""
        e = self.experiment_data(name)
        entries, missing = [], []
        for item in e.get("runs") or []:
            try:
                rid = to_run_id(item.get("run"))
            except (ValueError, AttributeError):
                missing.append(repr(item))
                continue
            if not self.has_run(rid):
                missing.append(str(rid))
                continue
            entries.append(ExperimentRun(rid, str(item.get("label") or f"run{rid}"), item.get("color"),
                                         item.get("linestyle"), self.get_run(rid)))
        if missing:
            raise KeyError(f"experiment {name!r} references unregistered run(s) {', '.join(missing)}; "
                           f"register them or fix {self.path}")
        return Experiment(name, str(e.get("description") or ""), e.get("question"), tuple(entries),
                          tuple(e.get("figures") or ()))

    def add_experiment(self, name: str, description: str, runs: Iterable[Mapping[str, Any] | Sequence[Any]],
                       *, question: str | None = None, figures: Iterable[str] = (), replace: bool = False,
                       save: bool = True) -> Experiment:
        """Create an experiment, or overwrite an existing one with ``replace=True``.

        ``runs`` items are ``{run, label, color, linestyle}`` dicts or ``(run, label[, color[, linestyle]])``
        tuples.  Every run must be registered and have a label, which is used in plot legends.
        """
        name = str(name).strip()
        if not name:
            raise ValueError("experiment name must not be empty")
        if not str(description or "").strip():
            raise ValueError("an experiment needs a --description")
        runs = [dict(r) if isinstance(r, Mapping) else dict(zip(("run", "label", "color", "linestyle"), list(r)))
                for r in runs]
        figures = list(figures)
        with self._mutation(save):
            exps = self._doc["experiments"]
            if name in exps and not replace:
                raise ValueError(f"experiment {name!r} already exists (use replace=True / --replace)")
            items = []
            for d in runs:
                rid = to_run_id(d.get("run"))
                self._find_map(rid)
                if not str(d.get("label") or "").strip():
                    raise ValueError(f"experiment {name!r}: run {rid} needs a label")
                items.append({"run": rid, "label": str(d["label"]), "color": d.get("color"),
                              "linestyle": d.get("linestyle")})
            if not items:
                raise ValueError(f"experiment {name!r} needs at least one run")
            entry = _to_yaml({"description": description, "question": question, "runs": items, "figures": figures})
            for item in entry["runs"]:
                item.fa.set_flow_style()
            exps[name] = entry
        return self.experiment(name)

    # ------------------------------------------------------------- mutations
    def _touch(self, m: CommentedMap) -> None:
        m["updated"] = _now()

    def add_note(self, run_id: int | str, text: str, *, save: bool = True) -> None:
        """Append a dated note to a run."""
        text = str(text).strip()
        if not text:
            raise ValueError("note text must not be empty")
        with self._mutation(save):
            m = self._find_map(run_id)
            if not isinstance(m.get("notes"), list):
                m["notes"] = CommentedSeq()
            m["notes"].fa.set_block_style()
            m["notes"].append(_to_yaml({"date": _today(), "text": text}))
            self._touch(m)

    def set_status(self, run_id: int | str, status: str, note: str | None = None, *, job_id: str | int | None = None,
                   partition: str | None = None, nodes: int | None = None, superseded_by: int | str | None = None,
                   save: bool = True) -> RunRecord:
        """Change a run's status, record SLURM details and add a dated note.

        Transitions not listed in :data:`TRANSITIONS`, and ``completed`` without a ``*.hst``
        file, warn.  Raises ``FileNotFoundError`` if the run directory is missing, unless the
        new status is ``archived``.
        """
        status = str(status).strip().lower()
        if status not in STATUSES:
            raise ValueError(f"invalid status {status!r}; choose from {', '.join(STATUSES)}")
        with self._mutation(save):
            m = self._find_map(run_id)
            rid = m["id"]
            old = str(m.get("status"))
            d = None
            if status != "archived":  # archived data may be gone; fail before any warning is issued
                d = self._checked_run_dir(m, f"set its status to {status}")
            if old != status and status not in TRANSITIONS.get(old, frozenset()):
                warnings.warn(f"run {rid}: unusual status change {old} -> {status}", stacklevel=2)
            if superseded_by is not None:
                sid = to_run_id(superseded_by)
                self._find_map(sid)
                if sid == rid:
                    raise ValueError(f"run {rid} cannot supersede itself")
                m["superseded_by"] = sid
            if status == "superseded" and m.get("superseded_by") is None:
                warnings.warn(f"run {rid}: status superseded without superseded_by (pass --superseded-by ID)",
                              stacklevel=2)
            if d is not None:
                if status == "completed" and not any(d.glob("*.hst")):
                    warnings.warn(f"run {rid}: marked completed but {d} has no *.hst file", stacklevel=2)
            slurm = {"job_id": None if job_id is None else str(job_id), "partition": partition,
                     "nodes": None if nodes is None else int(nodes)}
            if any(v is not None for v in slurm.values()):
                if not isinstance(m.get("slurm"), Mapping):
                    m["slurm"] = _to_yaml({"job_id": None, "partition": None, "nodes": None})
                for k, v in slurm.items():
                    if v is not None:
                        m["slurm"][k] = v
            m["status"] = status
            text = f"status {old} -> {status}"
            if slurm["job_id"] is not None:
                text += f" (job {slurm['job_id']})"
            if note:
                text += f": {note.strip()}"
            self.add_note(rid, text, save=False)
        return self.get_run(rid)

    def _new_record(self, *, rid: int, purpose: str, status: str, author: str | None, data_path: Path,
                    inp: AthInput | None, tags: Iterable[str], pgen_source: str | None, pgen_commit: str | None,
                    parent: int | None = None, parent_sha: str | None = None, changes: dict | None = None,
                    template: dict | None = None, note: str | None = None) -> RunRecord:
        from .env import detect_environment

        now = _now()
        rec = RunRecord(
            id=rid, purpose=purpose, status=status, created=now, updated=now,
            author=author or _default_author(), machine=detect_environment().machine,
            data_path=os.path.abspath(data_path),
            problem_id=inp.problem_id if inp is not None else None,
            pgen={"source": pgen_source, "git_commit": pgen_commit},
            athinput_file=inp.path.name if inp is not None and inp.path is not None else None,
            athinput_sha256=inp.sha256 if inp is not None else None,
            template=template,
            params=extract_params(inp) if inp is not None else {},
            parent=parent, parent_athinput_sha256=parent_sha, changes_from_parent=changes,
            tags=[str(t) for t in tags],
            notes=[{"date": _today(), "text": note}] if note else [],
        )
        return rec

    def _append(self, rec: RunRecord) -> CommentedMap:
        m = _to_yaml(rec.to_dict())
        runs = self._doc["runs"]
        if isinstance(runs, CommentedSeq):
            runs.fa.set_block_style()
        runs.append(m)
        return m

    def _plan_new_run(self, purpose: str, *, template: str | Path | None, parent: int | str | None,
                      ov: dict[str, dict[str, Any]], data_root: str | Path | None, author: str | None,
                      tags: Iterable[str], pgen_source: str | None, pgen_commit: str | None,
                      note: str | None) -> _NewRunPlan:
        parent_id = parent_sha = None
        if parent is not None:
            parent_id = to_run_id(parent)
            pm = self._find_map(parent_id)
            self._checked_run_dir(pm, "clone it", verify_hash=False)  # the hash is compared below
            src_path = self.athinput_path(parent_id)
            if src_path is None or not src_path.exists():
                raise FileNotFoundError(f"parent run {parent_id}: athinput not found in {self.run_dir(parent_id)}")
            src = read_athinput(src_path)
            parent_sha = pm.get("athinput_sha256")
            if not parent_sha:
                raise ValueError(f"parent run {parent_id} has no recorded athinput_sha256, so its input cannot be "
                                 f"verified; start from the file with --template {src_path}")
            if src.sha256 != parent_sha:
                raise ValueError(
                    f"parent run {parent_id}: {src_path} changed after registration (sha256 {str(parent_sha)[:12]}... "
                    f"recorded, {src.sha256[:12]}... now), so it is not the input run {parent_id} used. Accept the edit "
                    f"with `shearpic runs rehash {parent_id} --note ...`, or start from the file with --template."
                )
        else:
            src_path = Path(template).expanduser()
            if not src_path.exists():
                raise FileNotFoundError(f"template {src_path} does not exist")
            src = read_athinput(src_path)
        for block, kv in ov.items():
            for key, value in list(kv.items()):
                if key not in src.blocks.get(block, {}):
                    warnings.warn(f"<{block}> {key} is not in {src.path.name}; it will be added. Check the spelling: "
                                  "Athena++ silently falls back to defaults for keys it does not know.", stacklevel=3)
                    continue
                old = src.blocks[block][key]
                if isinstance(old, float) and isinstance(value, int) and not isinstance(value, bool):
                    value = kv[key] = float(value)  # speed_of_light=100 must stay a float in the athinput
                if old == value and isinstance(old, bool) == isinstance(value, bool):
                    warnings.warn(f"override <{block}> {key} = {value!r} does not change the value", stacklevel=3)

        root = Path(data_root).expanduser() if data_root is not None else self._data_roots()[0]
        rid = self.next_id(extra_roots=[root])
        run_dir = Path(os.path.abspath(root / self.run_dir_format.format(id=rid)))
        clash = [p for r in (*self._data_roots(), root) for i, p in _run_dirs(r) if i == rid]
        if run_dir.exists() or clash:
            where = run_dir if run_dir.exists() else clash[0]
            raise FileExistsError(
                f"{where} already exists but run {rid} is not registered; refusing to touch it. "
                f"Register it with `shearpic runs adopt {where} --purpose ...` first."
            )
        for m in self._run_maps():
            if m.get("data_path") and os.path.abspath(str(m["data_path"])) == str(run_dir):
                raise FileExistsError(f"{run_dir} is already recorded as the directory of run {m.get('id')}")

        name = src.path.name
        if ov:
            with tempfile.TemporaryDirectory(prefix="shearpic-new-run-") as tmp:
                new_inp = write_athinput(src, Path(tmp) / name, overrides=ov)
        else:  # byte-identical copy
            new_inp = src
        new_inp = AthInput(blocks=new_inp.blocks, text=new_inp.text, path=run_dir / name, raw=new_inp.raw)
        changes = diff_athinput(src, new_inp)
        tmpl = None
        if parent_id is None:
            tmpl = {"path": os.path.abspath(src.path), "sha256": src.sha256, "changes": changes}
        rec = self._new_record(rid=rid, purpose=purpose, status="planned", author=author, data_path=run_dir,
                               inp=new_inp, tags=tags, pgen_source=pgen_source, pgen_commit=pgen_commit,
                               parent=parent_id, parent_sha=parent_sha,
                               changes=changes if parent_id is not None else None, template=tmpl, note=note)
        return _NewRunPlan(rec, run_dir, name, new_inp)

    def new_run(self, purpose: str, *, template: str | Path | None = None, parent: int | str | None = None,
                overrides: Mapping[str, Any] | Iterable[str] | None = None, data_root: str | Path | None = None,
                author: str | None = None, tags: Iterable[str] = (), pgen_source: str | None = None,
                pgen_commit: str | None = None, note: str | None = None, dry_run: bool = False) -> RunRecord:
        """Register a new ``planned`` run and create its directory with the athinput and ``run_info.yaml``.

        Parameters
        ----------
        template : path, optional
            Athinput file, or a directory containing one, to start from; it is only read.
        parent : int or str, optional
            Registered run whose athinput is cloned instead; it must match the recorded hash.
        overrides : mapping or iterable of str, optional
            ``{"block/key": value}`` or ``["block/key=value", ...]``; ints given for float keys stay floats.
        data_root : path, optional
            Parent directory of the new run directory; defaults to the first data root.
        dry_run : bool
            Return the record without creating or saving anything.

        Give exactly one of ``template`` and ``parent``.  Raises ``FileExistsError`` if a
        ``run<N>`` directory with the new id already exists; use :meth:`adopt_run` for it.
        """
        purpose = str(purpose or "").strip()
        if not purpose:
            raise ValueError("a purpose is required: say in one sentence what this run is for")
        if (template is None) == (parent is None):
            raise ValueError("give exactly one of template (an athinput file) or parent (a registered run id)")
        ov = parse_overrides(overrides)
        kw = dict(template=template, parent=parent, ov=ov, data_root=data_root, author=author, tags=tags,
                  pgen_source=pgen_source, pgen_commit=pgen_commit, note=note)
        if dry_run:
            return self._plan_new_run(purpose, **kw).rec

        created: Path | None = None
        appended = None
        plan = None
        try:
            with self.transaction():
                plan = self._plan_new_run(purpose, **kw)
                plan.run_dir.parent.mkdir(parents=True, exist_ok=True)
                plan.run_dir.mkdir()  # raises FileExistsError if it appeared in the meantime
                created = plan.run_dir
                (plan.run_dir / plan.name).write_text(plan.inp.text)
                header = (f"# Provenance copy of run {plan.rec.id} as registered on {plan.rec.created}.\n"
                          f"# The registry ({self.path}) is authoritative; this file is not updated afterwards.\n")
                (plan.run_dir / RUN_INFO_NAME).write_text(header + self.dumps(_to_yaml(plan.rec.to_dict())))
                appended = self._append(plan.rec)
                self._dirty = True
        except BaseException:
            if appended is not None and self._txn_depth:  # an outer transaction keeps the document, so undo here
                try:
                    self._doc["runs"].remove(appended)
                except ValueError:
                    pass
            if created is not None and plan is not None:
                for f in (created / plan.name, created / RUN_INFO_NAME):
                    f.unlink(missing_ok=True)
                try:
                    created.rmdir()
                except OSError:
                    pass
            raise
        plan.rec.registry = self
        return plan.rec

    def adopt_run(self, path: str | Path, purpose: str, run_id: int | str | None = None, *,
                  status: str | None = None, author: str | None = None, tags: Iterable[str] = (),
                  pgen_source: str | None = None, pgen_commit: str | None = None, note: str | None = None,
                  save: bool = True) -> RunRecord:
        """Register an existing run directory without writing anything into it.

        The id defaults to the number in a ``run<N>`` directory name, else the next free id.
        ``status`` defaults to ``running`` with a warning if a file was modified within
        :data:`RECENT_SECONDS`, else ``completed`` if a ``*.hst`` exists, else ``planned``.
        """
        purpose = str(purpose or "").strip()
        if not purpose:
            raise ValueError("a purpose is required: say in one sentence what this run was for")
        d = Path(path).expanduser()
        if not d.is_dir():
            raise FileNotFoundError(f"{d} is not a directory")
        d = Path(os.path.abspath(d))
        if status is not None and status not in STATUSES:
            raise ValueError(f"invalid status {status!r}; choose from {', '.join(STATUSES)}")
        found = sorted(d.glob("athinput.*"))
        inp = None
        if found:
            if len(found) > 1:
                warnings.warn(f"several athinput files in {d}; fingerprinting {found[0].name}", stacklevel=2)
            inp = read_athinput(found[0])
        else:
            warnings.warn(f"no athinput.* in {d}: params and hash are not recorded", stacklevel=2)
        if status is None:
            recent = _recent_file(d)
            if recent is not None:
                status = "running"
                warnings.warn(
                    f"{recent[1]} was modified {recent[0] / 60:.0f} minute(s) ago, so the simulation may still be "
                    "running: registering as 'running'. Pass --status completed if it has finished.", stacklevel=2)
            else:
                status = "completed" if any(d.glob("*.hst")) else "planned"
        text = "adopted existing directory"
        if note:
            text += f": {note.strip()}"
        with self._mutation(save):
            if run_id is not None:
                rid = to_run_id(run_id)
            else:
                mm = _RUN_NAME_RE.match(d.name)
                rid = int(mm.group(1)) if mm else self.next_id()
            if self.has_run(rid):
                raise ValueError(f"run id {rid} is already registered "
                                 f"(data_path {self._find_map(rid).get('data_path')}); pass a different id")
            existing = self.find_run_by_path(d)
            if existing is not None:
                raise ValueError(f"{d} is already registered as run {existing.id}")
            rec = self._new_record(rid=rid, purpose=purpose, status=status, author=author, data_path=d, inp=inp,
                                   tags=tags, pgen_source=pgen_source, pgen_commit=pgen_commit, note=text)
            self._append(rec)
        rec.registry = self
        return rec

    def rehash(self, run_id: int | str, note: str, *, save: bool = True) -> dict[str, list[Any]]:
        """Accept an edited athinput by re-recording its hash and params, with a note saying why.

        Returns the parameter changes ``{name: [old, new]}`` and warns unless the run is
        ``planned``.  Only the recorded ``data_path`` is used; see :meth:`move_run`.
        """
        note = str(note or "").strip()
        if not note:
            raise ValueError("explain why the athinput changed (note)")
        with self._mutation(save):
            m = self._find_map(run_id)
            rid = m["id"]
            d = self._checked_run_dir(m, "rehash it", strict=True)
            path = d / str(m["athinput_file"]) if m.get("athinput_file") else self.athinput_path(rid)
            if path is None or not path.exists():
                raise FileNotFoundError(f"run {rid}: athinput not found in {d}")
            if m.get("status") != "planned":
                warnings.warn(f"run {rid} is {m.get('status')}: changing its recorded inputs rewrites history",
                              stacklevel=2)
            inp = read_athinput(path)
            old = _plain(m.get("params") or {})
            new = extract_params(inp)
            changes = {k: [old.get(k), new.get(k)] for k in list(new) + [k for k in old if k not in new]
                       if old.get(k) != new.get(k)}
            m["athinput_sha256"] = inp.sha256
            m["athinput_file"] = path.name
            m["params"] = _to_yaml(new)
            if m.get("parent") is not None and self.has_run(m["parent"]):
                ppath = self.athinput_path(m["parent"])
                if ppath is not None and ppath.exists():
                    m["changes_from_parent"] = _to_yaml(diff_athinput(read_athinput(ppath), inp))
            summary = ", ".join(f"{k}: {a} -> {b}" for k, (a, b) in changes.items()) or "no parameter changes"
            self.add_note(rid, f"athinput re-registered ({summary}): {note}", save=False)
        return changes

    def move_run(self, run_id: int | str, new_path: str | Path, *, force: bool = False, note: str | None = None,
                 save: bool = True) -> RunRecord:
        """Record ``new_path`` as the run's ``data_path``; nothing is moved on disk.

        The athinput there must match the recorded hash unless ``force``, in which case the
        note marks the location as unverified.
        """
        new = Path(new_path).expanduser()
        if not new.is_dir():
            raise FileNotFoundError(f"{new} is not a directory")
        new = Path(os.path.abspath(new))
        with self._mutation(save):
            m = self._find_map(run_id)
            rid = m["id"]
            for other in self._run_maps():
                if other is not m and other.get("data_path") and _same_path(Path(str(other["data_path"])), new):
                    raise ValueError(f"{new} is already the directory of run {other.get('id')}")
            sha, name = m.get("athinput_sha256"), m.get("athinput_file")
            if not (sha and name):
                problem = "the record has no athinput hash to compare"
            elif not (new / str(name)).is_file():
                problem = f"{name} is missing from {new}"
            elif _athinput_sha(new / str(name)) != sha:
                problem = f"{name} in {new} differs from the registered one (sha256 {str(sha)[:12]}... recorded)"
            else:
                problem = None
            if problem and not force:
                raise ValueError(f"run {rid}: refusing to record {new} as its directory: {problem}. Check that this "
                                 "is the right directory; use force / --force to record it anyway.")
            if problem:
                warnings.warn(f"run {rid}: recording {new} although {problem}", stacklevel=2)
            old = m.get("data_path")
            m["data_path"] = str(new)
            text = f"moved: data_path {old} -> {new}"
            if problem:
                text += f" (unverified: {problem})"
            if note:
                text += f": {str(note).strip()}"
            self.add_note(rid, text, save=False)
        return self.get_run(rid)

    # -------------------------------------------------------- lock / unlock
    def lock_run(self, run_id: int | str, *, force: bool = False, save: bool = True) -> int:
        """Remove write permission from every file and directory of a run; returns the number changed.

        Unless ``force``, refuses runs that are planned, submitted or running and directories
        with a file modified within :data:`RECENT_SECONDS`.  If some paths cannot be changed
        the record is marked ``locked: partial``.
        """
        with self._mutation(save):
            m = self._find_map(run_id)
            rid = m["id"]
            status = m.get("status")
            if status in ("submitted", "running") and not force:
                raise ValueError(f"run {rid} is {status}; Athena++ still needs to write there (use force / --force)")
            if status == "planned" and not force:
                raise ValueError(f"run {rid} is planned; lock a run after it has finished (use force / --force)")
            d = self._checked_run_dir(m, "lock it")
            if not force:
                recent = _recent_file(d)
                if recent is not None:
                    raise ValueError(
                        f"run {rid}: {recent[1]} was modified {recent[0] / 60:.0f} minute(s) ago; the simulation may "
                        "still be writing. Wait until nothing has changed for an hour, or use force / --force.")
            n, failures = _chmod_tree(d, remove=stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
            self._record_chmod(m, "locked", failures, "write permission removed from the run directory",
                               "made read-only")
        return n

    def unlock_run(self, run_id: int | str, *, save: bool = True) -> int:
        """Restore owner write permission recursively; returns the number of paths changed."""
        with self._mutation(save):
            m = self._find_map(run_id)
            d = self._checked_run_dir(m, "unlock it")
            n, failures = _chmod_tree(d, add=stat.S_IWUSR)
            self._record_chmod(m, "unlocked", failures, "owner write permission restored", "made writable")
        return n

    def _record_chmod(self, m: CommentedMap, verb: str, failures: list[tuple[str, str]], ok_text: str,
                      what: str) -> None:
        rid = m["id"]
        if failures:
            first = "; ".join(f"{p} ({why})" for p, why in failures[:3])
            more = f" and {len(failures) - 3} more" if len(failures) > 3 else ""
            text = f"{verb} partially: {len(failures)} path(s) could not be {what}: {first}{more}"
            m["locked"] = "partial"
            warnings.warn(f"run {rid}: {text}", stacklevel=3)
        else:
            text = f"{verb}: {ok_text}"
            m["locked"] = verb == "locked"
        self.add_note(rid, text, save=False)

    # ----------------------------------------------------------------- check
    def check(self, data_roots: Sequence[str | Path] | None = None, stale_days: float = STALE_DAYS) -> list[Problem]:
        """Check the registry against the file system.

        Errors cover bad or duplicate ids, empty purposes, invalid statuses, missing or changed
        athinputs, missing run directories, ``completed`` runs without ``*.hst``, references to
        unknown runs and unlabelled experiment entries.  Warnings cover unregistered ``run<N>``
        directories, a missing ``data_path`` with a same-name directory under a data root,
        archived runs without data, submitted or running runs not updated for ``stale_days``,
        and inconsistent superseded or locked states.
        """
        import datetime as _dt

        roots = self._data_roots(data_roots)
        problems: list[Problem] = []
        err = lambda rid, msg: problems.append(Problem("error", rid, msg))  # noqa: E731
        warn = lambda rid, msg: problems.append(Problem("warning", rid, msg))  # noqa: E731

        seen: dict[int, int] = {}
        for i, m in enumerate(self._doc["runs"]):
            if not isinstance(m, Mapping):
                err(None, f"runs[{i}] is not a mapping")
                continue
            rid = m.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                err(None, f"runs[{i}] has a missing or non-integer id ({rid!r})")
                continue
            seen[rid] = seen.get(rid, 0) + 1
        for rid, count in seen.items():
            if count > 1:
                err(rid, f"id {rid} is used by {count} records")
        known = set(seen)

        registered_dirs: set[Path] = set()
        now = _dt.datetime.now().astimezone()
        for m in self._run_maps():
            rid = m.get("id")
            if not isinstance(rid, int) or isinstance(rid, bool):
                continue
            if not str(m.get("purpose") or "").strip():
                err(rid, "empty purpose")
            status = m.get("status")
            if status not in STATUSES:
                err(rid, f"invalid status {status!r} (choose from {', '.join(STATUSES)})")
            for key in ("parent", "superseded_by"):
                ref = m.get(key)
                if ref is not None and ref not in known:
                    err(rid, f"{key} references unknown run {ref}")
            if status == "superseded" and m.get("superseded_by") is None:
                warn(rid, "status superseded but superseded_by is not set")

            d = self._resolve_dir(m, roots)
            recorded = self._recorded_dir(m)
            if not d.is_dir():
                if status == "archived":
                    warn(rid, f"run directory {d} not found (archived; `shearpic runs move {rid} NEW_PATH` if it moved)")
                else:
                    err(rid, f"run directory {d} does not exist (`shearpic runs move {rid} NEW_PATH` if it moved)")
            else:
                if recorded is not None and not _same_path(d, recorded):
                    warn(rid, f"recorded directory {recorded} does not exist; using {d} (same name under a data "
                              f"root). Record it with `shearpic runs move {rid} {d}`")
                registered_dirs.add(d.resolve())
                sha = m.get("athinput_sha256")
                name = m.get("athinput_file")
                if sha and name:
                    p = d / str(name)
                    if not p.exists():
                        err(rid, f"athinput {name} missing from {d}")
                    else:
                        now_sha = _athinput_sha(p)
                        if now_sha != sha:
                            err(rid, f"{name} changed after registration (sha256 {str(sha)[:12]}... recorded, "
                                     f"{now_sha[:12]}... now); if intended: `shearpic runs rehash {rid} --note ...`")
                if status == "completed" and not any(d.glob("*.hst")):
                    err(rid, f"status completed but no *.hst file in {d}")
                if m.get("locked") == "partial":
                    warn(rid, "only partially locked (some paths could not be made read-only; see its notes)")
                elif m.get("locked") and os.access(d, os.W_OK):
                    warn(rid, "marked locked but the directory is writable (run `shearpic runs lock` again)")

            if status in ("submitted", "running"):
                updated = _parse_time(m.get("updated"))
                if updated is not None and (now - updated).total_seconds() > stale_days * 86400:
                    warn(rid, f"{status} but not updated for {(now - updated).days} days "
                              "(check the job and `shearpic runs set-status`)")

        for name, e in self._doc["experiments"].items():
            if not isinstance(e, Mapping):
                err(None, f"experiment {name!r} is not a mapping")
                continue
            if not str(e.get("description") or "").strip():
                warn(None, f"experiment {name!r} has no description")
            for j, item in enumerate(e.get("runs") or []):
                if not isinstance(item, Mapping) or item.get("run") is None:
                    err(None, f"experiment {name!r} entry {j} has no run id")
                    continue
                try:
                    ref = to_run_id(item["run"])
                except ValueError:
                    ref = item["run"]
                if ref not in known:
                    err(None, f"experiment {name!r} references unknown run {ref}")
                if not str(item.get("label") or "").strip():
                    err(None, f"experiment {name!r} entry for run {ref} has no label")

        for root in roots:
            for num, cand in _run_dirs(root):
                if cand.resolve() in registered_dirs:
                    continue
                if num in known:
                    rec = next((m for m in self._run_maps() if m.get("id") == num), {})
                    warn(num, f"{cand} has the number of registered run {num}, which is recorded at "
                              f"{rec.get('data_path')} (stale copy after a move, or a different zero padding?): "
                              f"delete it, or `shearpic runs move {num} {cand}` if it is the real run")
                else:
                    warn(num, f"{cand} is not registered (`shearpic runs adopt {cand} --purpose ...`)")
        return problems


def _parse_time(value: Any):
    import datetime as _dt

    if value is None:
        return None
    try:
        if isinstance(value, _dt.datetime):
            t = value
        elif isinstance(value, _dt.date):
            t = _dt.datetime(value.year, value.month, value.day)
        else:
            t = _dt.datetime.fromisoformat(str(value))
    except ValueError:
        return None
    return t.astimezone() if t.tzinfo is None else t


def _chmod_tree(root: Path, *, remove: int = 0, add: int = 0) -> tuple[int, list[tuple[str, str]]]:
    """Set ``mode = (mode & ~remove) | add`` on ``root`` and everything below it, skipping symlinks.

    Failures do not stop the walk.  Returns ``(number changed, [(path, reason), ...])``.
    """
    changed = 0
    failures: list[tuple[str, str]] = []

    def apply(p: str) -> None:
        nonlocal changed
        try:
            if os.path.islink(p):
                return
            mode = stat.S_IMODE(os.lstat(p).st_mode)
            new = (mode & ~remove) | add
            if new != mode:
                os.chmod(p, new)
                changed += 1
        except OSError as err:
            failures.append((p, err.strerror or str(err)))

    def onerror(err: OSError) -> None:
        failures.append((str(err.filename or root), err.strerror or str(err)))

    if add:  # parents first, in case the added bits are needed to reach their children
        apply(str(root))
    for dirpath, dirnames, filenames in os.walk(root, onerror=onerror):
        for name in dirnames + filenames:
            apply(os.path.join(dirpath, name))
    if remove:
        apply(str(root))
    return changed, failures


# ======================================================= module-level helpers
def load_registry(path: str | Path | None = None, create: bool = False) -> Registry:
    """Load the run registry; see :meth:`Registry.load`."""
    return Registry.load(path, create=create)


def init_registry(path: str | Path | None = None) -> Registry:
    """Create an empty registry file; ``FileExistsError`` if it already exists."""
    p = registry_path(path)
    if p.exists():
        raise FileExistsError(f"registry {p} already exists")
    return Registry.load(p, create=True)


def _as_registry(registry: Registry | str | Path | None) -> Registry:
    return registry if isinstance(registry, Registry) else Registry.load(registry)


def new_run(purpose: str, *, registry: Registry | str | Path | None = None, **kw) -> RunRecord:
    """Load the registry and call :meth:`Registry.new_run`."""
    return _as_registry(registry).new_run(purpose, **kw)


def adopt_run(path: str | Path, purpose: str, run_id: int | str | None = None, *,
              registry: Registry | str | Path | None = None, **kw) -> RunRecord:
    """Load the registry and call :meth:`Registry.adopt_run`."""
    return _as_registry(registry).adopt_run(path, purpose, run_id, **kw)


def set_status(run_id: int | str, status: str, note: str | None = None, *,
               registry: Registry | str | Path | None = None, **kw) -> RunRecord:
    """Load the registry and call :meth:`Registry.set_status`."""
    return _as_registry(registry).set_status(run_id, status, note, **kw)


def lock_run(run_id: int | str, *, registry: Registry | str | Path | None = None, **kw) -> int:
    """Load the registry and call :meth:`Registry.lock_run`."""
    return _as_registry(registry).lock_run(run_id, **kw)


def unlock_run(run_id: int | str, *, registry: Registry | str | Path | None = None, **kw) -> int:
    """Load the registry and call :meth:`Registry.unlock_run`."""
    return _as_registry(registry).unlock_run(run_id, **kw)


def move_run(run_id: int | str, new_path: str | Path, *, registry: Registry | str | Path | None = None,
             **kw) -> RunRecord:
    """Load the registry and call :meth:`Registry.move_run`."""
    return _as_registry(registry).move_run(run_id, new_path, **kw)


def check(registry: Registry | str | Path | None = None, **kw) -> list[Problem]:
    """Load the registry and call :meth:`Registry.check`."""
    return _as_registry(registry).check(**kw)
