"""The ``shearpic`` command-line tool for inspecting runs and editing the run registry.

::

    shearpic env                                   machine, CPUs, data roots and registry file
    shearpic info RUN                              parameters, derived scales and outputs of a run
    shearpic runs init                             create an empty runs.yaml
    shearpic runs new --purpose TEXT --template athinput.kh_org [--set problem/M_A=20 ...]
    shearpic runs new --purpose TEXT --from 3 --set particles/speed_of_light=100
    shearpic runs list [--status running] [--tag cscan]
    shearpic runs show ID
    shearpic runs set-status ID submitted --job-id 1234567 [--partition skx --nodes 4]
    shearpic runs note ID "restarted from out3.00004.rst after node failure"
    shearpic runs lock ID [--force] | unlock ID    make the run directory read-only or writable
    shearpic runs adopt PATH --purpose TEXT        register an existing directory
    shearpic runs move ID NEW_PATH [--force]       record a new run directory location
    shearpic runs rehash ID --note TEXT            accept an edited athinput
    shearpic runs check                            consistency check
    shearpic exp list | show NAME
    shearpic exp add NAME --description D [--question Q] --run 1:'$c=50$':C0:- --run 2:'$c=100$':C1:--

``--registry PATH`` selects the registry file.  The exit code is 1 when ``runs check`` finds
errors and 2 for usage or user errors, which are printed without a traceback.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

__all__ = ["main", "build_parser", "parse_run_spec"]

_USER_ERRORS = (KeyError, ValueError, FileExistsError, FileNotFoundError, PermissionError)


# ================================================================== helpers
def _out(text: str = "") -> None:
    print(text, file=sys.stdout)


def _err(text: str) -> None:
    print(text, file=sys.stderr)


def _fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _truncate(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _table(rows: list[list[str]], header: list[str]) -> str:
    widths = [max(len(r[i]) for r in [header, *rows]) for i in range(len(header))]
    line = lambda r: "  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip()  # noqa: E731
    return "\n".join([line(header), line(["-" * w for w in widths]), *map(line, rows)])


def _registry(args):
    from .registry import load_registry

    return load_registry(getattr(args, "registry", None))


def parse_run_spec(text: str) -> dict[str, Any]:
    """Parse ``ID:LABEL[:COLOR[:LINESTYLE]]`` from ``exp add --run``.

    Colours with a ``tab:`` or ``xkcd:`` prefix are kept whole and the rest is the line style,
    so ``3:ref:tab:red::`` gives a dotted line.  Labels cannot contain ``:``.
    """
    from .registry import to_run_id

    parts = text.split(":", 2)
    if len(parts) < 2 or not parts[1].strip():
        raise ValueError(f"--run {text!r} must look like ID:LABEL[:COLOR[:LINESTYLE]]")
    run = to_run_id(parts[0])
    label = parts[1]
    color = linestyle = None
    rest = parts[2] if len(parts) == 3 else ""
    if rest:
        m = re.match(r"^((?:tab|xkcd):[^:]*|[^:]*)(?::(.*))?$", rest, flags=re.S)
        color = m.group(1) or None
        linestyle = m.group(2) or None
    return {"run": run, "label": label, "color": color, "linestyle": linestyle}


def _print_changes(changes: dict[str, list[Any]], indent: str = "    ") -> None:
    exact = lambda v: repr(v) if isinstance(v, float) else _fmt(v)  # noqa: E731  show 100.0 and 100 differently
    for key, (old, new) in changes.items():
        _out(f"{indent}{key}: {exact(old)} -> {exact(new)}")


# ================================================================= commands
def cmd_env(args) -> int:
    from .env import detect_environment
    from .registry import RegistryError, registry_path

    _out(detect_environment().describe())
    try:
        p = registry_path(getattr(args, "registry", None))
    except RegistryError as err:
        _out(f"registry     : not found -- {err}")
    else:
        state = "exists" if p.is_file() else "MISSING: create it with `shearpic runs init`"
        _out(f"registry     : {p} ({state})")
    return 0


def _resolve_info_target(args):
    """``(run_dir, record or None)`` for ``shearpic info RUN``."""
    from .env import resolve_run
    from .registry import RegistryError, to_run_id

    reg = None
    try:
        reg = _registry(args)
    except RegistryError as err:
        _err(f"warning: registry not usable ({err})")
    try:
        rid = to_run_id(args.run)
    except ValueError:
        rid = None
    if rid is not None and reg is not None and reg.has_run(rid):
        d = reg.run_dir(rid)
        if not d.is_dir():
            raise FileNotFoundError(f"run {rid} is registered but its directory {d} does not exist")
        return d, reg.get_run(rid)
    p = Path(args.run).expanduser()
    d = p if p.is_dir() and (rid is None or not str(args.run).isdigit()) else resolve_run(args.run)
    rec = reg.find_run_by_path(d) if reg is not None else None
    return d, rec


def _last_hst_time(path: Path) -> float | None:
    """Time of the last data row of a history file, reading only its tail."""
    with open(path, "rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - 16384))
        tail = fh.read().decode(errors="replace").splitlines()
    for line in reversed(tail):
        s = line.strip()
        if s and not s.startswith("#"):
            try:
                return float(s.split()[0])
            except ValueError:
                return None
    return None


def _output_lines(run_dir: Path, cfg) -> list[str]:
    from .io.athdf import list_snapshots, snapshot_number
    from .io.particles import list_particle_files
    from .io.trajectory import find_trajectory_files

    lines = []
    streams: dict[str, list[int]] = {}
    for p in list_snapshots(run_dir):
        stream = p.name.rsplit(".", 2)[0].rsplit(".", 1)[-1]
        streams.setdefault(stream, []).append(snapshot_number(p))
    if not streams:
        lines.append("athdf        : none")
    for stream, nums in sorted(streams.items()):
        m = re.fullmatch(r"out(\d+)", stream)
        dt = None
        if m and cfg is not None:
            block = cfg.athinput.blocks.get(f"output{m.group(1)}", {})
            dt = block.get("dt")
        text = f"athdf {stream:<7}: {len(nums)} snapshots (#{min(nums):05d}-#{max(nums):05d})"
        if isinstance(dt, (int, float)) and not isinstance(dt, bool):
            text += f", t = {min(nums) * dt:g}-{max(nums) * dt:g} (number x dt={dt:g})"
        lines.append(text)

    hst = sorted(run_dir.glob("*.hst"))
    if hst:
        for h in hst:
            try:
                t_last = _last_hst_time(h)
            except OSError:
                t_last = None
            lines.append(f"history      : {h.name} (last t = {_fmt(t_last)})")
    else:
        lines.append("history      : none")
    traj = find_trajectory_files(run_dir)
    lines.append(f"trajectories : {len(traj)} file(s)"
                 + (f" in {sorted({str(p.parent.relative_to(run_dir)) for p in traj})}" if traj else ""))
    par = list_particle_files(run_dir, warn_mixed=False)
    kinds: dict[str, set[str]] = {}
    for p in par:
        if (m := re.search(r"\.(\d{5})\.par\.(tab|bin)$", p.name)):
            kinds.setdefault(m.group(1), set()).add(m.group(2))
    lines.append(f"particles    : {len(par)} file(s)" + (f" from {len(kinds)} output(s)" if par else ""))
    both = sorted(num for num, k in kinds.items() if len(k) > 1)
    if both:
        lines.append(f"               output(s) {', '.join(both)} exist in both .par.tab and .par.bin "
                     "(choose one format when reading)")
    rst = sorted(run_dir.glob("*.rst"))
    lines.append(f"restarts     : {len(rst)} file(s)" + (f", latest {rst[-1].name}" if rst else ""))
    return lines


def cmd_info(args) -> int:
    from .config import RunConfig

    run_dir, rec = _resolve_info_target(args)
    _out(f"run directory: {run_dir}")
    if rec is not None:
        _out(f"registered   : run {rec.id} [{rec.status}] {rec.purpose}")
    else:
        _out("registered   : no (`shearpic runs adopt PATH --purpose ...` to register)")
    cfg = None
    try:
        if rec is not None and rec.athinput_file and (run_dir / rec.athinput_file).exists():
            cfg = RunConfig.from_athinput(run_dir / rec.athinput_file, run_dir=run_dir)
        else:
            cfg = RunConfig.from_run_dir(run_dir)
    except FileNotFoundError as err:
        _err(f"warning: {err}")
    if cfg is not None:
        _out("")
        _out(cfg.describe())
    _out("")
    _out("available outputs")
    for line in _output_lines(run_dir, cfg):
        _out(f"  {line}")
    return 0


def cmd_runs_init(args) -> int:
    from .registry import init_registry, load_registry, registry_path

    p = registry_path(getattr(args, "registry", None))
    if p.exists():
        reg = load_registry(p)
        _out(f"registry {p} already exists ({len(reg.runs)} run(s)); nothing changed")
        return 0
    reg = init_registry(p)
    _out(f"created empty registry {reg.path}")
    _out("next: `shearpic runs new --purpose ... --template athinput...` or `shearpic runs adopt PATH --purpose ...`")
    return 0


def cmd_runs_new(args) -> int:
    reg = _registry(args)
    rec = reg.new_run(
        args.purpose, template=args.template, parent=args.parent, overrides=args.set or None,
        data_root=args.data_root, author=args.author, tags=args.tag or (), pgen_source=args.pgen_source,
        pgen_commit=args.pgen_commit, note=args.note, dry_run=args.dry_run,
    )
    verb = "would create" if args.dry_run else "created"
    _out(f"{verb} run {rec.id} [{rec.status}] at {rec.data_path}")
    _out(f"  athinput : {rec.athinput_file} (sha256 {rec.athinput_sha256[:12]}...)")
    if rec.parent is not None:
        _out(f"  changes from parent run {rec.parent}:" + ("" if rec.changes_from_parent else " none"))
        _print_changes(rec.changes_from_parent or {})
    elif rec.template:
        _out(f"  template : {rec.template['path']}")
        if rec.template.get("changes"):
            _out("  changes from template:")
            _print_changes(rec.template["changes"])
    if args.dry_run:
        _out("dry run: nothing was written")
    else:
        _out(f"next: submit the job from {rec.data_path}, then "
             f"`shearpic runs set-status {rec.id} submitted --job-id JOBID`")
    return 0


def cmd_runs_list(args) -> int:
    reg = _registry(args)
    recs = reg.runs
    if args.status:
        recs = [r for r in recs if r.status == args.status]
    if args.tag:
        recs = [r for r in recs if args.tag in (r.tags or [])]
    if not recs:
        if not reg.exists:
            _out(f"no runs registered: {reg.path} does not exist (create it with `shearpic runs init`)")
        else:
            _out(f"no runs match in {reg.path}" if reg.runs else f"no runs registered in {reg.path}")
        return 0
    rows = []
    for r in sorted(recs, key=lambda r: (r.id if isinstance(r.id, int) else -1)):
        p = r.params or {}
        rows.append([_fmt(r.id), _fmt(r.status), (r.created or "-")[:10], _fmt(p.get("c")), _fmt(p.get("q_mc")),
                     _fmt(p.get("M_A")), _fmt(p.get("shear_strength")), _lock_mark(r.locked)
                     + _truncate(r.purpose, 60)])
    _out(_table(rows, ["id", "status", "created", "c", "q_mc", "M_A", "S", "purpose"]))
    return 0


def cmd_runs_show(args) -> int:
    reg = _registry(args)
    rec = reg.get_run(args.id)
    _out(reg.run_record_yaml(rec.id).rstrip())
    d = reg.run_dir(rec.id)
    if not d.is_dir():
        state = f"MISSING; `shearpic runs move {rec.id} NEW_PATH` if it moved"
    elif rec.data_path and Path(rec.data_path).is_absolute() and not Path(rec.data_path).is_dir():
        state = f"exists; the recorded data_path does not -- `shearpic runs move {rec.id} {d}` to record it"
    else:
        state = "exists"
    _out(f"# resolved run directory: {d} ({state})")
    return 0


def _lock_mark(locked: Any) -> str:
    return "L? " if locked == "partial" else ("L " if locked else "")


# The commands below run inside reg.transaction() so that what they print is what was saved.
def cmd_runs_set_status(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        old = reg.get_run(args.id).status
        new = reg.set_status(args.id, args.status, args.note, job_id=args.job_id, partition=args.partition,
                             nodes=args.nodes, superseded_by=args.superseded_by)
    _out(f"run {new.id}: {old} -> {new.status}")
    return 0


def cmd_runs_note(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        reg.add_note(args.id, args.text)
        rid = reg.get_run(args.id).id
    _out(f"run {rid}: note added")
    return 0


def cmd_runs_lock(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        n = reg.lock_run(args.id, force=args.force)
        rec, d = reg.get_run(args.id), reg.run_dir(args.id)
    state = "partially locked" if rec.locked == "partial" else "locked"
    _out(f"run {rec.id}: {state} {d} ({n} path(s) made read-only)")
    return 0


def cmd_runs_unlock(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        n = reg.unlock_run(args.id)
        rec, d = reg.get_run(args.id), reg.run_dir(args.id)
    state = "partially unlocked" if rec.locked == "partial" else "unlocked"
    _out(f"run {rec.id}: {state} {d} ({n} path(s) made writable)")
    return 0


def cmd_runs_adopt(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        rec = reg.adopt_run(args.path, args.purpose, args.id, status=args.status, author=args.author,
                            tags=args.tag or (), pgen_source=args.pgen_source, pgen_commit=args.pgen_commit,
                            note=args.note)
    _out(f"adopted {rec.data_path} as run {rec.id} [{rec.status}]")
    return 0


def cmd_runs_move(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        old = reg.get_run(args.id).data_path
        rec = reg.move_run(args.id, args.path, force=args.force, note=args.note)
    _out(f"run {rec.id}: data_path {old} -> {rec.data_path}")
    return 0


def cmd_runs_rehash(args) -> int:
    reg = _registry(args)
    with reg.transaction():
        changes = reg.rehash(args.id, args.note)
        rid = reg.get_run(args.id).id
    _out(f"run {rid}: athinput re-registered")
    _print_changes(changes)
    return 0


def cmd_runs_check(args) -> int:
    reg = _registry(args)
    problems = reg.check()
    for p in problems:
        _out(str(p))
    n_err = sum(p.level == "error" for p in problems)
    n_warn = len(problems) - n_err
    _out(f"{reg.path}: {len(reg.runs)} run(s), {n_err} error(s), {n_warn} warning(s)")
    return 1 if n_err else 0


def cmd_exp_list(args) -> int:
    reg = _registry(args)
    names = reg.experiment_names
    if not names:
        _out(f"no experiments in {reg.path}")
        return 0
    rows = []
    for name in names:
        e = reg.experiment_data(name)
        ids = [str(item.get("run")) for item in (e.get("runs") or []) if isinstance(item, dict)]
        rows.append([name, ",".join(ids), _truncate(e.get("description") or "", 60)])
    _out(_table(rows, ["name", "runs", "description"]))
    return 0


def cmd_exp_show(args) -> int:
    reg = _registry(args)
    e = reg.experiment(args.name)
    _out(f"experiment : {e.name}")
    _out(f"description: {e.description}")
    _out(f"question   : {e.question or '-'}")
    rows = [[str(r.run), r.record.status, r.label, _fmt(r.color), _fmt(r.linestyle), _truncate(r.record.purpose, 50)]
            for r in e]
    _out(_table(rows, ["run", "status", "label", "color", "linestyle", "purpose"]))
    if e.figures:
        _out("figures    : " + ", ".join(e.figures))
    return 0


def cmd_exp_add(args) -> int:
    reg = _registry(args)
    runs = [parse_run_spec(s) for s in args.run]
    with reg.transaction():
        e = reg.add_experiment(args.name, args.description, runs, question=args.question,
                               figures=args.figure or (), replace=args.replace)
    _out(f"experiment {e.name!r} saved with runs {', '.join(map(str, e.run_ids))}")
    return 0


# =================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--registry", metavar="PATH", default=argparse.SUPPRESS,
                        help="registry file (default: $PARTICLE_ACCEL_REGISTRY or <repo>/runs.yaml)")

    parser = argparse.ArgumentParser(prog="shearpic", parents=[common],
                                     description="Analysis and run bookkeeping for MHD-PIC shear simulations.")
    sub = parser.add_subparsers(dest="command", metavar="COMMAND")

    p = sub.add_parser("env", parents=[common], help="print the detected machine, CPUs and data paths")
    p.set_defaults(func=cmd_env)

    p = sub.add_parser("info", parents=[common], help="parameters and available outputs of a run")
    p.add_argument("run", metavar="RUN", help="registered id (7, run0007) or a run directory path")
    p.set_defaults(func=cmd_info)

    # ------------------------------------------------------------- runs
    runs = sub.add_parser("runs", parents=[common], help="run registry (runs.yaml)")
    rsub = runs.add_subparsers(dest="runs_command", metavar="ACTION")

    p = rsub.add_parser("init", parents=[common], help="create an empty registry file (once per project)")
    p.set_defaults(func=cmd_runs_init)

    p = rsub.add_parser("new", parents=[common], help="register a new run and create its directory")
    p.add_argument("--purpose", required=True, help="one sentence: what is this run for?")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--template", metavar="PATH", help="athinput file to start from (not modified)")
    src.add_argument("--from", dest="parent", metavar="ID",
                     help="clone the athinput of a registered run (it must match its recorded hash)")
    p.add_argument("--set", action="append", metavar="BLOCK/KEY=VALUE",
                   help="override an athinput value (repeatable), e.g. --set particles/speed_of_light=100")
    p.add_argument("--tag", action="append", metavar="TAG", help="tag (repeatable)")
    p.add_argument("--data-root", metavar="DIR", help="where to create the run directory (default: data root)")
    p.add_argument("--author", help="default: git config user.name or $USER")
    p.add_argument("--pgen-source", metavar="S", help="problem generator file, e.g. src/pgen/kh_driven.cpp")
    p.add_argument("--pgen-commit", metavar="HASH", help="git commit of the Athena++ fork used")
    p.add_argument("--note", help="initial note")
    p.add_argument("--dry-run", action="store_true", help="show what would happen; write nothing")
    p.set_defaults(func=cmd_runs_new)

    p = rsub.add_parser("list", parents=[common], help="table of registered runs")
    p.add_argument("--status", choices=_statuses())
    p.add_argument("--tag")
    p.set_defaults(func=cmd_runs_list)

    p = rsub.add_parser("show", parents=[common], help="full record of one run")
    p.add_argument("id", metavar="ID")
    p.set_defaults(func=cmd_runs_show)

    p = rsub.add_parser("set-status", parents=[common], help="change status (records a dated note)")
    p.add_argument("id", metavar="ID")
    p.add_argument("status", metavar="STATUS", help=" | ".join(_statuses()))
    p.add_argument("--note")
    p.add_argument("--job-id")
    p.add_argument("--partition")
    p.add_argument("--nodes", type=int)
    p.add_argument("--superseded-by", metavar="ID")
    p.set_defaults(func=cmd_runs_set_status)

    p = rsub.add_parser("note", parents=[common], help="append a dated note")
    p.add_argument("id", metavar="ID")
    p.add_argument("text", metavar="TEXT")
    p.set_defaults(func=cmd_runs_note)

    p = rsub.add_parser("lock", parents=[common], help="remove write permission from the run directory")
    p.add_argument("id", metavar="ID")
    p.add_argument("--force", action="store_true",
                   help="lock even if the run is planned/submitted/running or a file changed in the last hour")
    p.set_defaults(func=cmd_runs_lock)

    p = rsub.add_parser("unlock", parents=[common], help="restore owner write permission")
    p.add_argument("id", metavar="ID")
    p.set_defaults(func=cmd_runs_unlock)

    p = rsub.add_parser("adopt", parents=[common], help="register an existing run directory (writes nothing there)")
    p.add_argument("path", metavar="PATH")
    p.add_argument("--purpose", required=True)
    p.add_argument("--id", help="run id (default: from a run<N> directory name, else the next id)")
    p.add_argument("--status", choices=_statuses(),
                   help="default: running if a file changed in the last hour, else completed if a .hst exists, "
                        "else planned")
    p.add_argument("--tag", action="append", metavar="TAG")
    p.add_argument("--author")
    p.add_argument("--pgen-source", metavar="S")
    p.add_argument("--pgen-commit", metavar="HASH")
    p.add_argument("--note")
    p.set_defaults(func=cmd_runs_adopt)

    p = rsub.add_parser("move", parents=[common], help="record a new location of a run directory")
    p.add_argument("id", metavar="ID")
    p.add_argument("path", metavar="NEW_PATH", help="existing directory holding the run (nothing is moved)")
    p.add_argument("--force", action="store_true", help="record it even if its athinput does not match the record")
    p.add_argument("--note", help="why it moved")
    p.set_defaults(func=cmd_runs_move)

    p = rsub.add_parser("rehash", parents=[common], help="accept an intentionally edited athinput")
    p.add_argument("id", metavar="ID")
    p.add_argument("--note", required=True, help="why the input changed")
    p.set_defaults(func=cmd_runs_rehash)

    p = rsub.add_parser("check", parents=[common], help="consistency check (exit 1 on errors)")
    p.set_defaults(func=cmd_runs_check)

    # -------------------------------------------------------------- exp
    exp = sub.add_parser("exp", parents=[common], help="experiments: named groups of runs")
    esub = exp.add_subparsers(dest="exp_command", metavar="ACTION")

    p = esub.add_parser("list", parents=[common], help="list experiments")
    p.set_defaults(func=cmd_exp_list)

    p = esub.add_parser("show", parents=[common], help="show one experiment")
    p.add_argument("name", metavar="NAME")
    p.set_defaults(func=cmd_exp_show)

    p = esub.add_parser("add", parents=[common], help="create an experiment")
    p.add_argument("name", metavar="NAME")
    p.add_argument("--description", required=True)
    p.add_argument("--question")
    p.add_argument("--run", action="append", required=True, metavar="ID:LABEL[:COLOR[:LINESTYLE]]",
                   help="repeatable; e.g. --run 1:'$c=50$':C0:-  --run 2:'$c=100$':tab:red:--")
    p.add_argument("--figure", action="append", metavar="PATH")
    p.add_argument("--replace", action="store_true", help="overwrite an existing experiment")
    p.set_defaults(func=cmd_exp_add)

    parser._shearpic_subparsers = {"runs": runs, "exp": exp}  # help for a bare `shearpic runs`
    return parser


def _statuses() -> list[str]:
    from .registry import STATUSES

    return list(STATUSES)


def _show_warning(message, category, filename, lineno, file=None, line=None) -> None:
    _err(f"warning: {message}")


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point of the ``shearpic`` console script; returns the exit code."""
    from .registry import RegistryError

    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:  # argparse usage errors (2) and --help (0)
        return exc.code if isinstance(exc.code, int) else (0 if exc.code is None else 2)
    func = getattr(args, "func", None)
    if func is None:
        target = parser._shearpic_subparsers.get(args.command, parser)
        target.print_help(sys.stderr)
        return 2
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        warnings.showwarning = _show_warning
        try:
            return int(func(args))
        except RegistryError as err:
            _err(f"error: {err}")
            return 2
        except _USER_ERRORS as err:
            msg = err.args[0] if isinstance(err, KeyError) and err.args else err
            _err(f"error: {msg}")
            return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
