#!/usr/bin/env python
r"""Make Figures 1-7 of arXiv:2512.12720 with one command, and collect the paper's statistics.

Each selected figure is made by its own script, run as a subprocess with the figure's preset
(``--list`` shows the commands); ``--compute`` first runs the compute stages of Figs 1-3,
which read the raw snapshots.  Script output is shown and saved to ``<figures>/logs/<fig>.log``,
and the printed tables and JSON sidecars are collected in ``<figures>/paper_statistics.json``.
Runs are found under ``PARTICLE_ACCEL_DATA`` and products under ``PARTICLE_ACCEL_OUTPUT``;
``--bundle DIR`` points both at a bundle written by ``pack_small_data.py``.  A pre-flight
check reports missing runs, files and products (``--check`` stops after it), and the exit
status is 1 if any figure failed.

Usage::

    python analysis/reproduce_figures.py --bundle ~/particle-accel-small-data --formats pdf,png
    python analysis/reproduce_figures.py --figures 4,7 --out /tmp/figs
    python analysis/reproduce_figures.py --compute --workers 4      # from the full data
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `from common import ...` also when imported with importlib
    sys.path.insert(0, str(HERE))

from common import DEFAULT_PRESETS, load_preset, provenance, write_json  # noqa: E402

__all__ = [
    "FIGURES", "Figure", "FigureInputs", "parse_figures", "figure_inputs", "product_run_name", "run_command",
    "parse_printed_tables", "preflight", "missing_runs", "fatal_problems", "main",
]

# Keys of a figure's JSON sidecar that describe the machine/run of the command, not the figure;
# they are recorded once at the top of paper_statistics.json instead of per figure.
PROVENANCE_KEYS = ("created_utc", "command", "shearpic_version", "repository", "python", "host")


# ============================================================================ figure table
@dataclass(frozen=True)
class Figure:
    """One paper figure: its preset and the argv (after the script name) of its stages."""

    key: str                              # 'fig1'
    number: int
    title: str
    preset: str                           # entry in paper_figures.yaml
    script: str                           # file name in analysis/
    plot: tuple[str, ...]                 # plot stage: products + small files only
    compute: tuple[str, ...] | None       # compute stage that reads raw snapshots (None: nothing to compute locally)
    both: tuple[str, ...]                 # compute + plot
    note: str = ""


FIGURES: dict[str, Figure] = {f.key: f for f in (
    Figure("fig1", 1, "initial and x-averaged shear profiles", "fig1_shear_profile", "shear_profile.py",
           ("profile", "--stage", "plot"), ("profile", "--stage", "compute"), ("profile", "--stage", "all")),
    Figure("fig2", 2, "KHI flow maps with LIC", "fig2_flow_evolution", "shear_profile.py",
           ("flow", "--stage", "plot"), ("flow", "--stage", "compute"), ("flow", "--stage", "all")),
    Figure("fig3", 3, "time-averaged turbulence spectra", "fig3_turbulence", "steady_state.py",
           ("turbulence", "--stage", "plot"), ("turbulence", "--stage", "compute"), ("turbulence", "--stage", "all")),
    Figure("fig4", 4, "energy densities, driven vs decaying", "fig4_steady_state", "steady_state.py",
           ("energy",), None, ("energy",)),
    Figure("fig5", 5, "particle energy spectra", "fig5_particle_spectrum", "plot_spectrum.py",
           (), None, (), note="particle spectra from full particle outputs: energy_spectrum.py on HPC"),
    Figure("fig6", 6, "orbit and phase-space distributions", "fig6_phase_space", "phase_space.py",
           ("plot",), None, ("plot",), note="histograms from full particle outputs: phase_space.py compute on HPC"),
    Figure("fig7", 7, "power delivered to the particles", "fig7_power", "steady_state.py",
           ("power",), None, ("power",)),
)}


def parse_figures(spec: str | Sequence[str] | None) -> list[str]:
    """Figure keys from ``'all'``, ``'1,4,7'``, ``'fig2'``, ``'4-7'`` (or a list of such parts), in figure order."""
    if spec is None:
        return list(FIGURES)
    parts = [spec] if isinstance(spec, str) else list(spec)
    tokens = [t.strip().lower() for p in parts for t in str(p).split(",") if t.strip()]
    wanted: set[int] = set()
    for tok in tokens:
        if tok == "all":
            wanted.update(f.number for f in FIGURES.values())
            continue
        m = re.fullmatch(r"(?:fig)?(\d+)(?:-(?:fig)?(\d+))?", tok)
        if not m:
            raise ValueError(f"cannot parse figure {tok!r}; use e.g. 'all', '1,4,7', 'fig2' or '4-7'")
        lo = int(m.group(1))
        hi = int(m.group(2)) if m.group(2) else lo
        for n in range(lo, hi + 1):
            if f"fig{n}" not in FIGURES:
                raise ValueError(f"there is no figure {n}; available: {', '.join(FIGURES)}")
            wanted.add(n)
    if not wanted:
        raise ValueError("no figures selected")
    return [f.key for f in FIGURES.values() if f.number in wanted]


# ============================================================================ inputs of a figure
@dataclass
class FigureInputs:
    """What a figure's plot stage reads.

    ``runs``: runs whose ``athinput.*`` and ``*.hst`` are needed; ``files``: ``(run, path or glob)``
    inside run directories; ``computed``: ``(run, file name)`` products of a local compute stage;
    ``hpc_products``: ``(run, glob)`` products that only an HPC job can make.
    """

    runs: list[str] = field(default_factory=list)
    files: list[tuple[str, str]] = field(default_factory=list)
    computed: list[tuple[str, str]] = field(default_factory=list)
    hpc_products: list[tuple[str, str]] = field(default_factory=list)


def _unique(items: Iterable[Any]) -> list[Any]:
    out: list[Any] = []
    for x in items:
        if x not in out:
            out.append(x)
    return out


def figure_inputs(key: str, preset: Mapping[str, Any]) -> FigureInputs:
    """Inputs of figure ``key`` for its preset (see :class:`FigureInputs`)."""
    fi = FigureInputs()
    if key in ("fig1", "fig2"):
        from shear_profile import flow_product_path, profile_product_path

        run, output = str(preset["run"]), str(preset.get("output", "out2"))
        times = [float(t) for t in preset["times"]]
        fi.runs = [run]
        if key == "fig1":
            fi.computed = [(run, profile_product_path(Path("."), output, times).name)]
        else:
            fi.computed = [(run, flow_product_path(Path("."), output, t).name) for t in times]
    elif key == "fig3":
        run = str(preset["run"])
        fi.runs = [run]
        # name as written by steady_state.turbulence_product_path (checked in the tests)
        fi.computed = [(run, f"turbulence_spectra.{preset.get('output', 'out2')}.npz")]
    elif key == "fig4":
        fi.runs = _unique([str(preset["driven"]), str(preset["decaying"])])
    elif key == "fig5":
        run = str(preset["run"])
        fi.runs = [run]
        if preset.get("source", "legacy") == "legacy":
            d = preset.get("legacy_dir", "energy_spectrum_data")
            fi.files = [(run, f"{d}/histogram_frame_*.csv"), (run, f"{d}/histogram_metadata.csv"),
                        (run, f"{d}/frame_summary.csv")]
        else:
            fi.hpc_products = [(run, f"spectrum_{preset.get('variable', 'gamma')}_*.npz")]
    elif key == "fig6":
        hp, tp = dict(preset["histogram"]), dict(preset["trajectory"])
        fi.runs = _unique([str(hp["run"]), str(tp["run"])])
        if hp.get("source", "legacy") == "legacy":
            fi.files.append((str(hp["run"]), str(hp.get("legacy_file", "phase_space_histograms.npz"))))
        else:
            fi.hpc_products.append((str(hp["run"]), "phase_space_*.npz"))
        fi.files.append((str(tp["run"]), str(tp["file"])))
    elif key == "fig7":
        fi.runs = [str(preset["run"])]
    else:
        raise KeyError(key)
    return fi


def product_run_name(run: str) -> str:
    """Name of a run's product directory under the output root (``run364`` -> ``run0364``)."""
    from shearpic.env import run_output_dir

    return run_output_dir(Path(run).name, create=False).name


# optional (informational) files: their absence does not make a figure incomplete
OPTIONAL_FILES = ("frame_summary.csv",)


def _is_optional(pattern: str) -> bool:
    return Path(pattern).name in OPTIONAL_FILES


# ============================================================================ pre-flight check
def missing_runs(keys: Sequence[str], presets: Path, data_roots: str | None) -> dict[str, dict[str, Any]]:
    """Runs of the selected figures that cannot be found: ``{run: {"figures": [...], "error": str}}``."""
    from shearpic.env import resolve_run

    out: dict[str, dict[str, Any]] = {}
    for key in keys:
        try:
            fi = figure_inputs(key, load_preset(FIGURES[key].preset, presets))
        except Exception:  # noqa: BLE001 - a broken preset is reported by preflight()
            continue
        for run in _unique([*fi.runs, *(r for r, _ in fi.files)]):
            try:
                resolve_run(run, roots=data_roots) if data_roots else resolve_run(run)
            except (FileNotFoundError, ValueError) as err:
                entry = out.setdefault(run, {"figures": [], "error": str(err)})
                if key not in entry["figures"]:
                    entry["figures"].append(key)
    return out


def fatal_problems(keys: Sequence[str], presets: Path, data_roots: str | None, output_root: Path,
                   figures_dir: Path) -> list[str]:
    """Problems that make running any script pointless: missing presets file, runs or a broken output root."""
    from shearpic.env import detect_environment

    if not Path(presets).is_file():
        return [f"presets file {presets} does not exist"]
    fatal = []
    roots = data_roots.split(os.pathsep) if data_roots else [str(r) for r in detect_environment().data_roots]
    existing = [r for r in roots if r and Path(r).expanduser().is_dir()]
    if not existing:
        fatal.append(f"data root {os.pathsep.join(roots)} does not exist; set PARTICLE_ACCEL_DATA or pass "
                     "--data-root / --bundle")
    else:
        for run, info in missing_runs(keys, presets, data_roots).items():
            fatal.append(f"run {run} (needed by {', '.join(info['figures'])}) not found under "
                         f"{os.pathsep.join(existing)}; set PARTICLE_ACCEL_DATA or pass --data-root / --bundle")
    for what, path in (("output root", Path(output_root)), ("figure directory", Path(figures_dir))):
        blocker = next((p for p in (path, *path.parents) if p.exists()), None)
        if blocker is not None and not blocker.is_dir():
            fatal.append(f"{what} {path} is not a directory ({blocker} is a file)")
        elif blocker is not None and not os.access(blocker, os.W_OK) and not path.is_dir():
            fatal.append(f"{what} {path} cannot be created ({blocker} is not writable)")
    return fatal


def preflight(keys: Sequence[str], presets: Path, data_roots: str | None, output_root: Path, *,
              compute: bool = False, allow_download: bool = False) -> dict[str, list[str]]:
    """Problems per figure (missing runs/files/products, online-only placeholders); empty lists are fine."""
    from shearpic.env import resolve_run
    from shearpic.io._util import is_dataless

    problems: dict[str, list[str]] = {}
    for key in keys:
        fig = FIGURES[key]
        out: list[str] = []
        try:
            preset = load_preset(fig.preset, presets)
            fi = figure_inputs(key, preset)
        except Exception as err:  # noqa: BLE001 - reported, the script will fail the same way
            problems[key] = [f"preset {fig.preset}: {err}"]
            continue
        run_dirs: dict[str, Path] = {}
        for run in _unique([*fi.runs, *(r for r, _ in fi.files)]):
            try:
                run_dirs[run] = resolve_run(run, roots=data_roots) if data_roots else resolve_run(run)
            except (FileNotFoundError, ValueError) as err:
                out.append(f"run {run}: {err}")
        for run in fi.runs:
            d = run_dirs.get(run)
            if d is None:
                continue
            for pattern, what in (("athinput.*", "athinput"), ("*.hst", "history file")):
                found = sorted(d.glob(pattern))
                if not found:
                    out.append(f"{run}: no {what} ({pattern})" + (" -- needed for S" if what != "athinput" else ""))
                out += [f"{p}: online-only placeholder" for p in found if is_dataless(p) and not allow_download]
        for run, pattern in fi.files:
            d = run_dirs.get(run)
            if d is None:
                continue
            found = sorted(d.glob(pattern))
            if not found and not _is_optional(pattern):
                out.append(f"{run}: missing {pattern}")
            out += [f"{p}: online-only placeholder" for p in found if is_dataless(p) and not allow_download]
        if not compute:
            for run, name in fi.computed:
                p = output_root / product_run_name(run) / "products" / name
                if not p.is_file():
                    out.append(f"missing product {p} (run pack_small_data.py, or --compute with the raw snapshots)")
        for run, pattern in fi.hpc_products:
            d = output_root / product_run_name(run) / "products"
            if not sorted(d.glob(pattern)):
                out.append(f"no HPC products {d / pattern} (see {fig.note or 'the script docstring'})")
        problems[key] = out
    return problems


# ============================================================================ running scripts
def run_command(cmd: Sequence[str], *, env: Mapping[str, str] | None = None, log: Path | None = None,
                echo: bool = True, timeout: float | None = None) -> tuple[int, str]:
    """Run ``cmd``, echo its merged stdout/stderr line by line, save it to ``log``; return (exit code, text)."""
    lines: list[str] = []
    fh = None
    if log is not None:
        log.parent.mkdir(parents=True, exist_ok=True)
        fh = open(log, "w", encoding="utf-8")
        fh.write("$ " + shlex.join(cmd) + "\n")
    timer = None
    killed = threading.Event()
    try:
        proc = subprocess.Popen(list(cmd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
                                env=dict(env) if env is not None else None, errors="replace")
        if timeout is not None:
            def _kill() -> None:
                killed.set()
                proc.kill()

            timer = threading.Timer(timeout, _kill)
            timer.start()
        assert proc.stdout is not None
        for line in proc.stdout:
            lines.append(line)
            if echo:
                sys.stdout.write(line)
                sys.stdout.flush()
            if fh:
                fh.write(line)
        code = proc.wait()
        if killed.is_set():
            msg = f"\n[reproduce_figures] killed after {timeout:g} s\n"
            lines.append(msg)
            if fh:
                fh.write(msg)
            code = code or -9
    except OSError as err:
        lines.append(f"[reproduce_figures] could not start {cmd[0]}: {err}\n")
        code = 127
    finally:
        if timer is not None:
            timer.cancel()
        if fh:
            fh.close()
    return code, "".join(lines)


_NUMBER = re.compile(r"^[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?$")


def _value(text: str) -> Any:
    text = text.strip()
    if _NUMBER.match(text):
        try:
            return int(text) if re.fullmatch(r"[+-]?\d+", text) else float(text)
        except ValueError:  # pragma: no cover
            return text
    return text


def parse_printed_tables(text: str) -> dict[str, dict[str, Any]]:
    """Tables printed by ``common.print_table``: ``{title: {key: value}}`` (numbers converted).

    A table is an empty line, an unindented title line and rows ``'  <key padded> : <value>'``.
    The key/value separator is the ``' : '`` column shared by all rows of the table, so keys
    or values may themselves contain ``' : '``.  Repeated titles get a suffix `` (2)``, ...
    """
    lines = text.splitlines()
    tables: dict[str, dict[str, Any]] = {}
    i = 0
    while i < len(lines) - 1:
        title = lines[i + 1] if i + 1 < len(lines) else ""
        is_start = (lines[i].strip() == "" and title.strip() and not title.startswith(" ")
                    and i + 2 < len(lines) and lines[i + 2].startswith("  ") and " : " in lines[i + 2])
        if not is_start:
            i += 1
            continue
        rows = []
        j = i + 2
        while j < len(lines) and lines[j].startswith("  ") and " : " in lines[j]:
            rows.append(lines[j][2:])
            j += 1
        common: set[int] | None = None
        for r in rows:
            cols = {m.start() for m in re.finditer(r"(?= : )", r)}
            common = cols if common is None else common & cols
        sep = min(common) if common else None
        table: dict[str, Any] = {}
        for r in rows:
            k, v = (r[:sep], r[sep + 3:]) if sep is not None else r.split(" : ", 1)
            table[k.rstrip()] = _value(v)
        name, n = title.strip(), 2
        while name in tables:
            name = f"{title.strip()} ({n})"
            n += 1
        tables[name] = table
        i = j
    return tables


def _sidecar(figures_dir: Path, preset: Mapping[str, Any], started: float) -> tuple[Path | None, dict[str, Any]]:
    name = preset.get("figure")
    if not name:
        return None, {}
    path = figures_dir / f"{name}.json"
    if not path.is_file() or path.stat().st_mtime < started - 1.0:
        return path, {}
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return path, {}
    return path, {k: v for k, v in data.items() if k not in PROVENANCE_KEYS}


def build_command(fig: Figure, args: argparse.Namespace, figures_dir: Path) -> list[str]:
    stage = fig.both if args.compute else fig.plot
    cmd = [sys.executable, str(HERE / fig.script), *stage, "--preset", fig.preset, "--presets", str(args.presets),
           "--out", str(figures_dir), "--formats", args.formats, "--dpi", str(args.dpi), "--workers", str(args.workers),
           "--backend", args.backend]
    if args.data_root:
        cmd += ["--data-root", args.data_root]
    if args.allow_download:
        cmd.append("--allow-download")
    return cmd


def child_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    if args.data_root:
        env["PARTICLE_ACCEL_DATA"] = args.data_root
    if args.output_root:
        env["PARTICLE_ACCEL_OUTPUT"] = str(args.output_root)
    env.setdefault("MPLBACKEND", "Agg")
    env["PYTHONUNBUFFERED"] = "1"
    return env


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="reproduce_figures.py", description=__doc__.split("\n\n")[0],
                                epilog="See the module docstring (python -m pydoc analysis/reproduce_figures.py).",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--figures", default="all", help="'all' (default), '1,4,7', 'fig2' or '4-7'")
    p.add_argument("--compute", action="store_true",
                   help="run the compute stages of Figs 1-3 first (needs the raw .athdf snapshots)")
    p.add_argument("--bundle", default=None, help="small-data bundle: --data-root BUNDLE/data --output-root BUNDLE/outputs")
    p.add_argument("--data-root", default=None, help="run directories (default: PARTICLE_ACCEL_DATA)")
    p.add_argument("--output-root", default=None,
                   help="output root holding run0NNN/products (default: PARTICLE_ACCEL_OUTPUT)")
    p.add_argument("--out", default=None, help="figure directory (default: <output root>/paper_figures)")
    p.add_argument("--presets", default=str(DEFAULT_PRESETS), help="presets YAML file (default: %(default)s)")
    p.add_argument("--formats", default="pdf,png", help="figure formats (default: %(default)s)")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--workers", default="auto", help="worker processes of each script (default: auto)")
    p.add_argument("--backend", default="auto", choices=["auto", "serial", "thread", "process", "mpi"])
    p.add_argument("--allow-download", action="store_true", help="let the scripts read online-only placeholders")
    p.add_argument("--timeout", type=float, default=None, help="kill a figure's script after this many seconds")
    p.add_argument("--fail-fast", action="store_true", help="stop at the first failing figure")
    p.add_argument("--check", action="store_true", help="only report missing inputs; run nothing")
    p.add_argument("--list", action="store_true", help="print the figure table and exit")
    p.add_argument("--quiet", action="store_true", help="do not echo the scripts' output (still logged)")
    return p


def _print_rows(title: str, header: Sequence[str], rows: Sequence[Sequence[Any]]) -> None:
    widths = [max(len(str(x)) for x in col) for col in zip(header, *rows)]
    print(f"\n{title}")
    print("  " + "  ".join(str(h).ljust(w) for h, w in zip(header, widths)))
    for r in rows:
        print("  " + "  ".join(str(x).ljust(w) for x, w in zip(r, widths)))


def main(argv: Sequence[str] | None = None, *, runner: Callable[..., tuple[int, str]] = run_command) -> int:
    args = build_parser().parse_args(argv)
    from shearpic.env import detect_environment

    try:
        keys = parse_figures(args.figures)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    args.presets = Path(args.presets).expanduser().resolve()
    if args.bundle:
        bundle = Path(args.bundle).expanduser().resolve()
        args.data_root = args.data_root or str(bundle / "data")
        args.output_root = args.output_root or str(bundle / "outputs")
    if args.output_root:
        args.output_root = Path(args.output_root).expanduser().resolve()
    output_root = args.output_root or detect_environment().output_root
    figures_dir = Path(args.out).expanduser().resolve() if args.out else Path(output_root) / "paper_figures"

    if args.list:
        rows = []
        for k in keys:
            f = FIGURES[k]
            rows.append([k, f.title, f.preset, " ".join(["python", f"analysis/{f.script}", *f.plot]) or f.script,
                         " ".join(f.both) if f.compute else (f.note or "-")])
        _print_rows("Figures", ["fig", "content", "preset", "plot command", "--compute / notes"], rows)
        return 0

    data_label = args.data_root or os.pathsep.join(map(str, detect_environment().data_roots))
    print(f"data root(s): {data_label}\noutput root : {output_root}\nfigures     : {figures_dir}\n"
          f"presets     : {args.presets}\nfigures     : {', '.join(keys)}{' (with compute stages)' if args.compute else ''}",
          flush=True)
    if not args.output_root and not os.environ.get("PARTICLE_ACCEL_OUTPUT"):
        print(f"note: PARTICLE_ACCEL_OUTPUT is not set, so products are read from and figures written to the default "
              f"output root {output_root}; set PARTICLE_ACCEL_OUTPUT (outside the repository) or pass --output-root "
              "/ --bundle", file=sys.stderr)
    if not args.check:
        fatal = fatal_problems(keys, args.presets, args.data_root, Path(output_root), figures_dir)
        if fatal:
            print(f"error: {fatal[0]}" + (f" (and {len(fatal) - 1} more problem(s); run --check)" if len(fatal) > 1
                                          else ""), file=sys.stderr)
            return 2
    problems = preflight(keys, args.presets, args.data_root, Path(output_root), compute=args.compute,
                         allow_download=args.allow_download)
    for k in keys:
        for msg in problems[k]:
            print(f"  [{k}] {msg}")
    if args.check:
        bad = [k for k in keys if problems[k]]
        _print_rows("Pre-flight check", ["fig", "status"], [[k, "ok" if not problems[k] else
                                                             f"{len(problems[k])} problem(s)"] for k in keys])
        return 1 if bad else 0

    figures_dir.mkdir(parents=True, exist_ok=True)
    stats_path = figures_dir / "paper_statistics.json"
    try:
        record = json.loads(stats_path.read_text()) if stats_path.is_file() else {}
    except (OSError, ValueError):
        record = {}
    record.setdefault("figures", {})
    env = child_env(args)
    results = []
    t_all = time.monotonic()
    for key in keys:
        fig = FIGURES[key]
        cmd = build_command(fig, args, figures_dir)
        print(f"\n{'=' * 100}\n[{key}] Fig. {fig.number}: {fig.title}\n$ {shlex.join(cmd)}\n{'=' * 100}", flush=True)
        started_wall = time.time()
        t0 = time.monotonic()
        code, text = runner(cmd, env=env, log=figures_dir / "logs" / f"{key}.log", echo=not args.quiet,
                            timeout=args.timeout)
        seconds = time.monotonic() - t0
        preset = load_preset(fig.preset, args.presets)
        sidecar_path, sidecar = _sidecar(figures_dir, preset, started_wall) if code == 0 else (None, {})
        entry = {
            "figure": fig.number, "title": fig.title, "status": "ok" if code == 0 else "failed", "exit_code": code,
            "seconds": round(seconds, 2), "command": shlex.join(cmd), "preset": fig.preset,
            "preflight_problems": problems[key], "log": str(figures_dir / "logs" / f"{key}.log"),
            "printed_tables": parse_printed_tables(text),
            "sidecar": str(sidecar_path) if sidecar_path else None, "metadata": sidecar,
        }
        if code != 0:
            entry["error_tail"] = text.strip().splitlines()[-15:]
        elif not sidecar:
            entry["status"] = "ok (no fresh JSON sidecar found)"
        record["figures"][key] = entry
        results.append((key, entry))
        if code != 0 and args.fail_fast:
            break

    record.update({
        "description": "Statistics printed by the figure scripts of arXiv:2512.12720 and stored in their JSON "
                       "sidecars; see analysis/README.md for the definitions.",
        "updated": provenance(data_roots=data_label, output_root=str(output_root), figures_dir=str(figures_dir),
                              presets=str(args.presets), compute=args.compute),
    })
    write_json(stats_path, record)
    total = time.monotonic() - t_all
    _print_rows(f"Summary ({total:.1f} s)", ["fig", "status", "seconds", "tables", "log"],
                [[k, e["status"], f"{e['seconds']:.1f}", len(e["printed_tables"]), e["log"]] for k, e in results])
    skipped = [k for k in keys if k not in dict(results)]
    if skipped:
        print(f"  not run (--fail-fast): {', '.join(skipped)}")
    print(f"\nwrote {stats_path}")
    failed = [k for k, e in results if e["exit_code"] != 0]
    if failed:
        print(f"FAILED: {', '.join(failed)} (see the logs above)", file=sys.stderr)
        return 1
    return 1 if skipped else 0


if __name__ == "__main__":
    from common import run_cli

    run_cli(main)
