#!/usr/bin/env python
r"""Build a portable small-data bundle from which Figures 1-7 of arXiv:2512.12720 can be made on a laptop.

The plot stages of the figures read only a few MB of small run files and analysis products.
This script copies those files from the full data (``--data-root``) and runs the compute
stages of Figs 1-3 once, writing their products into the bundle::

    <dest>/data/<run>/...                  athinput, .hst and particle files the presets need
    <dest>/outputs/run0NNN/products/*.npz  products of the compute stages (or copied HPC products)
    <dest>/MANIFEST.json                   size, sha256, source and figures of every file
    <dest>/logs/compute_<fig>.log          output of the compute stages

The file list follows the presets, so a preset copied for another run bundles that run.
Files are copied, never moved; identical files already in the bundle are left alone and
existing products are kept unless ``--recompute``.  Online-only cloud placeholders are skipped
unless ``--allow-download``, and a figure with a missing file makes the exit status 1.
``<dest>`` may not lie inside a data root, nor inside the repository without ``--force``.

Usage::

    python analysis/pack_small_data.py                                  # all figures -> ~/particle-accel-small-data
    python analysis/pack_small_data.py --figures 4-7 --dest /tmp/bundle
    python analysis/reproduce_figures.py --bundle ~/particle-accel-small-data --formats pdf,png
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import os
import shlex
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
if str(HERE) not in sys.path:  # `from common import ...` also when imported with importlib
    sys.path.insert(0, str(HERE))

from common import DEFAULT_PRESETS, load_preset, provenance, write_json  # noqa: E402
from reproduce_figures import (  # noqa: E402
    FIGURES, OPTIONAL_FILES, figure_inputs, parse_figures, product_run_name, run_command,
)

from shearpic.io._util import is_dataless  # noqa: E402

DEFAULT_DEST = Path("~/particle-accel-small-data")
MANIFEST = "MANIFEST.json"
BUNDLE_FORMAT = 1
RAW_SUFFIXES_NEVER_BUNDLED = (".athdf", ".xdmf", ".rst", ".par.tab", ".par.bin")


# ============================================================================ helpers
def sha256(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def human(n: float) -> str:
    for unit in ("B", "kB", "MB", "GB"):
        if abs(n) < 1000 or unit == "GB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1000.0
    return f"{n:.1f} GB"  # pragma: no cover


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def check_destination(dest: Path, data_roots: Sequence[Path], force: bool) -> None:
    """Refuse a bundle inside the raw data or (without ``force``) inside the git repository."""
    for root in data_roots:
        if _inside(dest, root) or _inside(root, dest / "data") or _inside(root, dest / "outputs"):
            raise SystemExit(f"refusing --dest {dest}: it overlaps the data root {root} (raw data are read-only)")
    if _inside(dest, REPO) and not force:
        raise SystemExit(f"refusing --dest {dest}: it is inside the git repository {REPO} (data never go into git); "
                         "choose a directory outside it, or pass --force")


def copy_file(src: Path, dst: Path) -> str:
    """Copy ``src`` to ``dst`` (data + mtime only, atomically); 'unchanged' if an identical copy exists."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and dst.stat().st_size == src.stat().st_size and sha256(dst) == sha256(src):
        return "unchanged"
    tmp = dst.with_name(dst.name + ".packing")
    shutil.copyfile(src, tmp)          # not copy2: never copy cloud-storage file flags
    st = src.stat()
    os.utime(tmp, ns=(st.st_atime_ns, st.st_mtime_ns))
    os.replace(tmp, dst)
    return "copied"


@dataclass
class Entry:
    """One bundled (or wanted) file."""

    rel: str                       # path relative to <dest>
    kind: str                      # 'raw' | 'product'
    figures: list[str] = field(default_factory=list)
    source: str | None = None      # absolute source path (raw files, copied HPC products)
    computed_by: str | None = None
    status: str = "planned"        # copied | unchanged | computed | exists | skipped-online-only | missing | failed
    size: int | None = None
    sha256: str | None = None

    def add_figure(self, key: str) -> None:
        if key not in self.figures:
            self.figures.append(key)
            self.figures.sort(key=lambda k: FIGURES[k].number if k in FIGURES else 99)


# ============================================================================ planning
@dataclass
class Plan:
    entries: dict[str, Entry] = field(default_factory=dict)
    computes: list[dict[str, Any]] = field(default_factory=list)
    problems: dict[str, list[str]] = field(default_factory=dict)

    def entry(self, rel: str, kind: str, figure: str, **kw: Any) -> Entry:
        e = self.entries.get(rel)
        if e is None:
            e = self.entries[rel] = Entry(rel=rel, kind=kind, **kw)
        e.add_figure(figure)
        return e


def build_plan(keys: Sequence[str], presets: Path, data_roots: str, source_outputs: Path, dest: Path) -> Plan:
    from shearpic.env import resolve_run

    plan = Plan()
    for key in keys:
        fig = FIGURES[key]
        problems = plan.problems.setdefault(key, [])
        preset = load_preset(fig.preset, presets)
        fi = figure_inputs(key, preset)
        run_dirs: dict[str, Path] = {}
        for run in dict.fromkeys([*fi.runs, *(r for r, _ in fi.files)]):
            try:
                run_dirs[run] = resolve_run(run, roots=data_roots)
            except (FileNotFoundError, ValueError) as err:
                problems.append(f"run {run}: {err}")
        for run in fi.runs:
            d = run_dirs.get(run)
            if d is None:
                continue
            for pattern in ("athinput.*", "*.hst"):
                found = sorted(p for p in d.glob(pattern) if p.is_file())
                if not found:
                    problems.append(f"{d}: no {pattern}")
                for p in found:
                    plan.entry(f"data/{d.name}/{p.name}", "raw", key, source=str(p))
        for run, pattern in fi.files:
            d = run_dirs.get(run)
            if d is None:
                continue
            found = sorted(p for p in d.glob(pattern) if p.is_file())
            if not found and Path(pattern).name not in OPTIONAL_FILES:
                problems.append(f"{d}: missing {pattern}")
            for p in found:
                plan.entry(f"data/{d.name}/{p.relative_to(d).as_posix()}", "raw", key, source=str(p))
        for run, pattern in fi.hpc_products:
            src_dir = source_outputs / product_run_name(run) / "products"
            found = sorted(p for p in src_dir.glob(pattern) if p.is_file())
            if not found:
                problems.append(f"no HPC products {src_dir / pattern} ({fig.note}; or use the legacy source)")
            for p in found:
                plan.entry(f"outputs/{product_run_name(run)}/products/{p.name}", "product", key, source=str(p))
        if fi.computed:
            products = {run: dest / "outputs" / product_run_name(run) / "products" for run, _ in fi.computed}
            (run,) = set(products)  # one run per compute stage
            cmd = [sys.executable, str(HERE / fig.script), *(fig.compute or ()), "--preset", fig.preset,
                   "--presets", str(presets), "--data-root", data_roots, "--products", str(products[run])]
            plan.computes.append({"figure": key, "run": run, "products_dir": products[run], "command": cmd,
                                  "products": [name for _, name in fi.computed]})
            for _, name in fi.computed:
                plan.entry(f"outputs/{product_run_name(run)}/products/{name}", "product", key,
                           computed_by=" ".join(["python", f"analysis/{fig.script}", *(fig.compute or ())]))
    return plan


def products_ready(job: Mapping[str, Any], preset: Mapping[str, Any]) -> bool:
    """True if every product of a compute job exists (and, for Fig. 3, holds all preset times)."""
    paths = [Path(job["products_dir"]) / n for n in job["products"]]
    if not all(p.is_file() for p in paths):
        return False
    if job["figure"] == "fig3":
        try:
            from shearpic.io.field_spectra import load_field_spectra

            _, meta = load_field_spectra(paths[0])
            have = [float(r["time"]) for r in meta.get("snapshots", [])]
        except Exception:  # noqa: BLE001 - unreadable: recompute
            return False
        return all(any(abs(h - float(t)) < 1.0 for h in have) for t in preset.get("times", []))
    return True


# ============================================================================ manifest
def load_manifest(dest: Path) -> dict[str, Any]:
    path = dest / MANIFEST
    try:
        return json.loads(path.read_text()) if path.is_file() else {}
    except (OSError, ValueError):
        return {}


def print_manifest(entries: Sequence[Entry], title: str) -> None:
    rows = [(e.rel, human(e.size) if e.size is not None else "-", e.status, ",".join(e.figures)) for e in entries]
    widths = [max([len(h)] + [len(r[i]) for r in rows]) for i, h in enumerate(("file", "size", "status", "figures"))]
    print(f"\n{title}")
    print("  " + "  ".join(h.ljust(w) for h, w in zip(("file", "size", "status", "figures"), widths)))
    for r in rows:
        print("  " + "  ".join(x.ljust(w) for x, w in zip(r, widths)))


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="pack_small_data.py", description=__doc__.split("\n\n")[0],
                                epilog="See the module docstring (python -m pydoc analysis/pack_small_data.py).",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dest", default=str(DEFAULT_DEST), help="bundle directory (default: %(default)s)")
    p.add_argument("--figures", default="all", help="'all' (default), '1,4,7', 'fig2' or '4-7'")
    p.add_argument("--data-root", default=None,
                   help="full data: directory (or ':'-separated list) with the run folders (default: PARTICLE_ACCEL_DATA)")
    p.add_argument("--source-outputs", default=None,
                   help="output root with HPC products for presets with source: npz (default: PARTICLE_ACCEL_OUTPUT)")
    p.add_argument("--presets", default=str(DEFAULT_PRESETS), help="presets YAML file (default: %(default)s)")
    p.add_argument("--workers", default="auto", help="worker processes of the compute stages (default: auto)")
    p.add_argument("--backend", default="auto", choices=["auto", "serial", "thread", "process"])
    p.add_argument("--allow-download", action="store_true",
                   help="copy/read online-only cloud placeholders (downloads them)")
    p.add_argument("--no-compute", action="store_true", help="copy files only; do not run the compute stages")
    p.add_argument("--recompute", action="store_true", help="rerun compute stages even if their products exist")
    p.add_argument("--dry-run", action="store_true", help="show the plan; copy and compute nothing")
    p.add_argument("--force", action="store_true", help="allow --dest inside the git repository")
    p.add_argument("--quiet", action="store_true", help="do not echo the compute stages' output (still logged)")
    return p


def main(argv: Sequence[str] | None = None, *, runner: Callable[..., tuple[int, str]] = run_command) -> int:
    args = build_parser().parse_args(argv)
    from shearpic.env import detect_environment

    try:
        keys = parse_figures(args.figures)
    except ValueError as err:
        print(f"error: {err}", file=sys.stderr)
        return 2
    env_info = detect_environment()
    data_roots = args.data_root or os.pathsep.join(map(str, env_info.data_roots))
    roots = [Path(os.path.expanduser(r)).resolve() for r in data_roots.split(os.pathsep) if r]
    source_outputs = Path(args.source_outputs).expanduser() if args.source_outputs else env_info.output_root
    presets = Path(args.presets).expanduser().resolve()
    dest = Path(args.dest).expanduser().resolve()
    check_destination(dest, roots, args.force)

    print(f"source data : {data_roots}\nbundle      : {dest}\npresets     : {presets}\nfigures     : {', '.join(keys)}")
    t_start = time.monotonic()
    plan = build_plan(keys, presets, data_roots, source_outputs, dest)

    # ------------------------------------------------------------------ raw files and HPC products
    for e in plan.entries.values():
        if e.source is None:
            continue
        src = Path(e.source)
        if any(src.name.endswith(s) for s in RAW_SUFFIXES_NEVER_BUNDLED):  # pragma: no cover - defensive
            raise RuntimeError(f"refusing to bundle raw simulation output {src}")
        e.size = src.stat().st_size
        if is_dataless(src) and not args.allow_download:
            e.status = "skipped-online-only"
            continue
        if args.dry_run:
            e.status = "would copy"
            continue
        e.status = copy_file(src, dest / e.rel)
    t_copy = time.monotonic() - t_start

    # ------------------------------------------------------------------ compute stages
    compute_log: dict[str, dict[str, Any]] = {}
    for job in plan.computes:
        key = job["figure"]
        preset = load_preset(FIGURES[key].preset, presets)
        cmd = [*job["command"], "--workers", str(args.workers), "--backend", args.backend]
        if args.allow_download:
            cmd.append("--allow-download")
        job_entries = [plan.entries[f"outputs/{product_run_name(job['run'])}/products/{n}"] for n in job["products"]]
        info: dict[str, Any] = {"command": shlex.join(cmd), "seconds": 0.0}
        if args.no_compute or args.dry_run:
            ready = products_ready(job, preset)
            for e in job_entries:
                e.status = "exists" if ready else ("would compute" if args.dry_run else "missing")
            info["status"] = "skipped (--dry-run)" if args.dry_run else "skipped (--no-compute)"
        elif products_ready(job, preset) and not args.recompute:
            for e in job_entries:
                e.status = "exists"
            info["status"] = "products exist (use --recompute to rerun)"
        else:
            print(f"\n{'=' * 100}\n[{key}] compute stage\n$ {shlex.join(cmd)}\n{'=' * 100}", flush=True)
            env = dict(os.environ, PARTICLE_ACCEL_DATA=data_roots, PARTICLE_ACCEL_OUTPUT=str(dest / "outputs"),
                       MPLBACKEND="Agg", PYTHONUNBUFFERED="1")
            t0 = time.monotonic()
            code, text = runner(cmd, env=env, log=dest / "logs" / f"compute_{key}.log", echo=not args.quiet)
            info["seconds"] = round(time.monotonic() - t0, 2)
            info["exit_code"] = code
            ok = code == 0 and products_ready(job, preset)
            info["status"] = "computed" if ok else "failed"
            for e in job_entries:
                e.status = "computed" if ok else ("failed" if not (dest / e.rel).is_file() else "stale")
            if not ok:
                tail = "\n    ".join(text.strip().splitlines()[-10:])
                plan.problems[key].append(f"compute stage failed (exit {code}); see {dest / 'logs' / f'compute_{key}.log'}"
                                          f"\n    {tail}")
        compute_log[key] = info

    # ------------------------------------------------------------------ sizes, hashes, status
    for e in plan.entries.values():
        target = dest / e.rel
        if not args.dry_run and target.is_file() and e.status not in ("skipped-online-only", "failed", "missing"):
            e.size = target.stat().st_size
            e.sha256 = sha256(target)
        elif e.status == "planned":
            e.status = "missing"
    for e in plan.entries.values():
        if e.status in ("skipped-online-only", "missing", "failed", "stale"):
            for key in e.figures:
                plan.problems.setdefault(key, []).append(
                    f"{e.rel}: {e.status}" + (f" (source {e.source})" if e.source else ""))

    entries = sorted(plan.entries.values(), key=lambda e: e.rel)
    print_manifest(entries, "Bundle manifest" + (" (dry run)" if args.dry_run else ""))
    total = sum(e.size or 0 for e in entries if e.status not in ("skipped-online-only", "missing", "failed"))
    print(f"  total: {human(total)} in {len(entries)} files")

    if args.dry_run:
        for key in keys:
            for msg in plan.problems.get(key, []):
                print(f"  [{key}] {msg}")
        return 1 if any(plan.problems.get(k) for k in keys) else 0

    # ------------------------------------------------------------------ MANIFEST.json (merged)
    old = load_manifest(dest)
    files: dict[str, dict[str, Any]] = {f["path"]: f for f in old.get("files", []) if (dest / f["path"]).is_file()}
    for e in entries:
        rec = files.get(e.rel, {})
        figs = sorted(set(rec.get("figures", [])) | set(e.figures), key=lambda k: FIGURES[k].number if k in FIGURES else 99)
        if (dest / e.rel).is_file() and e.status not in ("failed", "skipped-online-only", "missing"):
            files[e.rel] = {"path": e.rel, "kind": e.kind, "size": e.size, "sha256": e.sha256, "source": e.source,
                            "computed_by": e.computed_by, "figures": figs, "status": e.status}
        elif e.rel in files:
            files[e.rel]["figures"] = figs
    figures = dict(old.get("figures", {}))
    for key in keys:
        fig = FIGURES[key]
        previous = (old.get("figures", {}).get(key) or {}).get("compute")
        if compute_log.get(key, {}).get("status", "").startswith("products exist") and previous \
                and previous.get("status") == "computed":
            compute_log[key] = {**previous, "status": "computed (earlier run; products kept)"}
        figures[key] = {
            "figure": fig.number, "title": fig.title, "preset": fig.preset, "complete": not plan.problems.get(key),
            "problems": plan.problems.get(key, []), "compute": compute_log.get(key),
            "files": [e.rel for e in entries if key in e.figures],
            "plot_command": " ".join(["python", f"analysis/{fig.script}", *fig.plot]),
        }
    skipped = [{"path": e.rel, "source": e.source, "reason": "online-only cloud placeholder (use --allow-download)",
                "figures": e.figures} for e in entries if e.status == "skipped-online-only"]
    now = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    manifest = {
        "bundle_format": BUNDLE_FORMAT,
        "description": "Small-data bundle for Figures 1-7 of arXiv:2512.12720 (plot stages need no .athdf or "
                       "particle files). See analysis/README.md.",
        "created_utc": old.get("created_utc", now), "updated_utc": now,
        "source_data_roots": [str(r) for r in roots], "presets": str(presets),
        "presets_sha256": sha256(presets),
        "usage": {
            "environment": {"PARTICLE_ACCEL_DATA": str(dest / "data"), "PARTICLE_ACCEL_OUTPUT": str(dest / "outputs")},
            "command": "python analysis/reproduce_figures.py --formats pdf,png",
            "or": f"python analysis/reproduce_figures.py --bundle {dest} --out <figure directory>",
        },
        "total_size": sum(f["size"] or 0 for f in files.values()),
        "figures": dict(sorted(figures.items(), key=lambda kv: FIGURES[kv[0]].number if kv[0] in FIGURES else 99)),
        "files": [files[k] for k in sorted(files)],
        "skipped": skipped,
        "provenance": provenance(copy_seconds=round(t_copy, 2), total_seconds=round(time.monotonic() - t_start, 2)),
    }
    write_json(dest / MANIFEST, manifest)

    print(f"\nFigure status ({time.monotonic() - t_start:.1f} s; bundle {human(manifest['total_size'])})")
    for key in keys:
        status = "complete" if figures[key]["complete"] else "INCOMPLETE"
        c = compute_log.get(key)
        extra = f"; compute {c['status']}" + (f" in {c['seconds']:.1f} s" if c and c.get("seconds") else "") if c else ""
        print(f"  {key}: {status}{extra}")
        for msg in figures[key]["problems"]:
            print(f"      - {msg}")
    print(f"\nwrote {dest / MANIFEST}\nuse it with:\n  export PARTICLE_ACCEL_DATA={shlex.quote(str(dest / 'data'))}\n"
          f"  export PARTICLE_ACCEL_OUTPUT={shlex.quote(str(dest / 'outputs'))}\n"
          "  python analysis/reproduce_figures.py --formats pdf,png")
    return 0 if all(figures[k]["complete"] for k in keys) else 1


if __name__ == "__main__":
    from common import run_cli

    run_cli(main)
