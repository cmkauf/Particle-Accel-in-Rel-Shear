#!/usr/bin/env python
r"""HPC producer of particle energy spectra (the data of Figure 5 of arXiv:2512.12720).

Every particle of a full particle output is histogrammed in gamma = sqrt(1 + |u|^2/c^2),
gamma - 1 or p/(mc) = |u|/c, with u = p/m and the run's numerical speed of light c.  Each output
gives one product ``<products>/spectrum_<variable>_<basename>.<file_id>.<kind>_<NNNNN>.npz`` holding
integer counts, the particles below and above the bins, ``n_total``, the header time and the run
summary; ``<products>`` defaults to ``$PARTICLE_ACCEL_OUTPUT/run0NNN/products``.

The script exits with a non-zero status when an output is incomplete or inconsistent: block files,
particle number, times, or ``--hst-check`` against ``KE_cr / (np c^2)`` of the history file.  A
matching existing product is reused unless ``--force`` is given.  Under MPI only rank 0 writes.

Laptop, small runs::

    python analysis/energy_spectrum.py --run run364 --outputs 0:2 --workers 4

Stampede3, inside a job script such as ``slurm/analysis_node.sh`` (``ibrun python ...`` on
several nodes)::

    python analysis/energy_spectrum.py --run run364 --outputs all --kind tab \
        --variable gamma --edges log:1:100:500 --workers auto --hst-check

The products are plotted with ``analysis/plot_spectrum.py --source npz``.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import math
import sys
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `from common import ...` also when imported with importlib
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    add_common_arguments, check_same_run, find_run, load_preset, product_dir, provenance, require_keys, run_cli,
    run_summary, workers_arg,
)

from shearpic.config import RunConfig  # noqa: E402

PRESET = "fig5_particle_spectrum"
HST_REL_TOL = 1e-3


# ----------------------------------------------------------------------------- helpers
def parse_outputs(spec: str, available: Sequence[int]) -> list[int]:
    """Output numbers selected from ``available`` by ``'all'``, ``'3,5,8'`` or an inclusive range ``'a:b'``.

    Raises ValueError, naming the available outputs, for a number that does not exist, a reversed or
    empty range, or an empty specification.
    """
    available = sorted(int(n) for n in available)
    have = f"available: {format_numbers(available)}"
    spec = str(spec).strip()
    if spec == "all":
        return available
    wanted: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            a, b = part.split(":", 1)
            try:
                lo = int(a) if a.strip() else (available[0] if available else 0)
                hi = int(b) if b.strip() else (available[-1] if available else -1)
            except ValueError:
                raise ValueError(f"cannot parse the output range {part!r} (use 'a:b', {have})") from None
            if lo > hi:
                raise ValueError(f"output range {part!r} is reversed (first {lo} > last {hi}; {have})")
            sel = [n for n in available if lo <= n <= hi]
            if not sel:
                raise ValueError(f"output range {part!r} contains no particle output ({have})")
            wanted.extend(sel)
        else:
            try:
                n = int(part)
            except ValueError:
                raise ValueError(f"cannot parse the output number {part!r} ({have})") from None
            if n not in available:
                raise ValueError(f"particle output {n:05d} does not exist ({have})")
            wanted.append(n)
    if not wanted:
        raise ValueError(f"--outputs {spec!r} selects no particle output ({have})")
    return sorted(set(wanted))


def format_numbers(numbers: Sequence[int]) -> str:
    if not numbers:
        return "none"
    if list(numbers) == list(range(numbers[0], numbers[-1] + 1)):
        return f"{numbers[0]}..{numbers[-1]}"
    return ", ".join(str(n) for n in numbers)


def product_path(directory: Path, variable: str, stream, number: int) -> Path:
    """Product path ``spectrum_<variable>_<basename>.<file_id>.<kind>_<NNNNN>.npz``.

    ``stream`` is ``(basename, file_id, kind)`` or a mapping with those keys.
    """
    from shearpic.io.particles import stream_tag

    return Path(directory) / f"spectrum_{variable}_{stream_tag(stream)}_{int(number):05d}.npz"


def stream_dict(stream) -> dict[str, str]:
    """Metadata form ``{'basename', 'file_id', 'kind'}`` of a stream tuple."""
    basename, file_id, kind = stream
    return {"basename": basename, "file_id": file_id, "kind": kind}


_NOTES_SEEN: set[str] = set()


def run_mismatch(product_run, cfg: RunConfig, quiet: bool = False) -> str | None:
    """Why :func:`common.check_same_run` rejects a product's recorded run for ``cfg``, or None.

    Its notes, e.g. about an athinput edited only in ``tlim``, are printed once per process;
    ``quiet`` suppresses them.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stderr(buf):
            check_same_run(product_run, cfg, "existing product(s)")
    except ValueError as err:
        return str(err)
    finally:
        for line in buf.getvalue().splitlines():
            if line not in _NOTES_SEEN and not quiet:
                print(line, file=sys.stderr)
            _NOTES_SEEN.add(line)
    return None


def n_meshblocks(cfg: RunConfig) -> int | None:
    """Number of meshblocks, and so of block files per particle output, of a uniform mesh."""
    if cfg.meshblock is None:
        return None
    n = 1
    for nx, mb in zip(cfg.nx, cfg.meshblock):
        if mb <= 0 or nx % mb:
            return None
        n *= nx // mb
    return n


def particle_output_dt(cfg: RunConfig, kind: str, file_id: str | None = None) -> float | None:
    """Output cadence of a particle stream, or None.

    ``file_id = 'outN'`` selects ``<outputN>``; otherwise the first output block with ``file_type = kind`` is used.
    """
    if file_id:
        digits = "".join(ch for ch in file_id if ch.isdigit())
        for block in cfg.athinput.outputs:
            if digits and block["id"] == int(digits) and block.get("dt") is not None:
                return float(block["dt"])
    return cfg.output_dt(kind)


def same_time(a: float | None, b: float | None) -> bool:
    """Whether two header times agree to 1e-6 relative, the precision of float32 .par.bin headers."""
    if a is None or b is None:
        return False
    return abs(float(a) - float(b)) <= 1e-6 * max(1.0, abs(float(a)), abs(float(b)))


def check_existing(path: Path, variable: str, edges: np.ndarray, expected_n: int | None, *,
                   stream=None, file_time: float | None = None, cfg: RunConfig | None = None,
                   quiet: bool = False) -> tuple[bool, str]:
    """Whether ``path`` is a valid product for this request, and the reason if not.

    Compares variable, edges, ``n_total``, stream, the time with ``file_time`` (header of the output's
    first block file) and the run physics; a comparison is skipped when its reference is None.
    """
    from shearpic.io.spectrum_files import load_spectrum

    if not path.exists():
        return False, "missing"
    try:
        spec, meta = load_spectrum(path, with_metadata=True)
    except Exception as err:  # noqa: BLE001 - any unreadable file is simply recomputed
        return False, f"unreadable ({type(err).__name__})"
    if spec.variable != variable:
        return False, f"variable {spec.variable}"
    if spec.edges.shape != edges.shape or not np.array_equal(spec.edges, edges):
        return False, "different edges"
    if spec.counts is None:
        return False, "no counts"
    if expected_n is not None and spec.n_total != expected_n:
        return False, f"n_total {spec.n_total:.0f} != {expected_n}"
    if stream is not None and meta.get("stream") != stream_dict(stream):
        return False, f"different stream ({meta.get('stream')})"
    if file_time is not None and not same_time(spec.time, file_time):
        return False, f"different time (product {spec.time}, file header {file_time:.10g})"
    if cfg is not None:
        why = run_mismatch(meta.get("run"), cfg, quiet=quiet)
        if why:
            return False, f"different run parameters ({why})"
    return True, "valid"


def mpi_rank(backend: str) -> int:
    """This process's MPI rank when the map will run under MPI, else 0 without importing mpi4py."""
    from shearpic.parallel import mpi_world_size

    if backend == "mpi" or (backend == "auto" and mpi_world_size() > 1):
        try:
            from mpi4py import MPI
        except ImportError:
            return 0  # parallel_map raises a clear ImportError on every rank
        return int(MPI.COMM_WORLD.Get_rank())
    return 0


def hst_mean_gamma_minus_1(hst, cfg: RunConfig, time: float) -> dict[str, Any] | None:
    """Mean gamma - 1 from the history, ``KE_cr / (np c^2)``, at the row nearest ``time``.

    ``KE_cr`` is sum_p (gamma - 1) c^2 and ``np`` the particle number.  Returns None unless a row lies
    within half the history cadence.
    """
    if hst is None or len(hst) == 0 or "KE_cr" not in hst.columns or "np" not in hst.columns:
        return None
    t = hst["time"].to_numpy(np.float64)
    k = int(np.argmin(np.abs(t - time)))
    dt = cfg.output_dt("hst") or (float(np.median(np.diff(t))) if t.size > 1 else 0.0)
    if abs(t[k] - time) > 0.5 * dt + 1e-9:
        return None
    n = float(hst["np"].iloc[k])
    return {"time": float(t[k]), "mean_gamma_minus_1": float(hst["KE_cr"].iloc[k]) / (n * cfg.c**2), "np": n}


def hst_check(spec, ref: dict[str, Any] | None, rel_tol: float = HST_REL_TOL) -> dict[str, Any]:
    """Compare the histogram estimate of <gamma - 1> with the history value ``ref``."""
    if ref is None:
        return {"status": "not covered"}
    lo, hi = spec.mean_bounds("gamma_minus_1")
    try:
        est = spec.mean("gamma_minus_1")
    except ValueError:  # particles outside the bins: only the bounds are known
        est = math.nan
    val = ref["mean_gamma_minus_1"]
    rel = abs(est - val) / abs(val) if math.isfinite(est) and val else math.nan
    ok = (lo <= val * (1 + 1e-9) and val <= hi * (1 + 1e-9)) and (not math.isfinite(rel) or rel <= rel_tol)
    return {"status": "ok" if ok else "FAILED", "hst_time": ref["time"], "hst_mean_gamma_minus_1": val,
            "estimate": est, "bounds": [lo, hi], "rel_diff": rel, "rel_tol": rel_tol}


# ------------------------------------------------------------------------------- CLI
def build_parser(preset_name: str = PRESET, argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=preset_name)
    pre.add_argument("--presets", default=None)
    known, _ = pre.parse_known_args(argv)
    preset = load_preset(known.preset, known.presets)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="See the module docstring for details.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=preset.get("run"), help="run name, number or directory (default: %(default)s)")
    p.add_argument("--outputs", default="all", help="'all', a comma list '3,5' or an inclusive range 'a:b'")
    p.add_argument("--kind", default=preset.get("kind", "tab"), choices=["tab", "bin"])
    p.add_argument("--variable", default=preset.get("variable", "gamma"),
                   choices=["gamma", "gamma_minus_1", "p_over_mc"], help="histogram variable (default: %(default)s)")
    p.add_argument("--edges", default=preset.get("edges", "log:1:100:500"),
                   help="'log:LO:HI:N' or 'lin:LO:HI:N' in --variable (default: %(default)s)")
    p.add_argument("--file-id", default=preset.get("file_id"), help="output block of the particle files, e.g. out4")
    p.add_argument("--basename", default=preset.get("basename"), help="problem_id prefix of the particle files")
    p.add_argument("--expected-n", type=int, default=None, help="required particle number (default: cfg.n_par)")
    p.add_argument("--force", action="store_true", help="recompute outputs that already have a valid product")
    p.add_argument("--hst-check", action="store_true",
                   help="compare <gamma-1> with KE_cr/(np c^2) of the history file where it covers the output")
    p.add_argument("--progress", action="store_true", help="progress bar over the block files")
    add_common_arguments(p, preset_name)
    return p


def output_header_time(files: Sequence[Path], allow_download: bool) -> float | None:
    """Header time of an output's first block file; ``None`` if it is an online-only placeholder or unreadable."""
    from shearpic.io._util import is_dataless
    from shearpic.io.particles import read_particle_time

    if not files or (is_dataless(files[0]) and not allow_download):
        return None
    try:
        return read_particle_time(files[0])
    except (OSError, ValueError):
        return None


def main(argv: Sequence[str] | None = None) -> int:
    from shearpic.io._util import is_dataless
    from shearpic.io.history import read_hst
    from shearpic.io.particles import particle_output_files, particle_output_numbers, particle_output_stream
    from shearpic.io.spectrum_files import load_spectrum, save_spectrum
    from shearpic.physics.spectra import parse_edges, particle_snapshot_spectrum

    args = build_parser(argv=argv).parse_args(argv)
    if args.run is None:
        require_keys(load_preset(args.preset, args.presets), ["run"], args.preset,
                     "pass --run or use --preset fig5_particle_spectrum")
    rank = mpi_rank(args.backend)
    say = print if rank == 0 else (lambda *a, **k: None)

    run_dir = find_run(args.run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    cfg.require_particles()
    edges = parse_edges(args.edges)
    expected_n = int(args.expected_n) if args.expected_n is not None else int(cfg.n_par)
    available = particle_output_numbers(run_dir, args.kind, file_id=args.file_id, basename=args.basename)
    if not available:
        raise FileNotFoundError(f"no {args.kind} particle outputs (file_id={args.file_id!r}, "
                                f"basename={args.basename!r}) in {run_dir}")
    numbers = parse_outputs(args.outputs, available)
    stream = particle_output_stream(run_dir, args.kind, file_id=args.file_id, basename=args.basename)
    out_dir = product_dir(run_dir, args.products)
    n_blocks = n_meshblocks(cfg)
    hst = None
    if args.hst_check and rank == 0:
        if cfg.hst_path is None:
            say("--hst-check: no history file; the check is skipped")
        else:
            hst = read_hst(cfg.hst_path)

    # every MPI rank builds the same work list before anything is written
    status: dict[int, str] = {}
    todo = []
    for n in numbers:
        path = product_path(out_dir, args.variable, stream, n)
        if args.force:
            todo.append(n)
            continue
        files = particle_output_files(run_dir, n, args.kind, file_id=args.file_id, basename=args.basename)
        t_head = output_header_time(files, args.allow_download)
        ok, why = check_existing(path, args.variable, edges, expected_n, stream=stream, cfg=cfg, quiet=rank != 0,
                                 file_time=t_head)
        if ok and t_head is not None:
            status[n] = "skipped (valid product)"
        else:
            todo.append(n)
            if path.exists():
                say(f"output {n:05d}: recomputing {path.name}: "
                    + (why if t_head is not None or not ok else "header time unavailable (placeholder?)"))
    say(f"run {run_dir.name}: c = {cfg.c:g}, N = {expected_n}, stream {'.'.join(stream)}, {len(numbers)} output(s) "
        f"selected ({format_numbers(numbers)}), {len(todo)} to compute -> {out_dir}")

    rows: dict[int, dict[str, Any]] = {}
    failures: dict[int, str] = {}
    for n in todo:
        path = product_path(out_dir, args.variable, stream, n)
        try:
            files = particle_output_files(run_dir, n, args.kind, file_id=args.file_id, basename=args.basename)
            if n_blocks is not None and len(files) != n_blocks:
                raise ValueError(f"output {n:05d} has {len(files)} block files, the mesh has {n_blocks} meshblocks")
            if not args.allow_download and any(is_dataless(f) for f in files):
                raise FileNotFoundError(f"output {n:05d} has online-only cloud placeholders; make them available "
                                        "offline or pass --allow-download")
            t_head = output_header_time(files, True)
            spec = particle_snapshot_spectrum(run_dir, n, cfg, edges, variable=args.variable, kind=args.kind,
                                              backend=args.backend, n_workers=workers_arg(args.workers),
                                              progress=args.progress and rank == 0, on_error="raise",
                                              file_id=args.file_id, basename=args.basename, expected_n=expected_n)
        except Exception as err:  # noqa: BLE001 - record, keep every rank in step, report at the end
            failures[n] = f"{type(err).__name__}: {err}"
            status[n] = "FAILED"
            continue
        if rank != 0 or spec is None:
            continue
        if spec.underflow or spec.overflow:
            warnings.warn(f"output {n:05d}: {spec.underflow:.0f} particles below and {spec.overflow:.0f} above "
                          f"the bins [{edges[0]:g}, {edges[-1]:g}] of {args.variable}", stacklevel=1)
        check = hst_check(spec, hst_mean_gamma_minus_1(hst, cfg, spec.time)) if args.hst_check else None
        rows[n] = {"spec": spec, "hst": check}
        if check is not None and check["status"] == "FAILED":
            failures[n] = (f"hst check failed: <gamma-1> = {check['estimate']:.6g} (bounds {check['bounds'][0]:.6g}"
                           f"..{check['bounds'][1]:.6g}) vs KE_cr/(np c^2) = {check['hst_mean_gamma_minus_1']:.6g}")
            status[n] = "FAILED"
            continue
        metadata = {
            "product": "particle energy spectrum: integer counts per bin of all particles of one output",
            "output_number": n, "kind": args.kind, "stream": stream_dict(stream),
            "n_block_files": len(files), "file_time": t_head if t_head is not None else spec.time,
            "variable": args.variable,
            "edges_spec": args.edges, "expected_n": expected_n, "hst_check": check,
            "run": run_summary(cfg), **provenance(stage="compute"),
        }
        save_spectrum(path, spec, metadata=metadata)
        status[n] = "computed"

    if rank != 0:
        return 1 if failures else 0

    # --hst-check also covers products kept from an earlier run
    if args.hst_check:
        for n in numbers:
            if status.get(n) != "skipped (valid product)":
                continue
            spec = load_spectrum(product_path(out_dir, args.variable, stream, n))
            check = hst_check(spec, hst_mean_gamma_minus_1(hst, cfg, spec.time))
            rows[n] = {"spec": spec, "hst": check}
            if check is not None and check["status"] == "FAILED":
                failures[n] = (f"hst check failed (existing product): <gamma-1> = {check['estimate']:.6g} (bounds "
                               f"{check['bounds'][0]:.6g}..{check['bounds'][1]:.6g}) vs KE_cr/(np c^2) = "
                               f"{check['hst_mean_gamma_minus_1']:.6g}")
                status[n] = "FAILED (hst check)"

    # times must increase over all selected products, kept ones included
    times = {}
    for n in numbers:
        p = product_path(out_dir, args.variable, stream, n)
        if p.exists() and not str(status.get(n, "")).startswith("FAILED"):
            times[n] = load_spectrum(p).time
    ordered = [times[n] for n in sorted(times)]
    bad = [(a, b) for a, b in zip(sorted(times), sorted(times)[1:]) if not times[b] > times[a]]
    for a, b in bad:
        failures[b] = f"time {times[b]:.10g} of output {b:05d} does not exceed {times[a]:.10g} of output {a:05d}"

    header = f"{'output':>6} {'time':>12} {'n_total':>10} {'under':>6} {'over':>6} {'<g-1>':>9} {'hst':>9}  status"
    print("\n" + header)
    for n in numbers:
        p = product_path(out_dir, args.variable, stream, n)
        spec = rows.get(n, {}).get("spec")
        if spec is None and p.exists() and status.get(n) != "FAILED":
            spec = load_spectrum(p)
        chk = rows.get(n, {}).get("hst")
        if spec is None:
            print(f"{n:>6} {'-':>12} {'-':>10} {'-':>6} {'-':>6} {'-':>9} {'-':>9}  {status.get(n)}")
            continue
        try:
            mean = f"{spec.mean('gamma_minus_1'):.5f}"
        except ValueError:
            mean = "out-rng"
        hst_txt = "-" if chk is None or "hst_mean_gamma_minus_1" not in chk else f"{chk['hst_mean_gamma_minus_1']:.5f}"
        print(f"{n:>6} {spec.time:>12.6g} {spec.n_total:>10.0f} {spec.underflow:>6.0f} {spec.overflow:>6.0f} "
              f"{mean:>9} {hst_txt:>9}  {status.get(n)}")
    if ordered:
        print(f"times {ordered[0]:g} .. {ordered[-1]:g} ({'strictly increasing' if not bad else 'NOT increasing'})")
    for n, msg in sorted(failures.items()):
        print(f"ERROR output {n:05d}: {msg}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    run_cli(main)
