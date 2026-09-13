#!/usr/bin/env python
r"""Figure 6 of arXiv:2512.12720: a single-particle orbit and the ensemble phase space f(p_i, y).

The top row shows one tracked particle in the (p_i/mc, y) planes over ``--t-window``, coloured by
time.  The bottom row shows f(p_i/mc, y) = dN / (N d(p_i/mc) dy) [1/a] of all N particles of one
snapshot on a log grey scale, with N including particles outside the momentum range.  Each row
uses the speed of light of its own run.

``compute`` (HPC) histograms the particle output nearest ``--time`` into
``<products>/phase_space_<basename>.<file_id>.<kind>_<NNNNN>.npz`` and reuses a matching product
unless ``--force``.  ``plot`` draws the figure from such a product (``--hist-source npz``) or from
``phase_space_histograms.npz`` (``--hist-source legacy``), hatches momenta outside the histogram
and prints the fraction of particles beyond an axis limit that cuts into it.  ``all`` runs both::

    python analysis/phase_space.py plot --formats pdf,png
    python analysis/phase_space.py compute --run run364 --time 1200 --kind tab --workers auto
    python analysis/phase_space.py plot --hist-source npz --products .../run0364/products
"""

from __future__ import annotations

import argparse
import math
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # `from common import ...` also when imported with importlib
    sys.path.insert(0, str(HERE))

from common import (  # noqa: E402
    add_common_arguments, check_same_run, figure_dir, find_run, load_preset, print_table, product_dir, provenance,
    require_keys, run_cli, run_summary, save_figure, workers_arg,
)
from energy_spectrum import (  # noqa: E402
    mpi_rank, n_meshblocks, particle_output_dt, run_mismatch, same_time, stream_dict,
)

from shearpic.config import RunConfig  # noqa: E402

PRESET = "fig6_phase_space"
FIGSIZE = (8, 11)
FONT = 22
CMAP_TOP = 0.85  # fraction of the Greys colormap used (1 = black at vmax)
RC = {"axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.major.size": 3.5,
      "ytick.major.size": 3.5, "xtick.minor.visible": False, "ytick.minor.visible": False, "font.size": FONT,
      "axes.labelsize": FONT, "xtick.labelsize": FONT, "ytick.labelsize": FONT}


# ============================================================================ helpers
def run_label(cfg: RunConfig) -> str:
    return f"run{cfg.run_id}" if cfg.run_id is not None else Path(str(cfg.run_dir)).name


def run_parameters(cfg: RunConfig) -> dict[str, Any]:
    p = {"c": cfg.c, "q_mc": cfg.q_mc, "M_A": cfg.M_A, "B0": cfg.B0, "shear_amplitude": cfg.shear_amplitude,
         "shear_amplitude_source": cfg.shear_amplitude_source, "vp_par": cfg.vp_par, "r_g0": cfg.r_g0}
    return p


def refuse_dataless(path: Path, allow_download: bool) -> None:
    from shearpic.io._util import is_dataless

    if is_dataless(path) and not allow_download:
        raise FileNotFoundError(f"{path} is an online-only cloud placeholder; make it available offline or pass "
                                "--allow-download")


def y_edges_for(cfg: RunConfig, n_bins: int) -> np.ndarray:
    lo, hi = cfg.bounds[1]
    return np.linspace(float(lo), float(hi), int(n_bins) + 1)


def product_name(stream, number: int) -> str:
    """``phase_space_<basename>.<file_id>.<kind>_<NNNNN>.npz`` (``stream`` = ``(basename, file_id, kind)``)."""
    from shearpic.io.particles import stream_tag

    return f"phase_space_{stream_tag(stream)}_{int(number):05d}.npz"


#: product names ``phase_space_<stream tag>_NNNNN.npz`` and the older ``phase_space_NNNNN.npz``
PRODUCT_RE = re.compile(r"^phase_space_(?:(?P<tag>.+)_)?(?P<number>\d{5})\.npz$")


def nice_limit(value: float, step: float = 0.5) -> float:
    """Smallest multiple of ``step`` >= 1.1 * value."""
    return step * math.ceil(1.1 * float(value) / step - 1e-9)


TICK_EDGE_MARGIN = 0.12  # fraction of the axis span kept free of tick labels at each end


def axis_ticks(lo: float, hi: float, margin: float = TICK_EDGE_MARGIN) -> list[float]:
    """Round tick positions on [lo, hi] kept ``margin * span`` away from both ends.

    Adjacent panels share an edge, so a label at a limit would collide with the neighbour's.  The smallest
    step of 1, 2, 2.5, 3, 4 or 5 times a power of ten giving exactly 3 ticks is used, else the smallest
    giving 2 to 5.
    """
    span = float(hi) - float(lo)
    steps = [m * 10.0**e for e in range(-1, 4) for m in (1, 2, 2.5, 3, 4, 5)]
    fallback = None
    for step in sorted(set(steps)):
        first = math.ceil((lo + margin * span) / step - 1e-9) * step
        ticks = [float(t) for t in np.arange(first, hi - margin * span + 1e-9 * step, step)]
        ticks = [0.0 if abs(t) < 1e-9 * step else round(t, 10) for t in ticks]
        if len(ticks) == 3:
            return ticks
        if fallback is None and 2 <= len(ticks) <= 5:
            fallback = ticks
    return fallback or [0.5 * (lo + hi)]


def visible_range(dens: Mapping[str, Any], vmin: float, hist_range: Sequence[float]) -> tuple[float, float]:
    """Smallest symmetric range +-L beyond which every bin of every component is below ``vmin``.

    ``dens`` maps components to ``(H, x_edges, y_edges)`` as returned by ``PhaseSpaceHist.density``.  L is
    the largest |edge| of a momentum bin reaching ``vmin`` at some y, rounded up to a multiple of 0.5
    (L <= 5), 1 (L <= 10) or 2, and clipped to ``hist_range``, which is returned if no bin reaches ``vmin``.
    """
    need = 0.0
    for H, xe, _ in dens.values():
        cols = np.nonzero((np.asarray(H) >= vmin).any(axis=1))[0]
        if cols.size:
            need = max(need, float(np.max(np.abs(xe[cols]))), float(np.max(np.abs(xe[cols + 1]))))
    if need <= 0.0:
        return float(hist_range[0]), float(hist_range[1])
    step = 0.5 if need <= 5 else (1.0 if need <= 10 else 2.0)
    lim = step * math.ceil(need / step - 1e-9)
    return max(-lim, float(hist_range[0])), min(lim, float(hist_range[1]))


def displayed_fractions(hist, xlim: Sequence[float]) -> dict[str, float]:
    """Fractions of ``n_total`` in bins beyond each side of the displayed range ``xlim`` in p_i/(mc).

    A bin is displayed when its centre lies within ``xlim``.  Returns ``beyond_left``, ``beyond_right``,
    ``outside_histogram`` (outside the momentum or y bins, side unknown) and ``outside_displayed`` (all
    particles not displayed).
    """
    xe = hist.u_edges / hist.c
    xc = 0.5 * (xe[:-1] + xe[1:])
    per_u = hist.counts.sum(axis=1)
    n = float(hist.n_total)
    inside = float(per_u[(xc >= xlim[0]) & (xc <= xlim[1])].sum())
    return {"beyond_left": float(per_u[xc < xlim[0]].sum()) / n, "beyond_right": float(per_u[xc > xlim[1]].sum()) / n,
            "outside_histogram": hist.out_of_range_fraction(), "outside_displayed": 1.0 - inside / n}


def percent_text(fraction: float) -> str:
    v = 100.0 * float(fraction)
    if v < 1e-3:
        return "<0.001%"
    return f"{v:.2g}%" if v < 10 else f"{v:.0f}%"


def row_limits(orbit_need: float, hist_range: Sequence[float], xlim_traj, xlim_hist, separate: bool):
    """x limits of the orbit and histogram rows.

    By default the orbit row spans :func:`nice_limit` of its largest |p_i|/(mc) and the histogram row its
    momentum range.  Unless ``separate``, both rows get the union of the two.
    """
    t = tuple(xlim_traj) if xlim_traj is not None else (-nice_limit(orbit_need), nice_limit(orbit_need))
    h = tuple(xlim_hist) if xlim_hist is not None else tuple(hist_range)
    if separate:
        return t, h
    both = (min(t[0], h[0]), max(t[1], h[1]))
    return both, both


# ============================================================================ compute
def check_existing(path: Path, u_edges: np.ndarray, y_edges: np.ndarray, expected_n: int, *, stream=None,
                   file_time: float | None = None, cfg: RunConfig | None = None,
                   quiet: bool = False) -> tuple[bool, str]:
    """Whether ``path`` is a valid product for this request, and the reason if not.

    Compares components, edges, ``n_total``, stream, the time with ``file_time`` (header of the output's
    first block file) and the run physics; a comparison is skipped when its reference is None.
    """
    from shearpic.io.spectrum_files import load_phase_space

    if not path.exists():
        return False, "missing"
    try:
        hists, meta = load_phase_space(path, with_metadata=True)
    except Exception as err:  # noqa: BLE001 - any unreadable file is simply recomputed
        return False, f"unreadable ({type(err).__name__})"
    if set(hists) != {"x", "y", "z"}:
        return False, f"components {sorted(hists)}"
    for h in hists.values():
        if not (np.array_equal(h.u_edges, u_edges) and np.array_equal(h.y_edges, y_edges)):
            return False, "different edges"
        if h.n_total != expected_n:
            return False, f"n_total {h.n_total} != {expected_n}"
    if stream is not None and meta.get("stream") != stream_dict(stream):
        return False, f"different stream ({meta.get('stream')})"
    t = next(iter(hists.values())).time
    if file_time is not None and not same_time(t, file_time):
        return False, f"different time (product {t}, file header {file_time:.10g})"
    if cfg is not None:
        why = run_mismatch(meta.get("run"), cfg, quiet=quiet)
        if why:
            return False, f"different run parameters ({why})"
    return True, "valid"


def compute(args) -> int:
    from shearpic.io._util import is_dataless
    from shearpic.io.particles import parse_particle_filename, particle_output_files, select_particle_output
    from shearpic.io.spectrum_files import save_phase_space
    from shearpic.physics.phase_space import particle_snapshot_phase_space
    from shearpic.physics.spectra import parse_edges

    rank = mpi_rank(args.backend)
    say = print if rank == 0 else (lambda *a, **k: None)
    preset = load_preset(args.preset, args.presets)
    for arg, key in (("run", "run"), ("time", "time")):
        if getattr(args, arg) is None:
            require_keys(preset.get("histogram") or {}, [key], f"{args.preset} (histogram)",
                         f"pass --{arg} or use --preset fig6_phase_space")
    run_dir = find_run(args.run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    cfg.require_particles()
    u_edges = parse_edges(args.u_edges)
    y_edges = y_edges_for(cfg, args.y_bins)
    expected_n = int(args.expected_n) if args.expected_n is not None else int(cfg.n_par)
    dt = particle_output_dt(cfg, args.kind, args.file_id)
    number, t_file = select_particle_output(run_dir, args.time, kind=args.kind, file_id=args.file_id,
                                            basename=args.basename, dt=dt, tol=args.tol,
                                            allow_download=args.allow_download)
    files = particle_output_files(run_dir, number, args.kind, file_id=args.file_id, basename=args.basename)
    stream = parse_particle_filename(files[0]).stream
    path = product_dir(run_dir, args.products) / product_name(stream, number)
    args.computed_product = str(path)
    if not args.force:
        ok, why = check_existing(path, u_edges, y_edges, expected_n, stream=stream, file_time=t_file, cfg=cfg,
                                 quiet=rank != 0)
        if ok:
            say(f"{path} exists and matches the request (use --force to recompute)")
            return 0
        if path.exists():
            say(f"recomputing {path.name}: {why}")
    if not args.allow_download and any(is_dataless(f) for f in files):
        raise FileNotFoundError(f"output {number:05d} of {run_dir} has online-only cloud placeholders; make them "
                                "available offline or pass --allow-download")
    n_blocks = n_meshblocks(cfg)
    if n_blocks is not None and len(files) != n_blocks:
        raise ValueError(f"output {number:05d} has {len(files)} block files, the mesh has {n_blocks} meshblocks")
    say(f"{run_dir.name}: output {number:05d} at t = {t_file:.10g} (requested {args.time:g}, cadence {dt}), "
        f"{len(files)} block files, u bins {args.u_edges}, {args.y_bins} y bins")
    hists = particle_snapshot_phase_space(run_dir, number, cfg, u_edges, y_edges, kind=args.kind,
                                          file_id=args.file_id, basename=args.basename, backend=args.backend,
                                          n_workers=workers_arg(args.workers), on_error="raise",
                                          expected_n=expected_n, progress=args.progress and rank == 0)
    if hists is None:  # MPI rank != 0
        return 0
    oor = {k: h.out_of_range_fraction() for k, h in hists.items()}
    oor_y_max = {k: float(np.nanmax(h.out_of_range_fraction_y())) for k, h in hists.items()}
    metadata = {
        "product": "phase-space histograms (u_i, y) of all particles: integer counts, per-y totals, "
                   "out-of-range numbers and exact sums of u_i, u_i^2",
        "units": {"u_edges": "u_i = p_i/m [U0]", "y_edges": "[a]", "density": "counts/(du dy)/n_total [1/(U0 a)]"},
        "output_number": number, "file_time": t_file, "requested_time": args.time, "output_dt": dt,
        "kind": args.kind, "stream": stream_dict(stream),
        "n_block_files": len(files), "u_edges_spec": args.u_edges, "y_bins": args.y_bins,
        "out_of_range_fraction": oor, "out_of_range_fraction_y_max": oor_y_max,
        "run": run_summary(cfg), **provenance(stage="compute"),
    }
    save_phase_space(path, hists, metadata=metadata)
    print_table("phase-space product", {"file": str(path), "time": t_file, "n_total": expected_n,
                                        **{f"out-of-range u_{k}": v for k, v in oor.items()},
                                        **{f"max per-y out-of-range u_{k}": v for k, v in oor_y_max.items()}})
    return 0


# ============================================================================ plot: inputs
def load_histograms(args, preset: Mapping[str, Any]) -> tuple[dict[str, Any], RunConfig, dict[str, Any]]:
    """Histograms for the bottom row, their run configuration and a provenance report."""
    from dataclasses import replace

    from shearpic.io.spectrum_files import load_phase_space, read_legacy_phase_space_npz

    run_dir = find_run(args.hist_run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    cfg.require_particles()
    if args.hist_source == "legacy":
        path = run_dir / args.legacy_file
        refuse_dataless(path, args.allow_download)
        n_total = int(args.n_total) if args.n_total is not None else int(cfg.n_par)
        if n_total != cfg.n_par:
            warnings.warn(f"--n-total {n_total} differs from cfg.n_par = {cfg.n_par}", stacklevel=2)
        hists = read_legacy_phase_space_npz(path, n_total=n_total, time=float(args.time), run_id=cfg.run_id, c=cfg.c)
        report = {"source": "legacy", "file": str(path), "n_total": n_total,
                  "time_note": "the legacy file stores no time; taken from the preset",
                  "n_in_range": {k: h.n_in_range for k, h in hists.items()}}
    else:
        if args.product:
            path = Path(args.product)
            if not path.exists():
                raise FileNotFoundError(f"--product {path} does not exist")
            selection = {"explicit": True}
        else:
            path, selection = find_product(product_dir(run_dir, args.products, create=False), cfg, float(args.time),
                                           kind=args.kind, file_id=args.file_id, basename=args.basename,
                                           tol=args.tol, allow_download=args.allow_download)
        refuse_dataless(path, args.allow_download)
        hists, meta = load_phase_space(path, with_metadata=True)
        check_same_run(meta.get("run"), cfg, path.name)
        t_prod = next(iter(hists.values())).time
        if args.product and t_prod is not None and args.time is not None:
            st = meta.get("stream") or {}
            dt = particle_output_dt(cfg, st.get("kind", "tab"), st.get("file_id")) or meta.get("output_dt")
            if dt and abs(t_prod - float(args.time)) > 0.5 * dt:
                print(f"note: {path.name} is at t = {t_prod:g}, not near --time {args.time:g}; plotting it anyway",
                      file=sys.stderr)
        report = {"source": "npz", "file": str(path), "selection": selection,
                  "product_metadata": {k: meta.get(k) for k in (
                      "output_number", "stream", "file_time", "u_edges_spec", "y_bins", "created_utc",
                      "shearpic_version")}}
    for k, h in list(hists.items()):
        if h.c is None:
            hists[k] = replace(h, c=cfg.c)
        elif not math.isclose(h.c, cfg.c, rel_tol=1e-9):
            raise ValueError(f"histogram c = {h.c} differs from c = {cfg.c} of {run_dir}")
    return hists, cfg, report


def find_product(directory: Path, cfg: RunConfig, time: float, *, kind: str | None = None, file_id: str | None = None,
                 basename: str | None = None, tol: float | None = None,
                 allow_download: bool = False) -> tuple[Path, dict[str, Any]]:
    """The phase-space product of one particle stream nearest ``time``, and a selection report.

    Products are identified by their metadata ``stream`` and ``file_time``; the file name only pre-selects
    them.  ``kind``/``file_id``/``basename`` must leave exactly one stream, and the nearest product must lie
    within ``tol``, by default half the stream's output cadence.  A stream-tagged name wins a tie with an
    older name.
    """
    from shearpic.io.spectrum_files import load_metadata

    directory = Path(directory)
    hint = "run the compute stage, or point --products / PARTICLE_ACCEL_OUTPUT at the products"
    if not directory.is_dir():
        raise FileNotFoundError(f"products directory {directory} does not exist ({hint})")
    found: dict[tuple[str, str, str], list[tuple[float, bool, Path, dict]]] = {}
    for p in sorted(directory.glob("phase_space_*.npz")):
        m = PRODUCT_RE.match(p.name)
        if m is None:
            continue
        refuse_dataless(p, allow_download)
        meta = load_metadata(p)
        st = meta.get("stream") or {}
        stream = (st.get("basename"), st.get("file_id"), st.get("kind"))
        if None in stream or meta.get("file_time") is None:
            raise ValueError(f"{p.name}: the product records no particle stream or file time; recompute it with "
                             "phase_space.py compute")
        if m["tag"] is not None and m["tag"] != ".".join(stream):
            raise ValueError(f"{p.name}: the name says stream {m['tag']!r} but the product was made from "
                             f"{'.'.join(stream)!r} (renamed file?)")
        found.setdefault(stream, []).append((float(meta["file_time"]), m["tag"] is not None, p, meta))
    if not found:
        raise FileNotFoundError(f"no phase_space_<stream>_NNNNN.npz products in {directory} ({hint})")
    present = sorted(found)
    describe = "; ".join(f"basename={b!r} file_id={f!r} kind={k!r} (t = "
                         + ", ".join(f"{e[0]:g}" for e in sorted(found[(b, f, k)], key=lambda e: e[0])) + ")"
                         for b, f, k in present)
    chosen = [s for s in present if (kind is None or s[2] == kind) and (file_id is None or s[1] == file_id)
              and (basename is None or s[0] == basename)]
    if not chosen:
        raise FileNotFoundError(f"no phase-space products for kind={kind!r} file_id={file_id!r} basename={basename!r} "
                                f"in {directory}; present: {describe}")
    if len(chosen) > 1:
        raise ValueError(f"{directory} holds phase-space products of several particle streams ({describe}); "
                         "choose one with --kind, --file-id or --basename")
    stream = chosen[0]
    items = found[stream]
    dt = particle_output_dt(cfg, stream[2], stream[1]) or items[0][3].get("output_dt")
    limit = float(tol) if tol is not None else (0.5 * float(dt) if dt else 1e-6 * max(1.0, abs(time)))
    best = min(items, key=lambda e: (abs(e[0] - time), not e[1]))
    if abs(best[0] - time) > limit + 1e-9 * max(1.0, abs(time)):
        raise FileNotFoundError(f"no {'.'.join(stream)} phase-space product within {limit:g} of t = {time:g} in "
                                f"{directory}; its products are at t = "
                                + ", ".join(f"{e[0]:g}" for e in sorted(items, key=lambda e: e[0])))
    return best[2], {"stream": stream_dict(stream), "requested_time": time, "file_time": best[0],
                     "tolerance": limit, "stream_output_dt": dt,
                     "other_streams_present": [".".join(s) for s in present if s != stream]}


def load_orbit(args) -> tuple[Any, RunConfig, Path]:
    from shearpic.io.trajectory import read_trajectory

    run_dir = find_run(args.traj_run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    cfg.require_particles()
    path = run_dir / args.traj_file
    refuse_dataless(path, args.allow_download)
    traj = read_trajectory(path)
    t0, t1 = args.t_window
    win = traj.time_window(t0, t1)
    if len(win) < 2:
        raise ValueError(f"{path.name}: fewer than 2 samples in {t0:g} <= t <= {t1:g} (file covers "
                         f"{traj.t[0]:g}..{traj.t[-1]:g})")
    return win, cfg, path


# ============================================================================ plot: statistics
def orbit_statistics(win, cfg: RunConfig) -> dict[str, Any]:
    from shearpic.physics.relativity import lorentz_factor

    gamma = lorentz_factor(win.u, cfg.c)
    y = win.x[:, 1]
    layers = np.asarray(cfg.profile.layer_positions, dtype=float)
    y_mean = float(np.mean(y))
    layer = float(layers[int(np.argmin(np.abs(layers - y_mean)))])
    dt = np.diff(win.t)
    out = {"t_first": float(win.t[0]), "t_last": float(win.t[-1]), "n_samples": len(win),
           "dt_min": float(dt.min()), "dt_max": float(dt.max()), "gamma_min": float(gamma.min()),
           "gamma_max": float(gamma.max()), "max_abs_p_over_mc": (np.abs(win.u).max(axis=0) / cfg.c).tolist(),
           "y_min": float(y.min()), "y_max": float(y.max()), "y_mean": y_mean, "band_layer": layer}
    if win.B is not None:
        bmag = np.linalg.norm(win.B, axis=1)
        period = 2 * np.pi * gamma / (cfg.q_mc * np.maximum(bmag, 1e-30))
        out["median_B"] = float(np.median(bmag))
        out["samples_per_gyroperiod_median"] = float(np.median(period[:-1] / dt))
    return out


def histogram_statistics(hists: Mapping[str, Any], cfg: RunConfig, half_width: float,
                         fit_half_width: float) -> dict[str, Any]:
    from shearpic.physics.phase_space import layer_asymmetry

    layers = [float(v) for v in cfg.profile.layer_positions]
    Ly = float(cfg.L[1])
    out: dict[str, Any] = {"out_of_range_fraction": {}, "moments_source": {}, "layers": {}, "variance": {},
                           "mean_u": {}}
    for k, h in hists.items():
        out["out_of_range_fraction"][k] = h.out_of_range_fraction()
        out["moments_source"][k] = h.moments_source()
        out["layers"][k] = [layer_asymmetry(h, L, half_width=half_width, fit_half_width=fit_half_width)
                            for L in layers]
        mean, var = h.moments()
        yc = h.y_centers

        def near(centre, width):
            d = (yc - centre + 0.5 * Ly) % Ly - 0.5 * Ly
            return np.abs(d) <= width

        at_layers = np.zeros(yc.size, bool)
        for L in layers:
            at_layers |= near(L, fit_half_width)
        mid = 0.5 * (layers[0] + layers[-1])  # jet centre; the wind centre is half a box away (periodic)
        bulk = near(mid, fit_half_width) | near(mid + 0.5 * Ly, fit_half_width)
        v_layer = float(np.nanmax(var[at_layers]))
        v_bulk = float(np.nanmean(var[bulk]))
        out["variance"][k] = {"var_layer": v_layer, "y_of_var_layer": float(yc[at_layers][np.nanargmax(var[at_layers])]),
                              "var_bulk": v_bulk, "ratio": v_layer / v_bulk if v_bulk else math.nan,
                              "y_of_global_max": float(yc[np.nanargmax(var)])}
        n = h.n_y if h.moments_source() == "sums" else h.counts.sum(axis=0)
        out["mean_u"][k] = float(np.nansum(mean * n) / np.sum(n))
    return out


def print_statistics(orbit: Mapping[str, Any], hstats: Mapping[str, Any], hist_label: str, orbit_label: str) -> None:
    rows = {f"out-of-range fraction u_{k}": v for k, v in hstats["out_of_range_fraction"].items()}
    rows.update({f"<u_{k}> over all y [U0] ({hstats['moments_source'][k]})": v for k, v in hstats["mean_u"].items()})
    print_table(f"Fig. 6 ensemble ({hist_label})", rows)
    print("\n  layer asymmetry: <u_i> within the half width below / above each layer, d<u_i>/dy near the layer")
    for k, entries in hstats["layers"].items():
        for e in entries:
            sign = "p>0 below, p<0 above" if e["mean_below"] > 0 > e["mean_above"] else "other sign pattern"
            print(f"    u_{k} y_L = {e['layer']:+8.3f}: below {e['mean_below']:+8.3f}  above {e['mean_above']:+8.3f}  "
                  f"slope {e['slope']:+7.3f} U0/a  [{sign}; {e['source']}]")
    print("\n  momentum variance Var u_i(y) [U0^2]: max within the fit half width of the layers vs mean in the bulk")
    for k, v in hstats["variance"].items():
        print(f"    u_{k}: layer {v['var_layer']:9.1f} at y = {v['y_of_var_layer']:+7.2f}   bulk {v['var_bulk']:8.1f}   "
              f"ratio {v['ratio']:5.2f}   global max at y = {v['y_of_global_max']:+7.2f}")
    print_table(f"Fig. 6 orbit ({orbit_label})", {
        "window": f"{orbit['t_first']:.3f} .. {orbit['t_last']:.3f} ({orbit['n_samples']} samples, dt "
                  f"{orbit['dt_min']:.3f}-{orbit['dt_max']:.3f})",
        "gamma range": f"{orbit['gamma_min']:.3f} .. {orbit['gamma_max']:.3f}",
        "max |p_i|/(mc) (x, y, z)": ", ".join(f"{v:.3f}" for v in orbit["max_abs_p_over_mc"]),
        "y range": f"{orbit['y_min']:.2f} .. {orbit['y_max']:.2f} (mean {orbit['y_mean']:.2f})",
        "grey band at layer": orbit["band_layer"],
        "samples per gyro-period (local |B|)": orbit.get("samples_per_gyroperiod_median"),
    })


# ============================================================================ plot: figure
def _segments(x: np.ndarray, y: np.ndarray, Ly: float) -> tuple[np.ndarray, np.ndarray]:
    """Line segments between successive samples, without periodic wrap jumps (|dy| > Ly/2)."""
    seg = np.stack([np.column_stack([x[:-1], y[:-1]]), np.column_stack([x[1:], y[1:]])], axis=1)
    keep = np.abs(np.diff(y)) <= 0.5 * Ly
    return seg[keep], keep


def plot_figure(hists, cfg_h: RunConfig, win, cfg_t: RunConfig, *, t_window, xlim_hist, xlim_traj,
                min_particles: float, orbit: Mapping[str, Any], separate: bool = False, xlim_auto: str = "range"):
    """Draw Fig. 6 and return the figure with a record of its layout.

    Without ``xlim_hist`` the histogram's part of the bottom-row limits is its full momentum range
    (``xlim_auto='range'``) or :func:`visible_range` (``'visible'``).
    """
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.ticker import FixedLocator

    from shearpic.plotting.lines import plot_phase_space
    from shearpic.plotting.style import add_colorbar

    fig, axs = plt.subplots(2, 3, figsize=FIGSIZE, gridspec_kw={"height_ratios": [1, 3], "hspace": 0.03,
                                                                "wspace": 0.05})
    comps = ("x", "y", "z")
    a = 1.0  # shear-layer half width in code units

    # top row: the orbit, in p_i/(mc) with c of the trajectory's run
    Ly_t = float(cfg_t.L[1])
    y = win.x[:, 1]
    norm_t = Normalize(vmin=float(t_window[0]), vmax=float(t_window[1]))
    ylo_box, yhi_box = (float(v) for v in cfg_t.bounds[1])
    if y.min() >= 0:
        ylim_t = (0.0, yhi_box)
    elif y.max() <= 0:
        ylim_t = (ylo_box, 0.0)
    else:
        ylim_t = (ylo_box, yhi_box)
    dens = {k: hists[k].density("total", unit="mc") for k in comps}
    x_rng = (max(float(dens[k][1][0]) for k in comps), min(float(dens[k][1][-1]) for k in comps))
    vmax = max(float(d[0].max()) for d in dens.values())
    h0 = hists["x"]
    vmin = min_particles * h0.c / (h0.n_total * float(np.min(h0.du)) * float(np.min(h0.dy)))
    if not vmin < vmax:  # tiny test data: fewer than min_particles in every bin
        warnings.warn(f"no bin holds {min_particles:g} particles; colour scale from vmax/100", stacklevel=2)
        vmin = vmax / 100.0
    if xlim_hist is not None:
        hist_need, xlim_source = x_rng, "given"
    elif xlim_auto == "visible":
        hist_need, xlim_source = visible_range(dens, vmin, x_rng), "visible (every bin beyond is below vmin)"
    elif xlim_auto == "range":
        hist_need, xlim_source = x_rng, "histogram range"
    else:
        raise ValueError(f"xlim_auto must be 'visible' or 'range', not {xlim_auto!r}")
    xlim_traj, xlim_hist = row_limits(max(orbit["max_abs_p_over_mc"]), hist_need, xlim_traj, xlim_hist, separate)
    lc = None
    for j, comp in enumerate(comps):
        ax = axs[0, j]
        p = win.u[:, j] / cfg_t.c
        seg, keep = _segments(p, y, Ly_t)
        tmid = 0.5 * (win.t[:-1] + win.t[1:])[keep]
        lc = LineCollection(seg, cmap="plasma", norm=norm_t, linewidths=1.5, capstyle="round")
        lc.set_array(tmid)
        ax.add_collection(lc)
        ax.axvline(0.0, color="black", linestyle="--", linewidth=1, alpha=0.5)
        ax.axhspan(orbit["band_layer"] - a, orbit["band_layer"] + a, color="gray", alpha=0.2, linewidth=0)
        ax.set_xlim(*xlim_traj)
        ax.set_ylim(*ylim_t)
        ax.tick_params(direction="in", top=True, right=True)
        if j:
            ax.set_yticklabels([])
    axs[0, 0].set_ylabel(r"$y\ /\ a$")
    cb_t = add_colorbar(axs[0, 2], lc, location="right", label=r"$U_0 t/a$", size="5%", pad=0.05)
    cb_t.ax.tick_params(direction="in")

    # bottom row: f(p_i/mc, y) per unit p_i/(mc) per unit y, normalised by all particles
    greys = ListedColormap(plt.get_cmap("Greys")(np.linspace(0.0, CMAP_TOP, 256)), name="Greys_trunc")
    img = None
    shown: dict[str, dict[str, float]] = {}
    for j, comp in enumerate(comps):
        ax = axs[1, j]
        H, xe, ye = dens[comp]
        img = plot_phase_space(ax, H, xe, ye, log=True, cbar=None, cmap=greys, vmin=vmin, vmax=vmax,
                               interpolation="nearest", rasterized=True,
                               xlabel=rf"$p_{comp}\ /\ (mc)$", ylabel=r"$y\ /\ a$" if j == 0 else "")
        ax.set_xlim(*xlim_hist)
        ax.set_ylim(float(ye[0]), float(ye[-1]))
        for lo, hi in ((xlim_hist[0], xe[0]), (xe[-1], xlim_hist[1])):  # outside the histogram range
            if hi > lo:
                ax.axvspan(lo, hi, facecolor="none", edgecolor="0.75", hatch="///", linewidth=0)
        frac = displayed_fractions(hists[comp], xlim_hist)
        shown[comp] = frac
        ge = r"\geq " if frac["outside_histogram"] > 0 else ""
        for side, x, ha, arrow in (("beyond_left", 0.03, "left", "left"), ("beyond_right", 0.97, "right", "right")):
            if frac[side] > 0:
                pct = percent_text(frac[side]).replace("%", r"\%")
                txt = (rf"$\leftarrow\,{ge}{pct}$" if arrow == "left" else rf"${ge}{pct}\,\rightarrow$")
                ax.text(x, 0.015, txt, transform=ax.transAxes, ha=ha, va="bottom", fontsize=11,
                        bbox={"boxstyle": "round,pad=0.15", "facecolor": "white", "edgecolor": "none", "alpha": 0.85})
        ax.tick_params(direction="in", top=True, right=True)
        ax.minorticks_off()
        if j:
            ax.set_yticklabels([])
    cb_h = add_colorbar(axs[1, 2], img, location="right", label=r"$f(p_i/mc,\ y)\ \ [a^{-1}]$", size="5%", pad=0.05)
    cb_h.ax.tick_params(which="both", direction="in")

    # shared or separate x axes
    same = np.allclose(xlim_traj, xlim_hist)
    for row, lim in ((0, xlim_traj), (1, xlim_hist)):
        for ax in axs[row]:
            ax.xaxis.set_major_locator(FixedLocator(axis_ticks(*lim)))
    for ax in axs[0]:
        if same:
            ax.set_xticklabels([])
        else:
            ax.tick_params(labelbottom=False, labeltop=True)
    return fig, {"vmin": vmin, "vmax": vmax, "xlim_hist": list(xlim_hist), "xlim_traj": list(xlim_traj),
                 "ylim_traj": list(ylim_t), "hist_x_range": list(x_rng), "shared_x": bool(same),
                 "xlim_hist_source": xlim_source, "hist_x_needed": list(hist_need),
                 "fraction_outside_displayed_x": {k: v["outside_displayed"] for k, v in shown.items()},
                 "fraction_beyond_axis_in_histogram": {k: [v["beyond_left"], v["beyond_right"]]
                                                       for k, v in shown.items()},
                 "fraction_outside_histogram": {k: v["outside_histogram"] for k, v in shown.items()},
                 "fraction_definitions": "of all N particles; a bin is displayed when its centre lies within "
                                         "xlim_hist; outside_histogram = outside the momentum/y bins (side unknown)"}


def plot(args) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shearpic.plotting.style import paper_style

    preset = load_preset(args.preset, args.presets)
    hp, tp = preset.get("histogram") or {}, preset.get("trajectory") or {}
    for arg, section, sec_name, key in (("hist_run", hp, "histogram", "run"), ("time", hp, "histogram", "time"),
                                        ("traj_run", tp, "trajectory", "run"), ("traj_file", tp, "trajectory", "file")):
        if getattr(args, arg) is None:
            require_keys(section, [key], f"{args.preset} ({sec_name})",
                         f"pass --{arg.replace('_', '-')} or use --preset fig6_phase_space")
    hists, cfg_h, report = load_histograms(args, preset)
    win, cfg_t, traj_path = load_orbit(args)
    orbit = orbit_statistics(win, cfg_t)
    hstats = histogram_statistics(hists, cfg_h, args.layer_half_width, args.layer_fit_half_width)
    t_hist = next(iter(hists.values())).time
    p_h, p_t = run_parameters(cfg_h), run_parameters(cfg_t)
    print_statistics(orbit, hstats, f"{run_label(cfg_h)}, t = {t_hist:g}", run_label(cfg_t))

    if args.xlim_hist:
        xlim_h = tuple(args.xlim_hist)
    elif args.hist_source == "legacy" and hp.get("xlim") is not None:
        xlim_h = tuple(hp["xlim"])
    else:
        xlim_h = None  # compute product: --xlim-hist-auto
    xlim_t = tuple(args.xlim_traj) if args.xlim_traj else None
    with paper_style(RC):
        fig, layout = plot_figure(hists, cfg_h, win, cfg_t, t_window=args.t_window, xlim_hist=xlim_h,
                                  xlim_traj=xlim_t, min_particles=args.min_particles_per_bin, orbit=orbit,
                                  separate=args.separate_xlim,
                                  xlim_auto=args.xlim_hist_auto if args.hist_source == "npz" else "range")
        if any(v > 0 for v in (x for pair in layout["fraction_beyond_axis_in_histogram"].values() for x in pair)):
            print_table("Fig. 6 particles beyond the bottom-row axis limits (fraction of N; left, right)",
                        {f"u_{k}": f"{lr[0]:.3g}, {lr[1]:.3g} (outside displayed range "
                                   f"{layout['fraction_outside_displayed_x'][k]:.3g})"
                         for k, lr in layout["fraction_beyond_axis_in_histogram"].items()})
        note = (f"Top: orbit of one particle of {run_label(cfg_t)} (c = {cfg_t.c:g}, q/mc = {cfg_t.q_mc:g}, "
                f"M_A = {cfg_t.M_A:g} (B0 = {cfg_t.B0:g}), S = {cfg_t.shear_amplitude:.3g}), p_i/(mc) with c = "
                f"{cfg_t.c:g}. Bottom: all {next(iter(hists.values())).n_total} particles of {run_label(cfg_h)} "
                f"(c = {cfg_h.c:g}, q/mc = {cfg_h.q_mc:g}, M_A = {cfg_h.M_A:g}, S = {cfg_h.shear_amplitude:.3g}) at "
                f"t = {t_hist:g} a/U0, f = dN/(N d(p_i/mc) dy) [1/a] with c = {cfg_h.c:g}; "
                + ", ".join(f"{100 * v:.2f}%" for v in hstats["out_of_range_fraction"].values())
                + " of the particles (x, y, z) lie outside the momentum range.")
        metadata = {
            "figure": "Fig. 6 phase space",
            "caption_note": note,
            "histogram": {"run": run_summary(cfg_h), "parameters": p_h, "time": t_hist, **report},
            "trajectory": {"run": run_summary(cfg_t), "parameters": p_t, "file": str(traj_path),
                           "t_window": list(args.t_window)},
            "density": "f = counts / (N d(p_i/mc) dy), N = all particles incl. out of range; unit 'mc' = u/c of "
                       "the histogram's own run (Jacobian c applied)",
            "colour_scale": {"vmin": layout["vmin"], "vmax": layout["vmax"],
                             "vmin_definition": f"{args.min_particles_per_bin:g} particles in one bin"},
            "layout": layout,
            "statistics": {
                "definitions": {
                    "out_of_range_fraction": "(N - particles inside the momentum and y bins) / N",
                    "layers[].mean_below/above": f"particle-weighted <u_i> over the y bins with centres within "
                                                 f"{args.layer_half_width:g} a below/above the layer [U0]",
                    "layers[].slope": f"least-squares slope of per-bin <u_i>(y) within {args.layer_fit_half_width:g} "
                                      "a of the layer [U0/a]",
                    "variance": f"var_layer = max Var u_i over bins within {args.layer_fit_half_width:g} a of a layer; "
                                f"var_bulk = mean over bins within {args.layer_fit_half_width:g} a of the jet/wind "
                                "centres; 'bins' moments are truncated to the momentum range",
                    "orbit": "window samples; gamma = sqrt(1 + |u|^2/c^2); samples per gyro-period = "
                             "2 pi gamma / (q_mc |B_local| dt), median",
                },
                "histogram": hstats, "orbit": orbit,
            },
        }
        paths = save_figure(fig, figure_dir(args.out), args.name, formats=args.formats, dpi=args.dpi,
                            metadata=metadata)
        plt.close(fig)
    print(f"\ncaption note: {note}")
    for p in paths:
        print(f"wrote {p}")
    return 0


# ============================================================================ CLI
def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=PRESET)
    pre.add_argument("--presets", default=None)
    known, _ = pre.parse_known_args(argv)
    preset = load_preset(known.preset, known.presets)
    hp, tp = dict(preset.get("histogram", {})), dict(preset.get("trajectory", {}))

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="See the module docstring.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)

    def hist_options(p):
        p.add_argument("--time", type=float, default=hp.get("time"), help="snapshot time [a/U0] (default: %(default)s)")

    def stream_options(p, compute: bool):
        if compute:
            p.add_argument("--kind", default=hp.get("kind", "tab"), choices=["tab", "bin"],
                           help="particle output format to read (default: %(default)s)")
            p.add_argument("--file-id", default=hp.get("file_id"), help="particle output block, e.g. out4")
            p.add_argument("--basename", default=hp.get("basename"), help="problem_id prefix of the particle files")
        else:
            p.add_argument("--kind", default=None, choices=["tab", "bin"],
                           help="npz: stream format of the product (default: the only stream present)")
            p.add_argument("--file-id", default=None, help="npz: stream output block of the product, e.g. out4")
            p.add_argument("--basename", default=None, help="npz: stream problem_id prefix of the product")
        p.add_argument("--tol", type=float, default=None,
                       help="max |t - time| of the particle output / product (default: half the stream's cadence)")

    def compute_options(p):
        p.add_argument("--run", default=hp.get("run"), help="run of the particle outputs (default: %(default)s)")
        p.add_argument("--u-edges", default=hp.get("u_edges", "lin:-600:600:240"),
                       help="momentum bins in u = p/m [U0] (default: %(default)s)")
        p.add_argument("--y-bins", type=int, default=hp.get("y_bins", 100))
        p.add_argument("--expected-n", type=int, default=None, help="required particle number (default: cfg.n_par)")
        p.add_argument("--force", action="store_true")
        p.add_argument("--progress", action="store_true")

    def plot_options(p, hist_source_default):
        p.add_argument("--hist-source", default=hist_source_default, choices=["legacy", "npz"])
        p.add_argument("--hist-run", default=hp.get("run"))
        p.add_argument("--legacy-file", default=hp.get("legacy_file", "phase_space_histograms.npz"))
        p.add_argument("--n-total", type=int, default=hp.get("n_total"),
                       help="particles in the legacy snapshot (default: preset, else cfg.n_par)")
        p.add_argument("--product", default=None, help="explicit phase-space product file (npz source)")
        p.add_argument("--traj-run", default=tp.get("run"))
        p.add_argument("--traj-file", default=tp.get("file"))
        p.add_argument("--t-window", type=float, nargs=2, default=tp.get("t_window", [310, 330]))
        p.add_argument("--xlim-hist", type=float, nargs=2, default=None,
                       help="p/(mc) limits of the bottom row (default: legacy source: the preset xlim; npz: "
                            "--xlim-hist-auto)")
        p.add_argument("--xlim-hist-auto", default=hp.get("xlim_auto", "visible"), choices=["visible", "range"],
                       help="npz source without --xlim-hist: 'visible' = smallest symmetric range beyond which every "
                            "bin is below vmin, 'range' = the full histogram range (default: %(default)s)")
        p.add_argument("--xlim-traj", type=float, nargs=2, default=tp.get("xlim"), help="p/(mc) limits, top row")
        p.add_argument("--separate-xlim", action="store_true",
                       help="own x limits per row instead of one shared p/(mc) axis per column")
        p.add_argument("--min-particles-per-bin", type=float, default=hp.get("min_particles_per_bin", 10))
        p.add_argument("--layer-half-width", type=float, default=hp.get("layer_half_width", 10.0))
        p.add_argument("--layer-fit-half-width", type=float, default=hp.get("layer_fit_half_width", 4.0))
        p.add_argument("--name", default=preset.get("figure", "phase_space"))

    pc = sub.add_parser("compute", help="HPC: phase-space histograms of a full particle output")
    hist_options(pc)
    stream_options(pc, compute=True)
    compute_options(pc)
    add_common_arguments(pc, PRESET)
    pp = sub.add_parser("plot", help="Fig. 6 from a product or the legacy file")
    hist_options(pp)
    stream_options(pp, compute=False)
    plot_options(pp, hp.get("source", "legacy"))
    add_common_arguments(pp, PRESET)
    pa = sub.add_parser("all", help="compute, then plot the product just computed")
    hist_options(pa)
    stream_options(pa, compute=True)
    compute_options(pa)
    plot_options(pa, "npz")
    add_common_arguments(pa, PRESET)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser(argv).parse_args(argv)
    if args.stage == "compute":
        return compute(args)
    if args.stage == "plot":
        return plot(args)
    status = compute(args)
    if status or mpi_rank(args.backend) != 0:
        return status
    args.hist_run = args.run
    args.hist_source = "npz"
    args.product = args.computed_product  # exactly the product of this stream and time
    return plot(args)


if __name__ == "__main__":
    run_cli(main)
