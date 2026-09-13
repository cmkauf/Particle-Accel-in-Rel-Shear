"""Figures 3, 4 and 7 of arXiv:2512.12720: the statistically steady state of the driven runs.

Each subcommand takes its defaults from a preset in ``paper_figures.yaml``:

``turbulence``  Fig. 3 (``fig3_turbulence``)
    Time-averaged shell spectra E(k) [rho0 U0^2 a] of the kinetic fluctuations
    sqrt(rho) (v - <rho v>_x/<rho>_x) and the magnetic fluctuations B - <B>_x, with a power law
    fitted over ``--k-fit``.  ``--stage compute`` writes ``<products>/turbulence_spectra.<output>.npz``
    and ``--stage plot`` reads it together with the athinput and the .hst.
``energy``      Fig. 4 (``fig4_steady_state``)
    Volume-averaged energy densities eps_k (including the mean shear flow), eps_m and
    eps_p = m_cr KE_cr / V of a driven and a decaying run, from their history files and
    smoothed with a cubic smoothing spline.
``power``       Fig. 7 (``fig7_power``)
    dE_p/dt and the components ``P_ideal,i = sum_p m_cr q_mc v_i cE_i`` from the history
    file, as totals over the box [rho0 U0^3 a^2].

Usage::

    python analysis/steady_state.py turbulence --stage all --formats pdf,png
    python analysis/steady_state.py energy --formats pdf,png
    python analysis/steady_state.py power --formats pdf,png

    # on a compute node, spectra of every steady-state snapshot of the full run
    python analysis/steady_state.py turbulence --stage compute --t-range 200 2400 --workers auto
"""

from __future__ import annotations

import argparse
import math
import sys
import warnings
from functools import partial
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:  # also when imported with importlib (tests)
    sys.path.insert(0, str(HERE))

from common import (add_common_arguments, check_same_run, figure_dir, find_run, load_preset,  # noqa: E402
                    print_table, product_dir, provenance, require_keys, run_cli, run_summary, save_figure,
                    workers_arg)

from shearpic.config import RunConfig  # noqa: E402
from shearpic.io.field_spectra import load_field_spectra, save_field_spectra, snapshot_mhd_spectra  # noqa: E402
from shearpic.io.history import read_hst  # noqa: E402
from shearpic.physics.diagnostics import (energy_budget, history_diagnostics, smooth_series,  # noqa: E402
                                          smoothing_lam)
from shearpic.physics.turbulence import (average_spectra, fit_power_law, local_slopes,  # noqa: E402
                                         nyquist_wavenumber, power_law_amplitude, rebin_log)

COLORS = {"kin": "#D2665A", "mag": "#536493", "tot": "k", "p": "#6A0066", "ratio": "#6A0066",
          "Px": "#536493", "Py": "#FFD369", "Pz": "#D2665A"}

ESTIMATOR = {"velocity_mean": "favre", "weight": "sqrt_rho", "magnetic_mean": "reynolds", "prefactor": 0.5,
             "shells": "linear, centred on m*dk", "k_max": "nyquist", "fft": "numpy.fft.fftn(norm='forward')"}

FIG3_DEFAULTS = {"output": "out2", "fit_range": [3.0, 15.0], "fit_weighting": "log",
                 "systematic_fit_ranges": [[3.0, 10.0], [3.0, 15.0], [3.0, 20.0], [5.0, 15.0], [5.0, 20.0]],
                 "local_slope_window": 0.5, "max_local_slope_change": 0.5, "n_per_decade": 8,
                 "ratio_ranges": [[2.0, 30.0], [30.0, 80.0]], "xlim": [2 * math.pi / 40, None], "ylim": None,
                 "ratio_ylim": [1e-2, 9.9], "guide_offset": 4.0}
FIG4_DEFAULTS = {"t_max": 600.0, "smoothing_period": 16.7, "ylim_top": [0.001, 0.61], "ylim_bottom": [0.49, 0.88],
                 "ytick_top": 0.1, "ytick_bottom": 0.05}
FIG7_DEFAULTS = {"t_steady": [100.0, 600.0], "smoothing_period": 16.7, "xlim": [0.0, 600.0], "ylim": [-1.0, 4.0],
                 "ratio_threshold": 0.98}


# ============================================================================ helpers
SUBCOMMAND_PRESETS = {"turbulence": "fig3_turbulence", "energy": "fig4_steady_state", "power": "fig7_power"}


def _preset(args: argparse.Namespace, defaults: dict[str, Any], required: Sequence[str] = (),
            optional: dict[str, Any] | None = None) -> dict[str, Any]:
    """``defaults`` updated with the preset, which must hold ``required`` and every ``optional`` key not given as an option."""
    raw = load_preset(args.preset, args.presets)
    needed = [key for key, cli in (optional or {}).items() if cli is None] + list(required)
    command = getattr(args, "command", None)
    default = SUBCOMMAND_PRESETS.get(command)
    hint = f"`steady_state.py {command}` needs a preset like {default} (use --preset {default})" if default else ""
    require_keys(raw, needed, args.preset, hint)
    p = dict(defaults)
    p.update(raw)
    return p


def _formatter_for_step(step: float) -> str:
    """printf format with just enough decimals for ticks spaced by ``step``, e.g. '%.2f' for 0.05."""
    decimals = max(0, -int(math.floor(math.log10(step) + 1e-9)))
    if abs(round(step * 10 ** decimals) - step * 10 ** decimals) > 1e-9:
        decimals += 1
    return f"%.{decimals}f"


def _style(font_size: float, tick_size: float, *, tick_length: float = 3.5, tick_width: float = 0.8,
           minor_linear: bool = False) -> dict[str, Any]:
    """rcParams of one figure on top of the shearpic paper style."""
    return {
        # font.size stays at 10 pt because tight_layout pads scale with it
        "font.size": 10, "axes.labelsize": font_size, "legend.fontsize": font_size,
        "xtick.labelsize": tick_size, "ytick.labelsize": tick_size,
        "xtick.minor.visible": minor_linear, "ytick.minor.visible": minor_linear,
        "xtick.major.size": tick_length, "ytick.major.size": tick_length,
        "xtick.major.width": tick_width, "ytick.major.width": tick_width,
        "xtick.minor.size": 0.57 * tick_length, "ytick.minor.size": 0.57 * tick_length,
        "xtick.minor.width": 0.5 * tick_width, "ytick.minor.width": 0.5 * tick_width,
        "axes.linewidth": 0.8, "lines.linewidth": 2.0, "legend.frameon": False,
        "savefig.pad_inches": 0.1,
    }


STYLE_FIG3 = _style(22, 20)
STYLE_FIG4 = _style(16, 16, tick_length=5, tick_width=0.5)
STYLE_FIG7 = _style(17, 15, tick_length=5, tick_width=0.5)


def _save_styled(fig, rc: dict[str, Any], args, name: str, metadata: dict[str, Any]) -> list[Path]:
    """Save under the same rcParams the figure was drawn with (fonts are resolved when drawing)."""
    from shearpic.plotting import paper_style

    with paper_style(rc):
        return save_figure(fig, figure_dir(args.out), name, formats=args.formats, dpi=args.dpi, metadata=metadata)


def _window(t: np.ndarray, lo: float, hi: float) -> np.ndarray:
    sel = (t >= lo - 1e-9) & (t <= hi + 1e-9)
    if sel.sum() < 2:
        raise ValueError(f"fewer than 2 samples in [{lo}, {hi}] (data cover {t.min():g}..{t.max():g})")
    return sel


# ============================================================================ Fig. 3
FIT_WEIGHTINGS = ("log", "shell")


def turbulence_product_path(run_dir: Path, products: str | None, output: str, *, create: bool = True) -> Path:
    """``<products>/turbulence_spectra.<output>.npz``; ``create=False`` for read-only (plot) stages."""
    return product_dir(run_dir, products, create=create) / f"turbulence_spectra.{output}.npz"


def time_tolerance(t: float, dt: float | None) -> float:
    """Tolerance ``max(1e-3 dt, 1e-6 max(1, |t|))`` within which two snapshot times are the same time.

    The same rule as :func:`shearpic.io.athdf.select_snapshots`.
    """
    return max(1e-3 * float(dt) if dt else 0.0, 1e-6 * max(1.0, abs(float(t))))


def check_unique_times(records: Sequence[dict[str, Any]], dt: float | None, where: str) -> None:
    """Raise ValueError when two snapshot records share a file or a time, which a time average would count twice."""
    files: dict[str, float] = {}
    for rec in records:
        name = rec.get("file")
        if name is not None and name in files:
            raise ValueError(f"duplicate snapshot in {where}: {name} appears twice (t = {files[name]:.6g}); "
                             "a time average would count it twice")
        if name is not None:
            files[name] = rec["time"]
    order = sorted(records, key=lambda r: r["time"])
    for a, b in zip(order, order[1:]):
        if abs(b["time"] - a["time"]) <= time_tolerance(b["time"], dt):
            raise ValueError(f"duplicate snapshot time in {where}: t = {a['time']:.6g} ({a.get('file')}) and "
                             f"t = {b['time']:.6g} ({b.get('file')}); a time average would count it twice")


def _select_paths(args, cfg: RunConfig, output: str, times) -> list[Path]:
    from shearpic.io.athdf import Snapshot, list_snapshots, select_snapshots

    dt = cfg.output_dt("hdf5")
    if args.t_range is not None:
        lo, hi = args.t_range
        every = list_snapshots(cfg.run_dir, output)
        local = list_snapshots(cfg.run_dir, output, skip_dataless=not args.allow_download)
        if len(local) < len(every):
            skipped = sorted(set(every) - set(local))
            print(f"note: skipping {len(skipped)} online-only placeholder(s) (times unknown without downloading): "
                  f"{', '.join(p.name for p in skipped)}; pass --allow-download to include them")
        found = [(p, Snapshot.open(p).time) for p in local]
        found = [(p, t) for p, t in found if lo - 1e-3 <= t <= hi + 1e-3]
        if not found:
            raise FileNotFoundError(f"no local {output} snapshots with {lo} <= t <= {hi} in {cfg.run_dir}")
        check_unique_times([{"time": t, "file": p.name} for p, t in found], dt,
                           f"the {output} snapshots of {cfg.run_dir} with {lo:g} <= t <= {hi:g}")
        return [p for p, _ in found]
    paths = select_snapshots(cfg.run_dir, times, output=output, dt=dt, tol=args.tol,
                             allow_download=args.allow_download)
    first: dict[Path, float] = {}
    for t, p in zip(times, paths):
        if p in first:
            raise ValueError(f"requested times {first[p]:g} and {t:g} both resolve to {p.name}; "
                             "request distinct output times (see --tol)")
        first[p] = float(t)
    return paths


def _estimator_meta(dk: float | None, shape, L) -> dict[str, Any]:
    from shearpic.physics.turbulence import shell_width

    return {**ESTIMATOR, "dk": float(dk if dk is not None else shell_width(shape, L)),
            "k_nyquist": nyquist_wavenumber(shape, L)}


def compute_turbulence(args, preset: dict[str, Any]) -> Path | None:
    """Compute stage of Fig. 3: write the per-snapshot spectra to the product; None on MPI ranks > 0."""
    from shearpic.parallel import parallel_map

    run_dir = find_run(args.run or preset["run"], args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    output = preset["output"]
    times = args.times if args.times is not None else preset.get("times")
    dt = cfg.output_dt("hdf5")
    paths = _select_paths(args, cfg, output, times)
    shape = tuple(n for n in cfg.nx if n > 1)
    worker = partial(snapshot_mhd_spectra, cfg=cfg, dk=args.dk, k_max="nyquist",
                     velocity_mean=ESTIMATOR["velocity_mean"], weight=ESTIMATOR["weight"],
                     allow_download=args.allow_download)
    results = parallel_map(worker, paths, backend=args.backend, n_workers=workers_arg(args.workers),
                           mem_per_task="2GB", progress=True, desc="spectra")
    if results is None:  # MPI rank > 0
        return None
    estimator = _estimator_meta(args.dk, shape, tuple(l for l, n in zip(cfg.L, cfg.nx) if n > 1))
    records = [{"time": r["time"], "file": Path(r["path"]).name, "grid": r["grid"]} for r in results]
    series = [r["spectra"] for r in results]
    check_unique_times(records, dt, "the computed snapshots")

    path = turbulence_product_path(run_dir, args.products, output)
    if path.exists():
        series, records = _merge_existing(path, series, records, estimator, cfg)
        check_unique_times(records, dt, f"{path} merged with the new snapshots (delete it and recompute)")
    order = np.argsort([rec["time"] for rec in records])
    series = [series[i] for i in order]
    records = [records[i] for i in order]
    meta = provenance(kind="turbulence_spectra", run=run_summary(cfg), output=output, estimator=estimator,
                      snapshots=records, units={"k": "1/a", "E": "rho0 U0^2 a", "total": "rho0 U0^2"})
    save_field_spectra(path, series, meta)

    rows = {}
    for rec, spec in zip(records, series):
        g = rec["grid"]
        rows[f"t = {rec['time']:.4g}"] = (
            f"eps_k' spectrum {spec['kin'].total:.6f} vs grid {g['eps_kin_fluct']:.6f} "
            f"(rel {abs(spec['kin'].total / g['eps_kin_fluct'] - 1):.1e}); "
            f"eps_m' {spec['mag'].total:.6f} vs {g['eps_mag_fluct']:.6f} "
            f"(rel {abs(spec['mag'].total / g['eps_mag_fluct'] - 1):.1e}); beyond k_Nyq {spec['tot'].energy_beyond_kmax:.2e}")
    print_table(f"Fig. 3 compute: Parseval totals sum(E dk) + beyond = 0.5<rho u''^2>, 0.5<B'^2> [{path}]", rows)
    return path


def _merge_existing(path: Path, series, records, estimator, cfg):
    """Entries of the existing product that are not recomputed plus the new ones; an incompatible product is replaced."""
    try:
        old_series, old_meta = load_field_spectra(path)
    except (OSError, ValueError, KeyError) as err:
        warnings.warn(f"replacing unreadable product {path}: {err}")
        return series, records
    if not isinstance(old_series, list) or old_meta.get("estimator") != estimator:
        warnings.warn(f"replacing {path}: it was computed with a different estimator")
        return series, records
    try:
        check_same_run(old_meta.get("run"), cfg, what=str(path))
    except ValueError as err:
        warnings.warn(f"replacing {path}: {err}")
        return series, records
    if len(old_meta.get("snapshots", [])) != len(old_series):
        warnings.warn(f"replacing {path}: its snapshot records are inconsistent")
        return series, records
    dt = cfg.output_dt("hdf5")
    new_files = {rec["file"] for rec in records}
    keep = [i for i, rec in enumerate(old_meta["snapshots"])
            if rec.get("file") not in new_files
            and not any(abs(rec["time"] - new["time"]) <= time_tolerance(new["time"], dt) for new in records)]
    print(f"note: merging into {path.name}: kept {len(keep)} earlier snapshot(s), (re)computed {len(records)}")
    return [old_series[i] for i in keep] + list(series), [old_meta["snapshots"][i] for i in keep] + list(records)


def load_turbulence_product(path: Path, times: Sequence[float] | None = None, t_range=None, tol: float | None = None,
                            dt: float | None = None):
    """Spectra, snapshot records and metadata of the product for the requested times or ``t_range``.

    Each time is matched to the nearest stored snapshot within ``tol`` (default
    :func:`time_tolerance` with the output cadence ``dt``); ambiguous matches raise ValueError.
    """
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist; run `steady_state.py turbulence --stage compute` first "
                                "(or point PARTICLE_ACCEL_OUTPUT / --products at the directory holding it)")
    series, meta = load_field_spectra(path)
    records = meta["snapshots"]
    available = np.array([r["time"] for r in records])
    if t_range is not None:
        idx = [i for i, t in enumerate(available) if t_range[0] - 1e-3 <= t <= t_range[1] + 1e-3]
        if not idx:
            raise ValueError(f"no snapshots with {t_range[0]} <= t <= {t_range[1]} in {path} "
                             f"(available: {np.round(available, 3).tolist()})")
    else:
        idx = []
        for t in times:
            d = np.abs(available - t)
            limit = tol if tol is not None else time_tolerance(t, dt)
            if d.size == 0 or d.min() > limit:
                raise ValueError(f"t = {t} not in {path} within {limit:g} (available: {np.round(available, 3).tolist()}); "
                                 "run the compute stage for it")
            i = int(np.argmin(d))
            if i in idx:
                raise ValueError(f"requested times {times[idx.index(i)]:g} and {t:g} both match the snapshot at "
                                 f"t = {available[i]:.6g} in {path}")
            idx.append(i)
    selected = [records[i] for i in idx]
    check_unique_times(selected, dt, f"{path} (selected snapshots)")
    return [series[i] for i in idx], selected, meta


def _fit_or_nan(spec, k_lo: float, k_hi: float, weighting: str) -> float:
    try:
        return fit_power_law(spec, k_lo, k_hi, weighting=weighting)[0]
    except ValueError:
        return float("nan")


def turbulence_statistics(series, records, cfg: RunConfig, preset: dict[str, Any], hst=None) -> dict[str, Any]:
    """Statistics of Fig. 3: spectral slopes, E_m/E_k and the Parseval and history checks.

    Slopes are weighted least-squares fits of ln E against ln k on the linear shells of the
    time-averaged spectra over ``fit_range``; ``'log'`` weighting gives equal weight per log k.
    ``systematic`` holds the fits over ``systematic_fit_ranges`` with both weightings, and the
    E_m/E_k averages are means per log k.
    """
    k_lo, k_hi = (float(v) for v in preset["fit_range"])
    weighting = str(preset["fit_weighting"])
    if weighting not in FIT_WEIGHTINGS:
        raise ValueError(f"fit_weighting must be one of {FIT_WEIGHTINGS}, got {weighting!r}")
    window = float(preset["local_slope_window"])
    avg = {name: average_spectra([s[name] for s in series]) for name in ("kin", "mag", "tot")}
    shape = tuple(n for n in cfg.nx if n > 1)
    L = tuple(l for l, n in zip(cfg.L, cfg.nx) if n > 1)
    k_nyq = nyquist_wavenumber(shape, L)
    stats: dict[str, Any] = {"times": [r["time"] for r in records], "fit_range": [k_lo, k_hi], "fit_weighting": weighting,
                             "systematic_fit_ranges": [[float(a), float(b)] for a, b in preset["systematic_fit_ranges"]],
                             "local_slope_window_decades": window,
                             "dk": float(avg["tot"].dk[1]), "k_nyquist": k_nyq}
    other = "shell" if weighting == "log" else "log"
    ends = [(k_lo, min(k_hi, k_lo * 10 ** window)), (max(k_lo, k_hi / 10 ** window), k_hi)]
    for name in ("kin", "mag", "tot"):
        slope, err = fit_power_law(avg[name], k_lo, k_hi, weighting=weighting)
        per = [fit_power_law(s[name], k_lo, k_hi, weighting=weighting)[0] for s in series]
        stats[f"slope_{name}"] = slope
        stats[f"slope_{name}_stderr"] = err
        stats[f"slope_{name}_{other}_weighting"] = _fit_or_nan(avg[name], k_lo, k_hi, other)
        stats[f"slope_{name}_snapshots"] = per
        stats[f"slope_{name}_scatter"] = float(np.std(per, ddof=1)) if len(per) > 1 else float("nan")
        fits = [{"k_min": lo, "k_max": hi, "weighting": w, "slope": _fit_or_nan(avg[name], lo, hi, w)}
                for lo, hi in stats["systematic_fit_ranges"] for w in FIT_WEIGHTINGS]
        finite = [f["slope"] for f in fits if np.isfinite(f["slope"])] + [slope]
        stats[f"slope_{name}_systematic"] = {"min": float(min(finite)), "max": float(max(finite)), "fits": fits}
        stats[f"local_slopes_{name}"] = local_slopes(avg[name], window, k_min=2 * math.pi / cfg.Lx, k_max=k_nyq,
                                                     weighting=weighting)
        stats[f"slope_{name}_fit_range_ends"] = [row["slope"] for row in local_slopes(avg[name], ends, weighting=weighting)]
    change = abs(stats["slope_tot_fit_range_ends"][1] - stats["slope_tot_fit_range_ends"][0])
    stats["fit_range_ends_windows"] = [list(e) for e in ends]
    stats["fit_range_local_slope_change"] = float(change)
    stats["fit_range_single_power_law"] = bool(change <= float(preset["max_local_slope_change"]))
    stats["slope_tot_logbins"] = _fit_or_nan(rebin_log(avg["tot"], int(preset["n_per_decade"])), k_lo, k_hi, weighting)
    k = avg["kin"].k
    with np.errstate(divide="ignore"):
        dlnk = np.where(k > 0, avg["kin"].dk / k, 0.0)
    for lo, hi in preset["ratio_ranges"]:
        sel = (k > lo) & (k < hi) & (avg["kin"].E > 0)
        if not sel.any():  # range beyond the resolved shells
            stats[f"ratio_logmean_{lo:g}_{hi:g}"] = stats[f"ratio_energy_{lo:g}_{hi:g}"] = float("nan")
            continue
        ratio = avg["mag"].E[sel] / avg["kin"].E[sel]
        stats[f"ratio_logmean_{lo:g}_{hi:g}"] = float(np.dot(dlnk[sel], ratio) / dlnk[sel].sum())
        stats[f"ratio_energy_{lo:g}_{hi:g}"] = float(avg["mag"].bin_energy[sel].sum() / avg["kin"].bin_energy[sel].sum())
    shell = avg["kin"].k_edges[:-1] > 0
    stats["k_peak_kin"] = float(k[shell][np.argmax(avg["kin"].E[shell])])
    stats["eps_kin_fluct_mean"] = float(np.mean([s["kin"].total for s in series]))
    stats["eps_mag_fluct_mean"] = float(np.mean([s["mag"].total for s in series]))
    stats["parseval"] = [{"time": r["time"], "kin_spectrum": s["kin"].total, "kin_grid": r["grid"]["eps_kin_fluct"],
                          "mag_spectrum": s["mag"].total, "mag_grid": r["grid"]["eps_mag_fluct"],
                          "tot_in_bins": s["tot"].energy_in_bins, "tot_beyond_kmax": s["tot"].energy_beyond_kmax,
                          "eps_kin_grid": r["grid"]["eps_kin"], "eps_mag_grid": r["grid"]["eps_mag"]}
                         for s, r in zip(series, records)]
    if hst is not None:
        t = hst["time"].to_numpy()
        dt_hst = cfg.output_dt("hst") or (float(np.median(np.diff(t))) if t.size > 1 else 0.0)
        stats["hst_dt"] = dt_hst
        for row in stats["parseval"]:
            i = int(np.argmin(np.abs(t - row["time"]))) if t.size else None
            covered = i is not None and abs(t[i] - row["time"]) <= 0.5 * dt_hst + 1e-6 * max(1.0, abs(row["time"]))
            row["hst_time"] = float(t[i]) if covered else None
            row["eps_kin_hst"] = float(hst["eps_kin"].iloc[i]) if covered else None
            row["eps_mag_hst"] = float(hst["eps_mag"].iloc[i]) if covered else None
    return stats


def guide_label(slope: float, k_lo: float, k_hi: float) -> str:
    """Label of the fitted guide line, e.g. ``$\\propto k^{-1.6}$ ($3 \\leq k \\leq 15$)``."""
    return fr"$\propto k^{{{slope:.1f}}}$ (${k_lo:g} \leq k \leq {k_hi:g}$)"


def wavelength_ticks(xlim: Sequence[float]) -> tuple[list[float], list[float]]:
    """Major (lambda = 10^j) and minor (lambda = m 10^j) tick wavenumbers k = 2 pi/lambda inside ``xlim``."""
    lam_lo, lam_hi = 2 * math.pi / xlim[1], 2 * math.pi / xlim[0]
    major, minor = [], []
    for j in range(math.floor(math.log10(lam_lo)), math.ceil(math.log10(lam_hi)) + 1):
        for m in range(1, 10):
            lam = m * 10.0 ** j
            if lam_lo * (1 - 1e-9) <= lam <= lam_hi * (1 + 1e-9):
                (major if m == 1 else minor).append(2 * math.pi / lam)
    return sorted(major), sorted(minor)


def draw_turbulence(avg: dict, cfg: RunConfig, stats: dict[str, Any], preset: dict[str, Any]):
    """Fig. 3 layout: log-binned E_k, E_m, E_k + E_m with the fitted guide (top) and E_m/E_k (bottom)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

    from shearpic.plotting import paper_style

    n_dec = int(preset["n_per_decade"])
    disp = {name: rebin_log(spec, n_dec) for name, spec in avg.items()}
    k_nyq = stats["k_nyquist"]
    k_box = 2 * math.pi / cfg.Lx
    xlim = [preset["xlim"][0] if preset["xlim"][0] is not None else 0.75 * k_box,
            preset["xlim"][1] if preset["xlim"][1] is not None else k_nyq]
    k_lo, k_hi = stats["fit_range"]

    with paper_style(STYLE_FIG3):
        fig, (ax1, ax3) = plt.subplots(2, 1, figsize=(8, 7), gridspec_kw={"height_ratios": [2, 1], "hspace": 0})
        labels = {"kin": r"$E_k$", "mag": r"$E_m$", "tot": r"$E_k + E_m$"}
        for name in ("kin", "mag", "tot"):
            s = disp[name]
            ok = (s.k_edges[:-1] > 0) & (s.E > 0)
            ax1.loglog(s.k[ok], s.E[ok], color=COLORS[name], ls="-", lw=3, label=labels[name])
        slope = stats["slope_tot"]
        amp = power_law_amplitude(avg["tot"], slope, k_lo, k_hi, weighting=stats["fit_weighting"])
        amp *= float(preset["guide_offset"])
        kg = np.geomspace(k_lo, k_hi, 50)
        ax1.loglog(kg, amp * kg ** slope, "k--", lw=2)
        ax1.annotate(guide_label(slope, k_lo, k_hi), xy=(k_lo, 2.0 * amp * k_lo ** slope), fontsize=20,
                     ha="left", va="bottom")
        ax1.axvline(k_box, color="gray", ls="-", lw=5, alpha=0.6, label=r"$k = 2\pi/L_x$")
        ok = (disp["tot"].k_edges[:-1] > 0) & (disp["tot"].E > 0) & (disp["tot"].k <= xlim[1])
        if preset.get("ylim"):
            ylim = preset["ylim"]
        else:
            top = disp["tot"].E[ok].max()
            bottom = disp["mag"].E[ok & (disp["mag"].E > 0)].min()
            ylim = [10 ** math.floor(math.log10(bottom)), 10 ** math.ceil(math.log10(top) + 0.7)]
        ax1.set_ylim(*ylim)
        ax1.set_ylabel(r"$E(k)\ [\rho_0 U_0^2 a]$")
        ax1.legend(loc="lower left")
        ax1.tick_params(labelbottom=False)

        ratio = np.full(disp["kin"].E.shape, np.nan)
        good = (disp["kin"].E > 0) & (disp["kin"].k_edges[:-1] > 0)
        ratio[good] = disp["mag"].E[good] / disp["kin"].E[good]
        ax3.loglog(disp["kin"].k[good], ratio[good], color=COLORS["ratio"], ls="-", lw=3)
        ax3.axhline(1.0, color="k", ls="--", lw=2)
        ax3.axvline(k_box, color="gray", ls="-", lw=5, alpha=0.6)
        ax3.set_ylim(*preset["ratio_ylim"])
        ax3.set_xlabel(r"$k\ [1/a]$")
        ax3.set_ylabel(r"$E_m(k)\ /\ E_k(k)$")

        ax2 = ax1.twiny()
        ax2.set_xscale("log")
        for ax in (ax1, ax2, ax3):
            ax.set_xlim(*xlim)
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        # the top spine carries only the wavelength ticks
        ax1.tick_params(axis="x", which="both", top=False)
        major, minor = wavelength_ticks(xlim)
        ax2.xaxis.set_major_locator(FixedLocator(major))
        ax2.xaxis.set_minor_locator(FixedLocator(minor))
        ax2.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{2 * math.pi / x:.3g}"))
        ax2.xaxis.set_minor_formatter(NullFormatter())
        ax2.set_xlabel(r"$\lambda\ [a]$")
        fig.align_ylabels([ax1, ax3])
        fig.tight_layout()  # adjusts the margins; the GridSpec keeps hspace = 0
    return fig, disp


def plot_turbulence(args, preset: dict[str, Any]) -> list[Path]:
    """Plot stage of Fig. 3: time-averaged spectra from the product, statistics, figure + JSON sidecar."""
    import matplotlib.pyplot as plt

    run_dir = find_run(args.run or preset["run"], args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    path = turbulence_product_path(run_dir, args.products, preset["output"], create=False)
    times = args.times if args.times is not None else preset.get("times")
    series, records, meta = load_turbulence_product(path, times=times, t_range=args.t_range, tol=args.tol,
                                                    dt=cfg.output_dt("hdf5"))
    check_same_run(meta.get("run"), cfg, what=str(path))
    hst = history_diagnostics(cfg.hst_path, cfg) if cfg.hst_path else None
    stats = turbulence_statistics(series, records, cfg, preset, hst)
    avg = {name: average_spectra([s[name] for s in series]) for name in ("kin", "mag", "tot")}
    fig, disp = draw_turbulence(avg, cfg, stats, preset)

    k_lo, k_hi = stats["fit_range"]
    w = stats["fit_weighting"]
    other = "shell" if w == "log" else "log"
    ranges = ", ".join(f"[{a:g},{b:g}]" for a, b in stats["systematic_fit_ranges"])
    rows = {"snapshot times [a/U0]": ", ".join(f"{t:.4g}" for t in stats["times"])}
    for name, label in (("kin", "E_k"), ("mag", "E_m"), ("tot", "E_k+E_m")):
        sysm = stats[f"slope_{name}_systematic"]
        rows[f"slope {label} on {k_lo:g}<=k<={k_hi:g}"] = (
            f"{stats[f'slope_{name}']:.3f} ({w}-weighted fit of the time average; nominal stderr "
            f"{stats[f'slope_{name}_stderr']:.3f}); {other}-weighted {stats[f'slope_{name}_{other}_weighting']:.3f}; "
            f"snapshots {np.mean(stats[f'slope_{name}_snapshots']):.3f} +- {stats[f'slope_{name}_scatter']:.3f} (std)")
        rows[f"slope {label} systematic"] = (f"{sysm['min']:.2f} .. {sysm['max']:.2f} (fits over k ranges {ranges} "
                                             "x log/shell weighting)")
    rows["slope E_k+E_m, log bins"] = f"{stats['slope_tot_logbins']:.3f} ({preset['n_per_decade']}/decade, {w}-weighted)"
    e0, e1 = stats["fit_range_ends_windows"]
    s0, s1 = stats["slope_tot_fit_range_ends"]
    rows["local slope E_k+E_m at the fit-range ends"] = (
        f"{s0:.2f} ({e0[0]:.3g}<=k<={e0[1]:.3g}), {s1:.2f} ({e1[0]:.3g}<=k<={e1[1]:.3g}); "
        + f"change {stats['fit_range_local_slope_change']:.2f} "
        + (f"(<= {preset['max_local_slope_change']:g} tolerated)" if stats["fit_range_single_power_law"] else
           f"> {preset['max_local_slope_change']:g}: NOT a single power law"))
    for lo, hi in preset["ratio_ranges"]:
        rows[f"E_m/E_k, {lo:g}<k<{hi:g}"] = (f"{stats[f'ratio_logmean_{lo:g}_{hi:g}']:.3f} (mean per log k of <E_m>/<E_k>); "
                                            f"{stats[f'ratio_energy_{lo:g}_{hi:g}']:.3f} (ratio of shell-summed energies)")
    rows["k of E_k maximum"] = f"{stats['k_peak_kin']:.3f} (2 pi/Lx = {2 * math.pi / cfg.Lx:.3f})"
    rows["<eps_k'>, <eps_m'>"] = (f"{stats['eps_kin_fluct_mean']:.5f}, {stats['eps_mag_fluct_mean']:.5f} rho0 U0^2 "
                                  "(sum E dk + beyond, time average)")
    for row in stats["parseval"]:
        extra = ""
        if "eps_kin_hst" in row:
            if row["eps_kin_hst"] is None:
                extra = f"; grid eps_k {row['eps_kin_grid']:.5f}, eps_m {row['eps_mag_grid']:.5f} (hst not covered)"
            else:
                extra = (f"; grid eps_k {row['eps_kin_grid']:.5f} (hst {row['eps_kin_hst']:.5f}), "
                         f"eps_m {row['eps_mag_grid']:.5f} (hst {row['eps_mag_hst']:.5f})")
        rows[f"Parseval t = {row['time']:.4g}"] = (
            f"kin {row['kin_spectrum']:.6f} = grid {row['kin_grid']:.6f}, mag {row['mag_spectrum']:.6f} = grid "
            f"{row['mag_grid']:.6f}; beyond k_Nyq {row['tot_beyond_kmax']:.2e}{extra}")
    print_table(f"Fig. 3 statistics (run {cfg.run_id}; E(k) = shell sum/dk, dk = {stats['dk']:.4g})", rows)
    local = {}
    for rk, rm, rt in zip(stats["local_slopes_kin"], stats["local_slopes_mag"], stats["local_slopes_tot"]):
        local[f"{rt['k_lo']:.3g} <= k <= {rt['k_hi']:.3g}"] = (f"E_k+E_m {rt['slope']:+.2f}, E_k {rk['slope']:+.2f}, "
                                                               f"E_m {rm['slope']:+.2f} ({rt['n_bins']} shells)")
    print_table(f"Fig. 3 local slopes d ln E/d ln k ({w}-weighted fits in {stats['local_slope_window_decades']:g}-decade "
                "windows of the time average)", local)
    if not stats["fit_range_single_power_law"]:
        print(f"warning: the local slope of E_k+E_m changes by {stats['fit_range_local_slope_change']:.2f} across the fit "
              f"range {k_lo:g}<=k<={k_hi:g}; quote the systematic range, not a single power law", file=sys.stderr)

    sysm = stats["slope_tot_systematic"]
    metadata = {
        "figure": "Fig. 3", "run": run_summary(cfg), "product": str(path), "product_created": meta.get("created_utc"),
        "estimator": meta.get("estimator"), "statistics": stats,
        "display": {"n_per_decade": preset["n_per_decade"],
                    "guide": f"{preset['guide_offset']} x best-fit amplitude; slope of a {w}-weighted fit of the "
                             f"time-averaged total over {k_lo:g} <= k <= {k_hi:g}",
                    "guide_label": guide_label(stats["slope_tot"], k_lo, k_hi),
                    "top_axis": "wavelength 2 pi/k [a], ticks at round wavelengths"},
        "caption_notes": [
            "E(k) = sum over |k| shells of 0.5|F_k|^2 / dk with F = fftn(sqrt(rho) u'', B - <B>_x, norm='forward'); integrates to the fluctuation energy density.",
            f"Average over t = {', '.join(f'{t:.4g}' for t in stats['times'])} a/U0.",
            f"Dashed line: power law fitted with equal weight per log k over {k_lo:g} <= k <= {k_hi:g}, slope "
            f"{stats['slope_tot']:.2f}; fits over other ranges and weightings give {sysm['min']:.2f} to {sysm['max']:.2f}. "
            "The spectrum steepens continuously towards the dissipation range (local slopes in the statistics).",
            "Vertical line: k = 2 pi/L_x, the largest x-wavelength in the box; kx = 0 modes are removed by the x-average.",
        ],
    }
    paths = _save_styled(fig, STYLE_FIG3, args, preset["figure"], metadata)
    plt.close(fig)
    return paths


def run_turbulence(args) -> int:
    preset = _preset(args, FIG3_DEFAULTS, required=("output", "fit_range", "figure"),
                     optional={"run": args.run, "times": args.times if args.times is not None else args.t_range})
    if args.k_fit is not None:
        preset["fit_range"] = list(args.k_fit)
    if args.fit_weighting is not None:
        preset["fit_weighting"] = args.fit_weighting
    if args.n_per_decade is not None:
        preset["n_per_decade"] = args.n_per_decade
    if args.output_stream is not None:
        preset["output"] = args.output_stream
    if args.stage in ("compute", "all"):
        if compute_turbulence(args, preset) is None:
            return 0  # MPI rank > 0: rank 0 writes and plots
    if args.stage in ("plot", "all"):
        for p in plot_turbulence(args, preset):
            print(f"wrote {p}")
    return 0


# ============================================================================ Fig. 4
def energy_series(run: str, args, t_max: float, period: float) -> dict[str, Any]:
    run_dir = find_run(run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    if cfg.hst_path is None:
        raise FileNotFoundError(f"no history file {cfg.problem_id}.hst in {run_dir}")
    h = history_diagnostics(cfg.hst_path, cfg)
    h = h[h["time"] <= t_max + 1e-9].reset_index(drop=True)
    t = h["time"].to_numpy()
    if t.size < 5:
        raise ValueError(f"{run_dir}: fewer than 5 history samples with t <= {t_max}")
    out = {"cfg": cfg, "t": t, "dt_median": float(np.median(np.diff(t)))}
    for name in ("eps_kin", "eps_mag", "eps_p"):
        raw = h[name].to_numpy()
        out[name] = raw
        out[name + "_smooth"] = smooth_series(t, raw, cutoff_period=period)
    return out


def energy_statistics(driven: dict, decaying: dict, period: float) -> dict[str, Any]:
    stats: dict[str, Any] = {"smoothing_period": period}
    for label, d in (("driven", driven), ("decaying", decaying)):
        t = d["t"]
        s: dict[str, Any] = {"run_id": d["cfg"].run_id, "t_first": float(t[0]), "t_last": float(t[-1]),
                             "dt_median": d["dt_median"], "lam": smoothing_lam(period, d["dt_median"])}
        for name in ("eps_kin", "eps_mag", "eps_p"):
            s[f"{name}_t0"] = float(d[name][0])
            s[f"{name}_tmax"] = float(d[name][-1])
            s[f"{name}_tmax_smooth"] = float(d[name + "_smooth"][-1])
        i = int(np.argmin(d["eps_p_smooth"]))
        s["eps_p_min_smooth"] = float(d["eps_p_smooth"][i])
        s["t_eps_p_min_smooth"] = float(t[i])
        j = int(np.argmin(d["eps_p"]))
        s["eps_p_min_raw"] = float(d["eps_p"][j])
        s["t_eps_p_min_raw"] = float(t[j])
        stats[label] = s
    return stats


ENERGY_PANELS = {
    "top": (("driven eps_kin", "driven", "eps_kin"), ("decaying eps_kin", "decaying", "eps_kin"),
            ("driven eps_mag", "driven", "eps_mag"), ("decaying eps_mag", "decaying", "eps_mag")),
    "bottom": (("driven eps_p", "driven", "eps_p"), ("decaying eps_p", "decaying", "eps_p")),
}


def auto_ylim(curves: Sequence[np.ndarray], pad: float = 0.05) -> list[float]:
    """Limits enclosing all ``curves`` with ``pad`` x their range added on both sides.

    Non-negative data (energy densities) never get a negative lower limit.
    """
    lo = min(float(np.min(c)) for c in curves if np.size(c))
    hi = max(float(np.max(c)) for c in curves if np.size(c))
    span = hi - lo if hi > lo else 0.1 * max(abs(hi), 1e-12)
    bottom = lo - pad * span
    return [max(bottom, 0.0) if lo >= 0 else bottom, hi + pad * span]


def energy_panels(driven: dict[str, Any], decaying: dict[str, Any], preset: dict[str, Any], t_max: float) -> dict[str, Any]:
    """x range, y limits (preset, or autoscaled when null) and the fraction of each smoothed curve outside them.

    The fraction is taken over the history samples inside the plotted x range.
    """
    runs = {"driven": driven, "decaying": decaying}
    xmax = float(min(driven["t"][-1], decaying["t"][-1], t_max))
    out: dict[str, Any] = {"xlim": [0.0, xmax]}
    for panel, curves in ENERGY_PANELS.items():
        visible = {}
        for label, run, key in curves:
            t = runs[run]["t"]
            visible[label] = runs[run][key + "_smooth"][(t >= -1e-9) & (t <= xmax + 1e-9)]
        ylim = preset.get(f"ylim_{panel}")
        autoscaled = ylim is None
        ylim = auto_ylim(list(visible.values())) if autoscaled else [float(ylim[0]), float(ylim[1])]
        outside = {label: float(np.mean((v < ylim[0]) | (v > ylim[1]))) if v.size else 0.0 for label, v in visible.items()}
        out[panel] = {"ylim": ylim, "autoscaled": autoscaled, "fraction_outside_ylim": outside}
    return out


def draw_energy(driven: dict[str, Any], decaying: dict[str, Any], preset: dict[str, Any], t_max: float,
                panels: dict[str, Any] | None = None):
    """Fig. 4 layout: smoothed eps_k, eps_m (top) and eps_p (bottom), driven solid / decaying dashed."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FormatStrFormatter, MultipleLocator

    from shearpic.plotting import paper_style

    panels = panels or energy_panels(driven, decaying, preset, t_max)
    with paper_style(STYLE_FIG4):
        fig, axs = plt.subplots(2, 1, figsize=(5.5, 8), sharex=True, gridspec_kw={"height_ratios": [1, 1], "hspace": 0})
        ax = axs[0]
        ax.plot(driven["t"], driven["eps_kin_smooth"], color=COLORS["kin"], ls="-", lw=2, label=r"Driven $\varepsilon_k$")
        ax.plot(decaying["t"], decaying["eps_kin_smooth"], color=COLORS["kin"], ls="--", lw=2, label=r"Decaying $\varepsilon_k$")
        ax.plot(driven["t"], driven["eps_mag_smooth"], color=COLORS["mag"], ls="-", lw=2, label=r"Driven $\varepsilon_m$")
        ax.plot(decaying["t"], decaying["eps_mag_smooth"], color=COLORS["mag"], ls="--", lw=2, label=r"Decaying $\varepsilon_m$")
        ax.set_ylim(*panels["top"]["ylim"])
        ax.set_ylabel(r"$\varepsilon_\mathrm{MHD}\ /\ (\rho_0 U_0^2)$")
        ax.legend(fontsize=16, loc="center left", bbox_to_anchor=(0, 0.35))
        ax = axs[1]
        ax.plot(driven["t"], driven["eps_p_smooth"], color=COLORS["p"], ls="-", lw=2, label=r"Driven $\varepsilon_p$")
        ax.plot(decaying["t"], decaying["eps_p_smooth"], color=COLORS["p"], ls="--", lw=2, label=r"Decaying $\varepsilon_p$")
        ax.set_ylim(*panels["bottom"]["ylim"])
        ax.set_ylabel(r"$\varepsilon_p\ /\ (\rho_0 U_0^2)$")
        ax.set_xlabel(r"$U_0 t/a$")
        ax.legend(fontsize=16, loc="upper left")
        for ax, step, lim in ((axs[0], preset.get("ytick_top"), panels["top"]["ylim"]),
                              (axs[1], preset.get("ytick_bottom"), panels["bottom"]["ylim"])):
            if step and (lim[1] - lim[0]) / float(step) <= 20:   # else: matplotlib's automatic ticks
                ax.yaxis.set_major_locator(MultipleLocator(step))
                ax.yaxis.set_major_formatter(FormatStrFormatter(_formatter_for_step(step)))
            # with hspace = 0 a tick label on the edge shared by the panels would overprint the other panel's
            span = lim[1] - lim[0]
            edge = lim[0] if ax is axs[0] else lim[1]
            ticks = [v for v in ax.yaxis.get_major_locator().tick_values(*lim)
                     if lim[0] - 1e-9 * span <= v <= lim[1] + 1e-9 * span and abs(v - edge) > 0.02 * span]
            ax.yaxis.set_major_locator(FixedLocator(ticks))
            ax.set_xlim(*panels["xlim"])
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.align_ylabels(axs)
        fig.tight_layout()
    return fig


def run_energy(args) -> int:
    import matplotlib.pyplot as plt

    preset = _preset(args, FIG4_DEFAULTS, required=("ylim_top", "ylim_bottom", "figure"),
                     optional={"driven": args.driven, "decaying": args.decaying})
    t_max = float(args.t_max if args.t_max is not None else preset["t_max"])
    period = float(args.smoothing_period if args.smoothing_period is not None else preset["smoothing_period"])
    driven = energy_series(args.driven or preset["driven"], args, t_max, period)
    decaying = energy_series(args.decaying or preset["decaying"], args, t_max, period)
    stats = energy_statistics(driven, decaying, period)
    panels = energy_panels(driven, decaying, preset, t_max)
    stats["panels"] = panels
    fig = draw_energy(driven, decaying, preset, t_max, panels)

    rows = {}
    for label in ("driven", "decaying"):
        s = stats[label]
        rows[f"{label} run {s['run_id']} samples"] = f"t = {s['t_first']:g}..{s['t_last']:g}, dt = {s['dt_median']:g}, lam = {s['lam']:.1f}"
        for name in ("eps_kin", "eps_mag", "eps_p"):
            rows[f"{label} {name}(t=0), (t={s['t_last']:g})"] = (f"{s[name + '_t0']:.4f}, {s[name + '_tmax']:.4f} "
                                                              f"(smoothed {s[name + '_tmax_smooth']:.4f})")
        rows[f"{label} min eps_p"] = (f"{s['eps_p_min_smooth']:.4f} at t = {s['t_eps_p_min_smooth']:g} (smoothed); "
                                      f"{s['eps_p_min_raw']:.4f} at t = {s['t_eps_p_min_raw']:g} (raw)")
    for panel in ("top", "bottom"):
        p = panels[panel]
        fractions = p["fraction_outside_ylim"]
        rows[f"{panel} panel ylim"] = (f"[{p['ylim'][0]:.4g}, {p['ylim'][1]:.4g}]{' (autoscaled)' if p['autoscaled'] else ''}; "
                                       "fraction outside: " + ", ".join(f"{k} {v:.3f}" for k, v in fractions.items()))
        for label, frac in fractions.items():
            if frac > 0:
                print(f"warning: Fig. 4 {panel} panel: {frac:.1%} of the smoothed {label} curve lies outside ylim "
                      f"{p['ylim']}; set ylim_{panel}: null in the preset to autoscale", file=sys.stderr)
    rows["note"] = ("eps_kin = sum(i-KE)/V is the TOTAL gas kinetic energy density, dominated by the laminar mean "
                    f"shear flow ({stats['driven']['eps_kin_t0']:.3f} at t = 0); eps_m = sum(i-ME)/V; eps_p = m_cr KE_cr/V")
    print_table(f"Fig. 4 statistics [rho0 U0^2]; smoothing half-amplitude period {period:g} a/U0", rows)

    metadata = {
        "figure": "Fig. 4", "runs": {"driven": run_summary(driven["cfg"]), "decaying": run_summary(decaying["cfg"])},
        "statistics": stats,
        "definitions": {"eps_kin": "(1-KE + 2-KE + 3-KE)/V, total gas kinetic energy density incl. the mean shear flow",
                        "eps_mag": "(1-ME + 2-ME + 3-ME)/V", "eps_p": "m_cr KE_cr / V",
                        "smoothing": f"make_smoothing_spline with half-amplitude period {period:g} a/U0 on each run's own time axis"},
        "caption_notes": ["Curves are smoothed with a cubic smoothing spline (half-amplitude period "
                          f"{period:g} a/U0).", "eps_k includes the laminar mean shear flow."],
    }
    for p in _save_styled(fig, STYLE_FIG4, args, preset["figure"], metadata):
        print(f"wrote {p}")
    plt.close(fig)
    return 0


# ============================================================================ Fig. 7
def power_series(run: str, args, period: float) -> dict[str, Any]:
    run_dir = find_run(run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    if cfg.hst_path is None:
        raise FileNotFoundError(f"no history file {cfg.problem_id}.hst in {run_dir}")
    raw = read_hst(cfg.hst_path)
    h = history_diagnostics(raw, cfg)
    missing = [c for c in ("eps_p", "P_ideal", "P_ideal_x", "P_ideal_y", "P_ideal_z") if c not in h.columns]
    if missing:
        raise KeyError(f"{cfg.hst_path}: history lacks {missing}")
    t = h["time"].to_numpy()
    E_p = h["eps_p"].to_numpy() * cfg.V
    out = {"cfg": cfg, "t": t, "E_p": E_p, "dt_median": float(np.median(np.diff(t))), "hst_raw": raw}
    out["E_p_smooth"] = smooth_series(t, E_p, cutoff_period=period)
    out["dEp_dt"] = smooth_series(t, E_p, cutoff_period=period, derivative=1)
    out["dEp_dt_gradient"] = np.gradient(out["E_p_smooth"], t)
    for c in ("P_ideal", "P_ideal_x", "P_ideal_y", "P_ideal_z"):
        out[c] = h[c].to_numpy()
        out[c + "_smooth"] = smooth_series(t, out[c], cutoff_period=period)
    out["P_stir"] = h["P_stir"].to_numpy() if "P_stir" in h.columns else None
    return out


def power_statistics(d: dict[str, Any], t_steady: Sequence[float], period: float, ylim, threshold: float) -> dict[str, Any]:
    t = d["t"]
    sel = _window(t, *t_steady)
    ts = t[sel]
    m = lambda x: float(np.mean(x[sel]))  # noqa: E731
    stats: dict[str, Any] = {
        "t_steady": [float(ts[0]), float(ts[-1])], "n_samples": int(sel.sum()), "smoothing_period": period,
        "lam": smoothing_lam(period, d["dt_median"]), "V": float(d["cfg"].V),
        "mean_dEp_dt_spline": m(d["dEp_dt"]),
        "mean_dEp_dt_gradient": m(d["dEp_dt_gradient"]),
        "Delta_E_p_over_Delta_t": float((d["E_p"][sel][-1] - d["E_p"][sel][0]) / (ts[-1] - ts[0])),
        "max_abs_spline_minus_gradient": float(np.max(np.abs(d["dEp_dt"] - d["dEp_dt_gradient"]))),
        "mean_P_ideal_raw": m(d["P_ideal"]),
        "mean_P_ideal_smooth": m(d["P_ideal_smooth"]),
        "Pz_over_P_ideal_raw_means": m(d["P_ideal_z"]) / m(d["P_ideal"]),
    }
    ratio = d["P_ideal_z_smooth"][sel] / d["dEp_dt"][sel]
    stats["mean_Pz_smooth_over_dEp_dt"] = float(np.mean(ratio))
    stats["fraction_Pz_smooth_over_dEp_dt_below_threshold"] = float(np.mean(ratio < threshold))
    stats["threshold"] = threshold
    stats["min_Pz_smooth_over_dEp_dt"] = float(ratio.min())
    stats["t_min_Pz_smooth_over_dEp_dt"] = float(ts[int(np.argmin(ratio))])
    stats["fraction_Pz_raw_over_P_ideal_raw_below_threshold"] = float(np.mean(d["P_ideal_z"][sel] / d["P_ideal"][sel] < threshold))
    Pz = m(d["P_ideal_z"])
    stats["mean_Px_over_mean_Pz_raw"] = m(d["P_ideal_x"]) / Pz
    stats["mean_Py_over_mean_Pz_raw"] = m(d["P_ideal_y"]) / Pz
    stats["mean_abs_Px_over_Pz_smooth"] = float(np.mean(np.abs(d["P_ideal_x_smooth"][sel]) / d["P_ideal_z_smooth"][sel]))
    stats["mean_abs_Py_over_Pz_smooth"] = float(np.mean(np.abs(d["P_ideal_y_smooth"][sel]) / d["P_ideal_z_smooth"][sel]))
    stats["mean_abs_Px_over_Pz_raw"] = float(np.mean(np.abs(d["P_ideal_x"][sel]) / d["P_ideal_z"][sel]))
    stats["mean_abs_Py_over_Pz_raw"] = float(np.mean(np.abs(d["P_ideal_y"][sel]) / d["P_ideal_z"][sel]))
    stats["rms_Px_raw"] = float(np.sqrt(np.mean(d["P_ideal_x"][sel] ** 2)))
    budget = energy_budget(d["hst_raw"], d["cfg"], *t_steady)
    stats["energy_budget"] = budget
    stats["ratio_Delta_E_p_over_int_P_ideal"] = budget["ratio_Ep_over_P_ideal"]
    if d["P_stir"] is not None:
        stats["mean_P_stir"] = m(d["P_stir"])
        stats["fraction_dEp_dt_over_P_stir"] = (stats["mean_dEp_dt_spline"] / stats["mean_P_stir"]
                                                if stats["mean_P_stir"] != 0 else None)
    i = int(np.argmin(d["dEp_dt"]))
    stats["min_dEp_dt_smooth"] = float(d["dEp_dt"][i])
    stats["t_min_dEp_dt_smooth"] = float(t[i])
    stats["min_P_ideal_raw"] = float(d["P_ideal"].min())
    stats["t_min_P_ideal_raw"] = float(t[int(np.argmin(d["P_ideal"]))])
    stats["dEp_dt_clipped_by_ylim"] = bool(d["dEp_dt"].min() < ylim[0] or d["dEp_dt"].max() > ylim[1])
    return stats


def draw_power(d: dict[str, Any], preset: dict[str, Any]):
    """Fig. 7 layout: spline dE_p/dt (black) and smoothed P_ideal,x/y/z (dashed)."""
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FormatStrFormatter

    from shearpic.plotting import paper_style

    ylim = preset["ylim"]
    with paper_style(STYLE_FIG7):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(d["t"], d["dEp_dt"], color="k", ls="-", lw=2, label=r"$dE_p / dt$")
        ax.plot(d["t"], d["P_ideal_x_smooth"], color=COLORS["Px"], ls="--", lw=1.5, label=r"$P_{\mathrm{Ideal},x}$")
        ax.plot(d["t"], d["P_ideal_y_smooth"], color=COLORS["Py"], ls="--", lw=1.5, label=r"$P_{\mathrm{Ideal},y}$")
        ax.plot(d["t"], d["P_ideal_z_smooth"], color=COLORS["Pz"], ls="--", lw=1.5, label=r"$P_{\mathrm{Ideal},z}$")
        ax.legend(fontsize=18, loc="center right", bbox_to_anchor=(0.99, 0.45))
        ax.set_xlim(*preset["xlim"])
        ax.set_ylim(*ylim)
        ax.set_xlabel(r"$U_0 t/a$")
        ax.set_ylabel(r"$P\ [\rho_0 U_0^3 a^2]$")
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.text(0.03, 0.97, r"totals over the box ($L_z = a$)", transform=ax.transAxes, ha="left", va="top",
                fontsize=13)
        ax.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        fig.tight_layout()
    return fig


def run_power(args) -> int:
    import matplotlib.pyplot as plt

    preset = _preset(args, FIG7_DEFAULTS, required=("t_steady", "ylim", "figure"), optional={"run": args.run})
    period = float(args.smoothing_period if args.smoothing_period is not None else preset["smoothing_period"])
    t_steady = [float(v) for v in (args.t_steady if args.t_steady is not None else preset["t_steady"])]
    d = power_series(args.run or preset["run"], args, period)
    ylim = preset["ylim"]
    stats = power_statistics(d, t_steady, period, ylim, float(preset["ratio_threshold"]))
    fig = draw_power(d, preset)

    s = stats
    rows = {
        "steady window": f"{s['t_steady'][0]:g} <= t <= {s['t_steady'][1]:g} ({s['n_samples']} samples); "
                         f"smoothing half-amplitude period {period:g} (lam = {s['lam']:.1f})",
        "<dE_p/dt> (spline derivative)": f"{s['mean_dEp_dt_spline']:.4f} rho0 U0^3 a^2 (= {s['mean_dEp_dt_spline'] / s['V']:.3e} rho0 U0^3/a per volume)",
        "<dE_p/dt> (np.gradient of smoothed)": f"{s['mean_dEp_dt_gradient']:.4f}; max |spline - gradient| = {s['max_abs_spline_minus_gradient']:.4f} (all t)",
        "Delta E_p / Delta t": f"{s['Delta_E_p_over_Delta_t']:.4f}",
        "<P_ideal> raw, smoothed": f"{s['mean_P_ideal_raw']:.4f}, {s['mean_P_ideal_smooth']:.4f}",
        "<P_z>/<P_ideal> (raw)": f"{s['Pz_over_P_ideal_raw_means']:.4f}",
        "mean of P_z/(dE_p/dt) (smoothed)": f"{s['mean_Pz_smooth_over_dEp_dt']:.4f} (min {s['min_Pz_smooth_over_dEp_dt']:.3f} at t = {s['t_min_Pz_smooth_over_dEp_dt']:g})",
        f"fraction P_z/(dE_p/dt) < {s['threshold']:g} (smoothed)": f"{s['fraction_Pz_smooth_over_dEp_dt_below_threshold']:.3f} "
                                                                  f"(raw P_z/P_ideal: {s['fraction_Pz_raw_over_P_ideal_raw_below_threshold']:.3f})",
        "<P_x>/<P_z>, <P_y>/<P_z> (signed, raw)": f"{s['mean_Px_over_mean_Pz_raw']:+.4f}, {s['mean_Py_over_mean_Pz_raw']:+.4f}",
        "mean |P_x|/P_z, |P_y|/P_z (smoothed)": f"{s['mean_abs_Px_over_Pz_smooth']:.4f}, {s['mean_abs_Py_over_Pz_smooth']:.4f} "
                                               f"(raw {s['mean_abs_Px_over_Pz_raw']:.4f}, {s['mean_abs_Py_over_Pz_raw']:.4f})",
        "Delta E_p / int P_ideal dt": f"{s['ratio_Delta_E_p_over_int_P_ideal']:.4f} (energy_budget over the window)",
    }
    if "mean_P_stir" in s:
        frac = s["fraction_dEp_dt_over_P_stir"]
        rows["<P_stir> = <Pstir dV>"] = f"{s['mean_P_stir']:.3f}; <dE_p/dt>/<P_stir> = {'n/a' if frac is None else f'{frac:.4f}'}"
    rows["min smoothed dE_p/dt"] = (f"{s['min_dEp_dt_smooth']:.3f} at t = {s['t_min_dEp_dt_smooth']:g} (raw P_ideal min "
                                    f"{s['min_P_ideal_raw']:.2f} at t = {s['t_min_P_ideal_raw']:g}); "
                                    f"{'clipped' if s['dEp_dt_clipped_by_ylim'] else 'not clipped'} by ylim {tuple(ylim)}")
    print_table(f"Fig. 7 statistics (run {d['cfg'].run_id}; totals over the box, rho0 U0^3 a^2)", rows)

    metadata = {
        "figure": "Fig. 7", "run": run_summary(d["cfg"]), "statistics": stats,
        "definitions": {
            "dEp_dt": "analytic derivative of the smoothing spline of E_p = m_cr KE_cr (total CR kinetic energy)",
            "P_ideal_i": "hst Pideal_i = sum_p m_cr q_mc v_i cE_i (NGP fields), total power, smoothed with the same spline",
            "P_stir": "hst Pstir * dV",
            "mean_Pz_smooth_over_dEp_dt": "time mean over the window of smoothed P_z / spline dE_p/dt",
            "Pz_over_P_ideal_raw_means": "<P_z>/<P_ideal> with raw history columns",
        },
        "caption_notes": ["Curves are totals over the box (units rho0 U0^3 a^2; V = Lx Ly Lz with Lz = a).",
                          f"The roll-up dip (min dE_p/dt = {s['min_dEp_dt_smooth']:.2f} at t = {s['t_min_dEp_dt_smooth']:g}) is clipped by the y range.",
                          "dE_p/dt is not the sum of the dashed curves: Delta E_p / int P_ideal dt = "
                          f"{s['ratio_Delta_E_p_over_int_P_ideal']:.3f}."],
    }
    for p in _save_styled(fig, STYLE_FIG7, args, preset["figure"], metadata):
        print(f"wrote {p}")
    plt.close(fig)
    return 0


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="steady_state.py", description=__doc__.split("\n\n")[0],
                                     epilog="See the module docstring (python -m pydoc analysis/steady_state.py) for "
                                            "the physics, products and HPC commands.",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("turbulence", help="Fig. 3: time-averaged kinetic and magnetic energy spectra")
    p.add_argument("--stage", choices=["compute", "plot", "all"], default="all")
    p.add_argument("--run", default=None, help="run name/number/path (default: preset)")
    p.add_argument("--times", type=float, nargs="+", default=None, help="snapshot times [a/U0] (default: preset)")
    p.add_argument("--t-range", type=float, nargs=2, metavar=("TMIN", "TMAX"), default=None,
                   help="use every (local) snapshot with TMIN <= t <= TMAX instead of --times")
    p.add_argument("--output-stream", default=None, help="athdf output stream (default: preset, out2)")
    p.add_argument("--dk", type=float, default=None, help="shell width [1/a] (default: 2 pi / min(Lx, Ly))")
    p.add_argument("--tol", type=float, default=None,
                   help="max |t_snapshot - t| when matching --times to snapshots or product entries "
                        "(default: max(1e-3 dt_output, 1e-6 |t|))")
    p.add_argument("--k-fit", type=float, nargs=2, metavar=("KMIN", "KMAX"), default=None,
                   help="fit range for the spectral slope (default: preset, 3 15)")
    p.add_argument("--fit-weighting", choices=["log", "shell"], default=None,
                   help="slope fit weighting: equal weight per log k (log, preset default) or per linear shell")
    p.add_argument("--n-per-decade", type=int, default=None, help="log bins per decade for display (default: 8)")
    add_common_arguments(p, "fig3_turbulence")
    p.set_defaults(func=run_turbulence)

    p = sub.add_parser("energy", help="Fig. 4: energy densities of driven and decaying runs")
    p.add_argument("--driven", default=None, help="driven run (default: preset)")
    p.add_argument("--decaying", default=None, help="decaying run (default: preset)")
    p.add_argument("--t-max", type=float, default=None, help="last time [a/U0] (default: preset)")
    p.add_argument("--smoothing-period", type=float, default=None,
                   help="half-amplitude period of the smoothing spline [a/U0] (default: preset, 16.7)")
    add_common_arguments(p, "fig4_steady_state")
    p.set_defaults(func=run_energy)

    p = sub.add_parser("power", help="Fig. 7: power delivered to the particles")
    p.add_argument("--run", default=None, help="run name/number/path (default: preset)")
    p.add_argument("--t-steady", type=float, nargs=2, metavar=("T0", "T1"), default=None,
                   help="averaging window [a/U0] (default: preset, 100 600)")
    p.add_argument("--smoothing-period", type=float, default=None,
                   help="half-amplitude period of the smoothing spline [a/U0] (default: preset, 16.7)")
    add_common_arguments(p, "fig7_power")
    p.set_defaults(func=run_power)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    run_cli(main)
