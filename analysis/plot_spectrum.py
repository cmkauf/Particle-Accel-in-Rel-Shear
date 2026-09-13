#!/usr/bin/env python
r"""Figure 5 of arXiv:2512.12720: evolution of the particle energy distribution (1/N) dN/dgamma.

The figure shows f = (1/N) dN/dgamma of all N particles, one curve per particle output coloured by
time; ``--variable`` re-expresses the histograms in p/(mc) or gamma - 1 by mapping the bin edges.
For each output the script prints, and stores in the JSON sidecar, <gamma - 1> with its bin-edge
bounds and the history value ``KE_cr / (np c^2)``, the index alpha of f ~ gamma^-alpha fitted over
``--slope-range``, the cutoff gamma_cut where f first falls below ``--cutoff-level`` above the
peak, and the peak position and height.

``--source legacy`` reads ``<run>/energy_spectrum_data/histogram_frame_NNNNN_t_T.csv``, densities
normalised over the in-range particles, and recovers the integer counts from N = ``cfg.n_par``;
``--bin-variable`` names the binning variable, gamma or gamma - 1, which the CSV files do not record.
``--source npz`` reads the products of ``energy_spectrum.py`` for one particle stream,
chosen with ``--kind``/``--file-id``/``--basename`` when several are present; their recorded run
must match the run's athinput.

Examples::

    export PARTICLE_ACCEL_DATA=".../Cosmic Ray Viscosity/results"
    python analysis/plot_spectrum.py --formats pdf,png
    python analysis/plot_spectrum.py --source npz --kind bin --t-range 0 1200
    python analysis/plot_spectrum.py --variable p_over_mc --out /tmp/figs
"""

from __future__ import annotations

import argparse
import json
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
    add_common_arguments, check_same_run, figure_dir, find_run, load_preset, print_table, product_dir, require_keys,
    run_cli, run_summary, save_figure,
)

from shearpic.config import RunConfig  # noqa: E402

PRESET = "fig5_particle_spectrum"

# matplotlib default line widths and tick sizes with 24 pt tick labels and 28 pt axis labels
RC = {"savefig.pad_inches": 0.1, "axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8, "xtick.minor.width": 0.6,
      "ytick.minor.width": 0.6, "xtick.major.size": 3.5, "ytick.major.size": 3.5, "xtick.minor.size": 2.0,
      "ytick.minor.size": 2.0, "font.size": 24, "axes.labelsize": 28, "xtick.labelsize": 24,
      "ytick.labelsize": 24, "xtick.minor.visible": False, "ytick.minor.visible": False}
FIGSIZE = (10, 8)
#: tight_layout's default pad of 1.08 font sizes at font.size 10, in units of the 24 pt font used here
TIGHT_PAD = 1.08 * 10 / 24

_XSYM = {"gamma": r"\gamma", "gamma_minus_1": r"(\gamma-1)", "p_over_mc": r"(p/mc)"}
_XLABEL = {"gamma": r"$\gamma$", "gamma_minus_1": r"$\gamma-1$", "p_over_mc": r"$p/(mc)$"}


# ----------------------------------------------------------------------------- loading
def _refuse_dataless(paths: Sequence[Path], allow_download: bool) -> None:
    from shearpic.io._util import is_dataless

    blocked = [p for p in paths if is_dataless(p)]
    if blocked and not allow_download:
        names = ", ".join(p.name for p in blocked[:5]) + (" ..." if len(blocked) > 5 else "")
        raise FileNotFoundError(f"{len(blocked)} input file(s) are online-only cloud placeholders ({names}); "
                                "make them available offline or pass --allow-download")


def legacy_frames(directory: Path) -> list[tuple[int, Path]]:
    import re

    out = []
    for p in Path(directory).glob("histogram_frame_*.csv"):
        m = re.match(r"histogram_frame_(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def listed_legacy_frames(directory: Path) -> list[int] | None:
    """Frame numbers listed in ``histogram_metadata.csv`` (None if absent or unparsable)."""
    import pandas as pd

    meta = Path(directory) / "histogram_metadata.csv"
    if not meta.exists():
        return None
    try:
        return [int(v) for v in json.loads(pd.read_csv(meta)["frame_numbers"].iloc[0])]
    except Exception:  # noqa: BLE001 - informational only
        return None


def load_legacy(run_dir: Path, cfg: RunConfig, directory: str, bin_variable: str, n_total: int,
                allow_download: bool) -> tuple[list[Any], list[int], dict[str, Any]]:
    """Legacy CSV spectra with recovered counts, their frame numbers and a completeness report."""
    from shearpic.io.spectrum_files import read_legacy_histogram_csv, spectrum_counts_from_pdf

    d = run_dir / directory
    frames = legacy_frames(d)
    if not frames:
        raise FileNotFoundError(f"no histogram_frame_*.csv files in {d}")
    _refuse_dataless([p for _, p in frames], allow_download)
    specs, numbers, recovered = [], [], []
    for number, path in frames:
        spec = read_legacy_histogram_csv(path, variable=bin_variable)
        try:
            spec = spectrum_counts_from_pdf(spec, n_total)
            recovered.append(True)
        except ValueError as err:
            warnings.warn(f"{path.name}: exact counts not recovered ({err}); using the in-range pdf", stacklevel=2)
            recovered.append(False)
        specs.append(spec)
        numbers.append(number)
    listed = listed_legacy_frames(d)
    missing = sorted(set(listed) - set(numbers)) if listed is not None else []
    if missing:
        warnings.warn(f"{d}: frames {missing} are listed in histogram_metadata.csv but have no CSV", stacklevel=2)
    report = {"source": "legacy", "directory": str(d), "files": [p.name for _, p in frames],
              "listed_frames": listed, "missing_frames": missing, "counts_recovered": all(recovered),
              "n_total": n_total}
    return specs, numbers, report


def _stream_text(stream: tuple[str, str, str]) -> str:
    return f"basename={stream[0]!r} file_id={stream[1]!r} kind={stream[2]!r}"


def load_npz(out_dir: Path, cfg: RunConfig, bin_variable: str, allow_download: bool, *, kind: str | None = None,
             file_id: str | None = None, basename: str | None = None,
             t_range: Sequence[float] | None = None) -> tuple[list[Any], list[int], dict[str, Any]]:
    """Spectra of one particle stream written by energy_spectrum.py, their output numbers and a report.

    Products are identified by their metadata; the file names ``spectrum_<bin_variable>_<stream>_NNNNN.npz``
    and the older ``spectrum_<bin_variable>_NNNNN.npz`` only pre-select them, and the newer name wins
    for the same output.  ``kind``/``file_id``/``basename`` must leave exactly one stream, ``t_range``
    keeps t0 <= t <= t1, and every product's recorded run must match ``cfg``.
    """
    import re

    from shearpic.io.spectrum_files import load_spectrum

    out_dir = Path(out_dir)
    hint = ("run energy_spectrum.py first, or point --products / PARTICLE_ACCEL_OUTPUT at the products")
    if not out_dir.is_dir():
        raise FileNotFoundError(f"products directory {out_dir} does not exist ({hint})")
    var = re.escape(bin_variable)
    new_re = re.compile(rf"^spectrum_{var}_(?P<tag>.+)_(?P<number>\d{{5}})\.npz$")
    old_re = re.compile(rf"^spectrum_{var}_(?P<number>\d{{5}})\.npz$")
    candidates = []
    for path in sorted(out_dir.glob(f"spectrum_{bin_variable}_*.npz")):
        m = old_re.match(path.name) or new_re.match(path.name)
        if m:
            candidates.append((int(m["number"]), m.groupdict().get("tag"), path))
    if not candidates:
        raise FileNotFoundError(f"no spectrum_{bin_variable}_<stream>_NNNNN.npz products in {out_dir} ({hint})")
    _refuse_dataless([p for _, _, p in candidates], allow_download)

    by_stream: dict[tuple[str, str, str], dict[int, tuple[Any, dict, Path, bool]]] = {}
    for number, tag, path in candidates:
        spec, meta = load_spectrum(path, with_metadata=True)
        if spec.variable != bin_variable:  # e.g. spectrum_gamma_minus_1_* matched the spectrum_gamma_* glob
            continue
        st = meta.get("stream") or {}
        stream = (st.get("basename"), st.get("file_id"), st.get("kind"))
        if None in stream:
            raise ValueError(f"{path.name}: the product records no particle stream (basename, file_id, kind); "
                             "recompute it with energy_spectrum.py")
        if tag is not None and tag != ".".join(stream):
            raise ValueError(f"{path.name}: the name says stream {tag!r} but the product was made from "
                             f"{'.'.join(stream)!r} (renamed file?)")
        if "output_number" in meta and int(meta["output_number"]) != number:
            raise ValueError(f"{path.name}: the name says output {number:05d} but the product holds output "
                             f"{int(meta['output_number']):05d} (renamed file?)")
        entries = by_stream.setdefault(stream, {})
        if number in entries:
            if tag is None:  # keep the new-style name
                continue
            if entries[number][3]:
                raise ValueError(f"two products for output {number:05d} of {'.'.join(stream)}: "
                                 f"{entries[number][2].name}, {path.name}")
        entries[number] = (spec, meta, path, tag is not None)
    if not by_stream:
        raise FileNotFoundError(f"no products binned in {bin_variable} in {out_dir} ({hint})")
    present = sorted(by_stream)
    chosen = [s for s in present if (kind is None or s[2] == kind) and (file_id is None or s[1] == file_id)
              and (basename is None or s[0] == basename)]
    if not chosen:
        raise FileNotFoundError(f"no {bin_variable} products for kind={kind!r} file_id={file_id!r} "
                                f"basename={basename!r} in {out_dir}; present: "
                                + "; ".join(_stream_text(s) for s in present))
    if len(chosen) > 1:
        raise ValueError(f"{out_dir} holds {bin_variable} products of several particle streams ("
                         + "; ".join(f"{_stream_text(s)}: {len(by_stream[s])} outputs" for s in chosen)
                         + "); choose one with --kind, --file-id or --basename")
    stream = chosen[0]
    entries = dict(sorted(by_stream[stream].items()))
    renamed = [e[2].name for e in by_stream[stream].values() if not e[3]]

    # one run check per distinct recorded run; products of one job share it
    runs: dict[str, tuple[Any, list[str]]] = {}
    for spec, meta, path, _ in entries.values():
        key = json.dumps(meta.get("run"), sort_keys=True, default=str)
        runs.setdefault(key, (meta.get("run"), []))[1].append(path.name)
    for run, names in runs.values():
        check_same_run(run, cfg, f"{len(names)} product(s) ({names[0]}, ...)" if len(names) > 1 else names[0])

    if t_range is not None:
        t0, t1 = (float(v) for v in t_range)
        eps = 1e-9 * max(1.0, abs(t0), abs(t1))
        entries = {n: e for n, e in entries.items() if e[0].time is not None and t0 - eps <= e[0].time <= t1 + eps}
        if not entries:
            raise FileNotFoundError(f"no {'.'.join(stream)} products with {t0:g} <= t <= {t1:g} in {out_dir}")
    specs, numbers, short = [], [], []
    for number, (spec, meta, path, _) in entries.items():
        if spec.counts is not None and spec.n_total != cfg.n_par:
            short.append(number)
        specs.append(spec)
        numbers.append(number)
    gaps = sorted(set(range(numbers[0], numbers[-1] + 1)) - set(numbers))
    if gaps:
        warnings.warn(f"{out_dir}: no {'.'.join(stream)} products for outputs {gaps}", stacklevel=2)
    if short:
        warnings.warn(f"outputs {short}: n_total differs from cfg.n_par = {cfg.n_par}", stacklevel=2)
    report = {"source": "npz", "directory": str(out_dir), "files": [e[2].name for e in entries.values()],
              "stream": {"basename": stream[0], "file_id": stream[1], "kind": stream[2]},
              "other_streams_present": [".".join(s) for s in present if s != stream],
              "old_style_names": renamed, "t_range": list(t_range) if t_range is not None else None,
              "missing_outputs": gaps, "n_total_mismatch": short}
    return specs, numbers, report


def select_time_range(specs: Sequence[Any], numbers: Sequence[int], t_range) -> tuple[list[Any], list[int]]:
    """The spectra and output numbers with t0 <= time <= t1; FileNotFoundError if there are none."""
    if t_range is None:
        return list(specs), list(numbers)
    t0, t1 = (float(v) for v in t_range)
    eps = 1e-9 * max(1.0, abs(t0), abs(t1))
    keep = [k for k, s in enumerate(specs) if s.time is not None and t0 - eps <= s.time <= t1 + eps]
    if not keep:
        raise FileNotFoundError(f"no spectra with {t0:g} <= t <= {t1:g}")
    return [specs[k] for k in keep], [numbers[k] for k in keep]


def check_times(specs: Sequence[Any], numbers: Sequence[int]) -> np.ndarray:
    """Times ordered by frame/output number; raise unless all are known, unique and strictly increasing."""
    times = [s.time for s in specs]
    if any(t is None for t in times):
        raise ValueError(f"spectra without a time: frames {[n for n, t in zip(numbers, times) if t is None]}")
    t = np.asarray(times, dtype=np.float64)
    bad = np.nonzero(np.diff(t) <= 0)[0]
    if bad.size:
        pairs = ", ".join(f"{numbers[k]}:{t[k]:g} -> {numbers[k + 1]}:{t[k + 1]:g}" for k in bad[:5])
        raise ValueError(f"spectrum times are not unique and strictly increasing with the frame number ({pairs})")
    return t


# ----------------------------------------------------------------------------- statistics
def spectrum_statistics(specs: Sequence[Any], cfg: RunConfig, slope_range: Sequence[float], level: float,
                        hst=None) -> list[dict[str, Any]]:
    """Per-output <gamma - 1>, history mean, peak, power-law index alpha and cutoff gamma_cut."""
    from shearpic.physics.spectra import cutoff_value, power_law_index

    rows = []
    for s in specs:
        g = s.to("gamma")
        row: dict[str, Any] = {"time": g.time}
        try:
            row["mean_gamma_minus_1"] = g.mean("gamma_minus_1")
        except ValueError:
            row["mean_gamma_minus_1"] = math.nan
        row["mean_gamma_minus_1_bounds"] = list(g.mean_bounds("gamma_minus_1"))
        row["hst_mean_gamma_minus_1"] = None
        if hst is not None and len(hst) and "KE_cr" in hst.columns:
            t = hst["time"].to_numpy(np.float64)
            k = int(np.argmin(np.abs(t - g.time)))
            dt = cfg.output_dt("hst") or 0.0
            if abs(t[k] - g.time) <= 0.5 * dt + 1e-9:
                row["hst_mean_gamma_minus_1"] = float(hst["KE_cr"].iloc[k]) / (float(hst["np"].iloc[k]) * cfg.c**2)
        f = g.dN_dx("total")
        peak = int(np.argmax(f))
        row["peak_gamma"] = float(g.centers("geometric")[peak])
        row["peak_value"] = float(f[peak])
        try:
            row["alpha"] = power_law_index(g, *slope_range)
        except ValueError:
            row["alpha"] = math.nan
        try:
            row["gamma_cut"] = cutoff_value(g, level)
        except ValueError:
            row["gamma_cut"] = math.nan
        row["n_in_range"] = g.n_in_range
        row["overflow"] = g.overflow
        rows.append(row)
    for prev, row in zip(rows, rows[1:]):
        dt = row["time"] - prev["time"]
        row["dgamma_cut_dt"] = (row["gamma_cut"] - prev["gamma_cut"]) / dt if dt > 0 else math.nan
    if rows:
        rows[0]["dgamma_cut_dt"] = math.nan
    return rows


def cutoff_growth(rows: Sequence[dict[str, Any]], t_split: Sequence[float] = (300.0, 1000.0)) -> dict[str, Any]:
    """Growth of gamma_cut: mean rates between the split times and the end, and linear and power-law fits.

    ``rate_<a>_<b>`` is (gamma_cut(b) - gamma_cut(a)) / (b - a) from the outputs nearest a and b;
    ``linear_fit_from_<t0>`` holds the slope [U0/a] and max |residual| of a line through the outputs with
    t >= t0, and ``power_law_index_from_<t0>`` is d log gamma_cut / d log t over the same outputs.
    """
    t = np.array([r["time"] for r in rows], dtype=float)
    g = np.array([r["gamma_cut"] for r in rows], dtype=float)
    ok = np.isfinite(g) & (t > 0)
    out: dict[str, Any] = {}
    if ok.sum() < 3:
        return out
    t, g = t[ok], g[ok]
    marks = [float(x) for x in t_split] + [float(t[-1])]
    for a, b in zip(marks, marks[1:]):
        ia, ib = int(np.argmin(np.abs(t - a))), int(np.argmin(np.abs(t - b)))
        if t[ib] > t[ia]:
            out[f"rate_{t[ia]:g}_{t[ib]:g}"] = float((g[ib] - g[ia]) / (t[ib] - t[ia]))
    for t0 in t_split:
        sel = t >= t0
        if sel.sum() >= 3:
            coef = np.polyfit(t[sel], g[sel], 1)
            out[f"linear_fit_from_{t0:g}"] = {"slope": float(coef[0]),
                                              "max_abs_residual": float(np.max(np.abs(np.polyval(coef, t[sel]) - g[sel])))}
            out[f"power_law_index_from_{t0:g}"] = float(np.polyfit(np.log(t[sel]), np.log(g[sel]), 1)[0])
    return out


def print_statistics(rows: Sequence[dict[str, Any]], slope_range, level) -> None:
    print(f"\nFig. 5 statistics: alpha = -dlog f/dlog gamma over {slope_range[0]:g} <= gamma <= {slope_range[1]:g}; "
          f"gamma_cut: f = (1/N) dN/dgamma first < {level:g} above the peak")
    print(f"{'time':>7} {'<g-1>':>8} {'bounds':>17} {'hst':>8} {'peak':>7} {'f_peak':>7} {'alpha':>7} "
          f"{'g_cut':>7} {'dgcut/dt':>9}")
    for r in rows:
        hst = "-" if r["hst_mean_gamma_minus_1"] is None else f"{r['hst_mean_gamma_minus_1']:.5f}"
        lo, hi = r["mean_gamma_minus_1_bounds"]
        print(f"{r['time']:>7.5g} {r['mean_gamma_minus_1']:>8.5f} {lo:>8.5f}..{hi:<8.5f} {hst:>8} "
              f"{r['peak_gamma']:>7.4f} {r['peak_value']:>7.3g} {r['alpha']:>7.3f} {r['gamma_cut']:>7.3f} "
              f"{r['dgamma_cut_dt']:>9.2e}")


# ----------------------------------------------------------------------------- plot
def plot_figure(specs: Sequence[Any], times: np.ndarray, *, variable: str, style: str, xlim, ylim):
    """Fig. 5: one curve per output coloured by time from 0 to the latest time, colorbar on top."""
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from shearpic.plotting.lines import plot_spectrum
    from shearpic.plotting.style import add_colorbar

    fig, ax = plt.subplots(figsize=FIGSIZE)
    cmap = plt.get_cmap("cividis")
    norm = Normalize(vmin=0.0, vmax=float(np.max(times)))
    floor = ylim[0] * 1e-3 if ylim else None
    for s, t in zip(specs, times):
        plot_spectrum(ax, s.to(variable), normalize="total", style=style, floor=floor if style == "stairs" else None,
                      color=cmap(norm(t)), lw=2, set_labels=False)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if xlim:
        ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_xlabel(_XLABEL[variable])
    ax.set_ylabel(rf"$(1/N)\,dN/d{_XSYM[variable]}$")
    ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, pad=8)
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = add_colorbar(ax, sm, location="top", label=r"$U_0 t/a$", size="3%", pad=0.05)
    cbar.ax.tick_params(which="both", direction="in")
    fig.tight_layout(pad=TIGHT_PAD)
    return fig, ax


# ----------------------------------------------------------------------------- CLI
def build_parser(argv: Sequence[str] | None = None) -> argparse.ArgumentParser:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--preset", default=PRESET)
    pre.add_argument("--presets", default=None)
    known, _ = pre.parse_known_args(argv)
    preset = load_preset(known.preset, known.presets)

    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="See the module docstring for details.",
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default=preset.get("run"))
    p.add_argument("--source", default=preset.get("source", "legacy"), choices=["legacy", "npz"])
    p.add_argument("--kind", default=None, choices=["tab", "bin"],
                   help="npz: particle stream format to plot (default: the only stream present)")
    p.add_argument("--file-id", default=None, help="npz: particle stream output block, e.g. out4")
    p.add_argument("--basename", default=None, help="npz: particle stream problem_id prefix")
    p.add_argument("--t-range", type=float, nargs=2, default=None, metavar=("T0", "T1"),
                   help="plot only outputs with T0 <= t <= T1 [a/U0] (default: all)")
    p.add_argument("--legacy-dir", default=preset.get("legacy_dir", "energy_spectrum_data"))
    p.add_argument("--bin-variable", default=preset.get("variable"), choices=["gamma", "gamma_minus_1", "p_over_mc"],
                   help="variable the data are binned in (legacy CSVs do not record it; default: %(default)s)")
    p.add_argument("--variable", default=preset.get("plot_variable", "gamma"),
                   choices=["gamma", "gamma_minus_1", "p_over_mc"], help="plotted variable (default: %(default)s)")
    p.add_argument("--style", default=preset.get("style", "stairs"), choices=["stairs", "line"])
    p.add_argument("--xlim", type=float, nargs=2, default=None, help="default from the preset for --variable")
    p.add_argument("--ylim", type=float, nargs=2, default=preset.get("ylim"))
    p.add_argument("--n-total", type=int, default=None, help="particle number for legacy counts (default: cfg.n_par)")
    p.add_argument("--slope-range", type=float, nargs=2, default=preset.get("slope_range", [3.0, 6.0]))
    p.add_argument("--cutoff-level", type=float, default=preset.get("cutoff_level", 1e-4))
    p.add_argument("--name", default=preset.get("figure", "particle_spectrum"), help="figure file name (no suffix)")
    add_common_arguments(p, PRESET)
    return p


def main(argv: Sequence[str] | None = None) -> int:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from shearpic.io.history import read_hst
    from shearpic.plotting.style import paper_style

    args = build_parser(argv).parse_args(argv)
    preset = load_preset(args.preset, args.presets)
    hint = "pass the option or use --preset fig5_particle_spectrum"
    if args.run is None:
        require_keys(preset, ["run"], args.preset, "pass --run or use --preset fig5_particle_spectrum")
    if args.bin_variable is None:
        require_keys(preset, ["variable"], args.preset,
                     "pass --bin-variable (legacy CSVs do not record whether they are binned in gamma); " + hint)
    run_dir = find_run(args.run, args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    cfg.require_particles()
    n_total = int(args.n_total) if args.n_total is not None else int(cfg.n_par)

    if args.source == "legacy":
        specs, numbers, report = load_legacy(run_dir, cfg, args.legacy_dir, args.bin_variable, n_total,
                                             args.allow_download)
        specs, numbers = select_time_range(specs, numbers, args.t_range)
        report["t_range"] = args.t_range
    else:
        specs, numbers, report = load_npz(product_dir(run_dir, args.products, create=False), cfg,
                                          args.bin_variable, args.allow_download, kind=args.kind,
                                          file_id=args.file_id, basename=args.basename, t_range=args.t_range)
    times = check_times(specs, numbers)

    hst = None
    if cfg.hst_path is not None:
        from shearpic.io._util import is_dataless

        if not is_dataless(cfg.hst_path) or args.allow_download:
            hst = read_hst(cfg.hst_path)
    stats = spectrum_statistics(specs, cfg, args.slope_range, args.cutoff_level, hst)
    print_statistics(stats, args.slope_range, args.cutoff_level)
    growth = cutoff_growth(stats)
    print_table("gamma_cut growth (the 'd gamma_max/dt ~ const' claim)",
                {k: (f"slope {v['slope']:.4g} /(a/U0), max residual {v['max_abs_residual']:.3g}"
                     if isinstance(v, dict) else v) for k, v in growth.items()})
    print_table(f"Fig. 5 inputs ({report['source']})", {
        "run": run_dir.name, "c": cfg.c, "N (cfg.n_par)": cfg.n_par, "outputs": len(specs),
        **({"stream": ".".join(report["stream"].values())} if "stream" in report else {}),
        "times": f"{times[0]:g} .. {times[-1]:g}", "binned in": args.bin_variable, "plotted in": args.variable,
        "missing": report.get("missing_frames", report.get("missing_outputs")),
        "exact counts": report.get("counts_recovered", True),
    })

    xlim = args.xlim or (preset.get("xlim") if args.variable == "gamma" else preset.get(f"xlim_{args.variable}"))
    rc = dict(RC)
    with paper_style(rc):
        fig, _ = plot_figure(specs, times, variable=args.variable, style=args.style, xlim=xlim, ylim=args.ylim)
        metadata = {
            "figure": "Fig. 5 particle energy spectrum",
            "quantity": f"(1/N) dN/d{args.variable}, N = all particles",
            "run": run_summary(cfg), "inputs": report, "times": times, "output_numbers": numbers,
            "bin_variable": args.bin_variable, "plot_variable": args.variable, "style": args.style,
            "colour_scale": {"cmap": "cividis", "vmin": 0.0, "vmax": float(times.max())},
            "statistics": {
                "definitions": {
                    "mean_gamma_minus_1": "sum over bins of counts x midpoint(gamma-1) / N; bounds from bin edges",
                    "hst_mean_gamma_minus_1": "KE_cr / (np c^2) at the nearest history row within half its cadence",
                    "alpha": f"-slope of least squares log f vs log gamma, non-empty bins with geometric centres in "
                             f"[{args.slope_range[0]:g}, {args.slope_range[1]:g}], f = (1/N) dN/dgamma",
                    "gamma_cut": f"gamma above the peak where f first < {args.cutoff_level:g}, log-log interpolation "
                                 "between bin centres",
                    "dgamma_cut_dt": "finite difference of gamma_cut between successive outputs [U0/a]",
                },
                "per_output": stats,
                "gamma_cut_growth": growth,
            },
        }
        paths = save_figure(fig, figure_dir(args.out), args.name, formats=args.formats, dpi=args.dpi,
                            metadata=metadata)
        plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    return 0


if __name__ == "__main__":
    run_cli(main)
