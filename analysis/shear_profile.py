#!/usr/bin/env python
r"""Figures 1 and 2 of arXiv:2512.12720: the mean shear flow and the Kelvin-Helmholtz flow maps.

The gas is driven towards the reference flow (code units a = rho0 = U0 = 1)

    U_ref(y) = -S U0 [1 - tanh((y - y1)/a) + tanh((y - y2)/a)],     y1,2 = -+ Ly/4,

a jet at +S U0 between the layers and a wind at -S U0 outside.  As in the figure labels, u is
the gas velocity here.  The stirring relaxes the unweighted x-average <u_x>_x towards U_ref
sampled at the lower cell faces.  S is read from ``<problem> shear_strength`` and checked
against 1-KE(0) in the history file.

``profile`` (Fig. 1)
    (a) U_ref(y)/U0; (b) <u_x>_x(y, t)/U0 at the preset times.
``flow`` (Fig. 2)
    One row per time: rho [rho0]; |u| [U0], including the mean shear flow; |B| [sqrt(rho0) U0].
    The |u| and |B| panels carry line-integral-convolution textures of (vel1, vel2) and
    (Bcc1, Bcc2), which show directions only.

Both subcommands take ``--stage compute|plot|all``. The compute stage selects snapshots by
time (a requested time between outputs is an error), reads them in parallel and writes
``.npz`` products with JSON metadata to ``--products``. The plot stage needs only these
products and the run's athinput and history file; it prints the figure statistics and
writes the figure with a JSON sidecar to ``--out``::

    python analysis/shear_profile.py profile --formats pdf,png
    python analysis/shear_profile.py flow --stage compute --run run358 --times 40 80 --workers auto
    python analysis/shear_profile.py flow --stage plot --products path/to/run0358/products
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import warnings
from functools import partial
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

import shearpic  # noqa: E402
from shearpic.config import RunConfig  # noqa: E402

if __name__ != "__main__":
    # Worker processes cannot import this module by name when it is loaded with importlib,
    # so its worker functions are pickled by value, as loky does for a __main__ script.
    try:
        from joblib.externals import cloudpickle

        cloudpickle.register_pickle_by_value(sys.modules[__name__])
    except (ImportError, KeyError, ValueError):  # pragma: no cover - old cloudpickle / unusual import
        pass

FIG1_RC = {"axes.linewidth": 0.8, "xtick.major.width": 0.8, "ytick.major.width": 0.8,
           "xtick.minor.visible": False, "ytick.minor.visible": False, "legend.handlelength": 2.0,
           "lines.linewidth": 1.0, "savefig.pad_inches": 0.1}
FIG1_COLORS = [(1.0, 0.8, 0.8), (1.0, 0.0, 0.0), (0.2, 0.0, 0.0)]   # light red -> dark red
FIG2_RC = FIG1_RC


# ============================================================================ helpers
def stream_dt(cfg: RunConfig, output: str) -> float | None:
    """Output cadence of the ``<outputN>`` block of stream ``outN`` (e.g. 'out2' -> <output2> dt)."""
    digits = "".join(ch for ch in output if ch.isdigit())
    if digits:
        for block in cfg.athinput.outputs:
            if block["id"] == int(digits) and block.get("dt") is not None:
                return float(block["dt"])
    return cfg.output_dt("hdf5")


def time_tolerance(t: float, dt: float | None, tol: float | None = None) -> float:
    """Largest accepted |t_snapshot - t_requested|: ``tol`` or ``select_snapshots``' default."""
    from shearpic.io.athdf import default_time_tolerance

    return float(tol) if tol is not None else default_time_tolerance(t, dt)


def check_product_times(path: Path, have: Sequence[float], want: Sequence[float], dt: float | None,
                        tol: float | None) -> None:
    """Refuse a product whose snapshot times are not the requested ones (within the selection tolerance)."""
    have, want = [float(v) for v in have], [float(v) for v in want]
    if len(have) != len(want) or any(abs(h - w) > time_tolerance(w, dt, tol) for h, w in zip(have, want)):
        raise ValueError(f"{path} holds snapshot times {have}, not the requested {want}; recompute it with "
                         "--stage compute (or pass the --tol it was computed with)")


def format_time(t: float) -> str:
    """Label of a float32 snapshot time: 40.0003 -> '40', 12.5 -> '12.5'."""
    return f"{round(t):d}" if abs(t - round(t)) < 5e-3 else f"{t:.1f}"


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serialisable: {type(obj).__name__}")


def save_product(path: Path, arrays: Mapping[str, np.ndarray], metadata: Mapping[str, Any]) -> Path:
    """Write ``arrays`` plus a JSON ``metadata`` string atomically to a compressed ``.npz``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    meta = json.dumps(metadata, default=_json_default, allow_nan=True)
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, metadata=np.array(meta), **arrays)
    os.replace(tmp, path)
    return path


def load_product(path: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Arrays and metadata of a product written by :func:`save_product`."""
    with np.load(path, allow_pickle=False) as z:
        arrays = {k: z[k] for k in z.files if k != "metadata"}
        meta = json.loads(str(z["metadata"]))
    return arrays, meta


def hst_shear_amplitude(cfg: RunConfig) -> float | None:
    """S implied by the history file, sqrt(1-KE(0) / (1/2 sum U_shape^2 dV)), or None.

    ``U_shape`` is the unit-amplitude reference profile at the lower cell faces, as the problem
    generator initialises it.
    """
    from shearpic.io.history import read_hst

    if cfg.hst_path is None or not cfg.shear_amplitude:
        return None
    hst = read_hst(cfg.hst_path)
    if len(hst) == 0 or "1-KE" not in hst.columns or float(hst["time"].iloc[0]) != 0.0:
        return None
    shape = cfg.profile.on_cpp_grid(cfg.edges(1)) / cfg.shear_amplitude
    ke_unit = 0.5 * float(np.sum(shape**2)) * cfg.dx[1] * cfg.Lx * cfg.L[2]
    return math.sqrt(float(hst["1-KE"].iloc[0]) / ke_unit) if ke_unit > 0 else None


def flow_parameters(cfg: RunConfig) -> dict[str, Any]:
    """Parameters that set the flow and the particles, including M_A_effective = S U0 / v_A."""
    S = cfg.shear_amplitude
    p = {"c": cfg.c, "q_mc": cfg.q_mc, "M_A": cfg.M_A, "shear_amplitude": S, "B0": cfg.B0, "v_A": cfg.v_A,
         "M_A_effective": S / cfg.v_A if cfg.v_A else math.inf, "cr_mass": cfg.cr_mass,
         "shear_amplitude_source": cfg.shear_amplitude_source}
    return p


def caption_note(cfg: RunConfig, params: Mapping[str, Any]) -> str:
    """Caption text giving the run parameters and what the colours and textures show."""
    run = f"run{cfg.run_id}" if cfg.run_id is not None else str(cfg.run_dir)
    c = "-" if params["c"] is None else f"{params['c']:g} U0"
    q = "-" if params["q_mc"] is None else f"{params['q_mc']:g}"
    return (f"{run}: S = {params['shear_amplitude']:g} (jet/wind at +-{params['shear_amplitude']:g} U0), "
            f"B0 = {params['B0']:g} sqrt(rho0) U0 (athinput M_A = {params['M_A']:g} with respect to U0, i.e. "
            f"S U0 / v_A = {params['M_A_effective']:g}), c = {c}, q/mc = {q}, rho_p/rho0 = {params['cr_mass']:.3g}. "
            "Colours are in code units; |u| includes the mean shear flow; grey LIC textures show the in-plane "
            "directions of u and B.")


def _select(run_dir: Path, cfg: RunConfig, times: Sequence[float], output: str, args) -> list[Path]:
    from shearpic.io.athdf import select_snapshots

    return select_snapshots(run_dir, times, output=output, dt=stream_dt(cfg, output), tol=args.tol,
                            allow_download=args.allow_download)


def _parallel(fn, items, args, mem_per_task: str, desc: str):
    from shearpic.parallel import parallel_map

    return parallel_map(fn, items, backend=args.backend, n_workers=workers_arg(args.workers),
                        mem_per_task=mem_per_task, desc=desc)


# ============================================================================ Fig. 1
def read_x_mean_profile(path: str | Path, variable: str = "vel1") -> dict[str, Any]:
    """Worker: unweighted x-(z-)average of ``variable`` of one snapshot (float64), with its time."""
    from shearpic.io.athdf import Snapshot
    from shearpic.physics.forcing import x_mean_velocity

    snap = Snapshot.open(path)
    field = snap.read(variable, dtype=np.float64)[variable]
    return {"time": snap.time, "cycle": snap.cycle, "file": Path(path).name, "profile": x_mean_velocity(field)}


def profile_product_path(products: Path, output: str, times: Sequence[float]) -> Path:
    tag = "-".join(format_time(t) for t in times)
    return products / f"shear_profile_{output}_t{tag}.npz"


def compute_profiles(run_dir: Path, cfg: RunConfig, times: Sequence[float], output: str, args) -> Path | None:
    """Compute stage of Fig. 1; returns the product path (None on MPI ranks other than 0)."""
    paths = _select(run_dir, cfg, times, output, args)
    results = _parallel(read_x_mean_profile, paths, args, "400MB", "x-mean vel1")
    if results is None:
        return None
    S_hst = hst_shear_amplitude(cfg)
    arrays = {
        "time": np.array([r["time"] for r in results], dtype=np.float64),
        "time_requested": np.asarray(times, dtype=np.float64),
        "cycle": np.array([r["cycle"] for r in results], dtype=np.int64),
        "y": cfg.centers(1),
        "y_faces": cfg.edges(1),
        "profiles": np.stack([r["profile"] for r in results]),
        "U_ref_faces": cfg.profile.on_cpp_grid(cfg.edges(1)),
    }
    meta = {
        "product": "x-averaged gas velocity <vel1>_x(y) (unweighted, as used by the stirring)",
        "arrays": {"time": "actual snapshot times [a/U0]", "y": "cell centres [a]", "y_faces": "cell faces [a]",
                   "profiles": "(n_times, ny) <vel1>_x [U0], float64",
                   "U_ref_faces": "reference profile at the lower cell faces (as the C++ samples it) [U0]"},
        "run": run_summary(cfg), "output": output, "output_dt": stream_dt(cfg, output), "time_tolerance": args.tol,
        "files": [r["file"] for r in results], "shear_amplitude": cfg.shear_amplitude,
        "shear_amplitude_source": cfg.shear_amplitude_source, "shear_amplitude_hst": S_hst,
        **provenance(stage="compute"),
    }
    path = save_product(profile_product_path(product_dir(run_dir, args.products), output, times), arrays, meta)
    print(f"wrote {path}")
    return path


def profile_statistics(t: np.ndarray, profiles: np.ndarray, U_ref: np.ndarray, S: float, S_hst: float | None,
                       dt: float | None, bulk_cutoff: float = 0.9999) -> dict[str, Any]:
    """Consistency checks and mean-flow statistics of Fig. 1.

    Parameters
    ----------
    t, profiles, U_ref : (n,) snapshot times, (n, ny) <u_x>_x and (ny,) U_ref at the lower faces.
    S, S_hst : shear amplitude used and the one implied by the history file (or None).
    dt : output cadence; a snapshot within dt/2 of t = 0 is taken as the initial state.
    bulk_cutoff : jet (wind) cells have U_ref >= +bulk_cutoff S (<= -bulk_cutoff S).

    Returns
    -------
    dict
        ``initial_max_abs_dev`` (None without a t = 0 snapshot) with ``initial_check_passed``
        (< 0.05 S), ``S_check_passed`` (|S - S_hst| <= 1e-3 S), and per time the jet and wind
        means and the max and rms of <u_x>_x - U_ref.
    """
    t = np.asarray(t, dtype=float)
    dev = np.asarray(profiles, dtype=float) - np.asarray(U_ref, dtype=float)[None, :]
    jet = U_ref >= bulk_cutoff * S
    wind = U_ref <= -bulk_cutoff * S
    tol = 0.5 * dt if dt else 1e-3
    i0 = int(np.argmin(np.abs(t))) if t.size else None
    initial = float(np.abs(dev[i0]).max()) if i0 is not None and abs(t[i0]) <= tol else None
    stats: dict[str, Any] = {
        "definitions": {
            "initial_max_abs_dev": "max_y |<u_x>_x(t=0) - U_ref(lower faces)| [U0]; must be < 0.05 S",
            "jet_mean / wind_mean": f"mean of <u_x>_x over cells with U_ref >= +{bulk_cutoff} S / <= -{bulk_cutoff} S",
            "max_abs_dev / rms_dev": "max / rms over y of <u_x>_x - U_ref(lower faces) [U0]",
        },
        "S": S, "S_hst": S_hst,
        "S_check_passed": S_hst is None or abs(S - S_hst) <= 1e-3 * abs(S),
        "initial_max_abs_dev": initial,
        "initial_check_passed": initial is None or initial < 0.05 * abs(S),
        "per_time": [],
    }
    for k, tk in enumerate(t):
        stats["per_time"].append({
            "time": float(tk),
            "jet_mean": float(profiles[k][jet].mean()) if jet.any() else math.nan,
            "wind_mean": float(profiles[k][wind].mean()) if wind.any() else math.nan,
            "max_abs_dev": float(np.abs(dev[k]).max()),
            "rms_dev": float(np.sqrt(np.mean(dev[k] ** 2))),
        })
    return stats


def plot_profiles(cfg: RunConfig, arrays: Mapping[str, np.ndarray]):
    """Fig. 1: (a) analytic U_ref(y)/U0, (b) <u_x>_x(y, t)/U0 per snapshot."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    from shearpic.plotting.fields import plot_profile

    S = cfg.shear_amplitude
    ylo, yhi = cfg.bounds[1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 9))
    y_fine = np.linspace(ylo, yhi, 1000)
    plot_profile(ax1, y_fine, cfg.profile.U(y_fine), cfg, vertical=True, shade_layers=False, color="k", lw=2.0,
                 xlabel=r"$u_{x, 0}\ /\ U_0$", ylabel=r"$y/a$")
    ax1.set_title("(a)", fontsize=26, pad=10)

    t = arrays["time"]
    cmap = LinearSegmentedColormap.from_list("fig1_reds", FIG1_COLORS)
    colors = cmap(np.linspace(0.0, 1.0, len(t)))
    for k, tk in enumerate(t):
        plot_profile(ax2, arrays["y"], arrays["profiles"][k], cfg, vertical=True, shade_layers=False,
                     color=colors[k], lw=1.0, alpha=0.8, label=rf"$U_0 t/a$ = {tk:.0f}",
                     xlabel=r"$\langle u_x(x,y,t)\rangle_x\ /\ U_0$", ylabel="")
    ax2.legend(fontsize=24, loc="best", frameon=False)
    ax2.set_title("(b)", fontsize=26, pad=10)
    for ax in (ax1, ax2):
        ax.set_xlim(-1.15 * S, 1.15 * S)
        ax.set_ylim(ylo, yhi)
        ax.xaxis.label.set_fontsize(26)
        ax.yaxis.label.set_fontsize(26)
        ax.minorticks_off()
        ax.tick_params(axis="both", direction="in", top=True, right=True, labelsize=24)
    return fig


def cmd_profile(args, preset: Mapping[str, Any]) -> dict[str, Any]:
    from shearpic.plotting.style import paper_style

    require_keys(preset, [k for k in ("run", "times") if getattr(args, k) is None], args.preset,
                 "use --preset fig1_shear_profile (or pass --run and --times)")
    run_dir = find_run(args.run or preset["run"], args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    output = args.output or preset.get("output", "out2")
    times = [float(t) for t in (args.times or preset["times"])]
    name = args.name or preset.get("figure", "shear_profile")
    product = profile_product_path(product_dir(run_dir, args.products, create=False), output, times)
    result: dict[str, Any] = {"product": product}

    if args.stage in ("compute", "all"):
        if compute_profiles(run_dir, cfg, times, output, args) is None:
            return result  # MPI rank != 0
    if args.stage not in ("plot", "all"):
        return result

    if not product.exists():
        raise FileNotFoundError(f"{product} does not exist; run with --stage compute (or all) first, or set "
                                "PARTICLE_ACCEL_OUTPUT / --products to where the products are")
    arrays, meta = load_product(product)
    check_same_run(meta.get("run"), cfg, str(product))
    dt = stream_dt(cfg, output)
    check_product_times(product, arrays["time"], times, dt, args.tol if args.tol is not None
                        else meta.get("time_tolerance"))

    S_hst = hst_shear_amplitude(cfg)
    stats = profile_statistics(arrays["time"], arrays["profiles"], arrays["U_ref_faces"], cfg.shear_amplitude,
                               S_hst, dt)
    rows = {"S (athinput / used)": cfg.shear_amplitude,
            "S from hst 1-KE(0)": "n/a" if S_hst is None else f"{S_hst:.7f}",
            "max|<u_x>_x(0) - U_ref(faces)|": "n/a" if stats["initial_max_abs_dev"] is None
            else f"{stats['initial_max_abs_dev']:.3e}  (limit 0.05 S)"}
    for r in stats["per_time"]:
        rows[f"t = {r['time']:8.3f}: jet / wind mean"] = f"{r['jet_mean']:+.4f} / {r['wind_mean']:+.4f}"
        rows[f"t = {r['time']:8.3f}: max / rms |<u_x>_x - U_ref|"] = f"{r['max_abs_dev']:.4f} / {r['rms_dev']:.4f}"
    print_table(f"Fig. 1 statistics ({cfg.run_dir.name}; {stats['definitions']['jet_mean / wind_mean']})", rows)
    if not stats["S_check_passed"]:
        raise RuntimeError(f"shear amplitude mismatch: athinput S = {cfg.shear_amplitude} but hst 1-KE(0) implies "
                           f"S = {S_hst}; the problem generator may not read shear_strength")
    if not stats["initial_check_passed"]:
        raise RuntimeError(f"the t = 0 profile deviates from U_ref by {stats['initial_max_abs_dev']:.3g} "
                           f">= 0.05 S: wrong run, amplitude or layer positions")

    with paper_style(FIG1_RC):
        fig = plot_profiles(cfg, arrays)
        paths = save_figure(fig, figure_dir(args.out), name, formats=args.formats, dpi=args.dpi, metadata={
            "figure": "Fig. 1 (shear_profile)", "preset": args.preset, "run": run_summary(cfg),
            "times": arrays["time"], "snapshot_files": meta.get("files"), "product": str(product),
            "statistics": stats,
        })
    import matplotlib.pyplot as plt

    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    result.update(figure=paths, statistics=stats)
    return result


# ============================================================================ Fig. 2
FLOW_VARIABLES = ("rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3")
FLOW_FIELDS = ("rho", "umag", "bmag")
PERCENTILES = (0.1, 1.0, 50.0, 99.0, 99.9)


def clipping_fractions(field: np.ndarray, vmin: float, vmax: float) -> dict[str, float]:
    """Fractions of finite pixels below ``vmin`` and above ``vmax`` (saturated colours)."""
    a = np.asarray(field)
    a = a[np.isfinite(a)]
    n = max(a.size, 1)
    return {"below": float(np.count_nonzero(a < vmin)) / n, "above": float(np.count_nonzero(a > vmax)) / n}


def field_statistics(field: np.ndarray, limits: Sequence[float] | None = None) -> dict[str, Any]:
    a = np.asarray(field, dtype=np.float64)
    out: dict[str, Any] = {"min": float(a.min()), "max": float(a.max()),
                           "percentiles": dict(zip([f"p{p:g}" for p in PERCENTILES],
                                                   np.percentile(a, PERCENTILES).tolist()))}
    if limits is not None:
        out["limits"] = [float(v) for v in limits]
        out["clipped"] = clipping_fractions(a, *limits)
    return out


def periodic_axes(cfg: RunConfig) -> tuple[bool, bool]:
    """Whether the mesh is periodic along x and y (``<mesh> ix?_bc = ox?_bc = periodic``)."""
    mesh = cfg.athinput["mesh"]
    return tuple(str(mesh.get(f"ix{d}_bc", "")).strip() == "periodic" and str(mesh.get(f"ox{d}_bc", "")).strip()
                 == "periodic" for d in (1, 2))  # type: ignore[return-value]


def compute_flow_maps(path: str | Path, *, bounds, field_downsample: int, lic: Mapping[str, Any],
                      periodic: tuple[bool, bool], limits: Mapping[str, Sequence[float]]) -> dict[str, Any]:
    """Worker: block-mean rho, |u|, |B| and the LIC textures of one snapshot.

    Parameters
    ----------
    bounds : ((x1min, x1max), (x2min, x2max), ...) domain bounds from the RunConfig.
    field_downsample : block-mean factor of the stored colour fields.
    lic : ``downsample``, ``kernel_length``, ``niter``, ``mode_u``, ``mode_b`` and ``seed``.
    limits : colour limits per field, used only for the full-resolution clipping fractions.
    """
    from shearpic.io.athdf import Snapshot
    from shearpic.physics.fields import magnitude
    from shearpic.plotting.lic import block_mean, lic_texture

    snap = Snapshot.open(path)
    if snap.shape[2] != 1:
        raise ValueError(f"{Path(path).name} is 3D {snap.shape}; the flow maps need a 2D run")
    d = snap.read(list(FLOW_VARIABLES), dtype=np.float32, squeeze=True)
    ex, ey = snap.edges(0, bounds[0]), snap.edges(1, bounds[1])
    spacing = (float(ex[1] - ex[0]), float(ey[1] - ey[0]))
    fields = {
        "rho": d["rho"],
        "umag": magnitude(d["vel1"], d["vel2"], d["vel3"]),
        "bmag": magnitude(d["Bcc1"], d["Bcc2"], d["Bcc3"]),
    }
    stats = {k: field_statistics(v, limits.get(k)) for k, v in fields.items()}
    k = int(field_downsample)
    out: dict[str, Any] = {
        "time": snap.time, "cycle": snap.cycle, "file": Path(path).name, "shape": snap.shape, "stats": stats,
        "x_faces": ex[: (snap.shape[0] // k) * k + 1: k], "y_faces": ey[: (snap.shape[1] // k) * k + 1: k],
    }
    for name, f in fields.items():
        out[name] = block_mean(np.asarray(f, dtype=np.float64), k).astype(np.float32)
    del fields
    kl = int(lic["downsample"])
    lic_kw = dict(downsample=kl, kernel_length=int(lic["kernel_length"]), niter=int(lic["niter"]), spacing=spacing,
                  seed=lic.get("seed", 0), periodic=periodic, equalize=True, shade=False)
    out["tex_u"] = lic_texture(d["vel1"], d["vel2"], mode=lic["mode_u"], **lic_kw)
    out["tex_b"] = lic_texture(d["Bcc1"], d["Bcc2"], mode=lic["mode_b"], **lic_kw)
    out["tex_x_faces"] = ex[: (snap.shape[0] // kl) * kl + 1: kl]
    out["tex_y_faces"] = ey[: (snap.shape[1] // kl) * kl + 1: kl]
    return out


def flow_settings(args, preset: Mapping[str, Any]) -> dict[str, Any]:
    lic = dict(preset.get("lic", {}))
    require_keys(preset["limits"], FLOW_FIELDS, f"{args.preset}.limits")
    require_keys(preset["cmaps"], FLOW_FIELDS, f"{args.preset}.cmaps")
    settings = {
        "limits": {k: [float(v) for v in preset["limits"][k]] for k in FLOW_FIELDS},
        "cmaps": {k: preset["cmaps"][k] for k in FLOW_FIELDS},
        "field_downsample": int(args.field_downsample or preset.get("field_downsample", 2)),
        "lic": {
            "downsample": int(args.lic_downsample or lic.get("downsample", 2)),
            "kernel_length": int(args.kernel_length or lic.get("kernel_length", 64)),
            "niter": int(args.niter or lic.get("niter", 25)),
            "mode_u": lic.get("mode_u", "velocity"),
            "mode_b": lic.get("mode_b", "polarization"),
            "seed": int(lic.get("seed", 0)),
        },
        "lic_alpha": float(args.lic_alpha if args.lic_alpha is not None else lic.get("alpha", 0.1)),
        "shade": bool(lic.get("shade", True)) and not args.no_shade,
        "panel_width": float(preset.get("panel_width", 2.05)),
    }
    return settings


def flow_product_path(products: Path, output: str, t: float) -> Path:
    return products / f"flow_maps_{output}_t{format_time(t)}.npz"


def compute_flows(run_dir: Path, cfg: RunConfig, times: Sequence[float], output: str, settings, args):
    paths = _select(run_dir, cfg, times, output, args)
    worker = partial(compute_flow_maps, bounds=cfg.bounds, field_downsample=settings["field_downsample"],
                     lic=settings["lic"], periodic=periodic_axes(cfg), limits=settings["limits"])
    results = _parallel(worker, paths, args, "1GB", "flow maps")
    if results is None:
        return None
    params = flow_parameters(cfg)
    written = []
    for t_req, r in zip(times, results):
        arrays = {k: r[k] for k in ("rho", "umag", "bmag", "tex_u", "tex_b", "x_faces", "y_faces", "tex_x_faces",
                                   "tex_y_faces")}
        arrays["time"] = np.float64(r["time"])
        arrays["periodic"] = np.array(periodic_axes(cfg), dtype=bool)
        meta = {
            "product": "KHI flow maps: block-mean rho, |u|, |B| and periodic LIC textures (unshaded)",
            "arrays": {
                "rho, umag, bmag": f"({FLOW_FIELDS}) block means over {settings['field_downsample']}^2 cells, "
                                   "indexed (x, y), float32; umag = |(vel1, vel2, vel3)| includes the mean shear",
                "tex_u, tex_b": "histogram-equalised LIC textures of (vel1, vel2) and (Bcc1, Bcc2) in [0, 1], (x, y)",
                "x_faces, y_faces": "cell faces of the colour fields [a]",
                "tex_x_faces, tex_y_faces": "cell faces of the textures [a]",
                "periodic": "(x, y) periodicity of the mesh, used for the LIC and its shading",
            },
            "time": r["time"], "time_requested": t_req, "cycle": r["cycle"], "file": r["file"],
            "full_shape": r["shape"], "field_downsample": settings["field_downsample"], "lic": settings["lic"],
            "periodic": periodic_axes(cfg), "full_resolution_statistics": r["stats"],
            "run": run_summary(cfg), "flow_parameters": params, "caption_note": caption_note(cfg, params),
            "output": output, "output_dt": stream_dt(cfg, output), "time_tolerance": args.tol,
            **provenance(stage="compute"),
        }
        path = save_product(flow_product_path(product_dir(run_dir, args.products), output, t_req), arrays, meta)
        print(f"wrote {path}")
        written.append(path)
    return written


def plot_flow(cfg: RunConfig, products: Sequence[Mapping[str, np.ndarray]], settings: Mapping[str, Any]):
    """Fig. 2: one row per time; rho, |u| + LIC(u), |B| + LIC(B); colorbars in a row on top."""
    from shearpic.plotting.fields import plot_field
    from shearpic.plotting.lic import plot_lic
    from shearpic.plotting.style import add_colorbar, panel_grid, panel_label

    (x0, x1), (y0, y1) = cfg.bounds[0], cfg.bounds[1]
    pw = settings["panel_width"]
    wspace, left, right = 0.03, 0.75, 0.1
    fig, axes, caxes = panel_grid(len(products), 3, (y1 - y0) / (x1 - x0), width=left + right + 3 * pw + 2 * wspace,
                                  wspace=wspace, hspace=0.1, cbar_height=0.2, cbar_pad=0.06,
                                  margins=(left, right, 0.6, 0.6))
    lim, cm = settings["limits"], settings["cmaps"]
    labels = {"rho": r"$\rho\ [\rho_0]$", "umag": r"$|u|\ [U_0]$", "bmag": r"$|B|\ [\sqrt{\rho_0}\,U_0]$"}
    font, tick_font = 19, 16
    for i, p in enumerate(products):
        xf, yf = p["x_faces"], p["y_faces"]
        extent = [xf[0], xf[-1], yf[0], yf[-1]]
        images = {}
        images["rho"] = plot_field(axes[i, 0], p["rho"], extent=extent, cmap=cm["rho"], vmin=lim["rho"][0],
                                   vmax=lim["rho"][1], cbar=None, xlabel=None, ylabel=None)
        for j, (name, tex) in enumerate((("umag", "tex_u"), ("bmag", "tex_b")), start=1):
            # a single pre-blended RGBA image keeps the texture contrast the same in PDF and PNG
            images[name], _ = plot_lic(axes[i, j], p["tex_x_faces"], p["tex_y_faces"], color_field=p[name],
                                       texture=p[tex], color_extent=extent, cmap=cm[name], vmin=lim[name][0],
                                       vmax=lim[name][1], alpha=settings["lic_alpha"], shade=settings["shade"],
                                       periodic=tuple(bool(v) for v in p.get("periodic", (True, True))),
                                       cbar=None, xlabel=None, ylabel=None)
        for j in range(3):
            ax = axes[i, j]
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.minorticks_off()
            ax.tick_params(axis="both", which="both", direction="in", top=True, right=True, labelsize=tick_font)
            if i == len(products) - 1:
                ax.set_xlabel("x/a", fontsize=font)
        axes[i, 0].set_ylabel("y/a", fontsize=font)
        panel_label(axes[i, 0], rf"$t = {format_time(float(p['time']))}\ a/U_0$", fontsize=tick_font)
        if i == 0:
            for j, name in enumerate(FLOW_FIELDS):
                cb = add_colorbar(axes[0, j], images[name], location="top", label=labels[name], cax=caxes[j],
                                  labelpad=15)
                cb.ax.xaxis.label.set_fontsize(font)
                cb.ax.tick_params(labelsize=tick_font)
                cb.ax.minorticks_off()
    return fig


def cmd_flow(args, preset: Mapping[str, Any]) -> dict[str, Any]:
    from shearpic.plotting.style import paper_style

    require_keys(preset, [k for k in ("run", "times") if getattr(args, k) is None] + ["limits", "cmaps"],
                 args.preset, "use --preset fig2_flow_evolution")
    run_dir = find_run(args.run or preset["run"], args.data_root)
    cfg = RunConfig.from_run_dir(run_dir)
    output = args.output or preset.get("output", "out2")
    times = [float(t) for t in (args.times or preset["times"])]
    name = args.name or preset.get("figure", "flow_evolution")
    settings = flow_settings(args, preset)
    pdir = product_dir(run_dir, args.products, create=False)
    result: dict[str, Any] = {"products": [flow_product_path(pdir, output, t) for t in times]}

    if args.stage in ("compute", "all"):
        if compute_flows(run_dir, cfg, times, output, settings, args) is None:
            return result
    if args.stage not in ("plot", "all"):
        return result

    dt = stream_dt(cfg, output)
    products, metas = [], []
    for t, path in zip(times, result["products"]):
        if not path.exists():
            raise FileNotFoundError(f"{path} does not exist; run with --stage compute (or all) first, or set "
                                    "PARTICLE_ACCEL_OUTPUT / --products to where the products are")
        arrays, meta = load_product(path)
        check_same_run(meta.get("run"), cfg, str(path))
        check_product_times(path, [float(arrays["time"])], [t], dt, args.tol if args.tol is not None
                            else meta.get("time_tolerance"))
        for key in ("downsample", "kernel_length", "niter", "mode_u", "mode_b"):
            if meta["lic"].get(key) != settings["lic"][key]:
                warnings.warn(f"{path.name}: LIC {key} = {meta['lic'].get(key)} differs from the requested "
                              f"{settings['lic'][key]}; recompute to apply it", stacklevel=2)
        products.append(arrays)
        metas.append(meta)

    params = flow_parameters(cfg)
    clipping = []
    rows: dict[str, Any] = {}
    for p, meta in zip(products, metas):
        t = float(p["time"])
        entry = {"time": t, "displayed": {}, "full_resolution": {}}
        for f in FLOW_FIELDS:
            shown = clipping_fractions(p[f], *settings["limits"][f])
            entry["displayed"][f] = shown
            full = meta["full_resolution_statistics"][f]
            same = full.get("limits") == settings["limits"][f]
            entry["full_resolution"][f] = full.get("clipped") if same else None
            pct = full["percentiles"]
            rows[f"t = {format_time(t)}: {f:4s} p1/p50/p99"] = f"{pct['p1']:.3f} / {pct['p50']:.3f} / {pct['p99']:.3f}"
            fr = f" (full res {100 * full['clipped']['below']:.2f}% / {100 * full['clipped']['above']:.2f}%)" if same \
                else ""
            rows[f"t = {format_time(t)}: {f:4s} below/above {settings['limits'][f]}"] = (
                f"{100 * shown['below']:.2f}% / {100 * shown['above']:.2f}%{fr}")
        clipping.append(entry)
    print_table(f"Fig. 2 statistics ({cfg.run_dir.name}; colour-limit clipping = fraction of displayed "
                f"{settings['field_downsample']}x{settings['field_downsample']} block-mean pixels)", rows)
    print_table("Fig. 2 run parameters", {
        "S": params["shear_amplitude"], "B0": params["B0"], "M_A (athinput, wrt U0)": params["M_A"],
        "M_A effective = S U0 / v_A": params["M_A_effective"], "c": params["c"], "q/mc": params["q_mc"],
        "rho_p/rho0": params["cr_mass"]})
    note = caption_note(cfg, params)
    print(f"\ncaption note: {note}")

    with paper_style(FIG2_RC):
        fig = plot_flow(cfg, products, settings)
        paths = save_figure(fig, figure_dir(args.out), name, formats=args.formats, dpi=args.dpi, metadata={
            "figure": "Fig. 2 (flow_evolution)", "preset": args.preset, "run": run_summary(cfg),
            "flow_parameters": params, "caption_note": note, "times": [float(p["time"]) for p in products],
            "snapshot_files": [m["file"] for m in metas], "products": [str(p) for p in result["products"]],
            "settings": settings, "clipping": clipping,
            "full_resolution_statistics": [m["full_resolution_statistics"] for m in metas],
        })
    import matplotlib.pyplot as plt

    plt.close(fig)
    for p in paths:
        print(f"wrote {p}")
    result.update(figure=paths, clipping=clipping, flow_parameters=params)
    return result


# ============================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], epilog="see the module docstring "
                                     "(python -m pydoc analysis/shear_profile.py) for physics and commands",
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="figure", required=True)
    for cmd, preset, help_text in (("profile", "fig1_shear_profile", "Fig. 1: initial and x-averaged shear profiles"),
                                   ("flow", "fig2_flow_evolution", "Fig. 2: rho, |u| and |B| maps with LIC")):
        p = sub.add_parser(cmd, help=help_text, description=help_text)
        p.add_argument("--stage", choices=["compute", "plot", "all"], default="all",
                       help="compute products from raw snapshots, plot from products, or both (default)")
        p.add_argument("--run", default=None, help="run name, number or directory (default: preset)")
        p.add_argument("--times", type=float, nargs="+", default=None, help="snapshot times [a/U0] (default: preset)")
        p.add_argument("--output", default=None, help="athdf output stream, e.g. out2 (default: preset)")
        p.add_argument("--tol", type=float, default=None,
                       help="largest accepted |t_snapshot - t| [a/U0] when selecting snapshots and checking products "
                            "(default: max(1e-3 dt, 1e-6 t) for the stream cadence dt)")
        p.add_argument("--name", default=None, help="figure file name without extension (default: preset)")
        if cmd == "flow":
            p.add_argument("--field-downsample", type=int, default=None,
                           help="block-mean factor of the stored colour fields (default: preset or 2)")
            p.add_argument("--lic-downsample", type=int, default=None, help="block-mean factor of the LIC input")
            p.add_argument("--kernel-length", type=int, default=None, help="LIC kernel length in pixels")
            p.add_argument("--niter", type=int, default=None, help="LIC passes")
            p.add_argument("--lic-alpha", type=float, default=None, help="opacity of the LIC texture")
            p.add_argument("--no-shade", action="store_true", help="draw the LIC texture without relief shading")
        add_common_arguments(p, preset)
    return parser


def run(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """Run one subcommand; returns its products, figure paths and statistics (used by the tests)."""
    args = build_parser().parse_args(argv)
    preset = load_preset(args.preset, args.presets)
    print(f"shearpic {shearpic.__version__}; preset {args.preset}: {preset.get('description', '')}")
    if args.figure == "profile":
        return cmd_profile(args, preset)
    return cmd_flow(args, preset)


def main(argv: Sequence[str] | None = None) -> int:
    run(argv)
    return 0


if __name__ == "__main__":
    run_cli(main)
