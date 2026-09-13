"""Quickstart: one figure each for the three kinds of output shearpic reads.

* fields: U_x, omega_z and the x-averaged U_x profile of an HDF5 snapshot, with the particle
  count, x kinetic energy and stirring power compared with the history file;
* trajectory: Lorentz factor, shear-layer crossings and pitch angle of the most energetic of
  the first 20 tracked particles;
* spectra: CR energy spectra from ``energy_spectrum_data/*.csv`` or from full particle
  outputs, with their mean energy compared with the history file.

Usage::

    export PARTICLE_ACCEL_DATA=/path/to/results        # directory that contains run423, ...
    python examples/quickstart.py                       # runs 423, 404 and 369, t = 300
    python examples/quickstart.py --field-run /path/to/run0001 --time 150 --only fields
    python examples/quickstart.py --out /tmp/figs

Runs are given as a number, a name or a directory.  Figures go to ``<output root>/quickstart``
unless ``--out`` is given.
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np

from shearpic.config import RunConfig
from shearpic.env import output_dir, resolve_run, run_output_dir
from shearpic.io.athdf import Snapshot, list_snapshots
from shearpic.io.history import read_hst
from shearpic.io.particles import list_particle_files
from shearpic.io.spectrum_files import read_legacy_spectrum_dir, save_spectrum
from shearpic.io.trajectory import find_trajectory_files, read_trajectory
from shearpic.physics.fields import integrate, kinetic_energy_density, vorticity
from shearpic.physics.forcing import stir_power
from shearpic.physics.particles import crossing_events_for, cumulative_crossings, trajectory_diagnostics
from shearpic.physics.spectra import log_edges, particle_snapshot_spectrum

PARTS = ("fields", "trajectory", "spectra")


# ------------------------------------------------------------------- helpers
def find_run(run: str) -> Path:
    """Run directory from a number, a name or a path; exits with a message if it cannot be resolved."""
    try:
        return resolve_run(run)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}") from None


def run_name(run_dir: Path) -> str:
    """File-name prefix of a run's products: ``run0423`` for ``.../run423`` and ``.../run0423``."""
    return run_output_dir(run_dir, create=False).name


def load_config(run_dir: Path) -> RunConfig:
    """RunConfig of a run, printing its warnings as notes."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = RunConfig.from_run_dir(run_dir)
    for w in caught:
        print(f"  note: {w.message}")
    return cfg


def hst_row(hst, t: float, tol: float):
    """History row closest to time ``t`` (None if the closest row is further than ``tol``)."""
    i = int(np.argmin(np.abs(hst["time"].to_numpy() - t)))
    return hst.iloc[i] if abs(hst["time"].iloc[i] - t) <= tol else None


def relative_difference(a: float, b: float) -> str:
    return f"{abs(a - b) / abs(b):.1e}" if b else "-"


def layer_bands(ax, cfg: RunConfig, cutoff: float = 0.9) -> None:
    """Shade the shear layers (full width 2 a artanh(cutoff)) on a plot with y vertical."""
    half = 0.5 * cfg.profile.layer_width(cutoff)
    for y_layer in cfg.profile.layer_positions:
        ax.axhspan(y_layer - half, y_layer + half, color="0.9", lw=0, zorder=0)


# -------------------------------------------------------------- 1. fields
def fields_figure(run_dir: Path, time: float, out: Path) -> Path:
    import matplotlib.pyplot as plt

    from shearpic.plotting import paper_style, plot_field, plot_profile

    print(f"\n[fields] {run_dir}")
    cfg = load_config(run_dir)
    hdf5 = cfg.athinput.output("hdf5")
    if hdf5 is None:
        raise SystemExit(f"{run_dir} has no <output> block with file_type = hdf5")
    paths = list_snapshots(run_dir, f"out{hdf5['id']}")
    if not paths:
        raise SystemExit(f"no out{hdf5['id']} snapshots in {run_dir}")
    snaps = [Snapshot.open(p) for p in paths]  # metadata only: cheap
    snap = min(snaps, key=lambda s: abs(s.time - time))
    print(f"  snapshot {snap.path.name}: t = {snap.time:.2f}, grid {snap.shape}")

    # float64: float32 sums over millions of cells lose ~1e-4 relative accuracy
    names = [n for n in ("rho", "vel1", "vel2", "vel3", "np") if n in snap.variables]
    d = snap.read(names, dtype=np.float64)
    wz = vorticity((d["vel1"], d["vel2"], d["vel3"]), cfg.dx)[2]  # periodic central differences

    # --- grid output vs history file
    if cfg.hst_path is not None:
        hst = read_hst(cfg.hst_path)
        row = hst_row(hst, snap.time, tol=0.5 * (cfg.output_dt("hst") or 1.0))
    else:
        row = None
    ke_x = integrate(kinetic_energy_density(d["rho"], (d["vel1"],)), cfg)
    checks = [("1-KE  (sum 0.5 rho U_x^2 dV)", ke_x, None if row is None else row["1-KE"])]
    if "np" in d:  # np is a number density, so its volume integral is the particle count
        checks.append(("np    (sum np dV)", integrate(d["np"], cfg), None if row is None else row["np"]))
    if cfg.tau is not None:  # stir_power includes dV; the history Pstir omits it and depends on the pgen
        checks.append(("Pstir (stir_power vs Pstir*dV)", stir_power(d["rho"], d["vel1"], cfg),
                       None if row is None else row["Pstir"] * cfg.dV))
    print(f"  {'quantity':32s} {'snapshot':>14s} {'history':>14s} {'rel. diff':>9s}")
    for label, snap_value, hst_value in checks:
        if hst_value is None:
            print(f"  {label:32s} {snap_value:14.6g} {'-':>14s}")
        else:
            rel = relative_difference(snap_value, hst_value)
            print(f"  {label:32s} {snap_value:14.6g} {hst_value:14.6g} {rel:>9s}")

    # --- figure: U_x, omega_z, <U_x>_x(y)
    x, y = cfg.centers(0), cfg.centers(1)
    S = cfg.shear_amplitude
    with paper_style():
        fig, axs = plt.subplots(1, 3, figsize=(7.0, 7.5), gridspec_kw={"width_ratios": [1, 1, 1.3], "wspace": 0.35})
        plot_field(axs[0], d["vel1"], x, y, symmetric=True, vmax=1.2 * S, cbar_label=r"$U_x\ [U_0]$")
        w_max = float(np.percentile(np.abs(wz), 99.5))
        plot_field(axs[1], wz, x, y, symmetric=True, vmax=w_max, cbar_label=r"$\omega_z\ [U_0/a]$", ylabel=None)
        plot_profile(axs[2], y, d["vel1"], cfg, vertical=True, show_reference=True, label=r"$\langle U_x\rangle_x$",
                     xlabel=r"$U_x\ [U_0]$", ylabel="", color="C3")
        axs[2].legend(loc="upper right")
        fig.suptitle(rf"run {cfg.run_id if cfg.run_id is not None else run_dir.name}, $t = {snap.time:.0f}\,a/U_0$")
        path = out / f"{run_name(run_dir)}_fields_t{snap.time:05.0f}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    print(f"  wrote {path}")
    return path


# ----------------------------------------------------------- 2. trajectory
def trajectory_figure(run_dir: Path, out: Path) -> Path:
    import matplotlib.pyplot as plt

    from shearpic.plotting import format_axes, paper_style, plot_timeseries

    print(f"\n[trajectory] {run_dir}")
    cfg = load_config(run_dir)
    files = find_trajectory_files(run_dir)
    if not files:
        raise SystemExit(f"no trajectory_*.tab files under {run_dir}")
    trajs = [read_trajectory(p) for p in files[:20]]  # see examples/trajectories.py for many files
    traj = max(trajs, key=lambda tr: tr.u[-1] @ tr.u[-1])  # the one with the largest final |u|
    d = trajectory_diagnostics(traj, cfg)
    t, gamma = d["t"], d["gamma"]
    print(f"  {traj.path.name}: {len(traj)} samples, t = {t[0]:g}-{t[-1]:g}, "
          f"gamma {gamma[0]:.3f} -> {gamma[-1]:.3f} (max {gamma.max():.3f})")

    with paper_style():
        fig, axs = plt.subplots(3, 1, figsize=(6.0, 6.5), sharex=True)
        plot_timeseries(axs[0], t, gamma, label=r"$\gamma$ (sampled)", color="k")
        if d["dgamma_dt"] is not None:
            # gamma(0) plus the sampled work integral; shearpic.physics.particles explains why it drifts from gamma
            dt = np.diff(t)
            work = np.concatenate(([0.0], np.cumsum(0.5 * (d["dgamma_dt"][1:] + d["dgamma_dt"][:-1]) * dt)))
            plot_timeseries(axs[0], t, gamma[0] + work, label=r"$\gamma_0 + \int q\,\mathbf{v}\cdot c\mathbf{E}"
                            r"/(mc^2)\,dt$ (per-sample fields)", color="C1", lw=0.8)
        axs[0].set_ylabel(r"$\gamma$")
        axs[0].legend(loc="upper left")

        layer_bands(axs[1], cfg)
        axs[1].plot(t, traj.x[:, 1], ",", color="k")
        axs[1].set_ylabel(r"$y/a$")
        axs[1].set_ylim(*cfg.bounds[1])
        if cfg.profile.layer_positions:
            events = crossing_events_for(traj, cfg)  # hysteresis band |y - y_layer| <= a
            n_cross = cumulative_crossings(events, len(traj))
            twin = axs[1].twinx()
            twin.plot(t, n_cross, color="C0", lw=1.0)
            twin.set_ylabel("layer crossings", color="C0")
            print(f"  {events.count} layer crossings (hysteresis {events.hysteresis:g} a), "
                  f"{events.select(direction=+1).count} towards +y")

        if d["pitch_drift"] is not None:
            axs[2].plot(t, d["pitch_drift"], ".", ms=1, color="C2", label="E x B drift frame")
            axs[2].set_ylabel(r"$\cos\theta_{\rm pitch}$")
            axs[2].set_ylim(-1.05, 1.05)
        else:
            axs[2].text(0.5, 0.5, "7-column trajectory file: no B, cE", transform=axs[2].transAxes, ha="center")
        axs[2].set_xlabel(r"$t\ [a/U_0]$")
        format_axes(axs)
        fig.suptitle(f"{run_dir.name}: particle {traj.pid} (born in meshblock {traj.init_mbid})")
        path = out / f"{run_name(run_dir)}_trajectory_pid{traj.pid}.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    print(f"  wrote {path}")
    return path


# -------------------------------------------------------------- 3. spectra
_PAR_RE = re.compile(r"\.(\d{5})\.par\.(tab|bin)$")


def load_spectra(run_dir: Path, cfg: RunConfig, variable: str | None, out: Path):
    """Spectra from ``energy_spectrum_data/`` if present, else from the full particle outputs, else None."""
    legacy = run_dir / "energy_spectrum_data"
    if legacy.is_dir():
        # variable=None auto-detects gamma vs gamma-1 from the bin range and warns
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            specs = read_legacy_spectrum_dir(legacy, variable=variable)
        for w in caught:
            print(f"  note: {w.message}")
        print(f"  {len(specs)} legacy spectra (pdf only, normalised over in-range particles)")
        return specs
    outputs: dict[int, set[str]] = {}
    for p in list_particle_files(run_dir, warn_mixed=False):  # both formats of one output: handled below
        m = _PAR_RE.search(p.name)
        if m:
            outputs.setdefault(int(m.group(1)), set()).add(m.group(2))
    if not outputs:
        return None
    edges = log_edges(1.0e-2, 1.0e2, 200)
    specs = []
    for number in sorted(outputs)[-4:]:  # the last few particle outputs
        kind = "bin" if "bin" in outputs[number] else "tab"  # reading both would count particles twice
        spec = particle_snapshot_spectrum(run_dir, number, cfg, edges, variable="gamma_minus_1", kind=kind)
        save_spectrum(out / f"{run_name(run_dir)}_spectrum_{number:05d}.npz", spec)
        print(f"  output {number:05d}: {spec}; underflow {spec.underflow:g}, overflow {spec.overflow:g}")
        specs.append(spec)
    return specs


def spectra_figure(run_dir: Path, out: Path, variable: str | None) -> Path | None:
    import matplotlib.pyplot as plt

    from shearpic.plotting import format_axes, paper_style, plot_spectrum, plot_timeseries

    print(f"\n[spectra] {run_dir}")
    cfg = load_config(run_dir)
    specs = load_spectra(run_dir, cfg, variable, out)
    if not specs:
        print("  no energy_spectrum_data/ and no particle outputs: skipped")
        return None
    specs = [s.to("gamma_minus_1") for s in specs]  # exact change of variable (same particles per bin)

    # mean kinetic energy <gamma - 1> of the histogrammed particles, from the bins
    t_spec = np.array([np.nan if s.time is None else s.time for s in specs])
    mean_spec = np.array([np.sum(s.dN_dx("in_range") * s.widths * s.centers("arithmetic")) for s in specs])

    with paper_style():
        fig, axs = plt.subplots(1, 2, figsize=(7.0, 2.9))
        colors = plt.cm.viridis(np.linspace(0.0, 0.9, len(specs)))
        for s, color in zip(specs, colors):
            plot_spectrum(axs[0], s, label=f"$t={s.time:g}$" if s.time is not None else None, color=color)
        axs[0].set_ylim(bottom=1e-6)
        axs[0].legend(fontsize=7, ncol=2)

        axs[1].plot(t_spec, mean_spec, "o", color="C3", label="spectrum bins")
        if cfg.hst_path is not None and cfg.c is not None:
            hst = read_hst(cfg.hst_path)
            # KE_cr = sum_p (gamma-1) c^2 per unit mass, so KE_cr / (N c^2) = <gamma - 1>
            plot_timeseries(axs[1], hst["time"], hst["KE_cr"] / (hst["np"] * cfg.c**2), color="k",
                            label=r"history $KE_{\rm cr}/(N c^2)$")
            for ts, ms in zip(t_spec, mean_spec):
                row = hst_row(hst, ts, tol=0.5 * (cfg.output_dt("hst") or 1.0))
                if row is not None:
                    ref = row["KE_cr"] / (row["np"] * cfg.c**2)
                    print(f"  t = {ts:6.1f}: <gamma-1> spectrum {ms:.5f}, history {ref:.5f}")
        axs[1].set_xlabel(r"$t\ [a/U_0]$")
        axs[1].set_ylabel(r"$\langle\gamma-1\rangle$")
        axs[1].legend()
        format_axes(axs[1])
        fig.suptitle(f"{run_dir.name}: CR energy spectra")
        path = out / f"{run_name(run_dir)}_spectra.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
    print(f"  wrote {path}")
    return path


# ------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--field-run", default="423", metavar="RUN", help="run with HDF5 snapshots (default 423)")
    parser.add_argument("--time", type=float, default=300.0, help="snapshot time to show (nearest is used)")
    parser.add_argument("--trajectory-run", default="404", metavar="RUN",
                        help="run with trajectory files (default 404)")
    parser.add_argument("--spectrum-run", default="369", metavar="RUN", help="run with spectra (default 369)")
    parser.add_argument("--spectrum-variable", choices=("gamma", "gamma_minus_1"), default=None,
                        help="variable of legacy spectrum CSVs (default: detect from the bin range)")
    parser.add_argument("--only", choices=PARTS, action="append", help="make only this figure (repeatable)")
    parser.add_argument("--out", type=Path, default=None, help="output directory (default <output root>/quickstart)")
    args = parser.parse_args(argv)

    import matplotlib

    matplotlib.use("Agg")  # scripts only write files
    out = args.out.expanduser() if args.out is not None else output_dir("quickstart")
    out.mkdir(parents=True, exist_ok=True)
    parts = args.only or PARTS
    if "fields" in parts:
        fields_figure(find_run(args.field_run), args.time, out)
    if "trajectory" in parts:
        trajectory_figure(find_run(args.trajectory_run), out)
    if "spectra" in parts:
        spectra_figure(find_run(args.spectrum_run), out, args.spectrum_variable)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
