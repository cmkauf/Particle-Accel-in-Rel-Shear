"""Reduce every HDF5 snapshot of a run in parallel and check the result against the history file.

Each worker reads one snapshot and returns only small results: the volume integrals that
the history file also records (``1-KE`` to ``3-ME``, ``np``, ``vp1..3``, ``Pstir * dV``) and
x-averaged profiles of ``U_x``, ``rho``, ``np`` and the Reynolds and Maxwell stresses.  The
main process prints the relative differences to the history file and writes
``<run>_snapshot_scalars.csv``, ``<run>_snapshot_profiles.npz`` and a summary figure to
``run_output_dir(run)`` unless ``--out`` is given.

Usage::

    python examples/parallel_snapshots.py --run 423                     # all snapshots, auto workers
    python examples/parallel_snapshots.py --run 423 --workers 4 --benchmark
    python examples/parallel_snapshots.py --run /path/to/run0001 --stride 2 --out /tmp/test
    ibrun python examples/parallel_snapshots.py --run 1                 # MPI: one shard per rank

Under ``ibrun`` or ``mpirun`` each rank handles ``paths[rank::size]`` and rank 0 gathers the results;
otherwise a process pool is sized to the CPUs and ``--mem-per-task``.
"""

from __future__ import annotations

import argparse
import functools
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from shearpic.config import RunConfig
from shearpic.env import detect_environment, resolve_run, run_output_dir, warn_if_login_node
from shearpic.io.athdf import Snapshot, list_snapshots
from shearpic.io.history import read_hst
from shearpic.parallel import mpi_world_size, parallel_map, resolve_workers
from shearpic.physics.fields import maxwell_stress_xy, reynolds_stress_xy, x_average
from shearpic.physics.forcing import stir_power

PROFILES = ("vx", "rho", "n_cr", "reynolds_xy", "maxwell_xy")


# ------------------------------------------------------------------ worker
def analyse_snapshot(path: Path, cfg: RunConfig) -> dict:
    """Volume integrals and x-averaged profiles of one snapshot, computed in a worker process.

    Returns a dict with ``time``, ``file``, the integrals named like the history columns
    (energies in rho0 U0^2 a^3, ``Pstir*dV`` in rho0 U0^3 a^2) and ``profiles``, a dict of
    arrays of shape (ny,).
    """
    snap = Snapshot.open(path)
    have = set(snap.variables)
    names = [n for n in ("rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3", "np", "vp1", "vp2", "vp3")
             if n in have]
    d = snap.read(names, dtype=np.float64)  # float64: sums over millions of cells
    dV = cfg.dV
    out: dict = {"time": snap.time, "file": path.name}
    for i in (1, 2, 3):
        if f"vel{i}" in d:
            out[f"{i}-KE"] = 0.5 * float(np.sum(d["rho"] * d[f"vel{i}"] ** 2)) * dV
        if f"Bcc{i}" in d:
            out[f"{i}-ME"] = 0.5 * float(np.sum(d[f"Bcc{i}"] ** 2)) * dV
    if "np" in d:
        out["np"] = float(np.sum(d["np"])) * dV  # np is a number density: this is the particle count N
        for i in (1, 2, 3):  # vp_i is the cell-mean reduced momentum <u_i>: sum np vp_i dV = sum_p u_i
            if f"vp{i}" in d:
                out[f"vp{i}"] = float(np.sum(d["np"] * d[f"vp{i}"])) * dV
    if cfg.tau is not None:
        out["Pstir*dV"] = stir_power(d["rho"], d["vel1"], cfg)  # total power, dV included

    profiles = {"vx": x_average(d["vel1"]), "rho": x_average(d["rho"]),
                "reynolds_xy": x_average(reynolds_stress_xy(d["rho"], d["vel1"], d["vel2"]))}
    if "np" in d:
        profiles["n_cr"] = x_average(d["np"])
    if "Bcc1" in d and "Bcc2" in d:
        profiles["maxwell_xy"] = x_average(maxwell_stress_xy(d["Bcc1"], d["Bcc2"]))
    out["profiles"] = profiles
    return out


# ------------------------------------------------------------------ helpers
def find_run(run: str) -> Path:
    """Run directory from a number, a name or a path; exits with a message if it cannot be resolved."""
    try:
        return resolve_run(run)
    except (FileNotFoundError, ValueError) as err:
        raise SystemExit(f"error: {err}") from None


def mpi_rank() -> int:
    if mpi_world_size() > 1:
        from mpi4py import MPI

        return MPI.COMM_WORLD.Get_rank()
    return 0


def compare_with_history(scalars: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame | None:
    """Relative difference snapshot/history for every column both have (rows: snapshots)."""
    if cfg.hst_path is None:
        return None
    hst = read_hst(cfg.hst_path)
    # the history Pstir omits dV, and its definition depends on the problem generator
    hst = hst.assign(**{"Pstir*dV": hst["Pstir"] * cfg.dV}) if "Pstir" in hst.columns else hst
    t_hst = hst["time"].to_numpy()
    tol = 0.5 * (cfg.output_dt("hst") or 1.0)
    rows = []
    for _, snap in scalars.iterrows():
        i = int(np.argmin(np.abs(t_hst - snap["time"])))
        row = {"time": snap["time"], "hst_time": t_hst[i]}
        if abs(t_hst[i] - snap["time"]) <= tol:
            for col in scalars.columns:
                if col in hst.columns and col not in ("time",):
                    ref = hst[col].iloc[i]
                    row[col] = abs(snap[col] - ref) / abs(ref) if ref else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def summary_figure(scalars: pd.DataFrame, profiles: dict, cfg: RunConfig, path: Path) -> None:
    import matplotlib.pyplot as plt

    from shearpic.plotting import add_colorbar, format_axes, paper_style, plot_profile, plot_timeseries

    hst = read_hst(cfg.hst_path) if cfg.hst_path is not None else None
    t, y = scalars["time"].to_numpy(), cfg.centers(1)
    with paper_style():
        fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
        # (a) space-time diagram of the mean flow
        ax = axs[0, 0]
        S = cfg.shear_amplitude
        mesh = ax.pcolormesh(t, y, profiles["vx"].T, shading="nearest", cmap="RdBu_r", vmin=-1.2 * S, vmax=1.2 * S)
        ax.set(xlabel=r"$t\ [a/U_0]$", ylabel=r"$y/a$")
        format_axes(ax)
        add_colorbar(ax, mesh, location="right", label=r"$\langle U_x\rangle_x\ [U_0]$")

        # (b) energies: history (lines) vs snapshots (markers)
        ax = axs[0, 1]
        for col, color in (("1-KE", "C0"), ("2-KE", "C1"), ("1-ME", "C2"), ("2-ME", "C3")):
            if col not in scalars.columns:
                continue
            if hst is not None and col in hst.columns:
                plot_timeseries(ax, hst["time"], hst[col] / cfg.V, color=color, lw=0.8)
            ax.plot(t, scalars[col] / cfg.V, "o", color=color, ms=3, label=col)
        ax.set(yscale="log", xlabel=r"$t\ [a/U_0]$", ylabel=r"energy / $V$  $[\rho_0 U_0^2]$")
        ax.legend(fontsize=7, ncol=2)

        # (c) stresses at the last snapshot, shear layers shaded
        ax = axs[1, 0]
        plot_profile(ax, y, profiles["reynolds_xy"][-1], cfg, label=r"$\langle\rho\,\delta U_x\delta U_y\rangle_x$")
        if "maxwell_xy" in profiles:
            plot_profile(ax, y, profiles["maxwell_xy"][-1], cfg, shade_layers=False, color="C3",
                         label=r"$-\langle\delta B_x\delta B_y\rangle_x$")
        ax.set_ylabel(rf"stress at $t={t[-1]:.0f}$  $[\rho_0 U_0^2]$")
        ax.legend(fontsize=7)

        # (d) stirring power
        ax = axs[1, 1]
        if "Pstir*dV" in scalars.columns:
            if hst is not None and "Pstir" in hst.columns:
                plot_timeseries(ax, hst["time"], hst["Pstir"] * cfg.dV, color="k", lw=0.6, label=r"history Pstir$\,dV$")
            ax.plot(t, scalars["Pstir*dV"], "o", color="C3", ms=3, label="stir_power(snapshot)")
            ax.set(xlabel=r"$t\ [a/U_0]$", ylabel=r"$P_{\rm stir}\ [\rho_0 U_0^3 a^2]$")
            ax.legend(fontsize=7)
        else:
            ax.set_axis_off()
        fig.suptitle(f"{cfg.run_dir.name}: {len(t)} snapshots")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="423", help="run number, name (run0423) or directory (default 423)")
    parser.add_argument("--stream", default=None, help="snapshot stream, e.g. out2 (default: the hdf5 output)")
    parser.add_argument("--stride", type=int, default=1, help="use every N-th snapshot")
    parser.add_argument("--backend", default="auto", choices=("auto", "serial", "thread", "process", "mpi"))
    parser.add_argument("--workers", default="auto", help="number of worker processes or 'auto'")
    parser.add_argument("--mem-per-task", default="1GB",
                        help="peak memory of one worker, caps the number of workers (default 1GB: ~11 float64 "
                             "fields of a 1024x4096 grid plus temporaries)")
    parser.add_argument("--benchmark", action="store_true",
                        help="also run serially and report the speed-up (doubles the run time)")
    parser.add_argument("--no-plot", action="store_true", help="skip the figure")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default <output root>/run<NNNN>, see run_output_dir)")
    args = parser.parse_args(argv)
    workers = int(args.workers) if str(args.workers).isdigit() else args.workers

    rank = mpi_rank()
    run_dir = find_run(args.run)
    with warnings.catch_warnings():
        if rank != 0:
            warnings.simplefilter("ignore")
        cfg = RunConfig.from_run_dir(run_dir)
    stream = args.stream
    if stream is None:
        hdf5 = cfg.athinput.output("hdf5")
        stream = f"out{hdf5['id']}" if hdf5 is not None else None
    paths = list_snapshots(run_dir, stream)[:: args.stride]
    if not paths:
        raise SystemExit(f"no {stream or ''} snapshots in {run_dir}")

    if rank == 0:
        warn_if_login_node("parallel_snapshots.py")
        env = detect_environment()
        n = resolve_workers(len(paths), workers, args.mem_per_task)
        print(f"{run_dir}: {len(paths)} snapshots ({stream})")
        print(f"machine {env.machine}, {env.n_cpus} CPUs usable, backend {args.backend}, "
              f"{'%d MPI ranks' % mpi_world_size() if mpi_world_size() > 1 else '%d worker(s)' % n}")

    worker = functools.partial(analyse_snapshot, cfg=cfg)  # a picklable one-argument function
    t0 = time.perf_counter()
    results = parallel_map(worker, paths, backend=args.backend, n_workers=workers, mem_per_task=args.mem_per_task,
                           progress=rank == 0, desc="snapshots")
    elapsed = time.perf_counter() - t0
    if results is None:  # MPI rank > 0: rank 0 has gathered the results
        return 0
    print(f"analysed {len(results)} snapshots in {elapsed:.1f} s")

    if args.benchmark:
        t0 = time.perf_counter()
        serial = parallel_map(worker, paths, backend="serial")
        t_serial = time.perf_counter() - t0
        same = all(np.isclose(a["1-KE"], b["1-KE"], rtol=1e-12) for a, b in zip(results, serial))
        print(f"serial: {t_serial:.1f} s -> speed-up {t_serial / elapsed:.2f}x (identical results: {same})")

    scalars = pd.DataFrame([{k: v for k, v in r.items() if k != "profiles"} for r in results])
    names = [p for p in PROFILES if all(p in r["profiles"] for r in results)]
    profiles = {p: np.stack([r["profiles"][p] for r in results]) for p in names}  # (n_snapshots, ny)

    name = run_output_dir(run_dir, create=False).name  # 'run0423' for run423 and run0423
    out = args.out.expanduser() if args.out is not None else run_output_dir(run_dir)
    out.mkdir(parents=True, exist_ok=True)
    scalars.to_csv(out / f"{name}_snapshot_scalars.csv", index=False)
    np.savez(out / f"{name}_snapshot_profiles.npz", time=scalars["time"].to_numpy(), y=cfg.centers(1),
             **profiles)
    print(f"wrote {out / f'{name}_snapshot_scalars.csv'} and {name}_snapshot_profiles.npz")

    diff = compare_with_history(scalars, cfg)
    if diff is not None:
        print("relative difference |snapshot - history| / |history|:")
        print(diff.to_string(index=False, float_format="{:.1e}".format,
                             formatters={"time": "{:.2f}".format, "hst_time": "{:.2f}".format}))
        worst = diff.drop(columns=["time", "hst_time"]).abs().max()
        print("largest relative difference per column:", ", ".join(f"{k} {v:.1e}" for k, v in worst.items()))

    if not args.no_plot:
        import matplotlib

        matplotlib.use("Agg")
        fig_path = out / f"{name}_snapshot_summary.png"
        summary_figure(scalars, profiles, cfg, fig_path)
        print(f"wrote {fig_path}")
    return 0


if __name__ == "__main__":  # worker processes may import this file again
    raise SystemExit(main())
