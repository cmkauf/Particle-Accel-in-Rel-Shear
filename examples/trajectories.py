"""Diagnostics and shear-layer crossings for every tracked particle of a run (optionally a movie).

A worker reads each ``trajectory_*.tab`` file, computes ``trajectory_diagnostics`` and the
shear-layer crossings, and returns a one-line summary with a thinned time series.  The
outputs go to ``run_output_dir(run, "trajectories")`` unless ``--out`` is given:

``<run>_trajectory_summary.csv``
    one row per particle: final and maximum gamma, crossings per layer and direction, energy
    gain, the integral of the sampled power ``q_mc v.cE`` and the mean pitch cosine in the
    E x B frame.
``<run>_trajectories.png``
    gamma(t) and cumulative crossings, energy gain against crossings, and the sampled work
    integral against the change of gamma.
``<run>_trajectories.mp4`` or ``.gif``
    with ``--animate N``, a movie of the N particles with the highest maximum gamma.

Usage::

    python examples/trajectories.py --run 404
    python examples/trajectories.py --run 404 --hysteresis 7.5            # a band of a few gyroradii: guiding-centre crossings
    python examples/trajectories.py --run 404 --animate 3 --t-start 0 --t-end 200 --stride 5

A crossing of the layer at ``y_l`` is counted when the particle is seen at ``|y - y_l| > h`` on
the side opposite to where it last left that band.  With the default ``h = a``, a particle whose
gyroradius exceeds a counts every gyration across the layer; ``h`` of a few ``r_g`` counts
guiding-centre crossings.  Energy changes are taken from the sampled gamma.
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
from shearpic.env import resolve_run, run_output_dir, warn_if_login_node
from shearpic.io.trajectory import find_trajectory_files, read_trajectory
from shearpic.parallel import mpi_world_size, parallel_map
from shearpic.physics.particles import crossing_events_for, cumulative_crossings, trajectory_diagnostics

N_KEEP = 2000  # samples kept per particle for plotting


# ------------------------------------------------------------------ worker
def summarize_trajectory(path: Path, cfg: RunConfig, hysteresis: float | None) -> dict:
    """Reduce one trajectory file to a summary row and thinned time series, in a worker process."""
    traj = read_trajectory(path)
    d = trajectory_diagnostics(traj, cfg)
    t, gamma = d["t"], d["gamma"]
    row = {
        "file": str(path), "pid": traj.pid, "init_mbid": traj.init_mbid, "n_samples": len(traj),
        "t_start": t[0], "t_end": t[-1], "gamma_start": gamma[0], "gamma_end": gamma[-1],
        "gamma_max": gamma.max(), "t_gamma_max": t[np.argmax(gamma)],
        "delta_gamma": gamma[-1] - gamma[0],
        "r_g_end": np.nan if d["r_g"] is None else d["r_g"][-1],
    }
    if d["dgamma_dt"] is not None:
        # trapezoidal integral of q_mc v.cE / c^2 over the sample times, compared with delta_gamma
        row["delta_gamma_from_work"] = float(np.sum(0.5 * (d["dgamma_dt"][1:] + d["dgamma_dt"][:-1]) * np.diff(t)))
        row["mean_pitch_drift"] = float(np.nanmean(d["pitch_drift"]))
    n_cross = np.zeros(len(traj), dtype=np.int64)
    if cfg.profile.layer_positions:
        events = crossing_events_for(traj, cfg, hysteresis)
        n_cross = cumulative_crossings(events, len(traj))
        row["n_crossings"] = events.count
        row["hysteresis"] = events.hysteresis
        for layer in range(len(events.layer_positions)):
            row[f"n_cross_layer{layer}_up"] = events.select(layer=layer, direction=+1).count
            row[f"n_cross_layer{layer}_down"] = events.select(layer=layer, direction=-1).count
    step = max(1, len(traj) // N_KEEP)  # return a thinned series, not the full arrays
    return {"row": row, "t": t[::step], "gamma": gamma[::step], "n_cross": n_cross[::step]}


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


def summary_figure(results: list[dict], table: pd.DataFrame, run_name: str, path: Path, max_lines: int) -> None:
    import matplotlib.pyplot as plt

    from shearpic.plotting import format_axes, paper_style

    order = np.argsort(table["gamma_max"].to_numpy())[::-1][:max_lines]  # most energetic first
    with paper_style():
        fig, axs = plt.subplots(2, 2, figsize=(7.0, 5.6))
        colors = plt.cm.plasma(np.linspace(0.0, 0.85, len(order)))
        for k, color in zip(order, colors):
            r = results[k]
            axs[0, 0].plot(r["t"], r["gamma"], color=color, lw=0.7)
            axs[0, 1].plot(r["t"], r["n_cross"], color=color, lw=0.7)
        axs[0, 0].set(xlabel=r"$t\ [a/U_0]$", ylabel=r"$\gamma$")
        axs[0, 1].set(xlabel=r"$t\ [a/U_0]$", ylabel="cumulative layer crossings")

        if "n_crossings" in table.columns:
            axs[1, 0].plot(table["n_crossings"], table["delta_gamma"], "o", ms=3)
            axs[1, 0].set(xlabel=f"layer crossings (h = {table['hysteresis'].iloc[0]:g} a)",
                          ylabel=r"$\gamma_{\rm end}-\gamma_{\rm start}$")
        if "delta_gamma_from_work" in table.columns:
            ax = axs[1, 1]
            ax.plot(table["delta_gamma"], table["delta_gamma_from_work"], "o", ms=3)
            lim = [min(table["delta_gamma"].min(), table["delta_gamma_from_work"].min(), 0.0),
                   max(table["delta_gamma"].max(), table["delta_gamma_from_work"].max())]
            ax.plot(lim, lim, "k--", lw=0.7)
            ax.set(xlabel=r"$\Delta\gamma$ (sampled $\gamma$)",
                   ylabel=r"$\int q\,\mathbf{v}\cdot c\mathbf{E}/(mc^2)\,dt$ (samples)")
        format_axes(axs)
        fig.suptitle(f"{run_name}: {len(table)} tracked particles")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)


# -------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run", default="404", help="run number, name (run0404) or directory (default 404)")
    parser.add_argument("--pattern", default="**/trajectory_*.tab", help="glob for trajectory files in the run")
    parser.add_argument("--backend", default="auto", choices=("auto", "serial", "thread", "process", "mpi"))
    parser.add_argument("--workers", default="auto", help="number of worker processes or 'auto'")
    parser.add_argument("--hysteresis", type=float, default=None,
                        help="crossing hysteresis half width in units of a (default: the layer half width a)")
    parser.add_argument("--max-lines", type=int, default=50, help="particles drawn in the time-series panels")
    parser.add_argument("--animate", type=int, default=0, metavar="N",
                        help="make a movie of the N most energetic particles")
    parser.add_argument("--t-start", type=float, default=None, help="movie start time")
    parser.add_argument("--t-end", type=float, default=None, help="movie end time")
    parser.add_argument("--stride", type=int, default=10, help="movie: use every N-th sample as a frame")
    parser.add_argument("--format", choices=("mp4", "gif"), default=None,
                        help="movie format (default mp4 if ffmpeg is available, else gif)")
    parser.add_argument("--out", type=Path, default=None,
                        help="output directory (default <output root>/run<NNNN>/trajectories)")
    args = parser.parse_args(argv)
    workers = int(args.workers) if str(args.workers).isdigit() else args.workers

    import matplotlib

    matplotlib.use("Agg")
    rank = mpi_rank()  # under ibrun/mpirun every rank runs this script; only rank 0 prints and writes
    run_dir = find_run(args.run)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = RunConfig.from_run_dir(run_dir)
    if rank == 0:
        for w in caught:
            print(f"note: {w.message}")
    files = find_trajectory_files(run_dir, args.pattern)
    if not files:
        raise SystemExit(f"no files matching {args.pattern!r} under {run_dir}")
    if rank == 0:
        warn_if_login_node("trajectories.py")
        print(f"{run_dir}: {len(files)} trajectory files; c = {cfg.c:g}, q/mc = {cfg.q_mc:g}, "
              f"r_g0 = {cfg.r_g0:g} a, T_gyro0 = {cfg.T_gyro0:.3g} a/U0")

    worker = functools.partial(summarize_trajectory, cfg=cfg, hysteresis=args.hysteresis)
    t0 = time.perf_counter()
    results = parallel_map(worker, files, backend=args.backend, n_workers=workers, progress=rank == 0,
                           desc="trajectories")
    if results is None:  # MPI rank > 0: rank 0 has gathered the results (and raises if a file failed)
        return 0
    print(f"processed {len(results)} trajectories in {time.perf_counter() - t0:.1f} s")

    name = run_output_dir(run_dir, create=False).name  # 'run0404' for run404 and run0404
    out = args.out.expanduser() if args.out is not None else run_output_dir(run_dir, "trajectories")
    out.mkdir(parents=True, exist_ok=True)
    table = pd.DataFrame([r["row"] for r in results])
    csv = out / f"{name}_trajectory_summary.csv"
    table.to_csv(csv, index=False)
    cols = [c for c in ("pid", "gamma_start", "gamma_end", "gamma_max", "n_crossings", "delta_gamma",
                        "delta_gamma_from_work", "mean_pitch_drift") if c in table.columns]
    top = table.sort_values("gamma_max", ascending=False)[cols].head(10)
    print(top.to_string(index=False, float_format="{:.3f}".format))
    print(f"wrote {csv}")

    fig_path = out / f"{name}_trajectories.png"
    summary_figure(results, table, name, fig_path, args.max_lines)
    print(f"wrote {fig_path}")

    if args.animate > 0:
        import matplotlib.animation as manim
        import matplotlib.pyplot as plt

        from shearpic.plotting import animate_trajectories, paper_style

        fmt = args.format or ("mp4" if manim.writers.is_available("ffmpeg") else "gif")
        chosen = table.sort_values("gamma_max", ascending=False)["file"].head(args.animate)
        trajs = [read_trajectory(p) for p in chosen]
        movie = out / f"{name}_trajectories.{fmt}"
        t0 = time.perf_counter()
        with paper_style():
            anim = animate_trajectories(trajs, cfg, t_start=args.t_start, t_end=args.t_end, stride=args.stride,
                                        outfile=movie, fps=20)
        plt.close(anim._fig)
        print(f"wrote {movie} ({anim.frame_times.size} frames, {time.perf_counter() - t0:.0f} s)")
    return 0


if __name__ == "__main__":  # required: worker processes may re-import this file
    raise SystemExit(main())
