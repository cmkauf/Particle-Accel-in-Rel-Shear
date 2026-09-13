"""Animations of tracked-particle trajectories.

Each frame shows the particle positions in the x-y plane and the reduced momenta
u_x, u_y, u_z against y (u = p/m = gamma v, in units of U0), with fading trails and the
shear layers shaded::

    trajs = load_trajectories(run_dir / "high_energy_particles")
    anim = animate_trajectories(trajs[:4], cfg, t_start=0, t_end=400, outfile="trajectories.mp4")
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

import numpy as np

from ..physics.relativity import lorentz_factor
from .style import format_axes

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig
    from ..io.trajectory import Trajectory

__all__ = ["animate_trajectories", "trail_alpha"]


def trail_alpha(n_points: int, n_history: int, faint: float = 0.05, bright: float = 0.3) -> np.ndarray:
    """Opacities of a trail of ``n_points <= n_history`` points, oldest first.

    The newest point is opaque and older ones fade linearly from ``bright`` to ``faint`` over a
    full trail of ``n_history`` points; shorter trails use the newest part of that ramp.
    """
    if not 1 <= n_points <= n_history:
        raise ValueError(f"need 1 <= n_points <= n_history, got {n_points}, {n_history}")
    ramp = np.linspace(faint, bright, n_history - 1)
    return np.append(ramp[ramp.size - (n_points - 1):], 1.0)


def _nearest_index(t: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Index of the sample of sorted ``t`` nearest to each value in ``times``."""
    if t.size == 1:
        return np.zeros(times.size, dtype=int)
    i = np.clip(np.searchsorted(t, times), 1, t.size - 1)
    return np.where(np.abs(times - t[i - 1]) <= np.abs(t[i] - times), i - 1, i)


def animate_trajectories(trajectories: Sequence["Trajectory"], cfg: "RunConfig", *,
                         t_start: float | None = None, t_end: float | None = None, stride: int = 1,
                         n_history: int = 7, momentum_limits: float | tuple[float, float] | None = None,
                         outfile: str | Path | None = None, fps: int = 30, size_by_gamma: bool = True,
                         marker_size: float = 12.0, layer_cutoff: float = 0.9,
                         colors: Sequence[Any] | None = None, figsize: tuple[float, float] = (14.0, 7.0),
                         dpi: float | None = None):
    """Build, and optionally save, a ``FuncAnimation`` of particle trajectories.

    The frame times are stored as ``anim.frame_times``; scripts should close the figure with
    ``plt.close(anim._fig)`` when done.

    Parameters
    ----------
    cfg : RunConfig
        Supplies the domain bounds, the shear-layer positions and ``c``.
    t_start, t_end : float, optional
        Time range [a/U0], restricted to the window covered by all trajectories.
    stride : int
        Use every ``stride``-th sample of the first trajectory as a frame; the others are
        sampled at the nearest time.
    n_history : int
        Points per trail, including the current position.
    momentum_limits : float or (float, float), optional
        Limits of the momentum panels; default +-1.05 times the 99.5th percentile of |u_i|.
    outfile : str or Path, optional
        Save to this path, ``.gif`` with Pillow and other suffixes with ffmpeg; paths inside
        ``cfg.run_dir`` are refused.
    size_by_gamma : bool
        Scale the marker area with the Lorentz factor gamma = sqrt(1 + u^2/c^2).
    marker_size : float
        Marker area at gamma = 1 [points^2].
    layer_cutoff : float
        Shaded layer width ``2 a artanh(layer_cutoff)``.
    colors : sequence, optional
        One colour per trajectory; default ``C0, C1, ...``.
    """
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation
    from matplotlib.colors import to_rgba

    trajs = list(trajectories)
    if not trajs:
        raise ValueError("no trajectories given")
    if any(len(tr) == 0 for tr in trajs):
        raise ValueError("empty trajectory in input")
    if stride < 1 or n_history < 1:
        raise ValueError("stride and n_history must be >= 1")
    if size_by_gamma and cfg.c is None:
        raise ValueError("size_by_gamma needs cfg.c (speed_of_light); pass size_by_gamma=False")
    if colors is None:
        colors = [f"C{i % 10}" for i in range(len(trajs))]
    if len(colors) < len(trajs):
        raise ValueError(f"{len(colors)} colours for {len(trajs)} trajectories")
    writer = _movie_writer(Path(outfile), cfg, fps) if outfile is not None else None  # fail before plotting

    lo =max(float(tr.t[0]) for tr in trajs)
    hi = min(float(tr.t[-1]) for tr in trajs)
    if t_start is not None:
        lo = max(lo, float(t_start))
    if t_end is not None:
        hi = min(hi, float(t_end))
    t_ref = trajs[0].t
    frame_times = t_ref[(t_ref >= lo) & (t_ref <= hi)][::stride]
    if frame_times.size == 0:
        raise ValueError(f"the trajectories share no samples in the window [{lo}, {hi}]")

    xs, us, gs = [], [], []
    for tr in trajs:
        idx = _nearest_index(tr.t, frame_times)
        xs.append(np.asarray(tr.x[idx], dtype=np.float64))
        us.append(np.asarray(tr.u[idx], dtype=np.float64))
        gs.append(lorentz_factor(us[-1], cfg.c) if size_by_gamma else np.ones(frame_times.size))

    if momentum_limits is None:
        L = 1.05 * float(np.percentile(np.abs(np.concatenate(us)), 99.5))
        momentum_limits = (-L, L) if L > 0 else (-1.0, 1.0)
    elif np.ndim(momentum_limits) == 0:
        momentum_limits = (-abs(float(momentum_limits)), abs(float(momentum_limits)))
    p_lo, p_hi = (float(v) for v in momentum_limits)

    (x_lo, x_hi), (y_lo, y_hi) = cfg.bounds[0], cfg.bounds[1]
    fig, axs = plt.subplots(1, 4, figsize=figsize, sharey=True)
    half = 0.5 * cfg.profile.layer_width(layer_cutoff)
    for ax in axs:
        for yl in cfg.profile.layer_positions:
            ax.axhspan(yl - half, yl + half, color="0.85", lw=0, zorder=0)
    axs[0].set(xlim=(x_lo, x_hi), ylim=(y_lo, y_hi), xlabel=r"$x/a$", ylabel=r"$y/a$")
    for ax, comp in zip(axs[1:], "xyz"):
        ax.axvline(0.0, color="k", ls="--", lw=0.8, alpha=0.5, zorder=1)
        ax.set(xlim=(p_lo, p_hi), xlabel=rf"$u_{comp} = p_{comp}/m\ [U_0]$")
    format_axes(axs)
    title = fig.suptitle("")

    rgba = [np.array(to_rgba(c)) for c in colors]
    scatters = [[ax.scatter([], [], s=marker_size, edgecolors="none", zorder=3) for ax in axs] for _ in trajs]

    def update(frame: int):
        start = max(0, frame - n_history + 1)
        window = slice(start, frame + 1)
        alpha = trail_alpha(frame + 1 - start, n_history)
        artists = []
        for k, panels in enumerate(scatters):
            x, u, g = xs[k][window], us[k][window], gs[k][window]
            face = np.tile(rgba[k], (alpha.size, 1))
            face[:, 3] = alpha
            points = (np.c_[x[:, 0], x[:, 1]], np.c_[u[:, 0], x[:, 1]], np.c_[u[:, 1], x[:, 1]],
                      np.c_[u[:, 2], x[:, 1]])
            for coll, xy in zip(panels, points):
                coll.set_offsets(xy)
                coll.set_sizes(marker_size * g)
                coll.set_facecolor(face)
                artists.append(coll)
        title.set_text(rf"$t = {frame_times[frame]:.1f}\ a/U_0$")
        return artists + [title]

    anim = FuncAnimation(fig, update, frames=frame_times.size, interval=1000.0 / fps, blit=False)
    anim.frame_times = frame_times
    if writer is not None:
        path = Path(outfile)
        path.parent.mkdir(parents=True, exist_ok=True)
        anim.save(str(path), writer=writer, dpi=dpi)
    return anim


def _movie_writer(path: Path, cfg: "RunConfig", fps: int):
    """Check that ``path`` may be written and return a matplotlib movie writer for it."""
    from matplotlib import animation as manim

    if cfg.run_dir is not None:
        run_dir = Path(cfg.run_dir).resolve()
        target = path.resolve()
        if target == run_dir or run_dir in target.parents:
            raise ValueError(f"refusing to write {path} inside the raw run directory {run_dir}; "
                             "use shearpic.env.output_dir(...) or another location")
    if path.suffix.lower() == ".gif":
        writer = manim.PillowWriter(fps=fps)
    else:
        if not manim.writers.is_available("ffmpeg"):
            raise RuntimeError("saving a movie needs ffmpeg, which matplotlib cannot find. Install it "
                               "(e.g. 'conda install ffmpeg', 'brew install ffmpeg', or 'module load ffmpeg' on "
                               "a cluster) or save as .gif instead")
        writer = manim.FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p"])
    return writer
