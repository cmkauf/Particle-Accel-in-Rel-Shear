"""Make the explanatory images used by README.md and analysis/README.md.

The concept images are drawn from analytic or synthetic inputs only, so they can be remade
anywhere without simulation data::

    python docs/make_images.py                        # docs/images/{domain,array_orientation,particle_motion}.png
    python docs/make_images.py --gallery FIGDIR       # also the figure thumbnails in docs/images/figures

FIGDIR is the ``--out`` directory of ``analysis/reproduce_figures.py --formats png``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shearpic.physics.forcing import ShearProfile  # noqa: E402
from shearpic.physics.relativity import drift_velocity, lorentz_factor, velocity  # noqa: E402
from shearpic.plotting import plot_field  # noqa: E402

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = HERE / "images"
MAX_BYTES = 150_000

# reproduce_figures.py file name -> thumbnail name
GALLERY = {
    "shear_profile": "fig1_shear_profile",
    "flow_evolution": "fig2_flow_evolution",
    "turbulence": "fig3_turbulence",
    "steady_state": "fig4_steady_state",
    "particle_spectrum": "fig5_particle_spectrum",
    "phase_space": "fig6_phase_space",
    "power": "fig7_power",
}

RC = {
    "font.size": 9,
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "font.family": "DejaVu Sans",
}

JET, WIND, LAYER = "#c0392b", "#2c6fbb", "#f3d9a4"


def save(fig, path: Path) -> Path:
    """Save a PNG without the matplotlib version stamp, so reruns give identical files."""
    fig.savefig(path, dpi=110, metadata={"Software": None})
    plt.close(fig)
    return path


# ------------------------------------------------------------------ domain
def domain_image(out: Path) -> Path:
    """Box, shear layers and the double-tanh profile U_ref(y) of the reference geometry."""
    Lx, Ly = 10 * np.pi, 40 * np.pi
    prof = ShearProfile("double_tanh", amplitude=1.0, a=1.0, y1=-Ly / 4, y2=Ly / 4)
    y = np.linspace(-Ly / 2, Ly / 2, 2001)
    half = prof.layer_width(0.9) / 2

    fig, (ax_box, ax_u, ax_zoom) = plt.subplots(1, 3, figsize=(7.6, 4.6),
                                                gridspec_kw={"width_ratios": [1.0, 1.1, 1.1]})
    ax_box.set_xlim(-Lx / 2, Lx / 2)
    ax_box.set_ylim(-Ly / 2, Ly / 2)
    ax_box.set_aspect("equal")
    for yl in prof.layer_positions:
        ax_box.axhspan(yl - half, yl + half, color=LAYER, zorder=0)
        ax_box.axhline(yl, color="0.4", lw=0.6, ls="--")
    for yy, color, sign, label in [(0.0, JET, 1, "jet  +S U0"), (Ly * 0.4, WIND, -1, "wind  -S U0"),
                                   (-Ly * 0.34, WIND, -1, "wind  -S U0")]:
        for xx in (-Lx / 4, Lx / 4):
            ax_box.annotate("", xy=(xx + sign * 5, yy), xytext=(xx - sign * 5, yy),
                            arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))
        ax_box.text(0, yy + 6, label, color=color, ha="center", fontsize=8)
    ax_box.text(Lx / 2 + 1.5, prof.y2, "y2 = Ly/4", va="center", fontsize=8)
    ax_box.text(Lx / 2 + 1.5, prof.y1, "y1 = -Ly/4", va="center", fontsize=8)
    ax_box.annotate("", xy=(Lx / 2 - 2, -Ly / 2 + 6), xytext=(-Lx / 2 + 2, -Ly / 2 + 6),
                    arrowprops=dict(arrowstyle="-|>", color="k", lw=1.0))
    ax_box.text(0, -Ly / 2 + 9, "B0 along x", ha="center", fontsize=8)
    ax_box.set_xlabel("x / a")
    ax_box.set_ylabel("y / a")
    ax_box.set_title("box  Lx x Ly, periodic")

    ax_u.plot(prof.U(y), y, color="k", lw=1.2)
    for yl in prof.layer_positions:
        ax_u.axhspan(yl - half, yl + half, color=LAYER, zorder=0)
    ax_u.axvline(0, color="0.6", lw=0.6)
    ax_u.set_ylim(-Ly / 2, Ly / 2)
    ax_u.set_xlim(-1.4, 1.4)
    ax_u.set_xlabel("U_ref(y) / U0")
    ax_u.set_yticklabels([])
    ax_u.set_title("cfg.profile.U(y), S = 1")

    yz = np.linspace(prof.y2 - 6, prof.y2 + 6, 400)
    ax_zoom.plot(prof.U(yz), yz, color="k", lw=1.2)
    ax_zoom.axhspan(prof.y2 - half, prof.y2 + half, color=LAYER, zorder=0)
    ax_zoom.axhline(prof.y2, color="0.4", lw=0.6, ls="--")
    edges = prof.y2 - 6 + 0.75 * np.arange(17)
    ax_zoom.plot(prof.on_cpp_grid(np.append(edges, edges[-1] + 0.75)), edges, "o", ms=2.5, color=WIND)
    ax_zoom.text(-1.3, prof.y2 - 5.6, "dots: U_ref at\nlower cell faces", fontsize=7, color=WIND)
    ax_zoom.set_xlim(-1.4, 1.4)
    ax_zoom.set_ylim(prof.y2 - 6, prof.y2 + 6)
    ax_zoom.set_xlabel("U_ref(y) / U0")
    ax_zoom.set_ylabel("y / a")
    ax_zoom.yaxis.set_label_position("right")
    ax_zoom.yaxis.tick_right()
    ax_zoom.set_title(f"one layer: |U| < 0.9 S over {2 * half:.1f} a")
    fig.tight_layout(w_pad=2.5)
    return save(fig, out / "domain.png")


# ------------------------------------------------------- array orientation
def array_orientation_image(out: Path) -> Path:
    """An (nx, ny) field drawn with plot_field next to a bare imshow of the same array."""
    nx, ny = 8, 16
    x = (np.arange(nx) + 0.5) * 0.5
    y = (np.arange(ny) + 0.5) * 0.5
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = Y / y.max() + 2.0 * np.exp(-((X - 3.0) ** 2 + (Y - 6.5) ** 2) / 0.4)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.4, 4.0), gridspec_kw={"width_ratios": [1.0, 1.6]})
    plot_field(ax1, field, x, y, cmap="viridis", cbar=False)
    ax1.set_xlabel("x  (first index i)")
    ax1.set_ylabel("y  (second index j)")
    ax1.set_title("plot_field(ax, f, x, y)\nx right, y up", color="#1e7b34")
    ax1.plot(3.0, 6.5, "+", color="w", ms=10, mew=1.5)

    ax2.imshow(field, cmap="viridis")
    ax2.set_xlabel("column = j  (y)")
    ax2.set_ylabel("row = i  (x), counted downwards")
    ax2.set_title("ax.imshow(f)\naxes swapped, x runs downwards", color="#b03a2e")
    fig.text(0.5, 0.02, "f.shape == (nx, ny) == (8, 16);   f[i, j] is the cell at (x_i, y_j);   "
             "marker: blob at x = 3, y = 6.5", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    return save(fig, out / "array_orientation.png")


# --------------------------------------------------------- particle motion
def boris_orbit(u0, cE, B, q_mc: float, c: float, dt: float, n: int) -> np.ndarray:
    """Positions, shape (n + 1, 3), of du/dt = q_mc (cE + v x B) in uniform fields, by the relativistic Boris push."""
    x, u = np.zeros(3), np.asarray(u0, float)
    xs = [x.copy()]
    for _ in range(n):
        u_minus = u + 0.5 * q_mc * dt * cE
        t = 0.5 * q_mc * dt * B / lorentz_factor(u_minus, c)
        u_prime = u_minus + np.cross(u_minus, t)
        u_plus = u_minus + np.cross(u_prime, 2.0 * t / (1.0 + t @ t))
        u = u_plus + 0.5 * q_mc * dt * cE
        x = x + velocity(u, c) * dt
        xs.append(x.copy())
    return np.array(xs)


def particle_motion_image(out: Path) -> Path:
    """u = p/m against v = u/gamma, and a gyro-orbit drifting with w = cE x B / B^2."""
    c = 1.0
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.4, 3.2), gridspec_kw={"width_ratios": [1.0, 1.2]})

    u = np.linspace(0, 5, 400)[:, None] * np.array([1.0, 0.0, 0.0])
    ax1.plot(u[:, 0], velocity(u, c)[:, 0], color="k", lw=1.4, label="|v| / c = |u| / (gamma c)")
    ax1.plot(u[:, 0], lorentz_factor(u, c) - 1, color=JET, lw=1.2, label="gamma - 1")
    ax1.plot([0, 1.3], [0, 1.3], color="0.5", lw=0.8, ls="--", label="v = u (non-relativistic)")
    ax1.axhline(1.0, color="0.7", lw=0.6)
    ax1.set_xlim(0, 5)
    ax1.set_ylim(0, 3)
    ax1.set_xlabel("|u| / c,   u = p/m  (stored by Athena++)")
    ax1.set_title("momentum u and velocity v")
    ax1.legend(loc="upper left", frameon=False)

    B = np.array([0.0, 0.0, 1.0])
    cE = np.array([0.0, 0.25, 0.0])
    w = drift_velocity(cE, B)
    xs = boris_orbit(u0=[0.0, 0.8, 0.0], cE=cE, B=B, q_mc=2.0, c=c, dt=0.02, n=900)
    ax2.plot(xs[:, 0], xs[:, 1], color="k", lw=0.9)
    ax2.plot(*xs[0, :2], "o", color=WIND, ms=4)
    ax2.annotate("", xy=(3.2, -1.0), xytext=(0.6, -1.0),
                 arrowprops=dict(arrowstyle="-|>", color=JET, lw=1.4))
    ax2.text(1.9, -1.45, f"drift w = cE x B / B^2 = {w[0]:.2f} c", color=JET, ha="center", fontsize=8)
    ax2.annotate("", xy=(-0.6, 0.9), xytext=(-0.6, 0.2),
                 arrowprops=dict(arrowstyle="-|>", color=WIND, lw=1.2))
    ax2.text(-0.5, 0.45, "cE", color=WIND, fontsize=8)
    ax2.text(-0.9, 1.25, "B out of the page", fontsize=8)
    ax2.set_aspect("equal")
    ax2.set_xlim(-1.0, xs[:, 0].max() + 0.3)
    ax2.set_ylim(-1.8, 1.6)
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("du/dt = q_mc (cE + v x B)")
    fig.tight_layout()
    return save(fig, out / "particle_motion.png")


# ----------------------------------------------------------------- gallery
def thumbnail(src: Path, dest: Path, max_size: tuple[int, int] = (520, 640)) -> Path:
    """Downscale a figure PNG and reduce it to 256 colours if it is still above MAX_BYTES."""
    from PIL import Image

    with Image.open(src) as im:
        im = im.convert("RGB")
        im.thumbnail(max_size, Image.LANCZOS)
        im.save(dest, optimize=True)
        if dest.stat().st_size > MAX_BYTES:
            im.quantize(colors=256, method=Image.MEDIANCUT, dither=Image.NONE).save(dest, optimize=True)
    return dest


def gallery(figdir: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    missing = [name for name in GALLERY if not (figdir / f"{name}.png").is_file()]
    if missing:
        raise FileNotFoundError(f"{figdir} has no {', '.join(m + '.png' for m in missing)}; "
                                "run analysis/reproduce_figures.py --formats png --out FIGDIR first")
    return [thumbnail(figdir / f"{name}.png", out / f"{thumb}.png") for name, thumb in GALLERY.items()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="image directory (default: %(default)s)")
    parser.add_argument("--gallery", type=Path, default=None, metavar="FIGDIR",
                        help="also write thumbnails of the PNG figures in FIGDIR to OUT/figures")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    with plt.rc_context(RC):
        written = [domain_image(args.out), array_orientation_image(args.out), particle_motion_image(args.out)]
    if args.gallery is not None:
        written += gallery(args.gallery, args.out / "figures")
    for path in written:
        print(f"wrote {path} ({path.stat().st_size / 1000:.0f} kB)")
    too_big = [p for p in written if p.stat().st_size > MAX_BYTES]
    if too_big:
        print(f"error: larger than {MAX_BYTES / 1000:.0f} kB: {', '.join(map(str, too_big))}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
