"""Paper style, axis formatting, colorbars, panel layouts and the magnetic-field colormap.

Importing this module leaves ``matplotlib.rcParams`` untouched.  The style in the packaged
``shearpic.mplstyle`` is applied with the :func:`paper_style` context manager or, for the
rest of a script, with :func:`use_paper_style`.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Iterator, Mapping

import matplotlib as mpl
import matplotlib.style as mstyle
import numpy as np

__all__ = [
    "STYLE_PATH", "paper_style", "use_paper_style", "format_axes", "add_colorbar", "panel_grid",
    "panel_label", "bfield_cmap", "register_colormaps", "BFIELD_COLORS",
]

STYLE_PATH = Path(__file__).with_name("shearpic.mplstyle")
"""Path of the packaged matplotlib style file."""

# black -> purple -> magenta -> orange -> yellow, for |B| maps
BFIELD_COLORS = (
    "#0a0a0a", "#1a0d2e", "#2d1b69", "#4a2c7d", "#663399", "#8b4cb8", "#b565a7",
    "#d4729a", "#e8847f", "#f39c6b", "#fcb653", "#ffd23f", "#ffed4e",
)


# ------------------------------------------------------------------ style
@contextlib.contextmanager
def paper_style(rc: Mapping[str, Any] | None = None) -> Iterator[None]:
    """Context manager applying the paper style; rcParams are restored on exit.

    ``rc`` holds extra rcParams applied on top, e.g. ``{"figure.figsize": (7, 3)}``.
    """
    styles: list[Any] = [str(STYLE_PATH)]
    if rc:
        styles.append(dict(rc))
    with mstyle.context(styles):
        yield


def use_paper_style(rc: Mapping[str, Any] | None = None) -> None:
    """Apply the paper style for the rest of the session; meant for top-level scripts."""
    mstyle.use(str(STYLE_PATH))
    if rc:
        mpl.rcParams.update(dict(rc))


# ------------------------------------------------------------------- axes
def format_axes(ax, minor: bool = True):
    """Inward ticks on all four sides, plus minor ticks if ``minor``.

    ``ax`` may be one Axes or a list or array of Axes; it is returned unchanged.
    """
    axes = np.ravel(ax) if isinstance(ax, (list, tuple, np.ndarray)) else [ax]
    for a in axes:
        a.tick_params(axis="both", which="both", direction="in", top=True, right=True)
        if minor:
            a.minorticks_on()
    return ax


def add_colorbar(ax, mappable, location: str = "top", label: str = "", *, size: str | None = None,
                 pad: float = 0.05, cax=None, labelpad: float | None = None, **kwargs):
    """Attach a colorbar in an axes carved out of ``ax`` and return the ``Colorbar``.

    Unlike ``fig.colorbar(mappable, ax=ax)``, image and colorbar stay aligned also with
    ``aspect='equal'``.  ``location='top'`` gives a horizontal bar with ticks and label above it.
    ``size`` is the thickness as a fraction of the axes, ``pad`` the gap in inches and
    ``labelpad`` the label distance in points.  With ``cax``, e.g. from :func:`panel_grid`, the
    colorbar is drawn into that axes and ``size`` and ``pad`` are ignored.  Other keywords go to
    ``Figure.colorbar``.
    """
    if location not in ("top", "right", "bottom", "left"):
        raise ValueError(f"location must be 'top', 'right', 'bottom' or 'left', not {location!r}")
    horizontal = location in ("top", "bottom")
    if cax is None:
        from mpl_toolkits.axes_grid1 import make_axes_locatable

        if size is None:
            size = "3%" if horizontal else "4%"
        cax = make_axes_locatable(ax).append_axes(location, size=size, pad=pad)
    cbar = ax.figure.colorbar(mappable, cax=cax, orientation="horizontal" if horizontal else "vertical",
                              **kwargs)
    if location == "top":
        cax.xaxis.set_ticks_position("top")
        cax.xaxis.set_label_position("top")
    elif location == "left":
        cax.yaxis.set_ticks_position("left")
        cax.yaxis.set_label_position("left")
    cax.tick_params(which="both", direction="in")
    if label:
        cbar.set_label(label, labelpad=labelpad)
    return cbar


# ----------------------------------------------------------------- layout
def panel_grid(n_rows: int, n_cols: int, panel_aspect: float, width: float, *, cbar_row: bool = True,
               wspace: float = 0.05, hspace: float = 0.1, cbar_height: float = 0.12, cbar_pad: float = 0.05,
               margins: tuple[float, float, float, float] = (0.6, 0.1, 0.5, 0.1), share: bool = True,
               label_outer: bool = True, **figure_kw):
    """Figure of identical fixed-shape panels with an optional colorbar row on top.

    Panels are placed in absolute inches, all with height ``panel_aspect`` times their width,
    and the figure height follows from the layout.  For maps of an ``Lx x Ly`` box pass
    ``panel_aspect = Ly / Lx``; with ``aspect='equal'`` the images fill their panels and share
    one length scale, which ``plt.subplots`` does not guarantee once colorbars take space.

    Parameters
    ----------
    width : float
        Figure width including the margins [inches].
    cbar_row : bool
        Add a horizontal colorbar axes above each column, as wide as the panels.
    wspace, hspace, cbar_height, cbar_pad : float
        Gaps between panels, colorbar thickness and its gap above the top row [inches].
    margins : (left, right, bottom, top)
        Space around the panels for tick and axis labels [inches].
    share : bool
        Share the x and y axes of all panels.
    label_outer : bool
        Keep x tick labels only on the bottom row and y tick labels only on the first column.
    **figure_kw
        Passed to ``plt.figure``.

    Returns
    -------
    fig, axes, cbar_axes
        ``axes`` has shape (n_rows, n_cols); ``cbar_axes`` has shape (n_cols,), or is None.
    """
    import matplotlib.pyplot as plt

    n_rows, n_cols = int(n_rows), int(n_cols)
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be at least 1")
    if not panel_aspect > 0 or not width > 0:
        raise ValueError("panel_aspect and width must be positive")
    left, right, bottom, top = (float(m) for m in margins)
    panel_w = (width - left - right - (n_cols - 1) * wspace) / n_cols
    if panel_w <= 0:
        raise ValueError(f"width {width} in leaves no room for {n_cols} panels with these margins and gaps")
    panel_h = panel_aspect * panel_w
    extra = (cbar_pad + cbar_height) if cbar_row else 0.0
    height = bottom + n_rows * panel_h + (n_rows - 1) * hspace + extra + top
    fig = plt.figure(figsize=(width, height), **figure_kw)

    axes = np.empty((n_rows, n_cols), dtype=object)
    for i in range(n_rows):
        y0 = bottom + (n_rows - 1 - i) * (panel_h + hspace)
        for j in range(n_cols):
            x0 = left + j * (panel_w + wspace)
            ax = fig.add_axes((x0 / width, y0 / height, panel_w / width, panel_h / height))
            if share and (i, j) != (0, 0):
                ax.sharex(axes[0, 0])
                ax.sharey(axes[0, 0])
            if label_outer:
                ax.tick_params(labelbottom=(i == n_rows - 1), labelleft=(j == 0))
            axes[i, j] = ax
    cbar_axes = None
    if cbar_row:
        cbar_axes = np.empty(n_cols, dtype=object)
        y0 = bottom + n_rows * panel_h + (n_rows - 1) * hspace + cbar_pad
        for j in range(n_cols):
            x0 = left + j * (panel_w + wspace)
            cbar_axes[j] = fig.add_axes((x0 / width, y0 / height, panel_w / width, cbar_height / height))
    return fig, axes, cbar_axes


def panel_label(ax, text: str, loc: str = "upper left", *, pad: float = 0.03, box: bool = True,
                box_alpha: float = 0.8, **text_kw):
    """Write ``text`` in a corner of ``ax``, e.g. a snapshot time, and return the ``Text``.

    ``loc`` is 'upper left', 'upper right', 'lower left' or 'lower right'.  ``pad`` is the
    distance from both edges of the corner as a fraction of the panel width.
    With ``box`` the text sits on a rounded white box of opacity ``box_alpha``.  ``text_kw`` are
    passed to ``Axes.text``.
    """
    vert, horiz = loc.split() if " " in loc else ("", "")
    if vert not in ("upper", "lower") or horiz not in ("left", "right"):
        raise ValueError(f"loc must be 'upper left', 'upper right', 'lower left' or 'lower right', not {loc!r}")
    bbox = ax.get_position()
    fig_w, fig_h = ax.figure.get_size_inches()
    ratio = (bbox.width * fig_w) / max(bbox.height * fig_h, 1e-12)
    px, py = pad, pad * ratio
    x = px if horiz == "left" else 1.0 - px
    y = 1.0 - py if vert == "upper" else py
    kw = dict(transform=ax.transAxes, ha=horiz, va="top" if vert == "upper" else "bottom", zorder=10)
    if box:
        kw["bbox"] = dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="none", alpha=box_alpha)
    kw.update(text_kw)
    return ax.text(x, y, text, **kw)


# -------------------------------------------------------------- colormaps
def bfield_cmap(n_colors: int = 256):
    """The 'shearpic_bfield' colormap for |B| maps, without registering it."""
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("shearpic_bfield", list(BFIELD_COLORS), N=n_colors)


def register_colormaps() -> dict[str, Any]:
    """Register 'shearpic_bfield' and 'shearpic_bfield_r' with matplotlib.

    Repeated calls are harmless.  Returns ``{name: Colormap}`` of the registered maps.
    """
    cmap = bfield_cmap()
    maps = {"shearpic_bfield": cmap, "shearpic_bfield_r": cmap.reversed(name="shearpic_bfield_r")}
    for name, cm in maps.items():
        if name not in mpl.colormaps:
            mpl.colormaps.register(cm, name=name)
    return {name: mpl.colormaps[name] for name in maps}

