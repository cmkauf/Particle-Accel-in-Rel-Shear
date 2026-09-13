"""2D field maps drawn with ``imshow`` on their cell-face extent, and x-averaged profiles.

Snapshot arrays are cell averages indexed (x, y), so an image of the whole box spans the
cell faces ``[x1min, x1max, x2min, x2max]``.  ``imshow`` receives ``field.T`` with
``origin='lower'``, putting x horizontal and y vertical.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np
from matplotlib.colors import LogNorm, Normalize

from .style import add_colorbar, format_axes

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = ["cell_edges", "image_extent", "plot_field", "plot_snapshot_field", "plot_profile"]

_UNIFORM_RTOL = 1e-4  # float32 coordinates are uniform only to ~1e-5 relative


# ------------------------------------------------------------ coordinates
def cell_edges(coord: Sequence[float] | np.ndarray, n: int) -> np.ndarray:
    """Float64 cell-face coordinates, shape (n + 1,), of a uniform grid with ``n`` cells.

    ``coord`` holds either the ``n`` cell centres or the ``n + 1`` faces.  A single centre or
    non-uniform spacing raises ``ValueError``; stretched grids need ``pcolormesh``.
    """
    c = np.asarray(coord, dtype=np.float64).ravel()
    if c.size not in (n, n + 1):
        raise ValueError(f"coordinate has {c.size} values, expected {n} centres or {n + 1} faces")
    if c.size < 2:
        raise ValueError("cannot infer the cell size from a single cell centre; pass extent=")
    step = np.diff(c)
    if not np.allclose(step, step[0], rtol=_UNIFORM_RTOL, atol=0.0):
        raise ValueError("coordinates are not uniformly spaced; imshow needs a uniform grid (use pcolormesh)")
    if c.size == n + 1:
        return c
    d = (c[-1] - c[0]) / (n - 1)
    edges = c[0] - 0.5 * d + d * np.arange(n + 1)
    edges[-1] = c[-1] + 0.5 * d
    return edges


def image_extent(shape: tuple[int, int], x=None, y=None) -> list[float]:
    """``imshow`` extent ``[left, right, bottom, top]`` of an (nx, ny) field.

    ``x`` and ``y`` may be cell centres or faces; a missing coordinate falls back to cell indices.
    """
    nx, ny = shape
    ex = cell_edges(x, nx) if x is not None else np.array([-0.5, nx - 0.5])
    ey = cell_edges(y, ny) if y is not None else np.array([-0.5, ny - 0.5])
    return [float(ex[0]), float(ex[-1]), float(ey[0]), float(ey[-1])]


def _as_2d(field, transpose: bool) -> np.ndarray:
    """Return an (nx, ny) array, dropping a trailing nz = 1 axis."""
    a = np.asarray(field)
    if a.ndim == 3:
        if a.shape[2] != 1:
            raise ValueError(f"field has shape {a.shape}: pass a 2D slice (e.g. field[:, :, k]) or a z-average")
        a = a[:, :, 0]
    if a.ndim != 2:
        raise ValueError(f"expected a 2D field, got shape {a.shape}")
    return a if transpose else a.T


def _color_scaling(data: np.ndarray, *, log: bool, symmetric: bool, vmin, vmax, warn: bool = True):
    """Colour normalisation of a field; returns ``(data_to_draw, norm)``."""
    if log and symmetric:
        raise ValueError("log=True and symmetric=True cannot be combined (use a SymLogNorm via norm=)")
    if log:
        bad = ~(data > 0)  # non-positive or NaN
        finite_bad = bad & np.isfinite(data)
        if bad.all():
            raise ValueError("log=True but the field has no positive values")
        if warn and finite_bad.any():
            warnings.warn(f"log colour scale: masking {int(finite_bad.sum())} non-positive value(s)", stacklevel=3)
        data = np.ma.masked_where(bad, data)
        return data, LogNorm(vmin=vmin, vmax=vmax)
    if symmetric:
        if vmax is None and vmin is None:
            finite = data[np.isfinite(data)]
            vmax = float(np.abs(finite).max()) if finite.size else 1.0
        else:
            vmax = float(max(abs(v) for v in (vmin, vmax) if v is not None))
        vmax = vmax if vmax > 0 else 1.0
        return data, Normalize(vmin=-vmax, vmax=vmax)
    return data, Normalize(vmin=vmin, vmax=vmax)


# ------------------------------------------------------------- field maps
def plot_field(ax, field, x=None, y=None, *, extent: Sequence[float] | None = None, cmap=None,
               log: bool = False, vmin: float | None = None, vmax: float | None = None,
               symmetric: bool = False, cbar: str | None = "top", cbar_label: str = "",
               xlabel: str = r"$x/a$", ylabel: str = r"$y/a$", title: str | None = None,
               transpose: bool = True, aspect: str | float = "equal", **imshow_kw):
    """Draw a 2D field with ``imshow`` on its cell-face extent and return the ``AxesImage``.

    Parameters
    ----------
    field : array_like, shape (nx, ny) or (nx, ny, 1)
        Field indexed (x, y), as read from a :class:`shearpic.io.athdf.Snapshot`.
    x, y : array_like, optional
        Cell centres or faces of a uniform grid [a], ignored with ``extent``; cell indices if omitted.
    extent : [left, right, bottom, top], optional
        Explicit image extent at the cell faces.
    cmap : str or Colormap, optional
        Default ``'viridis'``, or ``'RdBu_r'`` with ``symmetric``.
    log : bool
        Logarithmic colour scale; non-positive values are masked with a warning.
    symmetric : bool
        Colour limits ``[-m, m]`` with ``m = max|field|``, or ``max(|vmin|, |vmax|)`` if given.
    cbar : {'top', 'right', None}
        Colorbar position; the colorbar is available as ``image.colorbar``.
    transpose : bool
        Set to False if ``field`` is already an image indexed (y, x).
    **imshow_kw
        Passed to ``Axes.imshow``, e.g. ``interpolation`` or ``norm``.
    """
    data = _as_2d(field, transpose)
    extent = list(extent) if extent is not None else image_extent(data.shape, x, y)
    if len(extent) != 4:
        raise ValueError("extent must be [left, right, bottom, top]")
    norm = imshow_kw.pop("norm", None)
    if norm is None:
        data, norm = _color_scaling(data, log=log, symmetric=symmetric, vmin=vmin, vmax=vmax)
    if cmap is None:
        cmap = "RdBu_r" if symmetric else "viridis"
    im = ax.imshow(data.T, origin="lower", extent=extent, cmap=cmap, norm=norm, aspect=aspect, **imshow_kw)
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    format_axes(ax)
    if cbar:
        add_colorbar(ax, im, location=cbar, label=cbar_label)
    return im


def plot_snapshot_field(ax, snapshot_or_arrays, name: str, cfg: "RunConfig | None" = None, *,
                        z_index: int | None = None, **kw):
    """Plot variable ``name`` of a snapshot on its coordinates and return the ``AxesImage``.

    ``snapshot_or_arrays`` is a :class:`shearpic.io.athdf.Snapshot` or a dict of loaded
    (nx, ny, nz) arrays.  ``cfg`` supplies float64 domain bounds; without it a Snapshot's stored
    bounds are used and a dict is drawn on cell indices.  ``z_index`` selects the slice of a 3D
    field.  Other keywords go to :func:`plot_field`, with ``cbar_label`` defaulting to ``name``
    and, for a Snapshot, ``title`` to the snapshot time.
    """
    if hasattr(snapshot_or_arrays, "read") and hasattr(snapshot_or_arrays, "centers"):
        snap = snapshot_or_arrays
        arr = snap.read(name)[name]
        x = snap.centers(0, cfg.bounds[0] if cfg is not None else None)
        y = snap.centers(1, cfg.bounds[1] if cfg is not None else None)
        kw.setdefault("title", f"$t = {snap.time:.1f}\\,a/U_0$")
    elif isinstance(snapshot_or_arrays, Mapping):
        arr = np.asarray(snapshot_or_arrays[name])
        x = cfg.centers(0) if cfg is not None else None
        y = cfg.centers(1) if cfg is not None else None
    else:
        raise TypeError("snapshot_or_arrays must be a Snapshot or a mapping of arrays")
    if arr.ndim == 3 and arr.shape[2] > 1:
        if z_index is None:
            raise ValueError(f"{name} is 3D {arr.shape}; pass z_index= to choose a slice")
        arr = arr[:, :, z_index]
    kw.setdefault("cbar_label", name)
    return plot_field(ax, arr, x, y, **kw)


# ---------------------------------------------------------------- profiles
def plot_profile(ax, y, profile, cfg: "RunConfig | None" = None, *, label: str | None = None,
                 color: Any = None, linestyle: str = "-", vertical: bool = False,
                 shade_layers: bool = True, cutoff: float = 0.9, shade_color: Any = "0.88",
                 show_reference: bool = False, xlabel: str | None = None, ylabel: str | None = None,
                 **line_kw):
    """Plot an x-averaged profile against y, shading the shear layers, and return the line.

    Parameters
    ----------
    y : array_like, shape (ny,)
        Cell-centre y coordinates, ``cfg.centers(1)``.
    profile : array_like
        Profile of shape (ny,), or a field (nx, ny[, nz]) that is averaged over x and z here.
    cfg : RunConfig, optional
        Needed for the layer shading and ``show_reference``; also sets the limits to the domain.
    vertical : bool
        Put y on the vertical axis, e.g. next to a field map of the box.
    cutoff : float
        Shaded full layer width ``2 a artanh(cutoff)``, where ``|U_ref| < cutoff S U0``.
    show_reference : bool
        Also draw the reference shear profile ``U_ref(y)`` dashed, e.g. over ``<U_x>_x``.
    xlabel, ylabel : str, optional
        Axis labels; the coordinate axis defaults to ``$y/a$``.
    **line_kw
        Passed to ``Axes.plot``.
    """
    y = np.asarray(y, dtype=np.float64)
    prof = np.asarray(profile, dtype=np.float64)
    if prof.ndim == 3:
        prof = prof.mean(axis=(0, 2))
    elif prof.ndim == 2:
        prof = prof.mean(axis=0)
    if prof.shape != y.shape:
        raise ValueError(f"profile has shape {prof.shape} but y has shape {y.shape}")

    def draw(yy, vals, **k):
        return ax.plot(vals, yy, **k) if vertical else ax.plot(yy, vals, **k)

    if shade_layers and cfg is not None:
        half = 0.5 * cfg.profile.layer_width(cutoff)
        span = ax.axhspan if vertical else ax.axvspan
        for yl in cfg.profile.layer_positions:
            span(yl - half, yl + half, color=shade_color, lw=0, zorder=0)
    if show_reference:
        if cfg is None:
            raise ValueError("show_reference=True needs cfg")
        draw(y, cfg.profile.U(y), color="0.4", linestyle="--", lw=0.8, label=r"$U_{\rm ref}$")
    (line,) = draw(y, prof, color=color, linestyle=linestyle, label=label, **line_kw)
    lims = cfg.bounds[1] if cfg is not None else (float(y.min()), float(y.max()))
    coord_label = r"$y/a$"
    if vertical:
        if lims[1] > lims[0]:
            ax.set_ylim(*lims)
        ax.set_ylabel(ylabel if ylabel is not None else coord_label)
        if xlabel is not None:
            ax.set_xlabel(xlabel)
    else:
        if lims[1] > lims[0]:
            ax.set_xlim(*lims)
        ax.set_xlabel(xlabel if xlabel is not None else coord_label)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
    format_axes(ax)
    return line
