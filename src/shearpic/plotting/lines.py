"""Line plots: time series, particle energy spectra and 2D phase-space histograms.

:func:`plot_spectrum` accepts any object with the interface of
``shearpic.physics.spectra.Spectrum`` (``variable``, ``centers(kind)``, ``dN_dx(normalize)``
and, for stairs, ``edges``), so this module does not import the physics package.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from .fields import _color_scaling
from .style import add_colorbar, format_axes

__all__ = ["VARIABLE_LABELS", "plot_timeseries", "plot_spectrum", "plot_phase_space"]

VARIABLE_LABELS = {
    "gamma": r"\gamma",
    "gamma_minus_1": r"\gamma-1",
    "p_over_mc": r"p/(mc)",
}
"""TeX symbols (without ``$``) for the spectrum variables."""

_STYLE_KEYS = ("label", "color", "linestyle")


def _style_value(style: Any, key: str):
    if style is None:
        return None
    if isinstance(style, Mapping):
        return style.get(key)
    return getattr(style, key, None)


# ---------------------------------------------------------------- time series
def plot_timeseries(ax, t, values, *, label: str | None = None, color: Any = None,
                    linestyle: str | None = None, style: Mapping[str, Any] | Any = None, **line_kw):
    """Plot a time series, e.g. a history column, and return the line.

    ``t`` is in units of a/U0.  ``style`` may be a registry experiment entry such as
    ``{"run": 423, "label": "S=1", "color": "C0", "linestyle": "--"}``, whose other keys are
    ignored; explicit ``label``, ``color`` and ``linestyle`` take precedence over it.
    ``line_kw`` are passed to ``Axes.plot``.
    """
    kw = {k: _style_value(style, k) for k in _STYLE_KEYS}
    for key, val in (("label", label), ("color", color), ("linestyle", linestyle)):
        if val is not None:
            kw[key] = val
    if kw["linestyle"] is None:
        kw["linestyle"] = "-"
    kw = {k: v for k, v in kw.items() if v is not None}
    t = np.asarray(t, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    if t.shape != values.shape:
        raise ValueError(f"t has shape {t.shape} but values has shape {values.shape}")
    (line,) = ax.plot(t, values, **kw, **line_kw)
    format_axes(ax)
    return line


# -------------------------------------------------------------------- spectra
def _spectrum_labels(variable: str, normalize, compensate: float) -> tuple[str, str]:
    sym = VARIABLE_LABELS.get(variable, variable.replace("_", r"\_"))
    wrapped = f"({sym})" if any(ch in sym for ch in "-+/") else sym
    dndx = rf"dN/d{wrapped}"
    if normalize not in (None, False, "none"):
        dndx = rf"N^{{-1}}\,{dndx}"
    if compensate:
        dndx = rf"{wrapped}^{{{compensate:g}}}\,{dndx}"
    return f"${sym}$", f"${dndx}$"


def _bin_centers(spectrum, kind: str) -> np.ndarray:
    """Bin centres; geometric centres fall back to arithmetic ones for bins with a lower edge <= 0."""
    edges = getattr(spectrum, "edges", None)
    if kind == "geometric" and edges is not None:
        e = np.asarray(edges, dtype=np.float64)
        if e.ndim == 1 and e.size >= 2 and e[0] <= 0:
            lo, hi = e[:-1], e[1:]
            out = 0.5 * (lo + hi)
            pos = lo > 0
            out[pos] = np.sqrt(lo[pos] * hi[pos])
            return out
    return np.asarray(spectrum.centers(kind), dtype=np.float64)


def plot_spectrum(ax, spectrum, *, normalize: str | None = "total", compensate: float = 0.0,
                  label: str | None = None, centers: str = "geometric", set_labels: bool = True,
                  style: str = "line", floor: float | None = None, **line_kw):
    """Plot a particle spectrum ``dN/dx`` on log-log axes and return the line.

    Parameters
    ----------
    spectrum : Spectrum-like
        Typically ``shearpic.physics.spectra.Spectrum``; ``style='stairs'`` also needs ``edges``.
    normalize : str or None
        Passed to ``spectrum.dN_dx``; ``'total'`` gives ``(1/N) dN/dx``.
    compensate : float
        Plot ``x**compensate dN/dx`` with x at the bin centres; a power law ``x^-compensate`` is flat.
    centers : str
        Kind of bin centre passed to ``spectrum.centers``.
    set_labels : bool
        Label the axes from ``spectrum.variable``.
    style : {'line', 'stairs'}
        Points at the bin centres joined by lines, or the histogram constant across each bin.
    floor : float, optional
        With ``style='stairs'``, draw empty bins at this value instead of leaving gaps.
    **line_kw
        Passed to ``Axes.plot``.

    Empty bins are not drawn.  With geometric centres, bins whose lower edge is <= 0, as in
    ``gamma_minus_1`` spectra converted from ``gamma`` bins starting at 1, are placed at their
    arithmetic centre, since a geometric centre of 0 cannot be shown on a log axis.
    """
    if style not in ("line", "stairs"):
        raise ValueError("style must be 'line' or 'stairs'")
    xc = _bin_centers(spectrum, centers)
    y = np.asarray(spectrum.dN_dx(normalize), dtype=np.float64)
    if xc.shape != y.shape:
        raise ValueError(f"spectrum centres {xc.shape} and dN/dx {y.shape} differ in shape")
    if compensate:
        y = y * xc**compensate
    empty = ~(y > 0)
    if style == "line":
        x = xc
        y = np.where(empty, np.nan, y)
    else:
        edges = np.asarray(spectrum.edges, dtype=np.float64)
        if edges.shape != (y.size + 1,):
            raise ValueError("style='stairs' needs spectrum.edges with one entry more than dN/dx")
        if floor is not None and not floor > 0:
            raise ValueError("floor must be positive on a log axis")
        fill = np.nan if floor is None else float(floor)
        y = np.where(empty, fill, y)
        x = np.repeat(edges, 2)[1:-1]
        y = np.repeat(y, 2)
        if edges[0] <= 0:  # a log axis cannot show the first edge, so start at the bin centre
            x[0] = xc[0]
    (line,) = ax.plot(x, y, label=label, **line_kw)
    ax.set_xscale("log")
    ax.set_yscale("log")
    if set_labels:
        xlabel, ylabel = _spectrum_labels(str(spectrum.variable), normalize, compensate)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
    format_axes(ax, minor=False)  # log axes already carry minor ticks
    return line


# ---------------------------------------------------------------- phase space
def _is_uniform(edges: np.ndarray) -> bool:
    step = np.diff(edges)
    return bool(np.allclose(step, step[0], rtol=1e-6, atol=0.0))


def _is_geometric(edges: np.ndarray) -> bool:
    return bool(np.all(edges > 0) and _is_uniform(np.log(edges)))


def plot_phase_space(ax, H, x_edges, y_edges, *, log: bool = True, cbar: str | None = "right",
                     xlabel: str = "", ylabel: str = "", cbar_label: str = "", cmap: Any = "Greys",
                     vmin: float | None = None, vmax: float | None = None, title: str | None = None,
                     interpolation: str = "nearest", **kw):
    """Draw a 2D histogram with extents from the bin edges and return the image or mesh.

    Parameters
    ----------
    H : array_like, shape (len(x_edges) - 1, len(y_edges) - 1)
        Counts or densities indexed ``[x_bin, y_bin]``, as returned by ``np.histogram2d``.
    x_edges, y_edges : array_like
        Uniform edges are drawn with ``imshow``, others with ``pcolormesh``, where geometric
        edges also switch that axis to a log scale.
    log : bool
        Logarithmic colour scale; empty bins are left blank.
    cbar : {'right', 'top', None}
        Colorbar position; the colorbar is available as ``artist.colorbar``.
    interpolation : str
        ``imshow`` interpolation; ``'nearest'`` avoids thin lines along masked empty bins.
    **kw
        Passed to ``imshow`` or ``pcolormesh``, e.g. ``rasterized=True``.
    """
    H = np.asarray(H, dtype=np.float64)
    xe = np.asarray(x_edges, dtype=np.float64)
    ye = np.asarray(y_edges, dtype=np.float64)
    if H.shape != (xe.size - 1, ye.size - 1):
        raise ValueError(f"H has shape {H.shape}; expected {(xe.size - 1, ye.size - 1)} from the edges")
    data, norm = _color_scaling(H, log=log, symmetric=False, vmin=vmin, vmax=vmax, warn=False)
    if _is_uniform(xe) and _is_uniform(ye):
        artist = ax.imshow(data.T, origin="lower", extent=[xe[0], xe[-1], ye[0], ye[-1]], aspect="auto",
                           cmap=cmap, norm=norm, interpolation=interpolation, **kw)
    else:
        artist = ax.pcolormesh(xe, ye, data.T, cmap=cmap, norm=norm, shading="flat", **kw)
        if not _is_uniform(xe) and _is_geometric(xe):
            ax.set_xscale("log")
        if not _is_uniform(ye) and _is_geometric(ye):
            ax.set_yscale("log")
        ax.set_xlim(xe[0], xe[-1])
        ax.set_ylim(ye[0], ye[-1])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    format_axes(ax)
    if cbar:
        add_colorbar(ax, artist, location=cbar, label=cbar_label)
    return artist
