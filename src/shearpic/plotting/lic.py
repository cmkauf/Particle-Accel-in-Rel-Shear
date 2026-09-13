"""Line integral convolution (LIC) images of 2D vector fields, e.g. in-plane magnetic field lines.

LIC smears white noise along the streamlines of a vector field, so the texture shows the
field direction, usually drawn over a colour map of a scalar such as ``|B|``.  The
convolution is done by the optional package `rlic <https://github.com/neutrinoceros/rlic>`_
(``pip install shearpic[lic]``).  Arrays are indexed (x, y); the transposes to rlic's
(row, column) = (y, x) images happen internally.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .fields import _as_2d, cell_edges
from .style import add_colorbar, format_axes

__all__ = ["block_mean", "equalize_histogram", "lic_blend", "lic_texture", "plot_lic", "shade_texture"]

_LIC_MODES = ("polarization", "velocity")


def _import_rlic():
    try:
        import rlic
    except ImportError as err:  # pragma: no cover - depends on the environment
        raise ImportError("LIC plots need the optional package 'rlic': pip install shearpic[lic]") from err
    return rlic


def _periodic_axes(periodic: bool | Sequence[bool]) -> tuple[bool, bool]:
    """``(periodic in x, periodic in y)`` from a bool or a pair of bools."""
    if isinstance(periodic, (bool, np.bool_)):
        return bool(periodic), bool(periodic)
    values = tuple(bool(p) for p in periodic)
    if len(values) != 2:
        raise ValueError(f"periodic must be a bool or a pair (x, y) of bools, got {periodic!r}")
    return values  # type: ignore[return-value]


def block_mean(a: np.ndarray, factor: int) -> np.ndarray:
    """Average a 2D array over ``factor x factor`` blocks, giving shape (nx // factor, ny // factor).

    Trailing rows and columns that do not fill a whole block are dropped.
    """
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError(f"block_mean expects a 2D array, got shape {a.shape}")
    factor = int(factor)
    if factor < 1:
        raise ValueError("downsample factor must be a positive integer")
    if factor == 1:
        return a
    nx, ny = a.shape[0] // factor, a.shape[1] // factor
    if nx == 0 or ny == 0:
        raise ValueError(f"downsample factor {factor} is larger than the array {a.shape}")
    trimmed = a[: nx * factor, : ny * factor]
    return trimmed.reshape(nx, factor, ny, factor).mean(axis=(1, 3))


def equalize_histogram(image: np.ndarray, nbins: int = 256) -> np.ndarray:
    """Histogram equalisation: map values through their empirical CDF onto [0, 1].

    This stretches the contrast of LIC textures, whose values cluster around the noise mean.
    Non-finite pixels become NaN.
    """
    img = np.asarray(image, dtype=np.float64)
    finite = np.isfinite(img)
    out = np.full(img.shape, np.nan)
    if not finite.any():
        return out
    vals = img[finite]
    lo, hi = vals.min(), vals.max()
    if hi <= lo:
        out[finite] = 0.5
        return out
    hist, edges = np.histogram(vals, bins=nbins, range=(lo, hi))
    cdf = np.cumsum(hist, dtype=np.float64)
    cdf /= cdf[-1]
    centers = 0.5 * (edges[:-1] + edges[1:])
    out[finite] = np.interp(vals, centers, cdf)
    return out


def shade_texture(texture: np.ndarray, *, azdeg: float = 0.0, altdeg: float = 45.0, vert_exag: float = 5.0,
                  periodic: bool | Sequence[bool] = True) -> np.ndarray:
    """Relief shading of a texture indexed (x, y), lit from one side like ``LightSource.hillshade``.

    The texture is divided by its maximum and shaded with light-source azimuth ``azdeg`` and
    altitude ``altdeg`` [degrees] and vertical exaggeration ``vert_exag``.  The azimuth refers
    to rows ordered by increasing y, so with ``origin='lower'`` the default ``azdeg=0`` lights
    the relief from the bottom.  Along ``periodic`` axes the slopes are periodic central
    differences, which leaves no seam at the box edges; with ``periodic=False`` the result is
    ``hillshade(texture.T / max, vert_exag=vert_exag).T``.

    Returns float32 illumination in [0, 1], stretched over its own range; NaN pixels count as 0.
    """
    from matplotlib.colors import LightSource

    img = np.asarray(texture, dtype=np.float64)
    if img.ndim != 2 or min(img.shape) < 2:
        raise ValueError(f"shade_texture expects a 2D texture of at least 2 x 2 pixels, got shape {img.shape}")
    img = np.where(np.isfinite(img), img, 0.0)
    peak = img.max()
    if peak > 0:
        img = img / peak
    px, py = _periodic_axes(periodic)

    def derivative(f, axis, spacing, wrap):
        if wrap:
            return (np.roll(f, -1, axis) - np.roll(f, 1, axis)) / (2.0 * spacing)
        return np.gradient(f, spacing, axis=axis)

    # LightSource.hillshade on the (rows = y, columns = x) image: row spacing -1, column spacing +1
    elevation = vert_exag * img
    normal = np.empty(img.shape + (3,))
    normal[..., 0] = -derivative(elevation, 0, 1.0, px)
    normal[..., 1] = -derivative(elevation, 1, -1.0, py)
    normal[..., 2] = 1.0
    normal /= np.sqrt(np.sum(normal**2, axis=-1, keepdims=True))
    return LightSource(azdeg=azdeg, altdeg=altdeg).shade_normals(normal).astype(np.float32)


def lic_texture(vx: np.ndarray, vy: np.ndarray, *, downsample: int = 1, kernel_length: int = 64,
                niter: int = 25, mode: str = "polarization", spacing: Sequence[float] | None = None,
                seed: int | None = 0, equalize: bool = True, periodic: bool | Sequence[bool] = True,
                shade: bool = False, noise: np.ndarray | None = None) -> np.ndarray:
    """LIC texture of the in-plane field ``(vx, vy)``.

    Returns a float32 array indexed (x, y) of shape (nx // downsample, ny // downsample).

    Parameters
    ----------
    vx, vy : ndarray, shape (nx, ny) or (nx, ny, 1)
        Vector components on a uniform grid; only their direction matters.
    downsample : int
        Block-average the vectors over this many cells per axis first; the cost scales with the number of pixels.
    kernel_length : int
        Length of the sine-shaped kernel in pixels; a pass reaches ``kernel_length / 2`` each way.
    niter : int
        Number of convolution passes; more passes give smoother, longer streaks.
    mode : {'polarization', 'velocity'}
        ``'polarization'`` ignores the sign of the vectors, so streaks continue across field reversals.
    spacing : (dx, dy), optional
        Cell sizes, needed only when ``dx != dy``.
    seed : int or None
        Seed of the uniform white noise; ignored when ``noise`` is given.
    equalize : bool
        Apply :func:`equalize_histogram`, mapping the texture onto [0, 1].
    periodic : bool or (bool, bool)
        Periodic boundaries along x and/or y, so a periodic field gives a periodic texture; closed
        boundaries are faster and end streaks at the edges.
    shade : bool
        Apply :func:`shade_texture` at the end.
    noise : ndarray, optional
        Input texture on the downsampled grid, indexed (x, y), instead of seeded noise.
    """
    if mode not in _LIC_MODES:
        raise ValueError(f"mode must be one of {_LIC_MODES}, not {mode!r}")
    if int(kernel_length) < 2:
        raise ValueError("kernel_length must be at least 2")
    if int(niter) < 1:
        raise ValueError("niter must be at least 1")
    px, py = _periodic_axes(periodic)
    rlic = _import_rlic()
    u = _as_2d(vx, True)
    v = _as_2d(vy, True)
    if u.shape != v.shape:
        raise ValueError(f"vx {u.shape} and vy {v.shape} must have the same shape")
    u = block_mean(np.asarray(u, dtype=np.float64), downsample)
    v = block_mean(np.asarray(v, dtype=np.float64), downsample)
    if spacing is not None:
        dx, dy = float(spacing[0]), float(spacing[1])
        u, v = u / dx, v / dy
    u = np.where(np.isfinite(u), u, 0.0)
    v = np.where(np.isfinite(v), v, 0.0)

    ny_nx = (u.shape[1], u.shape[0])  # rlic images are (rows = y, columns = x)
    if noise is None:
        texture = np.random.default_rng(seed).random(ny_nx, dtype=np.float32)
    else:
        noise = np.asarray(noise)
        if noise.shape != u.shape:
            raise ValueError(f"noise has shape {noise.shape}, expected {u.shape} (the downsampled grid)")
        texture = np.ascontiguousarray(noise.T, dtype=np.float32)
    n = int(kernel_length)
    kernel = np.sin(np.pi * (np.arange(n) + 0.5) / n).astype(np.float32)
    kernel /= kernel.sum()  # unit sum keeps repeated passes within float32 range
    uu = np.ascontiguousarray(u.T, dtype=np.float32)   # horizontal component
    vv = np.ascontiguousarray(v.T, dtype=np.float32)   # vertical component

    if not (px or py):
        image = rlic.convolve(texture, uu, vv, kernel=kernel, uv_mode=mode, iterations=int(niter))
    else:
        # One pass moves at most n/2 pixels along a streamline, so a wrap of n//2 + 1 pixels gives
        # every interior pixel the samples of an infinite periodic grid.  The padding is renewed
        # after each pass because rlic's own iterations would let truncated edge streaks creep inwards.
        p = n // 2 + 1
        # (rows = y, columns = x); mode='wrap' also handles images smaller than the padding
        pad =((p, p) if py else (0, 0), (p, p) if px else (0, 0))
        uu = np.pad(uu, pad, mode="wrap")
        vv = np.pad(vv, pad, mode="wrap")
        rows = slice(pad[0][0], pad[0][0] + ny_nx[0])
        cols = slice(pad[1][0], pad[1][0] + ny_nx[1])
        image = texture
        for _ in range(int(niter)):
            image = rlic.convolve(np.pad(image, pad, mode="wrap"), uu, vv, kernel=kernel, uv_mode=mode,
                                  iterations=1)[rows, cols]
    image = np.ascontiguousarray(image.T)  # back to (x, y)
    if equalize:
        image = equalize_histogram(image).astype(np.float32)
    if shade:
        image = shade_texture(image, periodic=(px, py))
    return image


def plot_lic(ax, x, y, vx=None, vy=None, color_field=None, *, downsample: int = 2, cmap: Any = None,
             alpha: float = 0.35, vmin: float | None = None, vmax: float | None = None,
             log: bool = False, symmetric: bool = False, cbar: str | None = "top", cbar_label: str = "",
             kernel_length: int = 64, niter: int = 25, mode: str = "polarization",
             periodic: bool | Sequence[bool] = True, seed: int | None = 0, shade: bool = False,
             texture: np.ndarray | None = None, texture_cmap: Any = "gray",
             color_extent: Sequence[float] | None = None, xlabel: str | None = r"$x/a$", ylabel: str | None = r"$y/a$", title: str | None = None,
             aspect: str | float = "equal", **imshow_kw):
    """LIC texture of ``(vx, vy)`` drawn over a colour map of ``color_field``.

    ``x`` and ``y`` are the cell centres or faces of the vector grid.  ``color_field`` is a scalar
    indexed (x, y) covering the same domain at any resolution, or ``color_extent``
    ([left, right, bottom, top]); the texture is blended over it with opacity ``alpha``, or shown
    alone without it.  Colour options are as in :func:`shearpic.plotting.fields.plot_field` and
    texture options as in :func:`lic_texture`; ``**imshow_kw`` go to ``imshow`` except ``norm``,
    which normalises ``color_field``.  A precomputed ``texture`` replaces ``vx`` and ``vy`` and must
    not be shaded already when ``shade`` is set.  A ``None`` axis label is left unchanged.

    Returns ``(mappable, image)``: with ``color_field`` an undrawn ``ScalarMappable`` for colorbars
    and the blended RGBA image, which has the attribute ``lic_alpha``; without it None and the
    texture image.
    """
    if vx is None or vy is None:
        if texture is None:
            raise ValueError("plot_lic needs vx and vy, or a precomputed texture")
        texture = np.asarray(texture)
        if texture.ndim != 2:
            raise ValueError(f"texture must be 2D, got shape {texture.shape}")
        nx, ny = texture.shape
        k = 1
    else:
        u = _as_2d(vx, True)
        nx, ny = u.shape
        if texture is None:
            ex0, ey0 = cell_edges(x, nx), cell_edges(y, ny)
            spacing = (ex0[1] - ex0[0], ey0[1] - ey0[0])
            texture = lic_texture(vx, vy, downsample=downsample, kernel_length=kernel_length, niter=niter,
                                  mode=mode, spacing=spacing, periodic=periodic, seed=seed)
        texture = np.asarray(texture)
        k = nx // texture.shape[0] if texture.ndim == 2 and texture.shape[0] > 0 else 0
        if k < 1 or texture.shape != (nx // k, ny // k):
            raise ValueError(f"texture shape {texture.shape} does not match the grid {u.shape}")
    ex, ey = cell_edges(x, nx), cell_edges(y, ny)
    if shade:
        texture = shade_texture(texture, periodic=periodic)
    tex_extent = [ex[0], ex[texture.shape[0] * k], ey[0], ey[texture.shape[1] * k]]

    if color_field is None:
        tex_im = ax.imshow(texture.T, origin="lower", extent=tex_extent, cmap=texture_cmap,
                           alpha=1.0, aspect=aspect, interpolation="bilinear")
        ax.set_xlim(ex[0], ex[-1])
        ax.set_ylim(ey[0], ey[-1])
        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)
        format_axes(ax)
        return None, tex_im

    extent = list(color_extent) if color_extent is not None else [ex[0], ex[-1], ey[0], ey[-1]]
    if len(extent) != 4:
        raise ValueError("color_extent must be [left, right, bottom, top]")
    mappable, rgba, blend_extent = lic_blend(color_field, extent, texture, tex_extent, alpha=alpha, cmap=cmap,
                                             vmin=vmin, vmax=vmax, log=log, symmetric=symmetric,
                                             norm=imshow_kw.pop("norm", None), texture_cmap=texture_cmap)
    image = ax.imshow(rgba, origin="lower", extent=blend_extent, aspect=aspect, **imshow_kw)
    image.lic_alpha = float(alpha)
    ax.set_xlim(ex[0], ex[-1])
    ax.set_ylim(ey[0], ey[-1])
    if xlabel is not None:
        ax.set_xlabel(xlabel)
    if ylabel is not None:
        ax.set_ylabel(ylabel)
    if title is not None:
        ax.set_title(title)
    format_axes(ax)
    if cbar:
        add_colorbar(ax, mappable, location=cbar, label=cbar_label)
    return mappable, image


def _sample_nearest(values: np.ndarray, extent: Sequence[float], xc: np.ndarray, yc: np.ndarray) -> np.ndarray:
    """``values`` (indexed ``(x, y)`` over ``extent``) at the points ``xc`` x ``yc``; NaN outside."""
    nx, ny = values.shape[:2]
    x0, x1, y0, y1 = (float(e) for e in extent)
    i = np.floor((xc - x0) / (x1 - x0) * nx).astype(np.int64)
    j = np.floor((yc - y0) / (y1 - y0) * ny).astype(np.int64)
    inside_i, inside_j = (i >= 0) & (i < nx), (j >= 0) & (j < ny)
    out = np.full((xc.size, yc.size), np.nan)
    sub = np.asarray(values, dtype=np.float64)[np.ix_(i[inside_i], j[inside_j])]
    out[np.ix_(inside_i, inside_j)] = sub
    return out


def lic_blend(color_field: np.ndarray, color_extent: Sequence[float], texture: np.ndarray,
              texture_extent: Sequence[float], *, alpha: float = 0.35, cmap: Any = None, vmin: float | None = None,
              vmax: float | None = None, log: bool = False, symmetric: bool = False, norm=None,
              texture_cmap: Any = "gray"):
    """Composite a LIC texture over a colour-mapped field into a single RGBA image.

    Where the colour field is opaque, ``rgb = (1 - alpha) cmap(norm(field)) + alpha
    texture_cmap(texture)``, with the texture normalised over its finite range.  One image is
    drawn instead of two stacked ones because matplotlib's PDF and SVG backends merge stacked
    images and apply the texture alpha twice.

    Parameters
    ----------
    color_field : ndarray, shape (nx, ny)
        Scalar indexed (x, y) covering ``color_extent``.
    color_extent, texture_extent : [left, right, bottom, top]
        Cell-face extents of the two arrays, which may differ in extent and resolution.
    texture : ndarray, shape (mx, my)
        Texture indexed (x, y).
    cmap, vmin, vmax, log, symmetric, norm
        Colour mapping of ``color_field`` as in :func:`shearpic.plotting.fields.plot_field`.

    Returns
    -------
    mappable : ScalarMappable
        Norm, colormap and data of the colour field, for colorbars.
    rgba : ndarray, shape (ny', nx', 4)
        Rows along y, on the shared grid or else nearest-neighbour samples over the union of
        both extents at the finer resolution; pixels covered by neither array are transparent.
    extent : list
        ``[left, right, bottom, top]`` of ``rgba``.
    """
    import matplotlib as mpl
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    from .fields import _color_scaling

    if not 0.0 <= float(alpha) <= 1.0:
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    field = _as_2d(color_field, True)
    tex = np.asarray(texture, dtype=np.float64)
    if tex.ndim != 2:
        raise ValueError(f"texture must be 2D, got shape {tex.shape}")
    if norm is None:
        data, norm = _color_scaling(field, log=log, symmetric=symmetric, vmin=vmin, vmax=vmax)
    else:
        data = field
    if cmap is None:
        cmap = "RdBu_r" if symmetric else "viridis"
    cmap = mpl.colormaps[cmap] if isinstance(cmap, str) else cmap
    data = np.ma.masked_invalid(np.ma.asarray(data, dtype=np.float64))   # as imshow does
    norm.autoscale_None(data)
    mappable = ScalarMappable(norm=norm, cmap=cmap)
    mappable.set_array(data)

    ce = [float(v) for v in color_extent]
    te = [float(v) for v in texture_extent]
    same_grid = field.shape == tex.shape and np.allclose(ce, te, rtol=1e-9, atol=1e-12 * max(map(abs, ce + [1.0])))
    if same_grid:
        extent = ce
        col = data
        t = tex
    else:
        extent = [min(ce[0], te[0]), max(ce[1], te[1]), min(ce[2], te[2]), max(ce[3], te[3])]
        W, H = extent[1] - extent[0], extent[3] - extent[2]
        n_x = int(np.ceil(max(field.shape[0] * W / (ce[1] - ce[0]), tex.shape[0] * W / (te[1] - te[0])) - 1e-6))
        n_y = int(np.ceil(max(field.shape[1] * H / (ce[3] - ce[2]), tex.shape[1] * H / (te[3] - te[2])) - 1e-6))
        xc = extent[0] + (np.arange(n_x) + 0.5) * W / n_x
        yc = extent[2] + (np.arange(n_y) + 0.5) * H / n_y
        col = np.ma.masked_invalid(_sample_nearest(np.ma.filled(data, np.nan), ce, xc, yc))
        t = _sample_nearest(tex, te, xc, yc)
    finite = np.isfinite(tex)
    lo, hi = (float(tex[finite].min()), float(tex[finite].max())) if finite.any() else (0.0, 1.0)
    tex_cmap = mpl.colormaps[texture_cmap] if isinstance(texture_cmap, str) else texture_cmap
    tex_rgba = tex_cmap(Normalize(lo, hi)(np.ma.masked_invalid(t))).astype(np.float32)
    del t
    col_rgba = np.asarray(mappable.to_rgba(col), dtype=np.float32)   # masked pixels get the 'bad' colour
    a_t = np.float32(alpha) * tex_rgba[..., 3]
    a_c = col_rgba[..., 3]
    a_o = a_t + a_c * (1.0 - a_t)
    rgb = tex_rgba[..., :3] * a_t[..., None] + col_rgba[..., :3] * (a_c * (1.0 - a_t))[..., None]
    with np.errstate(invalid="ignore", divide="ignore"):
        rgb = np.where(a_o[..., None] > 0, rgb / np.where(a_o > 0, a_o, 1.0)[..., None], 0.0)
    rgba = np.concatenate([np.clip(rgb, 0.0, 1.0), a_o[..., None]], axis=-1).astype(np.float32, copy=False)
    return mappable, np.ascontiguousarray(rgba.transpose(1, 0, 2)), extent
