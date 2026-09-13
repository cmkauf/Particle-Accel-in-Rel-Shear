"""Periodic boundaries, noise input and relief shading of shearpic.plotting.lic."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from shearpic.plotting import lic  # noqa: E402

pytest.importorskip("rlic")


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def periodic_field(nx=48, ny=64, seed=1):
    """A smooth, doubly periodic in-plane field indexed (x, y), with long streamlines."""
    rng = np.random.default_rng(seed)
    X, Y = np.meshgrid(2 * np.pi * np.arange(nx) / nx, 2 * np.pi * np.arange(ny) / ny, indexing="ij")
    p = rng.uniform(0, 2 * np.pi, 4)
    vx = np.cos(Y + p[0]) + 0.4 * np.sin(2 * X + p[1]) + 0.3
    vy = np.sin(X + p[2]) + 0.3 * np.cos(3 * Y + p[3])
    return vx, vy


@pytest.mark.parametrize("mode", ["velocity", "polarization"])
@pytest.mark.parametrize("kernel_length", [15, 16])
def test_texture_is_wrap_invariant(mode, kernel_length):
    vx, vy = periodic_field()
    noise = np.random.default_rng(7).random(vx.shape).astype(np.float32)
    kw = dict(kernel_length=kernel_length, niter=4, mode=mode, equalize=False)
    base = lic.lic_texture(vx, vy, noise=noise, **kw)
    for shift in [(17, 0), (0, 29), (-11, 40)]:
        rolled = lic.lic_texture(np.roll(vx, shift, (0, 1)), np.roll(vy, shift, (0, 1)),
                                 noise=np.roll(noise, shift, (0, 1)), **kw)
        np.testing.assert_array_equal(rolled, np.roll(base, shift, (0, 1)))
    # closed boundaries are not wrap invariant (the test is sensitive)
    closed = lic.lic_texture(vx, vy, noise=noise, periodic=False, **kw)
    rolled = lic.lic_texture(np.roll(vx, (17, 29), (0, 1)), np.roll(vy, (17, 29), (0, 1)),
                             noise=np.roll(noise, (17, 29), (0, 1)), periodic=False, **kw)
    assert np.abs(rolled - np.roll(closed, (17, 29), (0, 1))).max() > 1e-3


def test_wrap_invariance_with_downsampling_and_one_periodic_axis():
    vx, vy = periodic_field(64, 96)
    noise = np.random.default_rng(3).random((32, 48)).astype(np.float32)
    kw = dict(downsample=2, kernel_length=12, niter=3, mode="polarization", equalize=False)
    base = lic.lic_texture(vx, vy, noise=noise, **kw)
    # a shift by 2k cells shifts the block-mean grid by k texels
    rolled = lic.lic_texture(np.roll(vx, 20, 0), np.roll(vy, 20, 0), noise=np.roll(noise, 10, 0), **kw)
    np.testing.assert_array_equal(rolled, np.roll(base, 10, 0))
    # periodic only in x: x shifts commute, y shifts do not
    kw["periodic"] = (True, False)
    base = lic.lic_texture(vx, vy, noise=noise, **kw)
    np.testing.assert_array_equal(
        lic.lic_texture(np.roll(vx, 20, 0), np.roll(vy, 20, 0), noise=np.roll(noise, 10, 0), **kw),
        np.roll(base, 10, 0))
    shifted_y = lic.lic_texture(np.roll(vx, 30, 1), np.roll(vy, 30, 1), noise=np.roll(noise, 15, 1), **kw)
    assert np.abs(shifted_y - np.roll(base, 15, 1)).max() > 1e-3


def _edge_vs_interior(tex):
    """Mean |jump| across the x boundary (last -> first column) and between interior neighbours."""
    edge = np.abs(tex[0] - tex[-1]).mean()
    interior = np.abs(np.diff(tex[8:-8], axis=0)).mean()
    return edge, interior


def test_texture_continuous_across_the_x_boundary():
    # uniform field along x: streaks are horizontal, so neighbouring columns are strongly correlated
    nx, ny = 96, 64
    vx, vy = np.ones((nx, ny)), 0.05 * np.sin(2 * np.pi * np.arange(ny) / ny)[None, :] * np.ones((nx, 1))
    tex = lic.lic_texture(vx, vy, kernel_length=24, niter=5, mode="velocity", equalize=False)
    edge, interior = _edge_vs_interior(tex)
    assert edge < 1.5 * interior
    # the columns next to the boundary are smoothed like interior ones (same variance along y)
    std = tex.std(axis=1)
    assert np.isclose(std[:3].mean(), std[40:56].mean(), rtol=0.25)
    assert np.isclose(std[-3:].mean(), std[40:56].mean(), rtol=0.25)

    closed = lic.lic_texture(vx, vy, kernel_length=24, niter=5, mode="velocity", equalize=False, periodic=False)
    edge_c, interior_c = _edge_vs_interior(closed)
    assert edge_c > 3 * interior_c                      # a visible seam with rlic's closed boundaries
    assert closed.std(axis=1)[0] > 1.3 * closed.std(axis=1)[40:56].mean()   # truncated streaks at the edge


def test_noise_and_periodic_validation():
    vx, vy = periodic_field(16, 16)
    with pytest.raises(ValueError, match="noise has shape"):
        lic.lic_texture(vx, vy, noise=np.zeros((4, 4)), kernel_length=4, niter=1)
    with pytest.raises(ValueError, match="periodic must be"):
        lic.lic_texture(vx, vy, periodic=(True, False, True), kernel_length=4, niter=1)
    # a kernel longer than the image still wraps consistently
    tex = lic.lic_texture(vx, vy, kernel_length=40, niter=2)
    assert tex.shape == (16, 16) and np.isfinite(tex).all()


def test_same_seed_same_texture_and_seed_matters():
    vx, vy = periodic_field(32, 32)
    a = lic.lic_texture(vx, vy, kernel_length=8, niter=2, seed=5)
    b = lic.lic_texture(vx, vy, kernel_length=8, niter=2, seed=5)
    c = lic.lic_texture(vx, vy, kernel_length=8, niter=2, seed=6)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_shade_texture_periodic_and_range():
    rng = np.random.default_rng(0)
    tex = rng.random((40, 60))
    shaded = lic.shade_texture(tex)
    assert shaded.shape == tex.shape and shaded.dtype == np.float32
    assert 0.0 <= shaded.min() and shaded.max() <= 1.0
    # periodic shading commutes with rolls (no seam at the edges)
    np.testing.assert_allclose(lic.shade_texture(np.roll(tex, (13, 21), (0, 1))), np.roll(shaded, (13, 21), (0, 1)),
                               atol=1e-6)
    open_shade = lic.shade_texture(tex, periodic=False)
    assert np.abs(lic.shade_texture(np.roll(tex, 13, 0), periodic=False) - np.roll(open_shade, 13, 0)).max() > 1e-3
    # without wrapping: exactly matplotlib's hillshade of the image with rows = increasing y
    from matplotlib.colors import LightSource

    ref = LightSource(azdeg=0, altdeg=45).hillshade((tex / tex.max()).T, vert_exag=5).T
    np.testing.assert_allclose(open_shade, ref, atol=1e-6)
    # periodic along x only: x rolls commute
    np.testing.assert_allclose(lic.shade_texture(np.roll(tex, 7, 0), periodic=(True, False)),
                               np.roll(lic.shade_texture(tex, periodic=(True, False)), 7, 0), atol=1e-6)
    # NaNs are shaded as zero-height pixels, not propagated
    tex[3, 4] = np.nan
    assert np.isfinite(lic.shade_texture(tex)).all()


def test_lic_texture_shade_matches_separate_step():
    vx, vy = periodic_field(32, 48)
    plain = lic.lic_texture(vx, vy, kernel_length=8, niter=2)
    np.testing.assert_allclose(lic.lic_texture(vx, vy, kernel_length=8, niter=2, shade=True),
                               lic.shade_texture(plain), atol=1e-6)


def test_plot_lic_precomputed_texture_without_vectors():
    nx, ny = 20, 80
    x = np.linspace(-5, 5, nx + 1)            # faces of the texture grid
    y = np.linspace(-20, 20, ny + 1)
    tex = np.random.default_rng(1).random((nx, ny)).astype(np.float32)
    color = np.linspace(0, 1, 10 * 40).reshape(10, 40)   # coarser colour field, explicit extent
    fig, ax = plt.subplots()
    mappable, image = lic.plot_lic(ax, x, y, color_field=color, texture=tex, color_extent=[-5, 4, -20, 18],
                                   cbar=None, alpha=0.2, shade=True, xlabel=None, ylabel=None, cmap="magma")
    assert len(ax.images) == 1 and image.lic_alpha == 0.2 and ax.get_xlabel() == ""
    assert mappable.get_clim() == (0.0, 1.0) and mappable.get_cmap().name == "magma"
    # union of the two extents at the finer (texture) resolution
    np.testing.assert_allclose(image.get_extent(), [-5, 5, -20, 20])
    rgba = np.asarray(image.get_array())
    assert rgba.shape == (ny, nx, 4)
    shaded = lic.shade_texture(tex)
    gray = plt.get_cmap("gray")((shaded - shaded.min()) / (shaded.max() - shaded.min()))[..., 0]
    xc, yc = -5 + (np.arange(nx) + 0.5) * 0.5, -20 + (np.arange(ny) + 0.5) * 0.5
    i, j = 3, 11                              # inside both: (1 - alpha) colour + alpha grey
    ci, cj = int((xc[i] + 5) / 0.9), int((yc[j] + 20) / 0.95)
    want = 0.8 * np.asarray(plt.get_cmap("magma")(color[ci, cj]))[:3] + 0.2 * gray[i, j]
    np.testing.assert_allclose(rgba[j, i, :3], want, atol=1e-6)
    assert rgba[j, i, 3] == pytest.approx(1.0)
    # outside the colour field (x > 4): only the texture, with opacity alpha over the background
    np.testing.assert_allclose(rgba[5, nx - 1, :3], gray[nx - 1, 5], atol=1e-6)
    assert rgba[5, nx - 1, 3] == pytest.approx(0.2)
    with pytest.raises(ValueError, match="needs vx and vy"):
        lic.plot_lic(ax, x, y)


def test_lic_blend_same_grid_is_exact_and_masks_bad_pixels():
    field = np.linspace(0.5, 1.5, 6 * 8).reshape(6, 8)
    field[2, 3] = np.nan
    tex = np.random.default_rng(3).random((6, 8))
    tex[4, 5] = np.nan
    extent = [0.0, 3.0, -2.0, 2.0]
    m, rgba, ext = lic.lic_blend(field, extent, tex, extent, alpha=0.1, cmap="viridis", vmin=0.5, vmax=1.5)
    assert ext == extent and rgba.shape == (8, 6, 4) and m.get_clim() == (0.5, 1.5)
    g = plt.get_cmap("gray")(np.ma.masked_invalid((tex - np.nanmin(tex)) / (np.nanmax(tex) - np.nanmin(tex))))[..., 0]
    want = 0.9 * plt.get_cmap("viridis")((field - 0.5) / 1.0)[..., :3] + 0.1 * g[..., None]
    ok = np.isfinite(field) & np.isfinite(tex)
    np.testing.assert_allclose(rgba.transpose(1, 0, 2)[ok][:, :3], want[ok], atol=1e-6)
    # NaN colour: transparent under the texture; NaN texture: pure colour
    assert rgba[3, 2, 3] == pytest.approx(0.1) and rgba[3, 2, 0] == pytest.approx(g[2, 3], abs=1e-6)
    np.testing.assert_allclose(rgba[5, 4, :3], plt.get_cmap("viridis")((field[4, 5] - 0.5))[:3], atol=1e-6)
    # autoscaled limits ignore the NaN, as imshow does
    assert lic.lic_blend(field, extent, tex, extent)[0].get_clim() == (0.5, 1.5)
    with pytest.raises(ValueError, match="alpha"):
        lic.lic_blend(field, extent, tex, extent, alpha=1.5)


def _bulk_row_contrast(gray: np.ndarray) -> float:
    """Std over the middle 30% of rows of the row-mean luminance (streaks run along x)."""
    h = gray.shape[0]
    return float(gray[int(0.35 * h):int(0.65 * h)].mean(axis=1).std())


def test_lic_texture_survives_pdf_output(tmp_path):
    """The PDF keeps the PNG's texture contrast, i.e. the texture alpha is not applied twice."""
    fitz = pytest.importorskip("fitz")
    from PIL import Image

    nx, ny = 64, 256
    x, y = np.linspace(-4, 4, nx + 1), np.linspace(-16, 16, ny + 1)
    tex = np.tile(((np.arange(ny) // 16) % 2).astype(np.float32), (nx, 1))  # horizontal stripes
    color = np.full((nx, ny), 0.5)
    fig = plt.figure(figsize=(2.0, 8.0))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    lic.plot_lic(ax, x, y, color_field=color, texture=tex, cmap="RdYlBu_r", vmin=0.0, vmax=1.0, alpha=0.1,
                 cbar=None, xlabel=None, ylabel=None, aspect="auto")
    ax.set_axis_off()
    fig.savefig(tmp_path / "panel.pdf", dpi=50)
    fig.savefig(tmp_path / "panel.png", dpi=50)
    plt.close(fig)

    png = np.asarray(Image.open(tmp_path / "panel.png").convert("L"), dtype=float)
    page = fitz.open(tmp_path / "panel.pdf")[0]
    pix = page.get_pixmap(dpi=50, colorspace=fitz.csGRAY)
    pdf = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(float)
    c_png, c_pdf = _bulk_row_contrast(png), _bulk_row_contrast(pdf)
    # stripes 0/1 at alpha 0.1: luminance steps of 0.1 * 255, i.e. a row std of ~12.7
    assert 10.0 < c_png < 15.0
    assert 0.8 * c_png < c_pdf < 1.25 * c_png
    # the PDF holds a single image for the panel
    assert len(page.get_images()) == 1
