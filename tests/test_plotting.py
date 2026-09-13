"""Tests for shearpic.plotting (synthetic data) plus one rendering test on run423."""

from __future__ import annotations

import subprocess
import sys
import textwrap
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib as mpl  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
from matplotlib.colors import LogNorm, same_color  # noqa: E402

from shearpic.config import RunConfig  # noqa: E402
from shearpic.io.trajectory import Trajectory  # noqa: E402
from shearpic.plotting import animation as anim_mod  # noqa: E402
from shearpic.plotting import fields, lic, lines, style  # noqa: E402

ATHINPUT = """\
<job>
problem_id = synth
<output1>
file_type = hst
dt = 0.5
<time>
tlim = 10.0
<mesh>
nx1 = 16
x1min = -5.0
x1max = 5.0
nx2 = 64
x2min = -20.0
x2max = 20.0
nx3 = 1
x3min = -0.5
x3max = 0.5
<meshblock>
nx1 = 16
nx2 = 64
nx3 = 1
<hydro>
iso_sound_speed = 5.0
<particles>
speed_of_light = 50.0
charge_over_mass_over_c = 200.0
<problem>
iprob = 0
shear_strength = 1.0
y1 = -10.0
y2 = 10.0
M_A = 10
tau = 0.5
vp_par = 50.0
cr_mass = 0.001
"""


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


@pytest.fixture
def cfg(tmp_path):
    run_dir = tmp_path / "run0001"
    run_dir.mkdir()
    path = run_dir / "athinput.synth"
    path.write_text(ATHINPUT)
    return RunConfig.from_athinput(path)


def _span_limits(patch, axis: int) -> tuple[float, float]:
    """Data-coordinate range of an axvspan/axhspan patch along ``axis`` (works for Polygon and Rectangle)."""
    verts = patch.get_patch_transform().transform(patch.get_path().vertices)
    return float(verts[:, axis].min()), float(verts[:, axis].max())


# ============================================================== style
def test_import_has_no_side_effects():
    code = textwrap.dedent("""
        import sys, matplotlib as mpl
        before = dict(mpl.rcParams)
        import shearpic.plotting
        import shearpic.plotting.style, shearpic.plotting.fields, shearpic.plotting.lines
        import shearpic.plotting.lic, shearpic.plotting.animation
        assert 'matplotlib.pyplot' not in sys.modules, 'pyplot imported'
        assert dict(mpl.rcParams) == before, 'rcParams changed'
        assert 'shearpic_bfield' not in mpl.colormaps, 'colormap registered at import'
    """)
    subprocess.run([sys.executable, "-c", code], check=True)


def test_paper_style_restores_rcparams():
    before = dict(mpl.rcParams)
    with style.paper_style(rc={"lines.linewidth": 3.0}):
        assert mpl.rcParams["xtick.direction"] == "in"
        assert mpl.rcParams["ytick.right"] is True
        assert mpl.rcParams["font.family"] == ["serif"]
        assert mpl.rcParams["font.serif"][:2] == ["Times New Roman", "STIX Two Text"]
        assert mpl.rcParams["mathtext.fontset"] == "stix"
        assert mpl.rcParams["lines.linewidth"] == 3.0
        fig, ax = plt.subplots()
        ax.plot([1, 2], [1, 2])
        fig.canvas.draw()  # fonts resolve (with fallbacks) without errors
    assert dict(mpl.rcParams) == before


def test_use_paper_style_global():
    with mpl.rc_context():
        style.use_paper_style(rc={"figure.dpi": 72})
        assert mpl.rcParams["xtick.top"] is True
        assert mpl.rcParams["figure.dpi"] == 72


def test_format_axes_and_colorbar():
    fig, axs = plt.subplots(1, 2)
    style.format_axes(axs)
    for ax in axs:
        assert ax.xaxis.get_tick_params(which="major")["direction"] == "in"
        assert ax.xaxis.get_major_ticks()[0].tick2line.get_visible()  # ticks on top
        assert ax.yaxis.get_major_ticks()[0].tick2line.get_visible()  # ticks on the right
    im = axs[0].imshow(np.arange(12.0).reshape(3, 4))
    cb = style.add_colorbar(axs[0], im, location="top", label="rho")
    assert cb.orientation == "horizontal" and im.colorbar is cb
    assert cb.ax.xaxis.get_label_position() == "top"
    cb2 = style.add_colorbar(axs[1], axs[1].imshow(np.ones((2, 2))), location="right")
    assert cb2.orientation == "vertical"
    with pytest.raises(ValueError):
        style.add_colorbar(axs[1], im, location="middle")


def test_register_colormaps_only_on_call():
    maps = style.register_colormaps()
    try:
        assert set(maps) == {"shearpic_bfield", "shearpic_bfield_r"}
        assert "shearpic_bfield" in mpl.colormaps
        style.register_colormaps()  # idempotent
        assert same_color(mpl.colormaps["shearpic_bfield"](0.0), style.BFIELD_COLORS[0])
        assert same_color(mpl.colormaps["shearpic_bfield"](1.0), style.BFIELD_COLORS[-1])
    finally:
        for name in maps:
            mpl.colormaps.unregister(name)


# ============================================================== fields
def test_cell_edges_and_extent_exact(cfg):
    x, y = cfg.centers(0), cfg.centers(1)
    np.testing.assert_allclose(fields.cell_edges(x, cfg.nx[0]), cfg.edges(0), rtol=0, atol=1e-12)
    np.testing.assert_array_equal(fields.cell_edges(cfg.edges(1), cfg.nx[1]), cfg.edges(1))
    ext = fields.image_extent((16, 64), x, y)
    np.testing.assert_allclose(ext, [-5.0, 5.0, -20.0, 20.0], rtol=0, atol=1e-12)
    assert fields.image_extent((4, 3)) == [-0.5, 3.5, -0.5, 2.5]
    with pytest.raises(ValueError, match="uniform"):
        fields.cell_edges([0.0, 1.0, 3.0], 3)
    with pytest.raises(ValueError, match="expected"):
        fields.cell_edges(np.arange(5.0), 3)
    with pytest.raises(ValueError, match="single"):
        fields.cell_edges([0.5], 1)


def test_plot_field_extent_and_orientation(cfg):
    x, y = cfg.centers(0), cfg.centers(1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    field = (X + 100.0 * Y)[:, :, None]  # (nx, ny, 1) like Snapshot.read
    fig, ax = plt.subplots()
    im = fields.plot_field(ax, field, x, y, cbar="right", cbar_label="f", title="T")
    np.testing.assert_allclose(im.get_extent(), [-5.0, 5.0, -20.0, 20.0], rtol=0, atol=1e-12)
    arr = np.asarray(im.get_array())
    assert arr.shape == (64, 16)  # image rows = y
    np.testing.assert_allclose(arr[0, :], field[:, 0, 0])  # bottom row = lowest y, x left->right
    np.testing.assert_allclose(arr[:, 0], field[0, :, 0])
    assert im.origin == "lower" and im.colorbar is not None
    assert ax.get_xlabel() == "$x/a$" and ax.get_ylabel() == "$y/a$" and ax.get_title() == "T"
    # image already indexed (y, x)
    im2 = fields.plot_field(plt.subplots()[1], field[:, :, 0].T, x, y, transpose=False, cbar=None)
    np.testing.assert_allclose(np.asarray(im2.get_array()), arr)
    np.testing.assert_allclose(im2.get_extent(), im.get_extent())
    # explicit extent, no coordinates
    im3 = fields.plot_field(plt.subplots()[1], field, extent=[0, 1, 0, 4], cbar=None)
    assert list(im3.get_extent()) == [0, 1, 0, 4]
    with pytest.raises(ValueError, match="2D slice"):
        fields.plot_field(ax, np.zeros((4, 4, 2)))


def test_plot_field_symmetric():
    f = np.array([[-1.0, 0.5], [3.0, -2.0]])
    ax = plt.subplots()[1]
    im = fields.plot_field(ax, f, symmetric=True, cbar=None)
    assert im.get_clim() == (-3.0, 3.0)
    assert im.get_cmap().name == "RdBu_r"
    im = fields.plot_field(ax, f, symmetric=True, vmax=1.5, cmap="PuOr", cbar=None)
    assert im.get_clim() == (-1.5, 1.5) and im.get_cmap().name == "PuOr"
    assert fields.plot_field(ax, f, cbar=None).get_cmap().name == "viridis"
    with pytest.raises(ValueError):
        fields.plot_field(ax, f, symmetric=True, log=True)


def test_plot_field_log_masks_nonpositive():
    f = np.array([[1.0, 10.0], [0.0, -1.0], [100.0, 5.0]])
    ax = plt.subplots()[1]
    with pytest.warns(UserWarning, match="2 non-positive"):
        im = fields.plot_field(ax, f, log=True, vmin=1.0, cbar="top")
    assert isinstance(im.norm, LogNorm)
    assert int(np.ma.getmaskarray(im.get_array()).sum()) == 2
    with pytest.raises(ValueError, match="no positive"):
        fields.plot_field(ax, -np.ones((2, 2)), log=True)


def test_plot_snapshot_field(cfg):
    arrays = {"rho": np.ones((16, 64, 1)), "big": np.ones((16, 64, 3))}
    im = fields.plot_snapshot_field(plt.subplots()[1], arrays, "rho", cfg)
    np.testing.assert_allclose(im.get_extent(), [-5, 5, -20, 20], atol=1e-12)
    assert im.colorbar.ax.get_xlabel() == "rho"
    with pytest.raises(ValueError, match="z_index"):
        fields.plot_snapshot_field(plt.subplots()[1], arrays, "big", cfg)
    fields.plot_snapshot_field(plt.subplots()[1], arrays, "big", cfg, z_index=1, cbar=None)

    class FakeSnapshot:  # duck-typed stand-in for shearpic.io.athdf.Snapshot
        time = 300.0

        def read(self, names):
            return {names: np.full((16, 64, 1), 2.0)}

        def centers(self, axis, bounds=None):
            return cfg.centers(axis)

    ax = plt.subplots()[1]
    im = fields.plot_snapshot_field(ax, FakeSnapshot(), "vel1", cbar=None)
    assert "300.0" in ax.get_title()
    np.testing.assert_allclose(im.get_extent(), [-5, 5, -20, 20], atol=1e-12)


def test_plot_profile_shading(cfg):
    y = cfg.centers(1)
    half = 0.5 * cfg.profile.layer_width(0.9)
    ax = plt.subplots()[1]
    field = np.broadcast_to(cfg.profile.U(y)[None, :, None], (16, 64, 1))
    line = fields.plot_profile(ax, y, field, cfg, label="<vx>", show_reference=True)
    np.testing.assert_allclose(line.get_xdata(), y)
    np.testing.assert_allclose(line.get_ydata(), cfg.profile.U(y))
    assert len(ax.patches) == 2
    spans = sorted(_span_limits(p, 0) for p in ax.patches)
    np.testing.assert_allclose(spans, [(-10 - half, -10 + half), (10 - half, 10 + half)], atol=1e-12)
    assert len(ax.lines) == 2 and ax.get_xlim() == (-20.0, 20.0)
    # vertical layout: y on the vertical axis
    ax2 = plt.subplots()[1]
    line2 = fields.plot_profile(ax2, y, cfg.profile.U(y), cfg, vertical=True, xlabel=r"$U$")
    np.testing.assert_allclose(line2.get_ydata(), y)
    spans2 = sorted(_span_limits(p, 1) for p in ax2.patches)
    np.testing.assert_allclose(spans2, [(-10 - half, -10 + half), (10 - half, 10 + half)], atol=1e-12)
    assert ax2.get_ylabel() == "$y/a$"
    with pytest.raises(ValueError):
        fields.plot_profile(ax2, y, np.zeros(3), cfg)


# =============================================================== lines
def test_plot_timeseries_style():
    ax = plt.subplots()[1]
    entry = {"run": 423, "label": "fiducial", "color": "tab:red", "linestyle": "--"}
    t = np.linspace(0, 10, 11)
    line = lines.plot_timeseries(ax, t, t**2, style=entry)
    assert line.get_label() == "fiducial" and line.get_linestyle() == "--"
    assert same_color(line.get_color(), "tab:red")
    line = lines.plot_timeseries(ax, t, t, style=entry, color="k", label="override")
    assert same_color(line.get_color(), "k") and line.get_label() == "override" and line.get_linestyle() == "--"
    assert lines.plot_timeseries(ax, t, t).get_linestyle() == "-"
    with pytest.raises(ValueError):
        lines.plot_timeseries(ax, t, t[:-1])


class StubSpectrum:
    """Minimal duck-typed Spectrum: log bins with counts."""

    def __init__(self, variable="gamma_minus_1"):
        self.variable = variable
        self.edges = np.logspace(-2, 2, 41)
        self.counts = np.arange(40, dtype=float)  # first bin empty
        self.counts[10] = 0.0

    def centers(self, kind="geometric"):
        assert kind == "geometric"
        return np.sqrt(self.edges[:-1] * self.edges[1:])

    def dN_dx(self, normalize="total"):
        d = self.counts / np.diff(self.edges)
        return d / self.counts.sum() if normalize == "total" else d


def test_plot_spectrum():
    spec = StubSpectrum()
    ax = plt.subplots()[1]
    line = lines.plot_spectrum(ax, spec, label="t=0")
    expected = spec.dN_dx("total")
    y = line.get_ydata()
    assert np.isnan(y[0]) and np.isnan(y[10])
    ok = expected > 0
    np.testing.assert_allclose(y[ok], expected[ok])
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    assert ax.get_xlabel() == r"$\gamma-1$"
    assert ax.get_ylabel() == r"$N^{-1}\,dN/d(\gamma-1)$"

    ax2 = plt.subplots()[1]
    line2 = lines.plot_spectrum(ax2, spec, normalize=None, compensate=2)
    x = spec.centers()
    np.testing.assert_allclose(line2.get_ydata()[ok], (spec.dN_dx(None) * x**2)[ok])
    assert ax2.get_ylabel() == r"$(\gamma-1)^{2}\,dN/d(\gamma-1)$"

    for variable, label in (("gamma", r"$\gamma$"), ("p_over_mc", r"$p/(mc)$")):
        ax3 = plt.subplots()[1]
        lines.plot_spectrum(ax3, StubSpectrum(variable))
        assert ax3.get_xlabel() == label


def test_plot_spectrum_with_zero_first_edge():
    """A gamma histogram starting at gamma = 1 becomes a gamma - 1 histogram whose first edge is 0."""
    from shearpic.physics.spectra import histogram

    rng = np.random.default_rng(3)
    gamma = 1.0 + rng.lognormal(0.0, 1.0, 5000)
    edges = np.concatenate([[1.0], 1.0 + np.logspace(-2, 2, 30)])
    spec = histogram(gamma, edges, "gamma").to("gamma_minus_1")
    assert spec.edges[0] == 0.0
    ax = plt.subplots()[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        line = lines.plot_spectrum(ax, spec)
    x = line.get_xdata()
    e = spec.edges
    assert x[0] == pytest.approx(0.5 * e[1])  # arithmetic centre of [0, e1]
    np.testing.assert_allclose(x[1:], np.sqrt(e[1:-1] * e[2:]))  # geometric elsewhere
    assert np.all(np.isfinite(x)) and np.all(x > 0)
    np.testing.assert_allclose(line.get_ydata()[spec.counts > 0], spec.dN_dx("total")[spec.counts > 0])
    # compensated form stays finite too
    line2 = lines.plot_spectrum(plt.subplots()[1], spec, compensate=1.0)
    ok = spec.counts > 0
    np.testing.assert_allclose(line2.get_ydata()[ok], (spec.dN_dx("total") * x)[ok])


def test_plot_phase_space_uniform_and_log_edges():
    rng = np.random.default_rng(1)
    px, yy = rng.normal(0, 50, 5000), rng.uniform(-20, 20, 5000)
    xe, ye = np.linspace(-150, 150, 31), np.linspace(-20, 20, 41)
    H, _, _ = np.histogram2d(px, yy, bins=[xe, ye])
    ax = plt.subplots()[1]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # empty bins must not warn
        im = lines.plot_phase_space(ax, H, xe, ye, xlabel="$u_x$", ylabel="$y/a$", cbar_label="N")
    assert list(im.get_extent()) == [-150, 150, -20, 20]
    assert isinstance(im.norm, LogNorm) and im.colorbar is not None
    np.testing.assert_array_equal(np.ma.getdata(im.get_array()), H.T)

    ge = np.logspace(0, 2, 21)
    H2, _, _ = np.histogram2d(rng.uniform(1, 100, 1000), rng.uniform(-20, 20, 1000), bins=[ge, ye])
    ax2 = plt.subplots()[1]
    mesh = lines.plot_phase_space(ax2, H2, ge, ye, log=False, cbar=None)
    assert ax2.get_xscale() == "log" and ax2.get_xlim() == (1.0, 100.0)
    assert mesh.get_array().size == H2.size
    with pytest.raises(ValueError, match="expected"):
        lines.plot_phase_space(ax2, H2.T, ge, ye)


# ================================================================= lic
def test_block_mean_and_equalize():
    a = np.arange(7 * 5, dtype=float).reshape(7, 5)
    b = lic.block_mean(a, 2)
    assert b.shape == (3, 2)
    assert b[0, 0] == a[:2, :2].mean() and b[2, 1] == a[4:6, 2:4].mean()
    assert lic.block_mean(a, 1) is a
    with pytest.raises(ValueError):
        lic.block_mean(a, 8)
    e = lic.equalize_histogram(np.random.default_rng(0).normal(size=(50, 50)))
    assert 0.0 <= np.nanmin(e) and np.nanmax(e) == pytest.approx(1.0)
    assert np.all(np.isnan(lic.equalize_histogram(np.full((2, 2), np.nan))))


def test_lic_import_error_hint(monkeypatch):
    monkeypatch.setitem(sys.modules, "rlic", None)
    with pytest.raises(ImportError, match=r"shearpic\[lic\]"):
        lic.lic_texture(np.ones((8, 8)), np.zeros((8, 8)))


def _circular_ratio(texture, X, Y, r_min=8.0, r_max=26.0):
    """Mean |derivative along circles| / mean |radial derivative| of a texture indexed (x, y)."""
    gx, gy = np.gradient(texture.astype(float))
    r = np.hypot(X, Y)
    sel = (r > r_min) & (r < r_max)
    tangential = np.abs(-Y * gx + X * gy)[sel] / r[sel]
    radial = np.abs(X * gx + Y * gy)[sel] / r[sel]
    return tangential.mean() / radial.mean()


def test_lic_orientation():
    pytest.importorskip("rlic")
    # uniform horizontal field on a non-square grid: streaks run along x (axis 0)
    vx, vy = np.ones((48, 32)), np.zeros((48, 32))
    tex = lic.lic_texture(vx, vy, kernel_length=15, niter=3, equalize=False)
    assert tex.shape == (48, 32)
    assert np.abs(np.diff(tex, axis=0)).mean() < 0.3 * np.abs(np.diff(tex, axis=1)).mean()

    # circular field (vx, vy) = (-y, x): streaks follow circles
    c = np.arange(64) - 31.5
    X, Y = np.meshgrid(c, c, indexing="ij")
    tex = lic.lic_texture(-Y, X, kernel_length=21, niter=4)
    assert 0.0 <= tex.min() and tex.max() <= 1.0
    assert _circular_ratio(tex, X, Y) < 0.5
    # the check is sensitive to orientation: the field an x<->y mix-up would produce fails it
    assert _circular_ratio(lic.lic_texture(X, -Y, kernel_length=21, niter=4), X, Y) > 0.8


def test_plot_lic_extents(cfg):
    pytest.importorskip("rlic")
    x, y = cfg.centers(0), cfg.centers(1)
    X, Y = np.meshgrid(x, y, indexing="ij")
    bx, by = np.cos(Y / 3.0), np.sin(X)
    ax = plt.subplots()[1]
    mappable, image = lic.plot_lic(ax, x, y, bx, by, np.hypot(bx, by), downsample=3, niter=2,
                                   kernel_length=8, cbar_label="|B|", alpha=0.35)
    # one blended image on the colour grid (16x64 cells); the texture has 5x21 texels covering 15x63 cells
    assert len(ax.images) == 1
    np.testing.assert_allclose(image.get_extent(), [-5, 5, -20, 20], atol=1e-12)
    rgba = np.asarray(image.get_array())
    assert rgba.shape == (64, 16, 4) and np.allclose(rgba[..., 3], 1.0)
    colour = mappable.to_rgba(np.hypot(bx, by))[..., :3].transpose(1, 0, 2)
    # the last column and row of cells are not covered by the texture: pure colour there
    np.testing.assert_allclose(rgba[:, 15, :3], colour[:, 15], atol=1e-6)
    np.testing.assert_allclose(rgba[63, :, :3], colour[63, :], atol=1e-6)
    # elsewhere 0.65 colour + 0.35 grey: the grey level of texel (0, 0) is recovered in cells 0..2 x 0..2
    grey = (rgba[:3, :3, :3] - 0.65 * colour[:3, :3]) / 0.35
    np.testing.assert_allclose(grey, grey[0, 0, 0], atol=1e-5)
    assert ax.get_xlim() == (-5.0, 5.0) and mappable.colorbar is not None
    assert mappable.colorbar.mappable is mappable
    # texture alone, reusing a precomputed texture
    tex = lic.lic_texture(bx, by, downsample=2, niter=2, kernel_length=8)
    none_im, tex_im2 = lic.plot_lic(plt.subplots()[1], x, y, bx, by, texture=tex)
    assert none_im is None and tex_im2.get_alpha() == 1.0
    with pytest.raises(ValueError, match="texture shape"):
        lic.plot_lic(plt.subplots()[1], x, y, bx, by, texture=np.zeros((5, 5)))


# =========================================================== animation
def test_trail_alpha():
    for n in range(1, 8):
        a = anim_mod.trail_alpha(n, 7)
        assert a.size == n and a[-1] == 1.0
    np.testing.assert_allclose(anim_mod.trail_alpha(7, 7)[:-1], np.linspace(0.05, 0.3, 6))
    np.testing.assert_allclose(anim_mod.trail_alpha(3, 7), [0.25, 0.3, 1.0])
    assert anim_mod.trail_alpha(1, 1).tolist() == [1.0]
    with pytest.raises(ValueError):
        anim_mod.trail_alpha(8, 7)


def _synthetic_trajectories(n_particles=3):
    rng = np.random.default_rng(2)
    trajs = []
    for k in range(n_particles):
        t = np.arange(0.0 + k, 20.0 - k, 0.5)  # different spans -> common window [2, 17.5]
        x = np.c_[rng.uniform(-5, 5, t.size), 15 * np.sin(0.1 * t + k), np.zeros(t.size)]
        u = rng.normal(0, 30 + 10 * k, (t.size, 3))
        trajs.append(Trajectory(t=t, x=x, u=u, pid=k, init_mbid=0))
    return trajs


@pytest.mark.filterwarnings("ignore:Animation was deleted")
def test_animate_trajectories(cfg, monkeypatch, tmp_path):
    trajs = _synthetic_trajectories()
    anim = anim_mod.animate_trajectories(trajs, cfg, t_start=1.0, stride=2, n_history=4)
    ft = anim.frame_times
    assert ft[0] == 2.0 and ft[-1] <= 17.5 and np.all(np.diff(ft) == 1.0)
    fig = anim._fig
    axs = fig.axes
    assert len(axs) == 4
    assert axs[0].get_xlim() == (-5.0, 5.0) and axs[0].get_ylim() == (-20.0, 20.0)
    lo, hi = axs[1].get_xlim()
    assert lo == -hi and hi > 0
    assert len(axs[0].patches) == 2  # two shaded shear layers
    for frame in (0, 1, 2, 5, ft.size - 1):
        anim._func(frame)
        n_expected = min(frame + 1, 4)
        for ax in axs:
            assert len(ax.collections) == len(trajs)
            for coll in ax.collections:
                assert coll.get_offsets().shape == (n_expected, 2)
                assert coll.get_facecolor().shape == (n_expected, 4)
                assert coll.get_facecolor()[-1, 3] == 1.0
                assert coll.get_sizes().size == n_expected
    # sizes follow gamma = sqrt(1 + u^2/c^2); particle 1 in the u_y panel at the last frame
    coll = axs[2].collections[1]
    idx = np.searchsorted(trajs[1].t, ft[-1])
    gamma = np.sqrt(1 + np.sum(trajs[1].u[idx] ** 2) / cfg.c**2)
    assert coll.get_sizes()[-1] == pytest.approx(12.0 * gamma)
    assert coll.get_offsets()[-1, 0] == pytest.approx(trajs[1].u[idx, 1])
    assert "17.0" in fig._suptitle.get_text()

    with pytest.raises(ValueError, match="share no samples"):
        anim_mod.animate_trajectories(trajs, cfg, t_start=50.0)
    with pytest.raises(ValueError, match="run directory"):
        anim_mod.animate_trajectories(trajs, cfg, t_end=3.0, outfile=cfg.run_dir / "movie.mp4")
    import matplotlib.animation as manim

    monkeypatch.setattr(manim.writers, "is_available", lambda name: False)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        anim_mod.animate_trajectories(trajs, cfg, t_end=3.0, outfile=tmp_path / "out" / "movie.mp4")


def test_animate_trajectories_gif(cfg, tmp_path):
    out = tmp_path / "anim" / "traj.gif"
    anim_mod.animate_trajectories(_synthetic_trajectories(2), cfg, t_end=3.5, outfile=out, fps=5,
                                  figsize=(4, 2), dpi=30, momentum_limits=100.0)
    assert out.stat().st_size > 0


# ================================================================ data
@pytest.mark.data
def test_render_run423_vorticity_and_lic(run423, tmp_path):
    pytest.importorskip("rlic")
    from shearpic.io.athdf import Snapshot
    from shearpic.physics.fields import vorticity, x_average

    cfg = RunConfig.from_run_dir(run423)
    snap = Snapshot.open(next(run423.glob("*.out2.00006.athdf")))
    d = snap.read(["vel1", "vel2", "Bcc1", "Bcc2"])
    x, y = snap.centers(0, cfg.bounds[0]), snap.centers(1, cfg.bounds[1])
    omega_z = vorticity((d["vel1"], d["vel2"], np.zeros_like(d["vel1"])), cfg.dx)[2]

    fig, axs = plt.subplots(1, 3, figsize=(7, 8))
    im_v = fields.plot_field(axs[0], d["vel1"], x, y, symmetric=True, cbar_label=r"$U_x$")
    im_w = fields.plot_field(axs[1], omega_z, x, y, symmetric=True, vmax=5.0, cbar_label=r"$\omega_z$")
    _, lic_im = lic.plot_lic(axs[2], x, y, d["Bcc1"], d["Bcc2"], np.hypot(d["Bcc1"], d["Bcc2"]), downsample=4,
                             niter=8, kernel_length=32, cbar_label=r"$|B_\perp|$")
    np.testing.assert_allclose(im_w.get_extent(), [-5 * np.pi, 5 * np.pi, -20 * np.pi, 20 * np.pi], atol=1e-9)
    assert np.asarray(lic_im.get_array()).shape == (4096, 1024, 4)   # blended on the full-resolution colour grid

    # orientation: the jet (|y| < 10 pi, U_x ~ +1) is in the middle rows of the rendered image
    prof = x_average(d["vel1"])
    jet = np.abs(y) < 5 * np.pi
    wind = np.abs(y) > 15 * np.pi
    assert prof[jet].mean() > 0.5 and prof[wind].mean() < -0.5
    rows = np.asarray(im_v.get_array()).mean(axis=1)  # image rows run along y
    assert rows[2048] > 0.5 and rows[10] < -0.5 and rows[-10] < -0.5

    out = tmp_path / "run423_vorticity_lic.png"
    fig.savefig(out, dpi=80)
    assert out.stat().st_size > 0


# ================================================================ panel layout
def test_panel_grid_identical_panels_and_colorbar_row():
    Lx, Ly = 10.0, 40.0
    fig, axes, caxes = style.panel_grid(2, 3, Ly / Lx, width=7.0, wspace=0.05, hspace=0.1, cbar_height=0.2,
                                        cbar_pad=0.05, margins=(0.7, 0.1, 0.5, 0.4))
    assert axes.shape == (2, 3) and caxes.shape == (3,)
    W, H = fig.get_size_inches()
    panel_w = (7.0 - 0.8 - 2 * 0.05) / 3
    assert np.isclose(H, 0.5 + 2 * 4 * panel_w + 0.1 + 0.05 + 0.2 + 0.4)
    sizes = {(round(a.get_position().width * W, 9), round(a.get_position().height * H, 9)) for a in axes.flat}
    assert len(sizes) == 1                                   # every panel has the same size in inches
    (w, h), = sizes
    assert np.isclose(w, panel_w) and np.isclose(h / w, Ly / Lx)
    # equal-aspect images fill their panels exactly (no shrinking), so all panels share one scale
    for a in axes.flat:
        a.imshow(np.zeros((40, 10)), extent=[-5, 5, -20, 20], aspect="equal", origin="lower")
    fig.canvas.draw()
    for a in axes.flat:
        box = a.get_window_extent()
        assert np.isclose(box.height / box.width, 4.0, rtol=1e-3)
    # colorbar axes: above the top row, aligned with the columns
    for j in range(3):
        cpos, ppos = caxes[j].get_position(), axes[0, j].get_position()
        assert np.isclose(cpos.x0, ppos.x0) and np.isclose(cpos.width, ppos.width)
        assert np.isclose((cpos.y0 - ppos.y1) * H, 0.05) and np.isclose(cpos.height * H, 0.2)
    # shared axes and outer tick labels only
    axes[1, 2].set_xlim(-3, 3)
    assert axes[0, 0].get_xlim() == (-3.0, 3.0)

    def shown(axis):
        return any(t.label1.get_visible() for t in axis.get_major_ticks())

    assert not shown(axes[0, 1].xaxis) and shown(axes[1, 1].xaxis)
    assert shown(axes[1, 0].yaxis) and not shown(axes[1, 2].yaxis)


def test_panel_grid_without_colorbar_row_and_validation():
    fig, axes, caxes = style.panel_grid(1, 2, 0.5, width=6.0, cbar_row=False, share=False, margins=(0, 0, 0, 0),
                                        wspace=0.0)
    assert caxes is None and axes.shape == (1, 2)
    assert np.allclose(fig.get_size_inches(), (6.0, 1.5))
    axes[0, 1].set_xlim(2, 3)
    assert axes[0, 0].get_xlim() != (2.0, 3.0)
    with pytest.raises(ValueError, match="no room"):
        style.panel_grid(1, 3, 1.0, width=0.5)
    with pytest.raises(ValueError, match="positive"):
        style.panel_grid(1, 1, -1.0, width=3.0)


def test_add_colorbar_into_given_axes_and_panel_label():
    fig, axes, caxes = style.panel_grid(1, 2, 2.0, width=5.0)
    im = axes[0, 0].imshow(np.arange(4.0).reshape(2, 2), vmin=0, vmax=3)
    n_axes = len(fig.axes)
    cb = style.add_colorbar(axes[0, 0], im, location="top", label=r"$\rho$", cax=caxes[0], labelpad=15)
    assert cb.ax is caxes[0] and len(fig.axes) == n_axes        # no new axes carved out of the panel
    assert cb.orientation == "horizontal" and caxes[0].xaxis.get_label_position() == "top"
    assert cb.ax.xaxis.label.get_text() == r"$\rho$" and cb.ax.xaxis.labelpad == 15
    txt = style.panel_label(axes[0, 1], "t = 40", loc="upper left", fontsize=12)
    assert txt.get_transform() == axes[0, 1].transAxes and txt.get_ha() == "left" and txt.get_va() == "top"
    x, y = txt.get_position()
    assert 0 < x < 0.1 and 0.9 < y < 1 and txt.get_bbox_patch() is not None
    txt2 = style.panel_label(axes[0, 1], "B", loc="lower right", box=False)
    assert txt2.get_ha() == "right" and txt2.get_va() == "bottom" and txt2.get_bbox_patch() is None
    with pytest.raises(ValueError, match="loc must be"):
        style.panel_label(axes[0, 1], "x", loc="center")
