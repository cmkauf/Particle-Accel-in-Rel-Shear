r"""Matplotlib helpers for shearpic figures: style, field maps, line plots, LIC and animations.

Importing this package neither imports ``matplotlib.pyplot`` nor changes ``rcParams``; the
functions are loaded from their submodules on first access.  Every plotting function takes
an explicit ``Axes`` and returns the artist it creates::

    from shearpic.plotting import paper_style, plot_field

    with paper_style():
        fig, ax = plt.subplots(figsize=(3, 8))
        plot_field(ax, vort_z, cfg.centers(0), cfg.centers(1), symmetric=True,
                   cbar_label=r"$\omega_z\ [U_0/a]$")
"""

from __future__ import annotations

import importlib

_LAZY = {
    "STYLE_PATH": "style", "paper_style": "style", "use_paper_style": "style", "format_axes": "style",
    "add_colorbar": "style", "bfield_cmap": "style", "register_colormaps": "style",
    "panel_grid": "style", "panel_label": "style",
    "cell_edges": "fields", "image_extent": "fields", "plot_field": "fields",
    "plot_snapshot_field": "fields", "plot_profile": "fields",
    "plot_timeseries": "lines", "plot_spectrum": "lines", "plot_phase_space": "lines",
    "block_mean": "lic", "equalize_histogram": "lic", "lic_texture": "lic", "plot_lic": "lic",
    "lic_blend": "lic", "shade_texture": "lic",
    "animate_trajectories": "animation", "trail_alpha": "animation",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(__all__))
