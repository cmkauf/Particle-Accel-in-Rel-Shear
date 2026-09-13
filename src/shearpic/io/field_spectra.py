"""Compute :class:`~shearpic.physics.turbulence.FieldSpectrum` products from snapshots and save or load them.

A product is an ``.npz`` file, read with ``allow_pickle=False``, with ``format_version``,
``layout``, ``names``, ``metadata_json`` and the FieldSpectrum fields of each name under keys
prefixed ``<name>__``. Layout ``'single'`` stores one dict of spectra; ``'series'`` stores
one per snapshot, with ``E``, ``total`` and ``energy_beyond_kmax`` stacked along a leading
snapshot axis and the times in ``times``. Unknown times are stored as NaN.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping, Sequence, Union

import numpy as np

from ..physics.turbulence import FieldSpectrum, fluctuations, mhd_energy_spectra

if TYPE_CHECKING:  # pragma: no cover
    from ..config import RunConfig

__all__ = ["FORMAT_VERSION", "save_field_spectra", "load_field_spectra", "snapshot_mhd_spectra"]

FORMAT_VERSION = 1
_SEP = "__"

SpectraDict = Mapping[str, FieldSpectrum]


def _json_default(obj: Any):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, tuple)):
        return list(obj)
    raise TypeError(f"metadata value of type {type(obj).__name__} is not JSON serialisable")


def _clean_json(obj: Any) -> Any:
    """Replace non-finite floats, which JSON cannot represent, by None."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, Mapping):
        return {str(k): _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean_json(v) for v in obj]
    return obj


def _check_name(name: str) -> None:
    if not isinstance(name, str) or not name or _SEP in name or "/" in name:
        raise ValueError(f"invalid spectrum name {name!r} (non-empty, no '{_SEP}' or '/')")


def save_field_spectra(path: str | Path, spectra: Union[SpectraDict, Sequence[SpectraDict]],
                       metadata: Mapping[str, Any] | None = None) -> Path:
    """Write spectra to the ``.npz`` file ``path``, used exactly as given.

    ``spectra`` is one dict of FieldSpectrum, e.g. a time average, or a list with one dict
    per snapshot sharing the same names and bins. ``metadata`` is a JSON-serialisable
    mapping; numpy values and paths are converted.
    """
    path = Path(path)
    data: dict[str, np.ndarray] = {"format_version": np.array(FORMAT_VERSION)}
    if isinstance(spectra, Mapping):
        layout, series = "single", [spectra]
    else:
        layout, series = "series", list(spectra)
        if not series:
            raise ValueError("empty list of spectra")
    names = list(series[0].keys())
    if not names:
        raise ValueError("no spectra to save")
    for name in names:
        _check_name(name)
    for i, snap in enumerate(series):
        if list(snap.keys()) != names:
            raise ValueError(f"snapshot {i} has spectra {list(snap.keys())}, expected {names}")
    data["layout"] = np.array(layout)
    data["names"] = np.array(names)
    if layout == "series":
        data["times"] = np.array([np.nan if s[names[0]].time is None else s[names[0]].time for s in series],
                                 dtype=np.float64)
    for name in names:
        first = series[0][name]
        for i, snap in enumerate(series[1:], start=1):
            s = snap[name]
            if s.k_edges.shape != first.k_edges.shape or not np.array_equal(s.k_edges, first.k_edges) \
                    or not np.array_equal(s.n_modes, first.n_modes):
                raise ValueError(f"spectrum {name!r} of snapshot {i} has different bins than snapshot 0")
        key = name + _SEP
        data[key + "k_edges"] = first.k_edges
        data[key + "n_modes"] = first.n_modes
        data[key + "kind"] = np.array(first.kind)
        if layout == "single":
            data[key + "E"] = first.E
            data[key + "total"] = np.array(first.total)
            data[key + "energy_beyond_kmax"] = np.array(first.energy_beyond_kmax)
            data[key + "time"] = np.array(np.nan if first.time is None else first.time)
        else:
            data[key + "E"] = np.stack([snap[name].E for snap in series])
            data[key + "total"] = np.array([snap[name].total for snap in series])
            data[key + "energy_beyond_kmax"] = np.array([snap[name].energy_beyond_kmax for snap in series])
    data["metadata_json"] = np.array(json.dumps(_clean_json(dict(metadata or {})), default=_json_default))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as fh:  # given a path, np.savez would append '.npz' to other suffixes
        np.savez(fh, **data)
    tmp.replace(path)
    return path


def load_field_spectra(path: str | Path) -> tuple[Union[dict[str, FieldSpectrum], list[dict[str, FieldSpectrum]]],
                                                  dict[str, Any]]:
    """Read ``(spectra, metadata)`` written by :func:`save_field_spectra`, spectra in the layout they were saved."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        if "format_version" not in f.files:
            raise ValueError(f"{path}: not a shearpic field-spectra file (no format_version)")
        version = int(f["format_version"])
        if version > FORMAT_VERSION:
            raise ValueError(f"{path}: format_version {version} is newer than supported ({FORMAT_VERSION})")
        layout = str(f["layout"])
        names = [str(n) for n in f["names"]]
        metadata = json.loads(str(f["metadata_json"])) if "metadata_json" in f.files else {}

        def t_or_none(value) -> float | None:
            value = float(value)
            return None if math.isnan(value) else value

        if layout == "single":
            out = {}
            for name in names:
                key = name + _SEP
                out[name] = FieldSpectrum(k_edges=f[key + "k_edges"], E=f[key + "E"], n_modes=f[key + "n_modes"],
                                          total=float(f[key + "total"]),
                                          energy_beyond_kmax=float(f[key + "energy_beyond_kmax"]),
                                          time=t_or_none(f[key + "time"]), kind=str(f[key + "kind"]))
            return out, metadata
        if layout != "series":
            raise ValueError(f"{path}: unknown layout {layout!r}")
        times = f["times"]
        series: list[dict[str, FieldSpectrum]] = [{} for _ in range(times.size)]
        for name in names:
            key = name + _SEP
            edges, modes, kind = f[key + "k_edges"], f[key + "n_modes"], str(f[key + "kind"])
            E, total, beyond = f[key + "E"], f[key + "total"], f[key + "energy_beyond_kmax"]
            for i in range(times.size):
                series[i][name] = FieldSpectrum(k_edges=edges, E=E[i], n_modes=modes, total=float(total[i]),
                                                energy_beyond_kmax=float(beyond[i]), time=t_or_none(times[i]),
                                                kind=kind)
        return series, metadata


def snapshot_mhd_spectra(path: str | Path, cfg: "RunConfig", *, dk: float | None = None,
                         k_max: float | str = "nyquist", velocity_mean: str = "favre", weight: str = "sqrt_rho",
                         allow_download: bool = False) -> dict[str, Any]:
    """Kinetic and magnetic fluctuation spectra and mean energies of one ``.athdf`` snapshot.

    Intended as a :func:`shearpic.parallel.parallel_map` worker. ``dk``, ``k_max``,
    ``velocity_mean`` and ``weight`` are passed to
    :func:`shearpic.physics.turbulence.mhd_energy_spectra`.

    Returns a dict with ``path``, ``time``, ``spectra`` (``kin``, ``mag``, ``tot``) and
    ``grid``, which holds ``rho_min``, ``rho_max`` and the volume-averaged energy densities
    ``eps_kin = <rho v^2/2>``, ``eps_mag = <B^2/2>`` and their fluctuating parts
    ``eps_kin_fluct`` and ``eps_mag_fluct``.
    """
    from ._util import is_dataless
    from .athdf import DatalessFileError, Snapshot

    path = Path(path)
    if is_dataless(path) and not allow_download:
        raise DatalessFileError(f"{path} is an online-only placeholder; make it available offline "
                                "or pass allow_download=True")
    snap = Snapshot.open(path)
    names = ["rho", "vel1", "vel2", "vel3", "Bcc1", "Bcc2", "Bcc3"]
    d = snap.read(names, dtype=np.float64, squeeze=True)
    rho = d.pop("rho")
    v = (d.pop("vel1"), d.pop("vel2"), d.pop("vel3"))
    B = (d.pop("Bcc1"), d.pop("Bcc2"), d.pop("Bcc3"))
    grid = {
        "eps_kin": float(np.mean(0.5 * rho * sum(c * c for c in v))),
        "eps_mag": float(np.mean(0.5 * sum(c * c for c in B))),
        "rho_min": float(rho.min()),
        "rho_max": float(rho.max()),
    }
    # real-space fluctuation energies, an independent check of the spectrum totals
    w, b = fluctuations(rho, v, B, velocity_mean=velocity_mean, weight=weight)
    grid["eps_kin_fluct"] = float(np.mean(0.5 * sum(c * c for c in w)))
    grid["eps_mag_fluct"] = float(np.mean(0.5 * sum(c * c for c in b)))
    del w, b
    spectra = mhd_energy_spectra(rho, v, B, cfg, dk=dk, k_max=k_max, velocity_mean=velocity_mean, weight=weight,
                                 time=snap.time)
    return {"path": str(path), "time": float(snap.time), "spectra": spectra, "grid": grid}
