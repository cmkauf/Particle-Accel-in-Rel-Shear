"""Save and load spectra and phase-space histograms, and read legacy spectrum files.

Products are ``.npz`` files, read with ``allow_pickle=False``, holding ``format_version``,
an optional ``metadata_json`` string and the Spectrum fields that are not None. Phase-space
files hold ``kind = 'phase_space'``, the list ``components`` and the PhaseSpaceHist fields
of each component under keys prefixed ``<component>__``. Both writers write to a temporary
file and rename it.

Legacy spectrum CSVs ``histogram_frame_NNNNN_t_T.csv`` have columns ``bin_centers,density``,
with the density normalised over the in-range particles and the centres the arithmetic
midpoints of log-spaced bins, either ``np.logspace(0, 2, 501)`` in gamma or
``np.logspace(-2, 2, 501)`` in gamma - 1. The files do not record which, so pass
``variable`` explicitly. The legacy ``phase_space_histograms.npz`` holds
``np.histogram2d(u_i, y, density=True)`` for each component, without time, run or particle
numbers.
"""

from __future__ import annotations

import json
import math
import os
import re
import warnings
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ..physics.phase_space import COMPONENTS, PhaseSpaceHist
from ..physics.spectra import VARIABLES, Spectrum

__all__ = ["FORMAT_VERSION", "save_spectrum", "load_spectrum", "load_metadata","read_legacy_histogram_csv",
           "read_legacy_spectrum_dir", "reconstruct_log_edges", "recover_integer_counts",
           "spectrum_counts_from_pdf", "save_phase_space", "load_phase_space", "read_legacy_phase_space_npz"]

FORMAT_VERSION = 1

_FRAME_RE = re.compile(r"histogram_frame_(\d+)")
_TIME_RE = re.compile(r"_t_([-+]?\d+(?:\.\d*)?(?:[eE][-+]?\d+)?)\.csv$")
_RUN_RE = re.compile(r"^run0*(\d+)$")


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, np.generic):
        obj = obj.item()
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    return obj


def _metadata_entry(metadata: Mapping[str, Any] | None) -> dict[str, np.ndarray]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping")
    return {"metadata_json": np.array(json.dumps(_jsonable(metadata), sort_keys=False))}


def _read_metadata(f) -> dict[str, Any]:
    return json.loads(str(f["metadata_json"])) if "metadata_json" in f.files else {}


def _atomic_savez(path: Path, data: Mapping[str, np.ndarray]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:  # given a path, np.savez would append '.npz' to other suffixes
            np.savez(fh, **data)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()
    return path


def _check_version(f, path: Path) -> int:
    if "format_version" not in f.files:
        raise ValueError(f"{path}: not a shearpic file (no format_version)")
    version = int(f["format_version"])
    if version > FORMAT_VERSION:
        raise ValueError(f"{path}: format_version {version} is newer than supported ({FORMAT_VERSION})")
    return version


def save_spectrum(path: str | Path, spec: Spectrum, *, metadata: Mapping[str, Any] | None = None) -> Path:
    """Write ``spec`` to the ``.npz`` file ``path``, used exactly as given.

    ``metadata`` is a JSON-serialisable mapping, where numpy values and paths are converted,
    stored as ``metadata_json``. Products belong in :func:`shearpic.env.run_output_dir`
    rather than in the run directory.
    """
    path = Path(path)
    data: dict[str, np.ndarray] = {
        "format_version": np.array(FORMAT_VERSION),
        "variable": np.array(spec.variable),
        "edges": spec.edges,
    }
    for name in ("counts", "pdf"):
        arr = getattr(spec, name)
        if arr is not None:
            data[name] = arr
    for name in ("n_total", "underflow", "overflow", "time", "c"):
        val = getattr(spec, name)
        if val is not None:
            data[name] = np.array(float(val))
    if spec.run_id is not None:
        data["run_id"] = np.array(str(spec.run_id))
        data["run_id_is_int"] = np.array(isinstance(spec.run_id, (int, np.integer)))
    data.update(_metadata_entry(metadata))
    return _atomic_savez(path, data)


def load_metadata(path: str | Path) -> dict[str, Any]:
    """The ``metadata_json`` entry of a spectrum or phase-space product, ``{}`` if none, without reading the arrays."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        _check_version(f, path)
        return _read_metadata(f)


def load_spectrum(path: str | Path, *, with_metadata: bool = False) -> Spectrum | tuple[Spectrum, dict[str, Any]]:
    """Read a Spectrum written by :func:`save_spectrum`, as ``(spectrum, metadata)`` if ``with_metadata``."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        _check_version(f, path)
        if "variable" not in f.files or "edges" not in f.files:
            raise ValueError(f"{path}: not a shearpic spectrum file")

        def opt(name):
            return f[name] if name in f.files else None

        def opt_float(name):
            return float(f[name]) if name in f.files else None

        run_id = None
        if "run_id" in f.files:
            run_id = str(f["run_id"])
            if "run_id_is_int" in f.files and bool(f["run_id_is_int"]):
                run_id = int(run_id)
        spec = Spectrum(
            variable=str(f["variable"]), edges=f["edges"], counts=opt("counts"), pdf=opt("pdf"),
            n_total=opt_float("n_total"), underflow=opt_float("underflow"), overflow=opt_float("overflow"),
            time=opt_float("time"), run_id=run_id, c=opt_float("c"),
        )
        meta = _read_metadata(f)
    return (spec, meta) if with_metadata else spec


def save_phase_space(path: str | Path, hists: Mapping[str, PhaseSpaceHist], *,
                     metadata: Mapping[str, Any] | None = None) -> Path:
    """Write ``{component: PhaseSpaceHist}`` to one ``.npz`` file; ``metadata`` as in :func:`save_spectrum`."""
    path = Path(path)
    if not hists:
        raise ValueError("nothing to save")
    data: dict[str, np.ndarray] = {"format_version": np.array(FORMAT_VERSION), "kind": np.array("phase_space")}
    comps = []
    for key, h in hists.items():
        if not isinstance(h, PhaseSpaceHist):
            raise TypeError(f"{key!r}: expected a PhaseSpaceHist, got {type(h).__name__}")
        if key != h.component:
            raise ValueError(f"key {key!r} does not match the histogram component {h.component!r}")
        comps.append(key)
        p = f"{key}__"
        data[p + "u_edges"], data[p + "y_edges"] = h.u_edges, h.y_edges
        data[p + "counts"] = h.counts.astype(np.int64)
        data[p + "n_total"] = np.array(h.n_total, dtype=np.int64)
        for name in ("out_of_range_y", "n_y"):
            if getattr(h, name) is not None:
                data[p + name] = getattr(h, name).astype(np.int64)
        for name in ("sum_u_y", "sum_u2_y"):
            if getattr(h, name) is not None:
                data[p + name] = getattr(h, name)
        for name in ("time", "c"):
            if getattr(h, name) is not None:
                data[p + name] = np.array(float(getattr(h, name)))
        if h.run_id is not None:
            data[p + "run_id"] = np.array(str(h.run_id))
            data[p + "run_id_is_int"] = np.array(isinstance(h.run_id, (int, np.integer)))
    data["components"] = np.array(",".join(comps))
    data.update(_metadata_entry(metadata))
    return _atomic_savez(path, data)


def load_phase_space(path: str | Path, *, with_metadata: bool = False):
    """Read ``{component: PhaseSpaceHist}`` written by :func:`save_phase_space`, with metadata if requested."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        _check_version(f, path)
        if "kind" not in f.files or str(f["kind"]) != "phase_space":
            raise ValueError(f"{path}: not a shearpic phase-space file")
        out = {}
        for comp in str(f["components"]).split(","):
            p = f"{comp}__"

            def opt(name, p=p):
                return f[p + name] if p + name in f.files else None

            run_id = None
            if p + "run_id" in f.files:
                run_id = str(f[p + "run_id"])
                if bool(f[p + "run_id_is_int"]):
                    run_id = int(run_id)
            out[comp] = PhaseSpaceHist(
                component=comp, u_edges=f[p + "u_edges"], y_edges=f[p + "y_edges"], counts=f[p + "counts"],
                n_total=int(f[p + "n_total"]), out_of_range_y=opt("out_of_range_y"), n_y=opt("n_y"),
                sum_u_y=opt("sum_u_y"), sum_u2_y=opt("sum_u2_y"),
                time=None if opt("time") is None else float(opt("time")), run_id=run_id,
                c=None if opt("c") is None else float(opt("c")),
            )
        meta = _read_metadata(f)
    return (out, meta) if with_metadata else out


def recover_integer_counts(probability, *, n_hint: int | None = None, max_min_count: int = 1000,
                           tol: float = 1e-6) -> tuple[np.ndarray, int]:
    """Integer counts ``n_k`` and ``N_in = sum n_k`` from normalised bin contents ``q_k = n_k / N_in``.

    For a density, ``q_k`` is the density times the bin width or area. If ``m`` is the
    smallest non-zero count then ``N_in = m / min(q)``; ``n_hint`` and then
    ``m = 1 .. max_min_count`` are tried, and a candidate is accepted when every ``q_k N_in``
    is within ``tol`` of an integer and the rounded counts sum to ``N_in``. Pass the particle
    number as ``n_hint`` for narrow distributions with no sparsely filled bin. Counts that
    share a common factor cannot be distinguished from the reduced ones.

    Returns ``(counts, n_in)``; raises ValueError if no candidate gives integer counts.
    """
    q = np.asarray(probability, dtype=np.float64)
    if not np.all(np.isfinite(q)) or np.any(q < 0):
        raise ValueError("probabilities must be finite and non-negative")
    pos = q[q > 0]
    if pos.size == 0:
        raise ValueError("histogram is empty")
    q_min = float(pos.min())
    best = math.inf
    candidates = [int(n_hint)] if n_hint else []
    candidates += [int(round(m / q_min)) for m in range(1, int(max_min_count) + 1)]
    for n_in in candidates:
        if n_in <= 0:
            continue
        n = q * n_in
        resid = float(np.max(np.abs(n - np.round(n))))
        best = min(best, resid)
        if resid < tol:
            counts = np.round(n).astype(np.int64)
            if int(counts.sum()) == n_in:
                return counts, n_in
    raise ValueError(f"could not recover integer counts (smallest integrality residual {best:.2e} >= {tol:g} "
                     f"for minimum counts 1..{max_min_count})")


def spectrum_counts_from_pdf(spec: Spectrum, n_total: int, **kw) -> Spectrum:
    """Spectrum with exact counts from a pdf-only Spectrum and the total particle number.

    The ``n_total - N_in`` particles outside the bins are counted as overflow; if there are
    any, the bins must start at the lower end of the domain, gamma = 1 or gamma - 1 = p/mc = 0.
    Raises ValueError otherwise, or if the counts cannot be recovered or exceed ``n_total``.
    """
    if spec.pdf is None:
        raise ValueError("spectrum has no pdf")
    kw.setdefault("n_hint", int(n_total))
    counts, n_in = recover_integer_counts(spec.pdf * spec.widths, **kw)
    n_total = int(n_total)
    missing = n_total - n_in
    if missing < 0:
        raise ValueError(f"the histogram holds {n_in} particles, more than n_total = {n_total}")
    domain_lo = {"gamma": 1.0, "gamma_minus_1": 0.0, "p_over_mc": 0.0}[spec.variable]
    if missing and spec.edges[0] > domain_lo + 1e-12:
        raise ValueError(f"{missing} particles are outside the bins [{spec.edges[0]:g}, {spec.edges[-1]:g}] of "
                         f"{spec.variable}; cannot tell underflow from overflow")
    return Spectrum(variable=spec.variable, edges=spec.edges, counts=counts, n_total=n_total, underflow=0.0,
                    overflow=float(missing), time=spec.time, run_id=spec.run_id, c=spec.c)


def read_legacy_phase_space_npz(path: str | Path, *, n_total: int, time: float | None,
                                run_id: int | str | None = None, c: float | None = None,
                                tol: float = 1e-6) -> dict[str, PhaseSpaceHist]:
    """Read a legacy ``phase_space_histograms.npz`` as integer histograms ``{component: PhaseSpaceHist}``.

    The stored ``n_ij / (N_in du_i dy_j)`` is multiplied by the bin areas and passed to
    :func:`recover_integer_counts`. ``n_total`` is the particle number of the snapshot,
    e.g. ``cfg.n_par``, and ``time``, ``run_id`` and ``c`` are not stored in the file. The
    ``y`` distribution of particles outside the momentum range is unknown, so
    ``out_of_range_y``, ``n_y`` and the moments are None.
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as f:
        needed = {"hist_px", "hist_py", "hist_pz", "edge_p", "edge_y"}
        missing = needed - set(f.files)
        if missing:
            raise ValueError(f"{path.name}: not a legacy phase-space file (missing {sorted(missing)})")
        u_edges = np.asarray(f["edge_p"], dtype=np.float64)
        y_edges = np.asarray(f["edge_y"], dtype=np.float64)
        area = np.diff(u_edges)[:, None] * np.diff(y_edges)[None, :]
        out = {}
        for comp in COMPONENTS:
            H = np.asarray(f[f"hist_p{comp}"], dtype=np.float64)
            if H.shape != area.shape:
                raise ValueError(f"{path.name}: hist_p{comp} has shape {H.shape}, edges give {area.shape}")
            counts, n_in = recover_integer_counts(H * area, n_hint=int(n_total), tol=tol)
            if n_in > int(n_total):
                raise ValueError(f"{path.name}: hist_p{comp} holds {n_in} particles, more than n_total = {n_total}")
            out[comp] = PhaseSpaceHist(component=comp, u_edges=u_edges, y_edges=y_edges, counts=counts,
                                       n_total=int(n_total), time=time, run_id=run_id, c=c)
    return out


def reconstruct_log_edges(centers, rtol: float = 1e-6) -> np.ndarray:
    """Edges of log-spaced bins from their arithmetic centres.

    For edges ``e_{i+1} = r e_i`` the centres are ``c_i = e_i (1 + r)/2``. Raises ValueError
    if the centre ratios vary by more than ``rtol``. Edges whose decimal exponents are round
    are regenerated with ``np.logspace`` so that they are exact.
    """
    cen = np.asarray(centers, dtype=np.float64)
    if cen.ndim != 1 or cen.size < 2:
        raise ValueError("need at least 2 bin centres")
    if np.any(cen <= 0) or np.any(np.diff(cen) <= 0):
        raise ValueError("bin centres must be positive and increasing for log-spaced bins")
    ratios = cen[1:] / cen[:-1]
    r = float(np.exp(np.mean(np.log(ratios))))
    dev = float(np.max(np.abs(ratios / r - 1.0)))
    if dev > rtol:
        raise ValueError(f"bin centres are not log-spaced (ratio varies by {dev:.2e} > {rtol:.0e})")
    edges = np.empty(cen.size + 1)
    edges[:-1] = 2.0 * cen / (1.0 + r)
    edges[-1] = edges[-2] * r
    lo, hi = math.log10(edges[0]), math.log10(edges[-1])
    lo_r, hi_r = round(lo, 6), round(hi, 6)
    if abs(lo - lo_r) < 1e-9 and abs(hi - hi_r) < 1e-9:
        exact = np.logspace(lo_r, hi_r, edges.size)
        if np.allclose(exact, edges, rtol=1e-9, atol=0):
            edges = exact
    return edges


def _parse_run_id(path: Path) -> int | None:
    for part in reversed(path.resolve().parts[:-1]):
        m = _RUN_RE.match(part)
        if m:
            return int(m.group(1))
    return None


def _read_legacy(path: Path, variable: str | None) -> tuple[Spectrum, bool]:
    df = pd.read_csv(path)
    missing = {"bin_centers", "density"} - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing columns {sorted(missing)} (found {list(df.columns)})")
    edges = reconstruct_log_edges(df["bin_centers"].to_numpy(np.float64))
    detected = variable is None
    if detected:
        variable = "gamma_minus_1" if edges[0] < 0.5 else "gamma"
    elif variable not in VARIABLES:
        raise ValueError(f"unknown variable {variable!r}; choose from {VARIABLES}")
    m = _TIME_RE.search(path.name)
    spec = Spectrum(variable=variable, edges=edges, pdf=df["density"].to_numpy(np.float64),
                    time=float(m.group(1)) if m else None, run_id=_parse_run_id(path))
    return spec, detected


def read_legacy_histogram_csv(path: str | Path, variable: str | None = None) -> Spectrum:
    """Read one legacy ``histogram_frame_NNNNN_t_T.csv`` as a pdf-only Spectrum.

    ``time`` is taken from the file name and ``run_id`` from a ``runNNN`` parent directory.
    With ``variable=None`` the variable is guessed from the bin range with a warning, a first
    edge below 0.5 meaning gamma - 1.
    """
    path = Path(path)
    spec, detected = _read_legacy(path, variable)
    if detected:
        warnings.warn(f"{path.name}: auto-detected spectrum variable {spec.variable!r} from the bin range "
                      f"[{spec.edges[0]:g}, {spec.edges[-1]:g}]; pass variable= to silence", stacklevel=2)
    return spec


def read_legacy_spectrum_dir(directory: str | Path, variable: str | None = None) -> list[Spectrum]:
    """Read every ``histogram_frame_*.csv`` in ``directory``, sorted by frame number.

    With ``variable=None`` the variable guessed for each file must agree across files.
    """
    directory = Path(directory)
    files = []
    for p in directory.glob("histogram_frame_*.csv"):
        m = _FRAME_RE.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    if not files:
        raise FileNotFoundError(f"no histogram_frame_*.csv files in {directory}")
    files.sort()
    specs = [_read_legacy(p, variable)[0] for _, p in files]
    found = {s.variable for s in specs}
    if len(found) > 1:
        raise ValueError(f"{directory}: files disagree on the spectrum variable ({sorted(found)}); pass variable=")
    if variable is None:
        warnings.warn(f"{directory.name}: auto-detected spectrum variable {specs[0].variable!r} from the bin range "
                      f"[{specs[0].edges[0]:g}, {specs[0].edges[-1]:g}]; pass variable= to silence", stacklevel=2)
    return specs
