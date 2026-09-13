"""Helpers shared by the figure scripts in this directory.

They read presets from ``paper_figures.yaml``, locate runs, products and figures, and write
the JSON provenance stored next to every figure and product. The physics lives in ``shearpic``.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import shearpic
from shearpic.config import RunConfig
from shearpic.env import output_dir, resolve_run, run_output_dir

HERE = Path(__file__).resolve().parent
DEFAULT_PRESETS = HERE / "paper_figures.yaml"

__all__ = [
    "add_common_arguments", "load_preset", "require_keys", "find_run", "figure_dir", "product_dir", "save_figure",
    "write_json", "provenance", "run_summary", "check_same_run", "print_table", "run_cli",
]


# --------------------------------------------------------------------- presets
def load_preset(name: str, path: str | Path | None = None) -> dict[str, Any]:
    """The preset ``name`` from ``paper_figures.yaml`` (or ``path``) as plain Python objects."""
    from ruamel.yaml import YAML

    path = Path(path) if path else DEFAULT_PRESETS
    with open(path) as fh:
        doc = YAML(typ="safe").load(fh) or {}
    if name not in doc:
        names = [k for k in doc if k != "schema_version"]
        raise KeyError(f"preset {name!r} not in {path}; available: {', '.join(names)}")
    return dict(doc[name])


def require_keys(preset: Mapping[str, Any], keys: Iterable[str], name: str, hint: str = "") -> None:
    """Raise a readable KeyError when a preset lacks keys a subcommand needs."""
    missing = [k for k in keys if k not in preset]
    if missing:
        raise KeyError(f"preset {name!r} has no {', '.join(missing)}" + (f"; {hint}" if hint else ""))


# --------------------------------------------------------------------- arguments
def add_common_arguments(parser: argparse.ArgumentParser, preset: str) -> argparse.ArgumentParser:
    """Options shared by every script: preset, data/output locations, parallelism, formats."""
    g = parser.add_argument_group("common options")
    g.add_argument("--preset", default=preset, help=f"preset name in the presets file (default: {preset})")
    g.add_argument("--presets", default=str(DEFAULT_PRESETS), help="presets YAML file (default: %(default)s)")
    g.add_argument("--data-root", default=None,
                   help="directory (or ':'-separated list) holding the run folders; default: PARTICLE_ACCEL_DATA")
    g.add_argument("--products", default=None,
                   help="directory for intermediate products (default: <output root>/<run>/products)")
    g.add_argument("--out", default=None, help="figure directory (default: <output root>/paper_figures)")
    g.add_argument("--formats", default="pdf", help="comma-separated figure formats, e.g. pdf,png (default: pdf)")
    g.add_argument("--dpi", type=int, default=300)
    g.add_argument("--workers", default="auto", help="number of worker processes or 'auto' (default)")
    g.add_argument("--backend", default="auto", choices=["auto", "serial", "thread", "process", "mpi"])
    g.add_argument("--allow-download", action="store_true",
                   help="allow reading online-only cloud placeholders (downloads them)")
    return parser


def workers_arg(value: str) -> int | str:
    return value if value == "auto" else int(value)


# --------------------------------------------------------------------- locations
def find_run(run: str | int | Path, data_root: str | None = None) -> Path:
    """Run directory from a name (``run364``), a number or a path."""
    return resolve_run(run, roots=data_root) if data_root else resolve_run(run)


def _warn_if_output_in_repo(path: Path) -> None:
    from shearpic.env import repo_root

    root = repo_root()
    if root is None or os.environ.get("PARTICLE_ACCEL_OUTPUT"):
        return
    try:
        Path(path).resolve().relative_to(root.resolve())
    except ValueError:
        return
    print(f"note: writing outputs inside the repository ({path}); set PARTICLE_ACCEL_OUTPUT to a directory "
          "outside the checkout (see analysis/README.md)", file=sys.stderr)


def figure_dir(out: str | None = None) -> Path:
    """Directory for figures (created): ``--out`` or ``<output root>/paper_figures``."""
    p = Path(out).expanduser() if out else output_dir("paper_figures", create=False)
    _warn_if_output_in_repo(p)
    p.mkdir(parents=True, exist_ok=True)
    return p


def product_dir(run_dir: Path, products: str | None = None, *, create: bool = True) -> Path:
    """Directory for a run's intermediate products: ``--products`` or ``<output root>/<run>/products``.

    Plot stages pass ``create=False`` so that a failed lookup leaves no empty directories.
    """
    p = Path(products).expanduser() if products else run_output_dir(run_dir, "products", create=False)
    if create:
        _warn_if_output_in_repo(p)
        p.mkdir(parents=True, exist_ok=True)
    return p


# --------------------------------------------------------------------- provenance
def _git_state(path: Path) -> dict[str, Any]:
    try:
        commit = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True,
                                timeout=5).stdout.strip() or None
        dirty = bool(subprocess.run(["git", "-C", str(path), "status", "--porcelain"], capture_output=True,
                                    text=True, timeout=5).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty": None}
    return {"commit": commit, "dirty": dirty if commit else None}


def run_summary(cfg: RunConfig) -> dict[str, Any]:
    """Parameters that identify a run in a provenance record."""
    keys = ("run_id", "problem_id", "run_dir", "nx", "c", "q_mc", "M_A", "B0", "vp_par", "cr_mass", "tau",
            "nu_iso", "shear_amplitude", "shear_amplitude_source", "n_par", "m_cr", "V")
    s = cfg.summary()
    out = {k: s[k] for k in keys if k in s}
    out["athinput_sha256"] = cfg.athinput.sha256
    out["layers"] = list(cfg.profile.layer_positions)
    return out


#: Parameters that must agree between a product and the run it is plotted with. A restart
#: that edits ``tlim`` changes the athinput hash but not these.
PHYSICS_KEYS = ("problem_id", "nx", "c", "q_mc", "M_A", "B0", "vp_par", "cr_mass", "tau", "nu_iso",
                "shear_amplitude", "n_par", "V")


def check_same_run(product_run: Mapping[str, Any] | None, cfg: RunConfig, what: str = "product") -> None:
    """Refuse a product whose recorded physics parameters differ from those of ``cfg``.

    ``product_run`` is the ``run`` entry written by :func:`run_summary`. Numbers are compared
    to a relative tolerance of 1e-6; a different athinput sha256 alone only prints a note.
    """
    if not product_run:
        return
    now = run_summary(cfg)
    diffs = []
    for key in PHYSICS_KEYS:
        if key not in product_run or key not in now:
            continue
        a, b = product_run[key], now[key]
        if isinstance(a, (list, tuple)) or isinstance(b, (list, tuple)):
            same = list(a) == list(b)
        elif isinstance(a, (int, float)) and isinstance(b, (int, float)) and not isinstance(a, bool):
            same = math.isclose(float(a), float(b), rel_tol=1e-6, abs_tol=1e-12)
        else:
            same = a == b
        if not same:
            diffs.append(f"{key}: {a} (product) vs {b} (run)")
    if diffs:
        raise ValueError(f"{what} was made for a different run than {cfg.run_dir}: " + "; ".join(diffs)
                         + " -- recompute it with --stage compute")
    sha = product_run.get("athinput_sha256")
    if sha and sha != cfg.athinput.sha256:
        print(f"note: {what} was made from an athinput with a different sha256 than {cfg.run_dir} "
              "(physics parameters agree, e.g. tlim edited at a restart)", file=sys.stderr)


def provenance(**extra: Any) -> dict[str, Any]:
    return {
        "created_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "command": " ".join([Path(sys.argv[0]).name, *sys.argv[1:]]),
        "shearpic_version": shearpic.__version__,
        "repository": _git_state(HERE),
        "python": platform.python_version(),
        "host": platform.node(),
        **extra,
    }


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


def write_json(path: str | Path, data: Mapping[str, Any]) -> Path:
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(_jsonable(data), indent=2, sort_keys=False) + "\n")
    os.replace(tmp, path)
    return path


def save_figure(fig, directory: str | Path, name: str, *, formats: str | Iterable[str] = "pdf", dpi: int = 300,
                metadata: Mapping[str, Any] | None = None) -> list[Path]:
    """Save ``fig`` as ``<directory>/<name>.<fmt>`` plus ``<name>.json`` with provenance."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    if isinstance(formats, str):
        formats = [f.strip() for f in formats.split(",") if f.strip()]
    paths = []
    for fmt in formats:
        p = directory / f"{name}.{fmt}"
        fig.savefig(p, dpi=dpi, bbox_inches="tight")
        paths.append(p)
    write_json(directory / f"{name}.json", provenance(outputs=[p.name for p in paths], **(metadata or {})))
    return paths


def print_table(title: str, rows: Mapping[str, Any]) -> None:
    """Print a title followed by aligned ``key : value`` rows."""
    print(f"\n{title}")
    width = max((len(k) for k in rows), default=0)
    for k, v in rows.items():
        if isinstance(v, float):
            v = f"{v:.6g}"
        print(f"  {k:<{width}} : {v}")


def run_cli(main, argv=None) -> None:
    """Run ``main(argv)``, reporting expected user errors in one line with exit code 2.

    ``SHEARPIC_DEBUG=1`` re-raises them with the full traceback.
    """
    from shearpic.io.athdf import DatalessFileError

    try:
        code = main(argv)
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
    except (FileNotFoundError, DatalessFileError, KeyError, ValueError) as err:
        if os.environ.get("SHEARPIC_DEBUG"):
            raise
        msg = err.args[0] if isinstance(err, KeyError) and err.args else err
        print(f"error: {msg}", file=sys.stderr)
        print("hint: check PARTICLE_ACCEL_DATA / PARTICLE_ACCEL_OUTPUT, --preset and --run "
              "(analysis/README.md, troubleshooting); SHEARPIC_DEBUG=1 shows the traceback", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(code or 0)
