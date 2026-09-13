"""shearpic: analysis of MHD-PIC simulations of particle acceleration in shear flows.

shearpic.io        readers for Athena++ outputs: athinput, .hst, .athdf and particle files
shearpic.physics   physics functions: relativity, fields, forcing, particles, spectra
shearpic.plotting  matplotlib helpers, imported only on request
shearpic.config    RunConfig, the parameters and derived scales of one run
shearpic.env       machine detection and data and output paths
shearpic.parallel  parallel_map with serial, thread, process and MPI backends
shearpic.registry  run bookkeeping backed by runs.yaml

Importing shearpic is cheap: it does not import matplotlib or touch rcParams.
"""

__version__ = "0.1.0"

_LAZY = {
    "RunConfig": ("shearpic.config", "RunConfig"),
    "detect_environment": ("shearpic.env", "detect_environment"),
    "resolve_run": ("shearpic.env", "resolve_run"),
    "run_output_dir": ("shearpic.env", "run_output_dir"),
    "init_registry": ("shearpic.registry", "init_registry"),
    "parallel_map": ("shearpic.parallel", "parallel_map"),
    "load_registry": ("shearpic.registry", "load_registry"),
}

__all__ = [*_LAZY, "__version__"]


def __getattr__(name):  # lazy imports keep the package import fast
    if name in _LAZY:
        import importlib

        module, attr = _LAZY[name]
        return getattr(importlib.import_module(module), attr)
    raise AttributeError(f"module 'shearpic' has no attribute {name!r}")
