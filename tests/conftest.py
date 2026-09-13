"""Shared pytest configuration.

Tests marked ``@pytest.mark.data`` read real simulation output and are skipped unless
``PARTICLE_ACCEL_TEST_DATA`` points at a directory holding the reference runs::

    PARTICLE_ACCEL_TEST_DATA=/path/to/results pytest -m data

Each data test is skipped when a run it reads is missing.
Each test gets ``PARTICLE_ACCEL_DATA``, ``PARTICLE_ACCEL_OUTPUT`` and ``PARTICLE_ACCEL_REGISTRY``
inside its own ``tmp_path``, and the session fails if the repository's ``runs.yaml`` changes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("PARTICLE_ACCEL_TEST_DATA"):
        return
    skip = pytest.mark.skip(reason="set PARTICLE_ACCEL_TEST_DATA to run tests on real simulation data")
    for item in items:
        if "data" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def data_root() -> Path:
    root = Path(os.environ["PARTICLE_ACCEL_TEST_DATA"])
    if not root.is_dir():
        pytest.skip(f"PARTICLE_ACCEL_TEST_DATA={root} is not a directory")
    return root


def _run(data_root: Path, name: str) -> Path:
    p = data_root / name
    if not p.is_dir():
        pytest.skip(f"{name} not available under {data_root}")
    return p


@pytest.fixture(scope="session")
def run423(data_root) -> Path:
    return _run(data_root, "run423")


@pytest.fixture(scope="session")
def run404(data_root) -> Path:
    return _run(data_root, "run404")


@pytest.fixture(scope="session")
def run369(data_root) -> Path:
    return _run(data_root, "run369")


@pytest.fixture(scope="session")
def run310(data_root) -> Path:
    return _run(data_root, "run310")


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch, tmp_path):
    """Point data, output and registry paths into ``tmp_path`` and clear SLURM/MPI variables."""
    for key in ("PARTICLE_ACCEL_MACHINE", "SLURM_JOB_ID", "SLURM_CPUS_ON_NODE", "SLURM_JOB_PARTITION",
                "SLURM_JOB_NUM_NODES", "SLURM_CLUSTER_NAME", "TACC_SYSTEM", "PMI_SIZE", "OMPI_COMM_WORLD_SIZE",
                "PMIX_SIZE", "SLURM_MEM_PER_NODE", "SLURM_MEM_PER_CPU"):
        monkeypatch.delenv(key, raising=False)
    # machine detection must not depend on whether the test host has `sbatch` on PATH
    monkeypatch.setattr("shearpic.env._have_sbatch", lambda: False)
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(tmp_path / "data"))
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "outputs"))
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(tmp_path / "runs.yaml"))
    monkeypatch.setenv("MPLBACKEND", "Agg")


REPO_REGISTRY = Path(__file__).resolve().parents[1] / "runs.yaml"


def _file_state(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


@pytest.fixture(scope="session", autouse=True)
def _repo_registry_untouched():
    """Fail the session if any test created, modified or deleted the repository's runs.yaml."""
    before = _file_state(REPO_REGISTRY)
    yield
    after = _file_state(REPO_REGISTRY)
    assert after == before, (f"the test session modified {REPO_REGISTRY} (sha256 {before} -> {after}); "
                             "a test is not using the isolated PARTICLE_ACCEL_REGISTRY")
