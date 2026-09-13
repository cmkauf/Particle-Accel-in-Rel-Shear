"""Unit tests for shearpic.io.trajectory (tracked-particle ``.tab`` files)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from shearpic.io.trajectory import (Trajectory, find_trajectory_files, load_trajectories, read_trajectory,
                                    stack_trajectories)
from shearpic.parallel import TaskError


def make_data(n=6, ncol=13, t=None, seed=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, n) if t is None else np.asarray(t, float)
    data = rng.normal(size=(t.size, ncol))
    data[:, 0] = t
    return data


def write_tab(path: Path, data: np.ndarray) -> Path:
    # the problem generator writes space-separated default-precision numbers, no header
    path.write_text("\n".join(" ".join(f"{v:.9g}" for v in row) for row in data) + "\n")
    return path


def test_read_13_columns(tmp_path):
    data = make_data(ncol=13)
    tr = read_trajectory(write_tab(tmp_path / "trajectory_initmbid_264_pid_19457.tab", data))
    assert isinstance(tr, Trajectory) and len(tr) == 6
    assert tr.has_fields
    np.testing.assert_allclose(tr.t, data[:, 0], rtol=1e-8)
    np.testing.assert_allclose(tr.x, data[:, 1:4], rtol=1e-8)
    np.testing.assert_allclose(tr.u, data[:, 4:7], rtol=1e-8)
    np.testing.assert_allclose(tr.B, data[:, 7:10], rtol=1e-8)
    np.testing.assert_allclose(tr.cE, data[:, 10:13], rtol=1e-8)
    assert tr.x.shape == tr.u.shape == tr.B.shape == tr.cE.shape == (6, 3)
    assert tr.pid == 19457 and tr.init_mbid == 264
    assert tr.n_dropped == 0


def test_read_7_columns(tmp_path):
    data = make_data(ncol=7)
    tr = read_trajectory(write_tab(tmp_path / "trajectory_initmbid_0_pid_7.tab", data))
    assert not tr.has_fields and tr.B is None and tr.cE is None
    np.testing.assert_allclose(tr.u, data[:, 4:7], rtol=1e-8)
    assert tr.pid == 7 and tr.init_mbid == 0


def test_unrecognised_filename_gives_no_ids(tmp_path):
    tr = read_trajectory(write_tab(tmp_path / "particle.tab", make_data(ncol=7)))
    assert tr.pid is None and tr.init_mbid is None


@pytest.mark.parametrize("ncol", [6, 10, 14])
def test_wrong_column_count(tmp_path, ncol):
    path = write_tab(tmp_path / "trajectory_initmbid_1_pid_2.tab", make_data(ncol=ncol))
    with pytest.raises(ValueError, match=f"expected 7 or 13 columns, found {ncol}"):
        read_trajectory(path)


def test_restart_duplicates_removed_with_warning(tmp_path):
    t = [0.0, 0.1, 0.2, 0.3, 0.2, 0.3, 0.4]
    data = make_data(t=t)
    path = write_tab(tmp_path / "trajectory_initmbid_1_pid_2.tab", data)
    with pytest.warns(UserWarning, match="dropped 2 rows"):
        tr = read_trajectory(path)
    np.testing.assert_allclose(tr.t, [0.0, 0.1, 0.2, 0.3, 0.4])
    assert tr.n_dropped == 2
    np.testing.assert_allclose(tr.x[2], data[4, 1:4], rtol=1e-8)  # last-written copy of t=0.2 kept


def test_partially_written_last_line_is_dropped(tmp_path):
    data = make_data(n=5)
    path = tmp_path / "trajectory_initmbid_1_pid_2.tab"
    text = "\n".join(" ".join(f"{v:.9g}" for v in row) for row in data) + "\n"
    path.write_text(text + " ".join(f"{v:.9g}" for v in data[0])[:25])  # interrupted mid-row
    with pytest.warns(UserWarning, match="incomplete last line"):
        tr = read_trajectory(path)
    assert len(tr) == 5 and tr.has_fields
    np.testing.assert_allclose(tr.u, data[:, 4:7], rtol=1e-8)
    # complete-looking last row without newline: the last number may be truncated, so it is dropped too
    path.write_text(text.rstrip("\n"))
    with pytest.warns(UserWarning, match="incomplete last line"):
        assert len(read_trajectory(path)) == 4


def test_nan_rows_dropped_and_empty_files(tmp_path):
    data = make_data(n=6)
    data[2, 5] = np.nan
    data[4, 11] = np.inf
    path = write_tab(tmp_path / "trajectory_initmbid_1_pid_2.tab", data)
    with pytest.warns(UserWarning, match="dropped 2 rows containing NaN"):
        tr = read_trajectory(path)
    np.testing.assert_allclose(tr.t, data[[0, 1, 3, 5], 0], rtol=1e-8)
    assert tr.n_dropped == 2 and np.all(np.isfinite(tr.cE))
    empty = tmp_path / "trajectory_initmbid_1_pid_3.tab"
    empty.write_text("")
    with pytest.raises(ValueError, match="trajectory_initmbid_1_pid_3.tab is empty"):
        read_trajectory(empty)
    empty.write_text("\n  \n")
    with pytest.raises(ValueError, match="is empty"):
        read_trajectory(empty)
    empty.write_text("0.5 nan nan nan nan nan nan\n")
    with pytest.warns(UserWarning, match="NaN"), pytest.raises(ValueError, match="is empty"):
        read_trajectory(empty)


def test_select_propagates_n_dropped(tmp_path):
    path = write_tab(tmp_path / "trajectory_initmbid_1_pid_2.tab", make_data(t=[0.0, 0.1, 0.2, 0.1, 0.2, 0.3]))
    with pytest.warns(UserWarning):
        tr = read_trajectory(path)
    assert tr.n_dropped == 2
    assert tr.select(tr.t > 0.05).n_dropped == 2
    assert tr.time_window(0.0, 0.2).n_dropped == 2


def test_load_trajectories_collect_errors(tmp_path):
    run = _write_run(tmp_path, n_files=3)
    files = find_trajectory_files(run)
    files[1].write_text("")  # unreadable
    with pytest.raises((ValueError, RuntimeError), match="is empty"):  # parallel_map may wrap the error
        load_trajectories(run, backend="serial")
    for backend in ("serial", "thread"):
        results = load_trajectories(run, backend=backend, n_workers=2, on_error="collect")
        assert len(results) == 3
        assert isinstance(results[1], TaskError) and "is empty" in str(results[1].error)
        good = [r for r in results if not isinstance(r, TaskError)]
        assert [tr.path for tr in good] == [files[0], files[2]]


def test_select_and_time_window(tmp_path):
    data = make_data(n=11)
    tr = read_trajectory(write_tab(tmp_path / "trajectory_initmbid_3_pid_4.tab", data))
    sub = tr.time_window(0.25, 0.65)
    np.testing.assert_allclose(sub.t, [0.3, 0.4, 0.5, 0.6])
    np.testing.assert_allclose(sub.cE, tr.cE[3:7])
    assert sub.pid == 4 and sub.init_mbid == 3 and sub.path == tr.path
    sl = tr.select(slice(0, 2))
    assert len(sl) == 2 and sl.B.shape == (2, 3)
    no_fields = Trajectory(tr.t, tr.x, tr.u).select(tr.t > 0.5)
    assert no_fields.B is None and len(no_fields) == 5


def _write_run(tmp_path, n_files=4, ncol=13, n=8) -> Path:
    run = tmp_path / "run0001"
    sub = run / "high_energy_particles"
    sub.mkdir(parents=True)
    t = np.linspace(0, 2, n)
    for p in range(n_files):
        write_tab(sub / f"trajectory_initmbid_{p + 10}_pid_{100 + p}.tab", make_data(t=t, ncol=ncol, seed=p))
    (sub / "notes.txt").write_text("not a trajectory")
    return run


@pytest.mark.parametrize("backend", ["serial", "thread"])
def test_load_trajectories_directory(tmp_path, backend):
    run = _write_run(tmp_path)
    files = find_trajectory_files(run)
    assert len(files) == 4 and all(f.suffix == ".tab" for f in files)
    trajs = load_trajectories(run, backend=backend, n_workers=2)
    assert [tr.path for tr in trajs] == files  # input order preserved
    assert sorted(tr.pid for tr in trajs) == [100, 101, 102, 103]
    # also from an explicit list of paths
    again = load_trajectories(files[:2], backend=backend)
    np.testing.assert_array_equal(again[1].u, trajs[1].u)
    # a single file given as str or Path
    for single in (files[3], str(files[3])):
        (one,) = load_trajectories(single, backend=backend)
        np.testing.assert_array_equal(one.x, trajs[3].x)


def test_load_trajectories_empty_directory(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_trajectories(tmp_path, backend="serial")


def test_stack_trajectories(tmp_path):
    trajs = load_trajectories(_write_run(tmp_path, n_files=3, n=5), backend="serial")
    st = stack_trajectories(trajs)
    assert st["x"].shape == (3, 5, 3) and st["u"].shape == (3, 5, 3)
    assert st["B"].shape == (3, 5, 3) and st["cE"].shape == (3, 5, 3)
    assert st["t"].shape == (5,)
    np.testing.assert_array_equal(st["u"][2], trajs[2].u)
    np.testing.assert_array_equal(st["pid"], [tr.pid for tr in trajs])


def test_stack_without_fields_omits_B(tmp_path):
    trajs = load_trajectories(_write_run(tmp_path, n_files=2, ncol=7), backend="serial")
    st = stack_trajectories(trajs)
    assert "B" not in st and "cE" not in st


def test_stack_mismatched_time_raises(tmp_path):
    trajs = load_trajectories(_write_run(tmp_path, n_files=2, n=6), backend="serial")
    shorter = trajs[1].select(slice(0, 5))
    with pytest.raises(ValueError, match="time grids differ"):
        stack_trajectories([trajs[0], shorter])
    shifted = Trajectory(trajs[1].t + 0.01, trajs[1].x, trajs[1].u)
    with pytest.raises(ValueError, match="time grids differ"):
        stack_trajectories([trajs[0], shifted])
    # trimming both to a common window makes them stackable again
    st = stack_trajectories([trajs[0].select(slice(0, 5)), shorter])
    assert st["x"].shape == (2, 5, 3)
