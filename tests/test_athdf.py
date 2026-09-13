"""Unit tests for shearpic.io.athdf on tiny synthetic Athena++-style HDF5 files."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from shearpic.io.athdf import Snapshot, list_snapshots, snapshot_number

NX = (8, 6, 4)          # root grid (nx1, nx2, nx3)
MB = (4, 3, 2)          # meshblock size -> 2 x 2 x 2 = 8 blocks
BOUNDS = ((-2.0, 2.0), (0.0, 3.0), (-1.0, 1.0))
PRIM = ["rho", "vel1", "vel2", "vel3", "np"]
BFIELD = ["Bcc1", "Bcc2", "Bcc3"]


def analytic(i, j, k):
    """Unique value per global cell so any mis-tiling is detected exactly."""
    return i + 10.0 * j + 100.0 * k


def write_athdf(path: Path, nx=NX, mb=MB, time=12.5, cycle=345, max_level=0, seed=0) -> Path:
    """Write a minimal uniform-mesh .athdf with shuffled meshblock order.

    Variable ``m`` of dataset ``prim`` holds ``analytic + 1000 m``; ``B`` variable m holds
    ``-(analytic + 1000 m)``.
    """
    rng = np.random.default_rng(seed)
    n1, n2, n3 = (n // b for n, b in zip(nx, mb))
    locs = np.array([(l1, l2, l3) for l3 in range(n3) for l2 in range(n2) for l1 in range(n1)], dtype=">i8")
    locs = locs[rng.permutation(len(locs))]
    nb = len(locs)
    prim = np.empty((len(PRIM), nb, mb[2], mb[1], mb[0]), dtype=np.float32)
    bfld = np.empty((len(BFIELD), nb, mb[2], mb[1], mb[0]), dtype=np.float32)
    for b, (l1, l2, l3) in enumerate(locs):
        k, j, i = np.meshgrid(np.arange(l3 * mb[2], (l3 + 1) * mb[2]), np.arange(l2 * mb[1], (l2 + 1) * mb[1]),
                              np.arange(l1 * mb[0], (l1 + 1) * mb[0]), indexing="ij")  # Athena order (z, y, x)
        f = analytic(i, j, k)
        for m in range(len(PRIM)):
            prim[m, b] = f + 1000.0 * m
        for m in range(len(BFIELD)):
            bfld[m, b] = -(f + 1000.0 * m)
    with h5py.File(path, "w") as h:
        a = h.attrs
        a["Coordinates"] = np.bytes_("cartesian")
        a["DatasetNames"] = np.array([b"prim", b"B"], dtype="S21")
        a["NumVariables"] = np.array([len(PRIM), len(BFIELD)], dtype=">i4")
        a["VariableNames"] = np.array([n.encode() for n in PRIM + BFIELD], dtype="S21")
        a["RootGridSize"] = np.array(nx, dtype=">i4")
        a["MeshBlockSize"] = np.array(mb, dtype=">i4")
        for d in range(3):
            a[f"RootGridX{d + 1}"] = np.array([*BOUNDS[d], 1.0], dtype=np.float32)
        a["Time"] = time
        a["NumCycles"] = cycle
        a["NumMeshBlocks"] = nb
        a["MaxLevel"] = max_level
        h["prim"] = prim
        h["B"] = bfld
        h["LogicalLocations"] = locs
        h["Levels"] = np.zeros(nb, dtype=">i4")
    return path


@pytest.fixture
def snap_path(tmp_path) -> Path:
    return write_athdf(tmp_path / "prob.out2.00003.athdf")


def expected(nx=NX, offset=0.0, sign=1.0):
    i, j, k = np.meshgrid(*(np.arange(n) for n in nx), indexing="ij")
    return sign * (analytic(i, j, k) + offset)


def test_metadata(snap_path):
    s = Snapshot.open(snap_path)
    assert s.time == 12.5 and s.cycle == 345
    assert s.variables == tuple(PRIM + BFIELD)
    assert s.shape == NX and s.block_shape == MB
    assert s.n_blocks == 8 and s.max_level == 0
    np.testing.assert_allclose(s.bounds, BOUNDS)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("nx, mb", [(NX, MB), ((12, 6, 2), (4, 3, 1)), ((4, 12, 1), (2, 3, 1))])
def test_assembly_exact_with_shuffled_blocks(tmp_path, seed, nx, mb):
    s = Snapshot.open(write_athdf(tmp_path / f"p.out2.{seed:05d}.athdf", nx=nx, mb=mb, seed=seed))
    data = s.read(["rho", "vel2", "np"])
    assert data["rho"].shape == nx  # (x, y, z) order
    np.testing.assert_array_equal(data["rho"], expected(nx))
    np.testing.assert_array_equal(data["vel2"], expected(nx, offset=2000.0))
    np.testing.assert_array_equal(data["np"], expected(nx, offset=4000.0))
    # (x, y, z) indexing: value at [i, j, k] = i + 10 j + 100 k
    k = nx[2] - 1
    assert data["rho"][3, 2, k] == 3 + 20 + 100 * k
    assert data["rho"].flags["C_CONTIGUOUS"]


def test_lookup_across_datasets(snap_path):
    s = Snapshot.open(snap_path)
    out = s.read(["Bcc2", "vel1"])
    np.testing.assert_array_equal(out["Bcc2"], expected(offset=1000.0, sign=-1.0))  # 2nd variable of 'B'
    np.testing.assert_array_equal(out["vel1"], expected(offset=1000.0))
    np.testing.assert_array_equal(s["Bcc3"], expected(offset=2000.0, sign=-1.0))
    assert set(s.read()) == set(PRIM + BFIELD)  # names=None reads everything


def test_dtype_and_squeeze(tmp_path):
    s = Snapshot.open(write_athdf(tmp_path / "flat.out1.00000.athdf", nx=(8, 6, 1), mb=(4, 3, 1)))
    d32 = s.read("rho")["rho"]
    assert d32.dtype == np.float32 and d32.shape == (8, 6, 1)
    d64 = s.read("rho", dtype=np.float64, squeeze=True)["rho"]
    assert d64.dtype == np.float64 and d64.shape == (8, 6)
    np.testing.assert_array_equal(d64, expected(nx=(8, 6, 1))[:, :, 0])
    raw = s.read("rho", dtype=None)["rho"]
    assert raw.dtype == np.float32


def test_missing_variable_keyerror_lists_available(snap_path):
    s = Snapshot.open(snap_path)
    with pytest.raises(KeyError) as err:
        s.read(["rho", "pressure"])
    msg = str(err.value)
    assert "pressure" in msg and "Bcc1" in msg and "rho" in msg


def test_refined_mesh_not_implemented(tmp_path):
    s = Snapshot.open(write_athdf(tmp_path / "amr.out2.00000.athdf", max_level=1))
    with pytest.raises(NotImplementedError, match="MaxLevel=1"):
        s.read("rho")


def test_edges_and_centers(snap_path):
    s = Snapshot.open(snap_path)
    e = s.edges(0)
    np.testing.assert_allclose(e, np.linspace(-2, 2, 9))
    np.testing.assert_allclose(s.centers(1), (np.arange(6) + 0.5) * 0.5)
    np.testing.assert_allclose(s.x, 0.5 * (e[:-1] + e[1:]))
    assert s.y.shape == (6,) and s.z.shape == (4,)
    np.testing.assert_array_equal(s.edges(2, bounds=(0.0, 4.0)), np.linspace(0.0, 4.0, 5))


def test_snapshot_is_hashable_and_equality_ignores_locations(snap_path):
    a, b = Snapshot.open(snap_path), Snapshot.open(snap_path)
    assert a == b and hash(a) == hash(b)
    assert {a: 1}[b] == 1
    assert "_locations" not in repr(a)


def test_files_opened_read_only_without_locking(snap_path, monkeypatch):
    import shearpic.io.athdf as athdf_mod

    calls = []
    real = h5py.File

    def spy(*args, **kwargs):
        calls.append((args[1:], kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(athdf_mod.h5py, "File", spy)
    s = Snapshot.open(snap_path)
    s.read("rho")
    assert len(calls) == 2
    assert all(args == ("r",) and kw.get("locking") is False for args, kw in calls)


def test_open_while_file_is_held_with_locking(snap_path):
    """yt keeps files open with the default locking flags; reading must still work."""
    with h5py.File(snap_path, "r", locking=True):
        s = Snapshot.open(snap_path)
        np.testing.assert_array_equal(s["rho"], expected())


def test_assemble_fills_requested_dtype_directly(snap_path):
    s = Snapshot.open(snap_path)
    with h5py.File(snap_path, "r") as f:
        blocks, loc = f["prim"][0], f["LogicalLocations"][:]
    out = s._assemble(blocks, loc, np.float64)
    assert out.dtype == np.float64 and out.flags["C_CONTIGUOUS"] and out.shape == NX
    np.testing.assert_array_equal(out, expected())
    assert s._assemble(blocks, loc).dtype == np.float32
    with pytest.raises(ValueError, match="do not tile"):
        s._assemble(blocks[:-1], loc[:-1])
    dup = loc.copy()
    dup[1] = dup[0]
    with pytest.raises(ValueError, match="do not tile"):
        s._assemble(blocks, dup)


def test_snapshot_is_picklable(snap_path):
    import pickle

    s = pickle.loads(pickle.dumps(Snapshot.open(snap_path)))
    np.testing.assert_array_equal(s["rho"], expected())


def test_list_snapshots_ordering_and_filter(tmp_path):
    names = ["prob.out2.00010.athdf", "prob.out2.00002.athdf", "prob.out1.00001.athdf",
             "prob.out2.00000.athdf", "prob.out1.00000.athdf"]
    for n in names:
        (tmp_path / n).touch()
    (tmp_path / "prob.out2.00001.xdmf").touch()
    (tmp_path / "prob.hst").touch()
    assert [p.name for p in list_snapshots(tmp_path, "out2")] == [
        "prob.out2.00000.athdf", "prob.out2.00002.athdf", "prob.out2.00010.athdf"]
    assert [p.name for p in list_snapshots(tmp_path)] == [
        "prob.out1.00000.athdf", "prob.out1.00001.athdf",
        "prob.out2.00000.athdf", "prob.out2.00002.athdf", "prob.out2.00010.athdf"]
    assert list_snapshots(tmp_path, "out5") == []


def test_snapshot_number():
    assert snapshot_number("org.stir.feedback.out2.00006.athdf") == 6
    assert snapshot_number(Path("/a/b.out1.12345.athdf")) == 12345
    with pytest.raises(ValueError):
        snapshot_number("prob.hst")


def test_old_hdf5_without_locking_option_falls_back(snap_path, monkeypatch):
    import shearpic.io.athdf as athdf

    real = h5py.File

    def old_file(path, mode="r", **kw):
        if "locking" in kw:
            raise ValueError("HDF5 version >= 1.12.1 or 1.10.x >= 1.10.7 required for file locking options.")
        return real(path, mode, **kw)

    monkeypatch.setattr(athdf.h5py, "File", old_file)
    snap = Snapshot.open(snap_path)
    assert snap.read("rho")["rho"].shape == snap.shape


# ------------------------------------------------------------------ select_snapshots
def _stream(tmp_path, times, numbers=None, stream="out2"):
    numbers = range(len(times)) if numbers is None else numbers
    for n, t in zip(numbers, times):
        write_athdf(tmp_path / f"prob.{stream}.{n:05d}.athdf", nx=(4, 3, 2), mb=(4, 3, 2), time=t)
    return tmp_path


def test_default_time_tolerance():
    from shearpic.io.athdf import default_time_tolerance

    assert default_time_tolerance(40.0, 10.0) == pytest.approx(0.01)
    assert default_time_tolerance(1e5, 1.0) == pytest.approx(0.1)          # 1e-6 relative dominates
    assert default_time_tolerance(0.0, None) == pytest.approx(1e-6)
    assert default_time_tolerance(300.0, None) == pytest.approx(3e-4)


def test_select_snapshots_nearest_not_first_within_tolerance(tmp_path):
    from shearpic.io.athdf import select_snapshots

    # file 00001 (the expected number for t ~ 10) is within the tolerance, but 00002 is nearer
    run = _stream(tmp_path, [0.0, 10.0, 10.004, 20.0008])
    got = select_snapshots(run, [10.003, 20.0, 0.0], dt=10.0)
    assert [p.name for p in got] == ["prob.out2.00002.athdf", "prob.out2.00003.athdf", "prob.out2.00000.athdf"]
    # float32 overshoot of an output time is accepted with the default tolerance (1e-3 dt) ...
    assert select_snapshots(run, [20.0], dt=10.0)[0].name == "prob.out2.00003.athdf"
    # ... a time between outputs is not
    with pytest.raises(FileNotFoundError, match=r"within 0.01 of t = 14 .*nearest local snapshot: prob.out2.00002.*"
                                                 r"t = 14 is not an output time of this stream \(dt = 10\)"):
        select_snapshots(run, [14.0], dt=10.0)
    assert select_snapshots(run, [10.004])[0].name == "prob.out2.00002.athdf"   # no dt: 1e-6 relative
    with pytest.raises(FileNotFoundError, match="within 1.00041e-05 of t = 10.0041"):
        select_snapshots(run, [10.0041])
    # an explicit tolerance overrides the default
    assert select_snapshots(run, [14.0], dt=10.0, tol=5.0)[0].name == "prob.out2.00002.athdf"


def test_select_snapshots_refuses_two_times_on_one_file(tmp_path):
    from shearpic.io.athdf import select_snapshots

    run = _stream(tmp_path, [0.0, 50.0, 100.0])
    with pytest.raises(ValueError, match=r"t = 40 and t = 60 both resolve to prob.out2.00001.athdf \(t = 50\)"):
        select_snapshots(run, [0, 40, 60], dt=50.0, tol=25.0)
    with pytest.raises(ValueError, match="both resolve"):
        select_snapshots(run, [50, 50], dt=50.0)
    with pytest.raises(FileNotFoundError, match=r"no \*.out3.\*.athdf"):
        select_snapshots(run, [0], output="out3")


def test_select_snapshots_placeholders(tmp_path, monkeypatch):
    import shearpic.io.athdf as athdf

    run = _stream(tmp_path, [0.0, 10.0, 20.0, 30.0])
    placeholders = {"prob.out2.00002.athdf", "prob.out2.00003.athdf"}
    monkeypatch.setattr(athdf, "is_dataless", lambda p: Path(p).name in placeholders)
    opened = []
    real_open = athdf.Snapshot.open.__func__

    def spy(cls, path):
        opened.append(Path(path).name)
        return real_open(cls, path)

    monkeypatch.setattr(athdf.Snapshot, "open", classmethod(spy))
    # local files only are read; a time only a placeholder can provide is refused
    assert athdf.select_snapshots(run, [10.0], dt=10.0)[0].name == "prob.out2.00001.athdf"
    assert set(opened) == {"prob.out2.00000.athdf", "prob.out2.00001.athdf"}
    with pytest.raises(athdf.DatalessFileError, match="prob.out2.00003.athdf .*--allow-download"):
        athdf.select_snapshots(run, [30.0], dt=10.0)
    # with allow_download only the expected placeholder is opened (downloaded)
    opened.clear()
    with pytest.warns(UserWarning, match="00003.athdf is an online-only placeholder"):
        got = athdf.select_snapshots(run, [30.0], dt=10.0, allow_download=True)
    assert got[0].name == "prob.out2.00003.athdf"
    assert opened == ["prob.out2.00000.athdf", "prob.out2.00001.athdf", "prob.out2.00003.athdf"]


def test_select_snapshots_off_cadence_never_downloads(tmp_path, monkeypatch):
    """A time that is not an output time fails at once, without touching placeholders."""
    import shearpic.io.athdf as athdf

    run = _stream(tmp_path, [0.0, 10.0, 20.0, 30.0])
    placeholders = {"prob.out2.00002.athdf", "prob.out2.00003.athdf"}
    monkeypatch.setattr(athdf, "is_dataless", lambda p: Path(p).name in placeholders)
    opened = []
    real_open = athdf.Snapshot.open.__func__

    def spy(cls, path):
        opened.append(Path(path).name)
        return real_open(cls, path)

    monkeypatch.setattr(athdf.Snapshot, "open", classmethod(spy))
    for allow in (False, True):
        opened.clear()
        with pytest.raises(FileNotFoundError, match="not an output time") as info:
            athdf.select_snapshots(run, [25.0], dt=10.0, allow_download=allow)
        assert not isinstance(info.value, athdf.DatalessFileError)
        assert not placeholders & set(opened)
