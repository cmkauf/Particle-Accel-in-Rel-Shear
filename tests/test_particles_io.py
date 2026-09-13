"""Unit tests for shearpic.io.particles: round trips against the Athena++ fork's particle writers.

The writers mimic ``src/particles/particles.cpp`` of MHD-PIC-in-Athena-.  ``FormattedTableOutput``
writes a two-line header and ``%g`` values separated by two spaces.  ``BinaryOutput`` writes
``float[12]`` (meshblock then mesh bounds), ``float[2]`` (time, dt), ``long npar`` and 44-byte
records ``{float xp, yp, zp, vpx, vpy, vpz, dpar, property; long pid; int init_mbid}``.
"""

from __future__ import annotations

import struct
import warnings
from pathlib import Path

import numpy as np
import pytest

from shearpic.io.particles import (PARBIN_HEADER_BYTES, PARBIN_RECORD, PARTICLE_FIELDS, ParticleBlock,
                                   list_particle_files, read_parbin, read_particle_block, read_partab)


def _particles(n=5, seed=0):
    rng = np.random.default_rng(seed)
    return {
        "init_mbid": rng.integers(0, 1024, n),
        "pid": rng.integers(0, 2**40, n),
        "x": rng.uniform(-15, 15, (n, 3)),
        "u": rng.normal(0, 50, (n, 3)),
    }


def _g(v) -> str:
    return f"{v:g}"  # C++ ostream default for double: %g, precision 6


def write_partab(path: Path, time: float, p: dict) -> Path:
    lines = [f"# Athena++ particle data at time = {_g(time)}",
             "born_meshblock  particle_id  x  y  z  vx  vy  vz"]
    for k in range(len(p["pid"])):
        vals = [str(p["init_mbid"][k]), str(p["pid"][k]), *(_g(v) for v in p["x"][k]), *(_g(v) for v in p["u"][k])]
        lines.append("  ".join(vals))
    path.write_text("\n".join(lines) + "\n")
    return path


def write_parbin(path: Path, time: float, dt: float, p: dict, truncate: int = 0) -> bytes:
    box = [-1.0, 1.0, -2.0, 2.0, -0.5, 0.5, -15.0, 15.0, -60.0, 60.0, -0.5, 0.5]
    buf = struct.pack("<12f", *box) + struct.pack("<2f", time, dt)
    n = len(p["pid"])
    buf += struct.pack("<q", n)
    for k in range(n):
        buf += struct.pack("<8fqi", *p["x"][k], *p["u"][k], 1.0, 0.0, int(p["pid"][k]), int(p["init_mbid"][k]))
    if truncate:
        buf = buf[:-truncate]
    path.write_bytes(buf)
    return buf


def test_record_layout_matches_fork_struct():
    assert PARBIN_RECORD.itemsize == 44 == struct.calcsize("<8fqi")
    assert PARBIN_RECORD.names == ("x", "y", "z", "ux", "uy", "uz", "dpar", "property", "pid", "init_mbid")
    assert PARBIN_RECORD.fields["pid"][1] == 32 and PARBIN_RECORD.fields["init_mbid"][1] == 40


def test_partab_round_trip(tmp_path):
    p = _particles()
    path = write_partab(tmp_path / "prob.block12.out4.00003.par.tab", 300.25, p)
    blk = read_partab(path)
    assert blk.time == 300.25 and blk.dt is None and len(blk) == 5
    np.testing.assert_array_equal(blk.pid, p["pid"])
    np.testing.assert_array_equal(blk.init_mbid, p["init_mbid"])
    np.testing.assert_allclose(blk.x, p["x"], rtol=1e-5, atol=1e-5)  # 6 significant digits
    np.testing.assert_allclose(blk.u, p["u"], rtol=1e-5, atol=1e-4)
    assert blk.x.dtype == np.float64 and blk.pid.dtype == np.int64
    assert read_particle_block(path).pid.tolist() == blk.pid.tolist()


def test_partab_empty_block(tmp_path):
    path = write_partab(tmp_path / "prob.block0.out4.00000.par.tab", 0.0, _particles(n=0))
    blk = read_partab(path)
    assert len(blk) == 0 and blk.x.shape == (0, 3) and blk.u.shape == (0, 3)


def test_partab_bad_headers(tmp_path):
    good = write_partab(tmp_path / "a.par.tab", 1.0, _particles(n=2)).read_text().splitlines()
    bad_time = tmp_path / "b.par.tab"
    bad_time.write_text("\n".join(["# Athena++ particle data"] + good[1:]) + "\n")
    with pytest.raises(ValueError, match="time ="):
        read_partab(bad_time)
    bad_cols = tmp_path / "c.par.tab"
    bad_cols.write_text("\n".join([good[0], "pid  x  y  z  vx  vy  vz  mbid"] + good[2:]) + "\n")
    with pytest.raises(ValueError, match="unexpected header"):
        read_partab(bad_cols)


def test_parbin_round_trip(tmp_path):
    p = _particles(n=7, seed=3)
    path = tmp_path / "prob.block5.out5.00002.par.bin"
    write_parbin(path, time=150.5, dt=1.25e-3, p=p)
    blk = read_parbin(path)
    assert len(blk) == 7
    assert blk.time == pytest.approx(150.5) and blk.dt == pytest.approx(1.25e-3, rel=1e-6)
    np.testing.assert_array_equal(blk.pid, p["pid"])  # int64 ids above 2^32 survive
    np.testing.assert_array_equal(blk.init_mbid, p["init_mbid"])
    np.testing.assert_allclose(blk.x, p["x"].astype(np.float32))
    np.testing.assert_allclose(blk.u, p["u"].astype(np.float32))
    assert blk.u.dtype == np.float64
    assert read_particle_block(path).time == blk.time


def test_parbin_zero_particles(tmp_path):
    path = tmp_path / "e.par.bin"
    write_parbin(path, 0.0, 0.1, _particles(n=0))
    assert len(read_parbin(path)) == 0


def test_parbin_truncated(tmp_path):
    path = tmp_path / "t.par.bin"
    write_parbin(path, 1.0, 0.1, _particles(n=4), truncate=10)
    with pytest.raises(ValueError, match="header says 4 particles but only 3 records"):
        read_parbin(path)


def test_read_particle_block_rejects_other_files(tmp_path):
    with pytest.raises(ValueError, match="not a particle file"):
        read_particle_block(tmp_path / "prob.out2.00000.athdf")


def test_list_particle_files(tmp_path):
    names = [f"prob.block{b}.out4.{n:05d}.par.tab" for b in (0, 1, 10) for n in (0, 3)]
    names += [f"prob.block{b}.out5.00003.par.bin" for b in (0, 2)]
    for name in names:
        (tmp_path / name).touch()
    (tmp_path / "prob.out2.00003.athdf").touch()
    (tmp_path / "prob.hst").touch()
    with pytest.warns(UserWarning, match="00003 .* both .par.tab and .par.bin"):
        assert len(list_particle_files(tmp_path)) == 8
    with pytest.warns(UserWarning, match="counts|twice"):
        three = list_particle_files(tmp_path, file_number=3)
    assert len(three) == 5 and all(".00003.par." in f.name for f in three)
    assert [f.name for f in list_particle_files(tmp_path, 3, kind="bin")] == [
        "prob.block0.out5.00003.par.bin", "prob.block2.out5.00003.par.bin"]
    assert len(list_particle_files(tmp_path, 0, kind="tab")) == 3
    assert list_particle_files(tmp_path, 7) == []


# ----------------------------------------------------------- kind='*' mixing
def test_list_particle_files_warns_only_for_the_same_output_in_both_formats(tmp_path):
    for b in (0, 1):
        (tmp_path / f"prob.block{b}.out4.00001.par.tab").touch()
        (tmp_path / f"prob.block{b}.out5.00002.par.bin").touch()
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # different outputs in different formats: fine
        assert len(list_particle_files(tmp_path)) == 4
        assert len(list_particle_files(tmp_path, 1)) == 2
    (tmp_path / "prob.block0.out5.00001.par.bin").touch()
    with pytest.warns(UserWarning, match="00001"):
        assert len(list_particle_files(tmp_path, 1)) == 3
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert len(list_particle_files(tmp_path, 1, kind="tab")) == 2  # an explicit kind never warns
        assert len(list_particle_files(tmp_path, 1, warn_mixed=False)) == 3


# ------------------------------------------------------------- field selection
def _parbin(tmp_path, n=7, seed=3):
    p = _particles(n=n, seed=seed)
    path = tmp_path / f"prob.block5.out5.{n:05d}.par.bin"
    write_parbin(path, time=150.5, dt=1.25e-3, p=p)
    return path, p


def test_parbin_fields_subset(tmp_path):
    path, p = _parbin(tmp_path)
    blk = read_parbin(path, fields=("u",))
    assert blk.x is None and blk.pid is None and blk.init_mbid is None
    assert len(blk) == 7 and blk.n == 7 and blk.time == pytest.approx(150.5)
    np.testing.assert_array_equal(blk.u, p["u"].astype(np.float32).astype(np.float64))
    # independent in-memory float64 copies, not views of the memory-mapped file
    assert type(blk.u) is np.ndarray and blk.u.dtype == np.float64 and blk.u.flags.owndata
    assert blk.u.flags.c_contiguous and blk.u.shape == (7, 3)
    ids = read_parbin(path, fields="pid")  # a single name is accepted
    assert ids.u is None and type(ids.pid) is np.ndarray and ids.pid.dtype == np.int64
    np.testing.assert_array_equal(ids.pid, p["pid"])
    counted = read_parbin(path, fields=())
    assert len(counted) == 7 and all(getattr(counted, f) is None for f in PARTICLE_FIELDS)
    full = read_parbin(path)  # default: all fields
    assert all(getattr(full, f) is not None for f in PARTICLE_FIELDS)
    np.testing.assert_array_equal(full.u, blk.u)
    assert full.init_mbid.dtype == np.int64
    with pytest.raises(ValueError, match="unknown particle fields"):
        read_parbin(path, fields=("u", "vx"))


def test_parbin_header_constant_and_short_files(tmp_path):
    assert PARBIN_HEADER_BYTES == 64
    path, _ = _parbin(tmp_path, n=3)
    assert path.stat().st_size == PARBIN_HEADER_BYTES + 3 * PARBIN_RECORD.itemsize
    short = tmp_path / "short.par.bin"
    short.write_bytes(path.read_bytes()[:40])
    with pytest.raises(ValueError, match="header"):
        read_parbin(short)
    trunc = tmp_path / "t.par.bin"
    write_parbin(trunc, 1.0, 0.1, _particles(n=4), truncate=10)
    with pytest.raises(ValueError, match="header says 4 particles but only 3 records"):
        read_parbin(trunc, fields=("u",))
    empty = tmp_path / "e.par.bin"
    write_parbin(empty, 0.0, 0.1, _particles(n=0))
    blk = read_parbin(empty, fields=("u", "pid"))
    assert len(blk) == 0 and blk.u.shape == (0, 3) and blk.pid.shape == (0,) and blk.x is None


def test_partab_fields_subset(tmp_path):
    p = _particles()
    path = write_partab(tmp_path / "prob.block12.out4.00003.par.tab", 300.25, p)
    blk = read_partab(path, fields=("pid", "u"))
    assert blk.x is None and blk.init_mbid is None and len(blk) == 5
    np.testing.assert_array_equal(blk.pid, p["pid"])
    np.testing.assert_allclose(blk.u, p["u"], rtol=1e-5, atol=1e-4)
    assert blk.u.dtype == np.float64 and blk.pid.dtype == np.int64
    assert len(read_partab(path, fields=())) == 5
    assert read_particle_block(path, fields=("x",)).u is None
    assert read_particle_block(tmp_path / "prob.block12.out4.00003.par.tab").u is not None
    empty = write_partab(tmp_path / "prob.block0.out4.00000.par.tab", 0.0, _particles(n=0))
    e = read_partab(empty, fields=("u",))
    assert len(e) == 0 and e.u.shape == (0, 3)
    with pytest.raises(ValueError, match="unknown particle fields"):
        read_partab(path, fields=("gamma",))


def test_particle_block_len_without_count():
    blk = ParticleBlock(time=0.0, x=None, u=np.zeros((4, 3)), pid=None, init_mbid=None)
    assert len(blk) == 4
    with pytest.raises(ValueError):
        len(ParticleBlock(0.0, None, None, None, None))


def test_parbin_u_only_peak_memory(tmp_path):
    """fields=('u',) allocates only the float64 (n, 3) momenta: no in-memory copy of the records."""
    import tracemalloc

    n = 200_000
    rec = np.zeros(n, dtype=PARBIN_RECORD)
    rec["ux"] = np.arange(n, dtype=np.float32)
    path = tmp_path / "big.block0.out5.00001.par.bin"
    with open(path, "wb") as fh:
        np.zeros(12, "<f4").tofile(fh)
        np.array([1.0, 0.1], "<f4").tofile(fh)
        np.array([n], "<i8").tofile(fh)
        rec.tofile(fh)
    del rec
    tracemalloc.start()
    try:
        blk = read_parbin(path, fields=("u",))
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    assert blk.u[-1, 0] == n - 1
    assert peak < 1.25 * 24 * n  # reading all 44-byte records into memory first would exceed 68 n


# ------------------------------------------------ streams, output numbers, header times
def test_parse_particle_filename():
    from shearpic.io.particles import parse_particle_filename

    n = parse_particle_filename("/x/org.stir.feedback.block1023.out4.00012.par.tab")
    assert (n.basename, n.gid, n.file_id, n.number, n.kind) == ("org.stir.feedback", 1023, "out4", 12, "tab")
    assert n.stream == ("org.stir.feedback", "out4", "tab")
    assert parse_particle_filename("org.stir.feedback.out2.00012.athdf") is None
    assert parse_particle_filename("prob.block1.out4.12.par.tab") is None


def test_list_filters_and_single_stream_selection(tmp_path):
    from shearpic.io.particles import particle_output_files, particle_output_numbers

    for name in ["prob.block0.out4.00000.par.tab", "prob.block1.out4.00000.par.tab", "prob.block0.out4.00002.par.tab",
                 "prob.block0.out5.00002.par.bin", "old.block0.out4.00002.par.tab"]:
        (tmp_path / name).touch()
    assert [p.name for p in list_particle_files(tmp_path, 2, file_id="out4", basename="prob")] == [
        "prob.block0.out4.00002.par.tab"]
    assert len(list_particle_files(tmp_path, None, "tab", file_id="out4")) == 4
    assert [p.name for p in particle_output_files(tmp_path, 0, "tab")] == ["prob.block0.out4.00000.par.tab",
                                                                           "prob.block1.out4.00000.par.tab"]
    with pytest.raises(ValueError, match="kind='tab' or kind='bin'"):
        particle_output_files(tmp_path, 2)
    with pytest.raises(ValueError, match="file_id=... or basename"):
        particle_output_files(tmp_path, 2, "tab")
    with pytest.raises(FileNotFoundError):
        particle_output_files(tmp_path, 7, "tab")
    with pytest.raises(ValueError, match="kind must be"):
        list_particle_files(tmp_path, 0, "hdf5")
    assert particle_output_numbers(tmp_path, "tab", basename="prob") == [0, 2]
    assert particle_output_numbers(tmp_path, "bin") == [2]
    with pytest.raises(ValueError, match="several particle outputs"):
        particle_output_numbers(tmp_path, "tab")
    assert particle_output_numbers(tmp_path / "none", "tab") == []


def test_read_particle_time_tab_and_bin(tmp_path):
    from shearpic.io.particles import read_particle_time

    p = _particles(3)
    tab = write_partab(tmp_path / "prob.block0.out4.00001.par.tab", 1200.00034, p)
    assert read_particle_time(tab) == pytest.approx(1200.00034)
    binp = tmp_path / "prob.block0.out5.00001.par.bin"
    write_parbin(binp, 150.5, 0.01, p)
    assert read_particle_time(binp) == pytest.approx(150.5)
    (tmp_path / "bad.block0.out4.00001.par.tab").write_text("no time\n")
    with pytest.raises(ValueError, match="time"):
        read_particle_time(tmp_path / "bad.block0.out4.00001.par.tab")
    with pytest.raises(ValueError, match="shorter"):
        (tmp_path / "s.block0.out5.00001.par.bin").write_bytes(b"12")
        read_particle_time(tmp_path / "s.block0.out5.00001.par.bin")
    with pytest.raises(ValueError, match="not a particle file"):
        read_particle_time(tmp_path / "x.txt")


def test_select_particle_output_by_time(tmp_path, monkeypatch):
    import shearpic.io._util as util
    from shearpic.io.particles import select_particle_output

    p = _particles(2)
    times = {0: 0.0, 1: 100.02, 2: 200.01, 4: 400.05}  # output 3 missing: numbers are not times / dt
    for n, t in times.items():
        for b in (0, 1):
            write_partab(tmp_path / f"prob.block{b}.out4.{n:05d}.par.tab", t, p)
    opened = []
    import shearpic.io.particles as pio

    real = pio.read_particle_time
    monkeypatch.setattr(pio, "read_particle_time", lambda path: opened.append(Path(path).name) or real(path))
    assert select_particle_output(tmp_path, 200, dt=100.0) == (2, pytest.approx(200.01))
    assert opened == ["prob.block0.out4.00002.par.tab"]  # the expected number is tried first
    assert select_particle_output(tmp_path, 395.0, dt=100.0)[0] == 4
    with pytest.raises(FileNotFoundError, match="closest is output"):
        select_particle_output(tmp_path, 300.0, dt=100.0)  # 3 is missing, 200 and 400 are > dt/2 away
    assert select_particle_output(tmp_path, 100.0, tol=0.1)[0] == 1  # no cadence: all headers, nearest
    with pytest.raises(FileNotFoundError):
        select_particle_output(tmp_path, 100.0)  # default tolerance without dt is 1e-6 relative
    # online-only placeholders are never opened unless allowed
    monkeypatch.setattr(util, "is_dataless", lambda path: "00002" in str(path))
    with pytest.raises(FileNotFoundError, match="placeholders"):
        select_particle_output(tmp_path, 200.0, dt=100.0)
    assert select_particle_output(tmp_path, 200.0, dt=100.0, allow_download=True)[0] == 2


def test_stream_tag_and_single_stream(tmp_path):
    from shearpic.io.particles import parse_particle_filename, particle_output_stream, stream_tag

    n = parse_particle_filename("org.stir.feedback.block7.out4.00012.par.tab")
    assert n.stream_tag == "org.stir.feedback.out4.tab" == stream_tag(("org.stir.feedback", "out4", "tab"))
    assert stream_tag({"basename": "prob", "file_id": "out5", "kind": "bin"}) == "prob.out5.bin"
    with pytest.raises(ValueError, match="incomplete particle stream"):
        stream_tag({"basename": "prob", "file_id": None, "kind": "tab"})
    for name in ["prob.block0.out4.00000.par.tab", "prob.block0.out5.00000.par.bin", "old.block0.out4.00001.par.tab"]:
        (tmp_path / name).touch()
    assert particle_output_stream(tmp_path, "bin") == ("prob", "out5", "bin")
    assert particle_output_stream(tmp_path, "tab", basename="old") == ("old", "out4", "tab")
    with pytest.raises(ValueError, match="several particle outputs"):
        particle_output_stream(tmp_path, "tab")
    with pytest.raises(FileNotFoundError, match="no particle files"):
        particle_output_stream(tmp_path, "bin", file_id="out9")
