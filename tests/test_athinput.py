"""Unit tests for shearpic.io.athinput (typed, block-aware athinput parsing and writing)."""

from __future__ import annotations

import hashlib

import pytest

from shearpic.io.athinput import AthInput, parse_athinput, parse_value, read_athinput, write_athinput

SAMPLE = """\
<comment>
problem   = Driven KH instability
reference =
configure = --prob kh_driven -b --eos isothermal -mpi -hdf5 --hdf5_path=$TACC_HDF5_DIR --cxx icpc

<job>
problem_id = org.stir.feedback  # problem ID: base name of output files

<output2>
file_type  = hdf5       # Binary data dump
variable   = prim       # variables to be output
dt         = 50.0       # time increment between outputs

<output1>
file_type  = hst        # History data dump
dt         = 0.5        # time increment between outputs

<output10>
file_type  = rst
dt         = 100.0

<time>
nlim       = -1         # cycle limit
tlim       = 600.0      # time limit

<mesh>
nx1        = 1024       # Number of zones in X1-direction
x1min      = -15.707963267948966192313216916398      # minimum value of X1
x1max      = 15.707963267948966192313216916398       # maximum value of X1
ix1_bc     = periodic   # inner-X1 boundary flag

<particles>
backreaction = true     # turn on/off the back reaction of the gas drag
delta_f_enable = false  # turn on/off the delta f method
charge_over_mass_over_c = 200.0     # charge of each particles
speed_of_light = 5e1   # speed of light which only used in particle module

<analysis>
enable = false   # whether do analysis
dt    = 10.0     # time increment between outputs

<problem>
pert = 1e-3       # initial perturbation magnitude
tur_index = -2    # initial power spectrum index
cr_mass = 5.0E-4  # cosmic ray mass density ratio
eta = +2.5
"""


@pytest.fixture
def inp() -> AthInput:
    return parse_athinput(SAMPLE)


# ------------------------------------------------------------------ parse_value
@pytest.mark.parametrize(
    "text, expected, typ",
    [
        ("1024", 1024, int),
        ("-1", -1, int),
        ("+7", 7, int),
        ("0.5", 0.5, float),
        ("-15.707963267948966", -15.707963267948966, float),
        ("1e-3", 1e-3, float),
        ("5.0E-4", 5.0e-4, float),
        ("true", True, bool),
        ("False", False, bool),
        ("periodic", "periodic", str),
        ("  hdf5 ", "hdf5", str),
    ],
)
def test_parse_value_types(text, expected, typ):
    value = parse_value(text)
    assert type(value) is typ
    assert value == expected


# ---------------------------------------------------------------------- parsing
def test_typed_values_with_trailing_comments(inp):
    assert inp.get("mesh", "nx1") == 1024 and isinstance(inp.get("mesh", "nx1"), int)
    assert inp.get("mesh", "x1min") == pytest.approx(-15.707963267948966, rel=1e-15)
    assert inp.get("mesh", "ix1_bc") == "periodic"
    assert inp.get("time", "nlim") == -1
    assert inp.get("particles", "backreaction") is True
    assert inp.get("particles", "delta_f_enable") is False
    assert inp.get("particles", "speed_of_light") == 50.0
    assert inp.get("problem", "pert") == 1e-3
    assert inp.get("problem", "tur_index") == -2
    assert inp.get("problem", "cr_mass") == 5e-4
    assert inp.get("problem", "eta") == 2.5
    assert inp.problem_id == "org.stir.feedback"


def test_get_default_and_require(inp):
    assert inp.get("mesh", "nx9", default=3) == 3
    assert inp.get("no_such_block", "nx1") is None
    with pytest.raises(KeyError, match="nx9"):
        inp.require("mesh", "nx9")
    assert "mesh" in inp and "nope" not in inp
    assert inp["mesh"]["nx1"] == 1024


def test_comment_block_keeps_raw_configure_line(inp):
    # the configure line contains '=' and '$' and must survive verbatim
    assert inp.configure == "--prob kh_driven -b --eos isothermal -mpi -hdf5 --hdf5_path=$TACC_HDF5_DIR --cxx icpc"
    assert inp.get("comment", "problem") == "Driven KH instability"
    assert inp.get("comment", "reference") == ""


def test_no_cross_block_collisions(inp):
    # `dt` appears in four blocks and each keeps its own value
    assert inp.get("output1", "dt") == 0.5
    assert inp.get("output2", "dt") == 50.0
    assert inp.get("output10", "dt") == 100.0
    assert inp.get("analysis", "dt") == 10.0
    assert inp.find("dt") == {"output2": 50.0, "output1": 0.5, "output10": 100.0, "analysis": 10.0}


def test_outputs_sorted_numerically_with_ids(inp):
    outs = inp.outputs
    assert [o["id"] for o in outs] == [1, 2, 10]  # numeric, not lexicographic ('10' < '2')
    assert [o["file_type"] for o in outs] == ["hst", "hdf5", "rst"]
    assert inp.output("hdf5")["dt"] == 50.0
    assert inp.output("hdf5", "prim")["id"] == 2
    assert inp.output("hdf5", "cons") is None
    assert inp.output("vtk") is None


def test_lookup_unique_and_ambiguous(inp):
    assert inp.lookup("tlim") == 600.0
    with pytest.raises(KeyError, match="4 blocks"):
        inp.lookup("dt")
    with pytest.raises(KeyError, match="0 blocks"):
        inp.lookup("does_not_exist")


def test_duplicate_key_warns_last_wins():
    with pytest.warns(UserWarning, match="duplicate key"):
        parsed = parse_athinput("<mesh>\nnx1 = 4\nnx1 = 8\n")
    assert parsed.get("mesh", "nx1") == 8


def test_read_athinput_from_file_and_directory(tmp_path):
    f = tmp_path / "athinput.kh_org"
    f.write_text(SAMPLE)
    by_file = read_athinput(f)
    by_dir = read_athinput(tmp_path)
    assert by_file.path == f and by_dir.path == f
    assert by_file.blocks == by_dir.blocks
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no athinput"):
        read_athinput(empty)


# --------------------------------------------------------------------- hashing
def test_sha256_is_hash_of_text_and_stable(tmp_path, inp):
    assert inp.sha256 == hashlib.sha256(SAMPLE.encode()).hexdigest()
    f = tmp_path / "athinput.x"
    f.write_text(SAMPLE)
    assert read_athinput(f).sha256 == inp.sha256
    assert parse_athinput(SAMPLE + "\n").sha256 != inp.sha256


# ---------------------------------------------------------------------- writing
def test_write_override_existing_key_preserves_comment(tmp_path, inp):
    dest = tmp_path / "athinput.new"
    out = write_athinput(inp, dest, {"output2": {"dt": 25.0}, "particles": {"backreaction": False}})
    text = dest.read_text()
    line = next(ln for ln in text.splitlines() if ln.strip().startswith("dt") and "25.0" in ln)
    assert "# time increment between outputs" in line
    assert out.get("output2", "dt") == 25.0
    assert out.get("output1", "dt") == 0.5  # other blocks untouched
    assert out.get("analysis", "dt") == 10.0
    assert out.get("particles", "backreaction") is False
    assert "false" in next(ln for ln in text.splitlines() if ln.strip().startswith("backreaction"))
    assert out.path == dest


def test_write_adds_new_key_to_existing_and_new_block(tmp_path, inp):
    dest = tmp_path / "athinput.new"
    out = write_athinput(inp, dest, {"problem": {"shear_strength": 2.0}, "mesh": {"nx2": 256},
                                     "newblock": {"alpha": 3, "flag": True}})
    assert out.get("problem", "shear_strength") == 2.0
    assert out.get("mesh", "nx2") == 256
    assert out.get("newblock", "alpha") == 3
    assert out.get("newblock", "flag") is True
    # new key appended inside <mesh>, i.e. before the next block header
    lines = dest.read_text().splitlines()
    i_mesh = lines.index("<mesh>")
    i_next = next(i for i in range(i_mesh + 1, len(lines)) if lines[i].startswith("<"))
    assert any(ln.startswith("nx2") for ln in lines[i_mesh:i_next])


def test_write_round_trip_equality(tmp_path, inp):
    dest = tmp_path / "athinput.copy"
    out = write_athinput(inp, dest)
    assert out.blocks == inp.blocks
    reparsed = read_athinput(dest)
    assert reparsed.blocks == inp.blocks
    assert reparsed.configure == inp.configure

    over = {"problem": {"pert": 2e-3, "tur_index": -3}, "time": {"tlim": 1200.0}}
    out2 = write_athinput(dest, tmp_path / "athinput.mod", over)
    expected = {b: dict(kv) for b, kv in inp.blocks.items()}
    for b, kv in over.items():
        expected[b].update(kv)
    assert read_athinput(tmp_path / "athinput.mod").blocks == expected == out2.blocks
    # writing is deterministic -> identical hashes
    out3 = write_athinput(dest, tmp_path / "athinput.mod2", over)
    assert out3.sha256 == out2.sha256


# ------------------------------------------------------------ byte-exact writing
@pytest.mark.parametrize("eol", ["\n", "\r\n"])
@pytest.mark.parametrize("tail", ["", "\n", "\n\n\n"])
def test_write_without_overrides_is_byte_identical(tmp_path, eol, tail):
    text = SAMPLE.rstrip("\n").replace("\n", eol) + tail.replace("\n", eol)
    src = tmp_path / "athinput.src"
    src.write_bytes(text.encode())
    inp = read_athinput(src)
    assert inp.text == text and inp.sha256 == hashlib.sha256(text.encode()).hexdigest()
    assert inp.get("mesh", "nx1") == 1024 and inp.configure.endswith("--cxx icpc")  # no stray \r
    out = write_athinput(inp, tmp_path / "athinput.copy")
    assert (tmp_path / "athinput.copy").read_bytes() == text.encode()
    assert out.blocks == inp.blocks
    # overrides keep the line endings and the trailing newlines too
    write_athinput(src, tmp_path / "athinput.mod", {"time": {"tlim": 900.0}, "newblock": {"k": 1}})
    mod = (tmp_path / "athinput.mod").read_bytes().decode()
    if eol == "\r\n":
        assert "\n" not in mod.replace("\r\n", "")
    body = mod.rstrip("\r\n")
    assert mod[len(body):] == tail.replace("\n", eol)
    assert read_athinput(tmp_path / "athinput.mod").get("newblock", "k") == 1


def test_write_keeps_padding_and_changes_only_the_value(tmp_path, inp):
    write_athinput(inp, tmp_path / "a", {"output2": {"dt": 25.0}, "particles": {"backreaction": False},
                                         "mesh": {"x1min": -1.0}})
    lines = (tmp_path / "a").read_text().splitlines()
    assert "dt         = 25.0       # time increment between outputs" in lines
    assert "backreaction = false    # turn on/off the back reaction of the gas drag" in lines
    assert "x1min      = -1.0                                    # minimum value of X1" in lines
    orig = SAMPLE.splitlines()
    changed = [i for i, (a, b) in enumerate(zip(orig, lines)) if a != b]
    assert len(lines) == len(orig) and len(changed) == 3


def test_write_overrides_every_occurrence_of_duplicate_key(tmp_path):
    text = "<mesh>\nnx1 = 4   # first\nnx2 = 8\nnx1 = 4   # again\n\n<time>\ntlim = 1.0\n"
    with pytest.warns(UserWarning, match="duplicate key"):
        src = parse_athinput(text)
    with pytest.warns(UserWarning, match="duplicate key"):
        out = write_athinput(src, tmp_path / "dup", {"mesh": {"nx1": 16}})
    written = (tmp_path / "dup").read_text()
    assert written == "<mesh>\nnx1 = 16  # first\nnx2 = 8\nnx1 = 16  # again\n\n<time>\ntlim = 1.0\n"
    assert out.get("mesh", "nx1") == 16
    assert written.count("nx1") == 2  # not appended a third time


def test_write_int_override_of_float_value_stays_float(tmp_path, inp):
    out = write_athinput(inp, tmp_path / "f", {"output2": {"dt": 100}, "time": {"tlim": "1200"},
                                               "mesh": {"nx1": 512}, "problem": {"eta": True}})
    text = (tmp_path / "f").read_text()
    assert "dt         = 100.0      # time increment" in text
    assert out.get("output2", "dt") == 100.0 and isinstance(out.get("output2", "dt"), float)
    assert isinstance(out.get("time", "tlim"), float) and out.get("time", "tlim") == 1200.0
    assert out.get("mesh", "nx1") == 512 and isinstance(out.get("mesh", "nx1"), int)  # int stays int
    assert out.get("problem", "eta") is True  # bool is not converted
