"""Unit tests for shearpic.io.history.read_hst (raw column names, restart de-duplication)."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import shearpic.io.history as history_mod
from shearpic.io._util import keep_last_written
from shearpic.io.history import ALIASES, HST_SCHEMA, find_hst, read_hst

# The first 30 columns of a real kh_driven history file.  Athena++ writes each header field
# with setw(13), so names that fill the field are glued to the next index
# ("[28]=Pideal_x[29]=Pideal_y").
NAMES = ["time", "dt", "mass", "1-mom", "2-mom", "3-mom", "1-KE", "2-KE", "3-KE", "1-ME", "2-ME", "3-ME",
         "np", "vp1", "vp2", "vp3", "vp1^2", "vp2^2", "vp3^2", "-BxBy", "dVxVy", "Pstir", "KE_cr", "Gamma",
         "Pideal", "r_g", "w_c", "Pideal_x", "Pideal_y", "Pideal_z"]
NCOL = len(NAMES)


def _header(names=NAMES, indices=None) -> str:
    indices = indices or range(1, len(names) + 1)
    return "# Athena++ history data\n# " + "".join(f"[{i}]={n}".ljust(13) for i, n in zip(indices, names)) + "\n"


def _rows(times, ncol=NCOL, fill=None) -> str:
    """Rows in Athena's " %13.5e" style; column k (0-based, k >= 2) holds t * (k - 1)."""
    lines = []
    for t in times:
        vals = [t, 1e-3] + [t * (k - 1) if fill is None else fill for k in range(2, ncol)]
        lines.append("".join(f" {v: .5e}" for v in vals))
    return "\n".join(lines) + "\n"


def test_header_style_is_glued():
    hdr = _header()
    assert "[28]=Pideal_x[29]=Pideal_y[30]=Pideal_z" in hdr
    assert "[20]=-BxBy   [21]" in hdr and "[17]=vp1^2   [18]" in hdr


def test_raw_names_preserved(tmp_path):
    f = tmp_path / "run.hst"
    f.write_text(_header() + _rows([0.0, 0.5, 1.0]))
    df = read_hst(f)
    assert list(df.columns) == NAMES
    assert df.attrs["n_dropped"] == 0 and df.attrs["path"] == str(f)
    np.testing.assert_allclose(df["time"], [0.0, 0.5, 1.0])
    for name in ("1-KE", "-BxBy", "vp1^2", "Pideal_x", "Pideal_y", "Pideal_z"):
        k = NAMES.index(name)
        np.testing.assert_allclose(df[name], (k - 1) * df["time"], err_msg=name)
    assert df[ALIASES["neg_BxBy"]].equals(df["-BxBy"])
    assert df[ALIASES["vp1_sq"]].equals(df["vp1^2"])


def test_restart_duplicates_removed_keep_last_written(tmp_path):
    f = tmp_path / "run.hst"
    first = _rows([0.0, 0.5, 1.0, 1.5, 2.0])
    # restart from t=1.0: rows 1.0..2.0 are rewritten (with different values) and continued
    second = _rows([1.0], fill=-1.0) + _rows([1.5, 2.0, 2.5])
    f.write_text(_header() + first + _header() + second)
    with pytest.warns(UserWarning, match="dropped 3 rows"):
        df = read_hst(f)
    np.testing.assert_allclose(df["time"], [0.0, 0.5, 1.0, 1.5, 2.0, 2.5])
    assert np.all(np.diff(df["time"]) > 0)
    assert df.attrs["n_dropped"] == 3
    assert df.loc[2, "1-KE"] == -1.0  # the row written after the restart wins
    assert df.index.tolist() == list(range(6))

    raw = read_hst(f, drop_duplicates=False)
    assert len(raw) == 9 and raw.attrs["n_dropped"] == 0


def test_restart_with_changed_columns_raises(tmp_path):
    f = tmp_path / "run.hst"
    f.write_text(_header() + _rows([0.0, 0.5]) + _header(NAMES[:-1]) + _rows([0.5, 1.0], ncol=NCOL - 1))
    with pytest.raises(ValueError, match="history columns change"):
        read_hst(f)
    renamed = NAMES[:-1] + ["Pideal_w"]
    f.write_text(_header() + _rows([0.0, 0.5]) + _header(renamed) + _rows([1.0]))
    with pytest.raises(ValueError, match=r"run.hst:6: the history columns change.*column 30 is 'Pideal_z' before"):
        read_hst(f)


def test_identical_repeated_header_is_ignored(tmp_path):
    f = tmp_path / "run.hst"
    f.write_text(_header() + _rows([0.0, 0.5]) + _header() + _rows([1.0, 1.5]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        df = read_hst(f)
    np.testing.assert_allclose(df["time"], [0.0, 0.5, 1.0, 1.5])
    assert list(df.columns) == NAMES


def test_partially_written_last_line_is_dropped(tmp_path):
    f = tmp_path / "run.hst"
    full = _header() + _rows([0.0, 0.5, 1.0])
    partial = _rows([1.5])[:40]  # the writer was interrupted in the middle of the row
    f.write_text(full + partial)
    with pytest.warns(UserWarning, match="incomplete last line"):
        df = read_hst(f)
    np.testing.assert_allclose(df["time"], [0.0, 0.5, 1.0])
    # a last row with all its columns but no newline yet is still incomplete (the last number may be cut)
    f.write_text(full + _rows([1.5]).rstrip("\n"))
    with pytest.warns(UserWarning, match="incomplete last line"):
        assert len(read_hst(f)) == 3
    # header only: empty frame with the right columns
    f.write_text(_header())
    empty = read_hst(f)
    assert len(empty) == 0 and list(empty.columns) == NAMES


def test_keep_last_written_mask():
    t = np.array([0, 1, 2, 3, 4, 2, 3, 4, 5, 5], float)
    keep = keep_last_written(t)
    assert t[keep].tolist() == [0, 1, 2, 3, 4, 5]
    assert keep.tolist() == [True, True, False, False, False, True, True, True, False, True]
    assert keep_last_written(np.arange(4.0)).all()


def test_column_count_mismatch(tmp_path):
    f = tmp_path / "bad.hst"
    f.write_text(_header() + _rows([0.0, 1.0], ncol=len(NAMES) + 1))
    with pytest.raises(ValueError, match="header has 30 columns but data rows have 31"):
        read_hst(f)


def test_header_index_gap(tmp_path):
    f = tmp_path / "gap.hst"
    f.write_text(_header(indices=[*range(1, 5), *range(6, NCOL + 2)]) + _rows([0.0]))
    with pytest.raises(ValueError, match="malformed history header"):
        read_hst(f)


def test_missing_header(tmp_path):
    f = tmp_path / "nohdr.hst"
    f.write_text("# Athena++ history data\n" + _rows([0.0]))
    with pytest.raises(ValueError, match=r"no '\[1\]=' column header"):
        read_hst(f)


def test_single_row_and_directory_lookup(tmp_path):
    f = tmp_path / "prob.hst"
    f.write_text(_header() + _rows([0.0]))
    df = read_hst(tmp_path)
    assert isinstance(df, pd.DataFrame) and len(df) == 1
    assert find_hst(tmp_path) == f
    (tmp_path / "other.hst").write_text(_header() + _rows([0.0]))
    with pytest.raises(ValueError, match="several .hst"):
        find_hst(tmp_path)
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="no .hst"):
        find_hst(empty)


def test_schema_caveats_units_and_exports():
    assert "ALIASES" in history_mod.__all__
    assert HST_SCHEMA["KE_cr"].units == "U0^2 per unit particle mass (sum of (gamma-1) c^2)"
    caveat = HST_SCHEMA["Pstir"].caveat
    assert "changed between problem-generator versions" in caveat
    assert "stir_power" in caveat
    src = open(history_mod.__file__).read()
    assert "test_data_regression" not in src
    assert "the last header wins" not in src


def test_schema_documents_key_columns():
    for name in ("time", "1-KE", "np", "vp1", "vp1^2", "-BxBy", "Pstir", "KE_cr", "Gamma", "Pideal", "r_g", "w_c"):
        assert name in HST_SCHEMA
    assert HST_SCHEMA["Pstir"].aggregation == "cell sum (no dV)"
    assert HST_SCHEMA["1-KE"].aggregation == "volume integral"
    assert HST_SCHEMA["-BxBy"].aggregation == "volume mean"
    assert HST_SCHEMA["KE_cr"].aggregation == "particle sum"
    assert set(ALIASES.values()) <= set(HST_SCHEMA)
