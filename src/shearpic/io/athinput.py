"""Reader and writer for Athena++ input files (``athinput.*``).

Values stay attached to their ``<block>``, since keys such as ``dt`` appear in several
blocks, and are converted to ``int``, ``float`` or ``bool`` when possible::

    inp = read_athinput("run0001/athinput.kh_org")
    inp.get("particles", "speed_of_light")    # 50.0
    [o["dt"] for o in inp.outputs]            # [0.5, 50.0, 100.0]
"""

from __future__ import annotations

import hashlib
import numbers
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

__all__ = ["AthInput", "read_athinput", "parse_athinput", "write_athinput", "parse_value"]

_BLOCK_RE = re.compile(r"^\s*<\s*([A-Za-z0-9_]+)\s*>")
_KV_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=\s*(.*?)\s*$")
_INT_RE = re.compile(r"^[+-]?\d+$")


def parse_value(text: str) -> Any:
    """Convert an athinput value string to bool, int, float or (fallback) str."""
    s = text.strip()
    low = s.lower()
    if low in ("true", "false"):
        return low == "true"
    if _INT_RE.match(s):
        return int(s)
    try:
        return float(s)
    except ValueError:
        return s


def _strip_comment(value: str) -> str:
    return value.split("#", 1)[0].strip()


@dataclass(frozen=True)
class AthInput:
    """Parsed athinput file. ``blocks[block][key]`` holds typed values."""

    blocks: Mapping[str, Mapping[str, Any]]
    text: str = ""
    path: Path | None = None
    raw: Mapping[str, Mapping[str, str]] = field(default_factory=dict)

    def get(self, block: str, key: str, default: Any = None) -> Any:
        return self.blocks.get(block, {}).get(key, default)

    def require(self, block: str, key: str) -> Any:
        try:
            return self.blocks[block][key]
        except KeyError:
            where = f" in {self.path}" if self.path else ""
            raise KeyError(f"<{block}> {key} is missing{where}") from None

    def __getitem__(self, block: str) -> Mapping[str, Any]:
        return self.blocks[block]

    def __contains__(self, block: str) -> bool:
        return block in self.blocks

    def find(self, key: str) -> dict[str, Any]:
        """Return ``{block: value}`` for every block that defines ``key``."""
        return {b: kv[key] for b, kv in self.blocks.items() if key in kv}

    def lookup(self, key: str) -> Any:
        """Value of ``key`` if it is defined in exactly one block, else raise."""
        hits = self.find(key)
        if len(hits) != 1:
            raise KeyError(
                f"'{key}' is defined in {len(hits)} blocks ({list(hits)}); use get(block, key)"
            )
        return next(iter(hits.values()))

    @property
    def outputs(self) -> list[dict[str, Any]]:
        """All ``<outputN>`` blocks sorted by N, each with an added ``'id'`` (int)."""
        out = []
        for name, kv in self.blocks.items():
            m = re.fullmatch(r"output(\d+)", name)
            if m:
                out.append({"id": int(m.group(1)), **kv})
        return sorted(out, key=lambda o: o["id"])

    def output(self, file_type: str, variable: str | None = None) -> dict[str, Any] | None:
        """First output block with the given ``file_type`` (and ``variable``), or None."""
        for o in self.outputs:
            if o.get("file_type") == file_type and (variable is None or o.get("variable") == variable):
                return o
        return None

    @property
    def problem_id(self) -> str | None:
        pid = self.get("job", "problem_id")
        return None if pid is None else str(pid)

    @property
    def configure(self) -> str | None:
        """The ``configure`` line from ``<comment>`` (how the executable was built)."""
        raw = self.raw.get("comment", {}).get("configure")
        return raw

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()


def parse_athinput(text: str, path: Path | None = None) -> AthInput:
    blocks: dict[str, dict[str, Any]] = {}
    raw: dict[str, dict[str, str]] = {}
    current: str | None = None
    for lineno, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        m = _BLOCK_RE.match(stripped)
        if m:
            current = m.group(1)
            blocks.setdefault(current, {})
            raw.setdefault(current, {})
            continue
        m = _KV_RE.match(stripped)
        if m is None or current is None:
            continue
        key = m.group(1)
        # <comment> values are kept verbatim; they may contain '#'
        value_text = m.group(2) if current == "comment" else _strip_comment(m.group(2))
        if key in blocks[current]:
            warnings.warn(f"{path or 'athinput'}:{lineno}: duplicate key <{current}> {key}; last value wins")
        raw[current][key] = value_text
        blocks[current][key] = value_text if current == "comment" else parse_value(value_text)
    return AthInput(blocks=blocks, text=text, path=path, raw=raw)


def read_athinput(path: str | Path) -> AthInput:
    """Read an athinput file. ``path`` may be the file or a run directory."""
    p = Path(path)
    if p.is_dir():
        candidates = sorted(p.glob("athinput.*"))
        if not candidates:
            raise FileNotFoundError(f"no athinput.* file in {p}")
        if len(candidates) > 1:
            warnings.warn(f"several athinput files in {p}; using {candidates[0].name}")
        p = candidates[0]
    with open(p, newline="") as fh:  # keep \r\n so text and sha256 match the file
        text = fh.read()
    return parse_athinput(text, path=p)


def _format_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _format_override(value: Any, source_value: Any) -> str:
    """Text of an override; an integer replacing a float value is written as a float (100 -> 100.0)."""
    if isinstance(source_value, float) and not isinstance(value, bool):
        if isinstance(value, numbers.Integral) or (isinstance(value, str) and _INT_RE.match(value.strip())):
            return repr(float(int(value)))
    return _format_value(value)


_LINE_SEP_RE = re.compile(r"(\r\n|\n|\r)")
_VALUE_RE = re.compile(r"^(?P<pre>\s*[A-Za-z0-9_]+\s*=[ \t]*)(?P<val>[^#]*?)(?P<gap>[ \t]*)(?P<comment>#.*)?$")


def _replace_value(line: str, new: str, block: str) -> str:
    """``line`` with only its value text replaced; key padding, spacing and comment are kept."""
    if block == "comment":  # values in <comment> may contain '#'
        m = re.match(r"^(\s*[A-Za-z0-9_]+\s*=[ \t]*)(.*?)([ \t]*)$", line)
        pre = m.group(1) if m.group(1)[-1].isspace() else m.group(1) + " "
        return pre + new + m.group(3)
    m = _VALUE_RE.match(line)
    pre, val, gap, comment = m.group("pre"), m.group("val"), m.group("gap"), m.group("comment")
    if not val and not pre[-1].isspace():  # "key =" with no value yet
        pre += " "
    if comment is None:
        return pre + new + gap
    if val:  # keep the comment in its column when the new value fits
        gap = " " * max(1, len(val) + len(gap) - len(new))
    return pre + new + (gap or " ") + comment


def write_athinput(
    source: AthInput | str | Path,
    dest: str | Path,
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> AthInput:
    """Copy an athinput file to ``dest``, replacing ``overrides[block][key]`` in place.

    ``source`` is a parsed AthInput or a path. Everything not overridden is kept byte for
    byte, so a copy without overrides is identical to the source. Only the value text of an
    existing key is replaced, at every occurrence in its block; an integer replacing a float
    is written as a float. New keys are appended to the first occurrence of their block and
    new blocks to the end of the file. Returns the parsed result of the written file.
    """
    src = source if isinstance(source, AthInput) else read_athinput(source)
    overrides = {b: dict(kv) for b, kv in (overrides or {}).items()}
    text = src.text
    eol = "\r\n" if "\r\n" in text else "\n"
    parts = _LINE_SEP_RE.split(text)
    # [line, separator] pairs; the last separator is "", as is the last line after a final newline
    items = [[parts[i], parts[i + 1] if i + 1 < len(parts) else ""] for i in range(0, len(parts), 2)]

    existing = {b: set(kv) for b, kv in src.blocks.items()}
    to_add = {b: {k: v for k, v in kv.items() if k not in existing.get(b, set())} for b, kv in overrides.items()}
    out: list[list[str]] = []

    def is_blank(item) -> bool:
        return not item[0].strip()

    def insert_new(lines: list[str]) -> None:
        """Insert lines before the trailing blank lines currently at the end of ``out``."""
        trailing = []
        while out and is_blank(out[-1]):
            trailing.append(out.pop())
        out.extend([ln, eol] for ln in lines)
        out.extend(reversed(trailing))

    def flush_block(block: str | None) -> None:
        if block is not None and to_add.get(block):
            insert_new([f"{k:<10} = {_format_value(v)}" for k, v in to_add.pop(block).items()])

    current: str | None = None
    for line, sep in items:
        m = _BLOCK_RE.match(line.strip())
        if m:
            flush_block(current)
            current = m.group(1)
            out.append([line, sep])
            continue
        kv = _KV_RE.match(line.strip())
        if (kv and current is not None and not line.lstrip().startswith("#")
                and kv.group(1) in overrides.get(current, {}) and kv.group(1) in existing.get(current, set())):
            key = kv.group(1)
            new = _format_override(overrides[current][key], src.blocks[current][key])
            out.append([_replace_value(line, new, current), sep])
            continue
        out.append([line, sep])
    flush_block(current)
    new_blocks = [(b, kv) for b, kv in to_add.items() if kv]
    if new_blocks:
        lines: list[str] = []
        for block, kv in new_blocks:
            lines += ["", f"<{block}>"] + [f"{k:<10} = {_format_value(v)}" for k, v in kv.items()]
        insert_new(lines)
    for item in out[:-1]:  # a former last line may now be followed by appended lines
        if not item[1]:
            item[1] = eol
    if out:
        out[-1][1] = ""
    new_text = "".join(line + sep for line, sep in out)
    dest = Path(dest)
    with open(dest, "w", newline="") as fh:
        fh.write(new_text)
    return parse_athinput(new_text, path=dest)
