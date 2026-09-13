"""The READMEs, their images and the example scripts must keep working with the package.

* The ``python`` blocks of README.md compile, and its ``shearpic`` commands parse and run in
  order in a temporary registry and data root.
* Every ``python analysis/...`` and ``python examples/...`` command in either README is accepted
  by that script's argument parser.
* Every image referenced by a README exists, and the Mermaid blocks use the diagram types
  GitHub renders.
* With ``PARTICLE_ACCEL_TEST_DATA`` (``-m data``) the README's Python blocks run in order
  against the real runs.
"""

from __future__ import annotations

import argparse
import ast
import builtins
import importlib.util
import os
import re
import shlex
import subprocess
import sys
import textwrap
import time
import warnings
from pathlib import Path

import numpy as np
import pytest

from shearpic import cli
from shearpic.registry import STATUSES

REPO = Path(__file__).resolve().parents[1]
README = REPO / "README.md"
ANALYSIS_README = REPO / "analysis" / "README.md"
READMES = (README, ANALYSIS_README)
EXAMPLES = REPO / "examples"
MAX_IMAGE_BYTES = 150_000
MERMAID_TYPES = ("flowchart ", "graph ", "sequenceDiagram", "stateDiagram-v2")

ATHINPUT = """\
<job>
problem_id = org.stir.feedback

<output1>
file_type  = hst
dt         = 0.5

<time>
tlim       = 600.0

<mesh>
nx1        = 16
x1min      = -1.5707963267948966
x1max      = 1.5707963267948966
nx2        = 32
x2min      = -6.283185307179586
x2max      = 6.283185307179586
nx3        = 1
x3min      = -0.5
x3max      = 0.5

<hydro>
iso_sound_speed = 5

<particles>
backreaction = true
charge_over_mass_over_c = 200.0
speed_of_light = 50.0

<problem>
iprob = 0
shear_strength = 1.0
y1 = -3.141592653589793
y2 = 3.141592653589793
M_A = 10
tau = 0.5
npx1 = 16
npx2 = 32
npx3 = 1
vp_par = 50.0
cr_mass = 0.0005
"""


# ------------------------------------------------------------------ helpers
def _blocks(language: str, path: Path = README) -> list[tuple[int, str]]:
    """``(first line number, code)`` of every fenced block of ``language``, including indented ones."""
    text = path.read_text(encoding="utf-8")
    pattern = rf"^([ \t]*)```{language}\n(.*?)^\1```"
    return [(text[: m.start()].count("\n") + 2, textwrap.dedent(m.group(2)))
            for m in re.finditer(pattern, text, flags=re.S | re.M)]


def _command_lines(path: Path = README) -> list[str]:
    """Every line of the bash blocks, with backslash continuations joined."""
    lines = []
    for _, code in _blocks("bash", path):
        lines += [line.strip() for line in re.sub(r"\\\n\s*", " ", code).splitlines() if line.strip()]
    return lines


def _shearpic_commands() -> list[str]:
    return [line for line in _command_lines() if line.startswith("shearpic ")]


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses look up their module while the file executes
    spec.loader.exec_module(module)
    return module


def _load_example(name: str):
    return _load_module(EXAMPLES / f"{name}.py", f"example_{name}")


def _age(root: Path, hours: float = 2.0) -> None:
    old = time.time() - hours * 3600
    for p in [root, *root.rglob("*")]:
        os.utime(p, (old, old))


def _write_trajectory(path: Path, n: int = 200, seed: int = 0) -> Path:
    """13-column trajectory: gyration in B = 0.1 x-hat, drifting across the layer at y = 3.14."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 20.0, n)
    data = np.zeros((n, 13))
    data[:, 0] = t
    data[:, 2] = -6.0 + 0.5 * t + 0.1 * rng.normal(size=n)
    data[:, 4:7] = 50.0 * np.column_stack([np.full(n, 0.3), np.cos(3 * t), np.sin(3 * t)])
    data[:, 7] = 0.1
    data[:, 11] = 0.01
    path.write_text("\n".join(" ".join(f"{v:.9g}" for v in row) for row in data) + "\n")
    return path


def _synthetic_trajectory_run(root: Path, name: str) -> Path:
    run = root / name
    (run / "high_energy_particles").mkdir(parents=True)
    (run / "athinput.kh_org").write_text(ATHINPUT)
    for k, pid in enumerate((11, 12)):
        _write_trajectory(run / "high_energy_particles" / f"trajectory_initmbid_{k}_pid_{pid}.tab", seed=k)
    return run


# ------------------------------------------------------- README: static checks
def test_readme_python_blocks_compile():
    blocks = _blocks("python")
    assert len(blocks) >= 8
    for line, code in blocks:
        try:
            ast.parse(code)
        except SyntaxError as err:  # pragma: no cover - the message is the point
            pytest.fail(f"README.md python block at line {line} does not compile: {err}")


def test_readme_script_snippet_is_self_contained():
    """The parallel_map ``script.py`` defines everything it uses and handles MPI's ``None``."""
    (line, code), = [(n, c) for n, c in _blocks("python") if c.startswith("# script.py")]
    tree = ast.parse(code)
    defined = {alias.asname or alias.name.split(".")[0] for node in ast.walk(tree)
               if isinstance(node, (ast.Import, ast.ImportFrom)) for alias in node.names}
    defined |= {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)}
    used = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)}
    missing = used - defined - set(dir(builtins)) - {"__name__"}
    assert not missing, f"script.py (README line {line}) uses undefined names {sorted(missing)}"
    assert "results is not None" in code


def test_readme_shearpic_commands_parse():
    parser = cli.build_parser()
    cmds = _shearpic_commands()
    assert any(c.startswith("shearpic runs init") for c in cmds)
    assert any(c.startswith("shearpic runs move") for c in cmds)
    for cmd in cmds:
        argv = shlex.split(cmd, comments=True)[1:]
        try:
            args = parser.parse_args(argv)
        except SystemExit:  # pragma: no cover
            pytest.fail(f"README command does not parse: {cmd}")
        assert getattr(args, "func", None) is not None, cmd


def test_readme_has_no_stale_advice():
    text = README.read_text(encoding="utf-8")
    assert "cd $PARTICLE_ACCEL_DATA/run0002" not in text  # the run may not be in the first data root
    assert "[50.0, 100]" not in text  # int overrides of float keys are written as floats
    assert 'output_dir("run0423"' not in text  # products of one run: run_output_dir


# --------------------------------------------- README: bookkeeping end to end
def test_readme_bookkeeping_commands_run_in_order(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    work = tmp_path / "work"
    template = tmp_path / "athinput.kh_org"
    template.write_text(ATHINPUT)
    results = tmp_path / "results" / "run423"  # existing data that is adopted
    results.mkdir(parents=True)
    (results / "athinput.kh_org").write_text(ATHINPUT)
    (results / "org.stir.feedback.hst").write_text("# [1]=time [2]=dt\n0.0 0.1\n")
    _age(results)
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    monkeypatch.setenv("PARTICLE_ACCEL_REGISTRY", str(tmp_path / "runs.yaml"))
    monkeypatch.setenv("PARTICLE_ACCEL_MACHINE", "local")
    monkeypatch.setenv("WORK", str(work))
    subs = {"path/to/athinput.kh_org": str(template), "/path/to/results/run423": str(results)}

    run2 = data / "run0002"
    ran = []
    for cmd in _shearpic_commands():
        for old, new in subs.items():
            cmd = cmd.replace(old, new)
        argv = [os.path.expandvars(a) for a in shlex.split(cmd, comments=True)[1:]]
        # what the Athena++ job and the user do between the documented commands
        if argv[:4] == ["runs", "set-status", "2", "completed"]:
            (run2 / "org.stir.feedback.hst").write_text("# [1]=time [2]=dt\n600.0 0.1\n")
        if argv[:3] == ["runs", "lock", "2"]:
            _age(run2)  # the job finished more than an hour ago
        if argv[:3] == ["runs", "move", "2"]:
            dest = Path(argv[3])
            dest.parent.mkdir(parents=True)
            subprocess.run(["cp", "-Rp", str(run2), str(dest)], check=True)
        capsys.readouterr()
        code = cli.main(argv)
        out, err = capsys.readouterr()
        assert code == 0, f"`{cmd}` exited with {code}:\n{out}\n{err}"
        ran.append((argv, out))
        if argv[:2] == ["runs", "check"]:
            assert "0 error(s)" in out
    subprocess.run(["chmod", "-R", "u+w", str(tmp_path)], check=False)  # let pytest clean up

    joined = "\n".join(out for _, out in ran)
    assert "created run 2 [planned] at " + str(run2) in joined  # the directory the README tells you to cd into
    assert "run 2: locked" in joined
    assert "adopted " + str(results) + " as run 423" in joined
    assert "experiment 'cscan' saved with runs 1, 2" in joined
    assert "particles/speed_of_light: [50.0, 100.0]" in (tmp_path / "runs.yaml").read_text()


# ------------------------------------------------- script commands in the READMEs
class _Parsed(BaseException):
    """Raised instead of running a script once its arguments have been parsed."""


def _script_commands() -> list[tuple[Path, Path, list[str]]]:
    """``(readme, script, argv)`` of every ``python analysis/...`` or ``python examples/...`` line."""
    found = []
    for readme in READMES:
        for line in _command_lines(readme):
            words = shlex.split(line, comments=True)
            if "python" not in words:
                continue
            for k, word in enumerate(words):
                m = re.search(r"(analysis|examples)/(\w+)\.py$", word)
                if m:
                    found.append((readme, REPO / m.group(1) / f"{m.group(2)}.py", words[k + 1:]))
                    break
    return found


def test_readme_script_commands_parse(tmp_path, monkeypatch):
    commands = _script_commands()
    scripts = {script.name for _, script, _ in commands}
    assert {"reproduce_figures.py", "pack_small_data.py", "energy_spectrum.py", "phase_space.py",
            "plot_spectrum.py", "steady_state.py", "shear_profile.py", "quickstart.py"} <= scripts

    # the preset file that analysis/README.md section 5 tells the reader to create
    presets = (REPO / "analysis" / "paper_figures.yaml").read_text()
    (tmp_path / "analysis").mkdir()
    (tmp_path / "analysis" / "my_figures.yaml").write_text(
        presets + "\nfig7_run425:\n  run: run425\n  t_steady: [100, 600]\n  ylim: [-1, 4]\n  figure: power_run425\n")
    monkeypatch.chdir(tmp_path)

    original = argparse.ArgumentParser.parse_args

    def parse_and_stop(self, args=None, namespace=None):
        raise _Parsed(original(self, args, namespace))

    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", parse_and_stop)
    for readme, script, argv in commands:
        module = _load_module(script, f"docs_{script.parent.name}_{script.stem}")
        try:
            module.main(argv)
        except _Parsed:
            continue
        except SystemExit as exc:  # pragma: no cover - the message is the point
            pytest.fail(f"{readme.relative_to(REPO)}: `{script.name} {shlex.join(argv)}` does not parse ({exc})")
        pytest.fail(f"{script.name} returned without parsing its arguments")  # pragma: no cover


# ------------------------------------------------------------ images and diagrams
def _image_references(readme: Path) -> list[str]:
    text = readme.read_text(encoding="utf-8")
    return re.findall(r"!\[[^\]]*\]\(([^)\s]+)\)", text) + re.findall(r"<img\s[^>]*src=\"([^\"]+)\"", text)


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_readme_images_exist(readme):
    refs = _image_references(readme)
    assert refs, f"{readme} references no images"
    for ref in refs:
        path = (readme.parent / ref).resolve()
        assert path.is_file(), f"{readme.relative_to(REPO)} references missing image {ref}"
        assert path.stat().st_size <= MAX_IMAGE_BYTES, f"{ref} is larger than {MAX_IMAGE_BYTES} bytes"


def test_analysis_readme_shows_every_figure():
    refs = {Path(ref).name for ref in _image_references(ANALYSIS_README)}
    assert {f"fig{n}_" for n in range(1, 8)} == {name[:5] for name in refs if name.startswith("fig")}


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_readme_code_fences_are_balanced(readme):
    fences = [line for line in readme.read_text(encoding="utf-8").splitlines() if line.lstrip().startswith("```")]
    assert len(fences) % 2 == 0, f"{readme.relative_to(REPO)} has an unclosed code fence"


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_readme_table_rows_have_the_header_column_count(readme):
    """A bare ``|`` inside a table cell, even in backticks, starts a new column on GitHub."""
    columns, in_code = None, False
    for number, line in enumerate(readme.read_text(encoding="utf-8").splitlines(), 1):
        if line.lstrip().startswith("```"):
            in_code = not in_code
        if in_code or not line.startswith("|"):
            columns = None
            continue
        n = len(re.findall(r"(?<!\\)\|", line))
        columns = columns or n
        assert n == columns, f"{readme.relative_to(REPO)}:{number}: {n - 1} cells instead of {columns - 1}"


@pytest.mark.parametrize("readme", READMES, ids=lambda p: str(p.relative_to(REPO)))
def test_mermaid_blocks_are_simple_and_valid(readme):
    blocks = _blocks("mermaid", readme)
    assert blocks, f"{readme} has no Mermaid diagram"
    for line, code in blocks:
        where = f"{readme.relative_to(REPO)}:{line}"
        rows = [row.strip() for row in code.splitlines() if row.strip()]
        assert rows[0].startswith(MERMAID_TYPES), f"{where}: unknown diagram type {rows[0]!r}"
        for row in rows:
            assert row.count('"') % 2 == 0, f"{where}: unbalanced quotes in {row!r}"
            assert "<" not in row.replace("<br/>", ""), f"{where}: HTML other than <br/> in {row!r}"
        if rows[0].startswith(("flowchart ", "graph ")):
            outside_labels = [re.sub(r'"[^"]*"', '""', row) for row in rows]
            assert sum(r.startswith("subgraph ") for r in outside_labels) == outside_labels.count("end"), where
            for row in outside_labels:
                for opening, closing in ("[]", "()", "{}"):
                    assert row.count(opening) == row.count(closing), f"{where}: unbalanced {opening} in {row!r}"
        if rows[0].startswith("stateDiagram-v2"):
            pairs = [pair for row in rows[1:] for pair in re.findall(r"(\w+|\[\*\])\s*-->\s*(\w+|\[\*\])", row)]
            states = {state for pair in pairs for state in pair} - {"[*]"}
            assert states <= set(STATUSES), f"{where}: {sorted(states - set(STATUSES))} are not run statuses"


def test_make_images_writes_small_pngs(tmp_path):
    from PIL import Image

    module = _load_module(REPO / "docs" / "make_images.py", "docs_make_images")
    assert module.main(["--out", str(tmp_path)]) == 0
    for name in ("domain", "array_orientation", "particle_motion"):
        png = tmp_path / f"{name}.png"
        assert png.is_file() and png.stat().st_size <= MAX_IMAGE_BYTES
        with Image.open(png) as new, Image.open(REPO / "docs" / "images" / png.name) as committed:
            assert new.size == committed.size, f"docs/images/{png.name} is out of date: run python docs/make_images.py"


# ------------------------------------------------------------ example scripts
@pytest.mark.parametrize("name", ["quickstart", "parallel_snapshots", "trajectories"])
def test_example_help(name, capsys):
    module = _load_example(name)
    with pytest.raises(SystemExit) as exc:
        module.main(["--help"])
    assert exc.value.code == 0
    assert "usage:" in capsys.readouterr().out


@pytest.mark.parametrize("name", ["quickstart", "parallel_snapshots", "trajectories"])
def test_example_find_run_errors_exit_with_a_message(name, tmp_path, monkeypatch):
    data = tmp_path / "data"
    (data / "run404").mkdir(parents=True)
    (data / "run0404").mkdir()
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    module = _load_example(name)
    with pytest.raises(SystemExit, match="error: run 404 is ambiguous"):
        module.find_run("404")
    with pytest.raises(SystemExit, match="error: run 9999 not found"):
        module.find_run("9999")
    assert module.find_run(str(data / "run404")) == data / "run404"


def test_trajectories_example_writes_to_normalised_run_directory(tmp_path, monkeypatch, capsys):
    data = tmp_path / "data"
    data.mkdir()
    _synthetic_trajectory_run(data, "run404")
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data))
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "outputs"))
    module = _load_example("trajectories")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert module.main(["--run", "404", "--backend", "serial"]) == 0
    out = tmp_path / "outputs" / "run0404" / "trajectories"
    assert (out / "run0404_trajectory_summary.csv").is_file()
    assert (out / "run0404_trajectories.png").is_file()
    assert not (tmp_path / "outputs" / "run404").exists()


def test_quickstart_spectra_do_not_warn_about_mixed_particle_formats(tmp_path):
    from shearpic.config import RunConfig
    from shearpic.io.particles import PARBIN_RECORD

    run = tmp_path / "run0007"
    run.mkdir()
    (run / "athinput.kh_org").write_text(ATHINPUT)
    u = np.array([[10.0, 0.0, 0.0], [0.0, 60.0, 0.0], [0.0, 0.0, 200.0]])
    rec = np.zeros(len(u), dtype=PARBIN_RECORD)
    rec["ux"], rec["uy"], rec["uz"] = u.T
    header = np.zeros(12, "<f4").tobytes() + np.array([5.0, 0.1], "<f4").tobytes() + np.array([len(u)], "<i8").tobytes()
    (run / "org.stir.feedback.block0.out3.00001.par.bin").write_bytes(header + rec.tobytes())
    rows = "\n".join(f"0 {i} 0 0 0 {ux} {uy} {uz}" for i, (ux, uy, uz) in enumerate(u))
    (run / "org.stir.feedback.block0.out4.00001.par.tab").write_text(
        "# Athena++ particle data at time = 5.0\nborn_meshblock particle_id x y z vx vy vz\n" + rows + "\n")
    cfg = RunConfig.from_run_dir(run)
    module = _load_example("quickstart")
    out = tmp_path / "out"
    out.mkdir()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        specs = module.load_spectra(run, cfg, None, out)
    assert not [w for w in caught if "both .par.tab and .par.bin" in str(w.message)]
    assert len(specs) == 1 and specs[0].n_total == 3  # one format only: no double counting
    assert (out / "run0007_spectrum_00001.npz").is_file()


# ------------------------------------------------------------ real data (-m data)
@pytest.mark.data
def test_readme_python_blocks_run_on_real_data(data_root, run423, run404, run369, tmp_path, monkeypatch):
    """Execute the README's Python blocks in order, as a user would."""
    monkeypatch.setenv("PARTICLE_ACCEL_DATA", str(data_root))
    registry = tmp_path / "runs.yaml"
    for argv in (["runs", "init"],
                 ["runs", "adopt", str(run423), "--purpose", "snapshots", "--status", "completed"],
                 ["runs", "adopt", str(run369), "--purpose", "spectra", "--status", "completed"],
                 ["exp", "add", "cscan", "--description", "docs test", "--run", "423:a", "--run", "369:b"]):
        assert cli.main(["--registry", str(registry), *argv]) == 0  # adopting only reads the run directories
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.syspath_prepend(str(work))
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    namespace: dict = {"__name__": "readme"}
    for line, code in _blocks("python"):
        name = re.match(r"#\s*(\w+\.py)\s*\n", code)
        if name:
            (work / name.group(1)).write_text(code)
            if "__main__" in code:
                r = subprocess.run([sys.executable, name.group(1)], cwd=work, capture_output=True, text=True,
                                   env=dict(os.environ), timeout=600)
                assert r.returncode == 0, f"README {name.group(1)} (line {line}) failed:\n{r.stderr[-3000:]}"
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                exec(compile(code, f"README.md:{line}", "exec"), namespace)
            except Exception as err:  # pragma: no cover - the message is the point
                pytest.fail(f"README python block at line {line} failed: {err!r}")
        plt.close("all")
    outputs = Path(os.environ["PARTICLE_ACCEL_OUTPUT"])
    assert (outputs / "run0423" / "vorticity_t300.png").is_file()
    assert (outputs / "run0423" / "mean_vx.npz").is_file()
    assert (outputs / "run0404" / "trajectory_samples_spectrum.png").is_file()
    assert namespace["events"].count > 0


@pytest.mark.data
def test_parallel_snapshots_example_on_run423(run423, tmp_path, monkeypatch):
    monkeypatch.setenv("PARTICLE_ACCEL_OUTPUT", str(tmp_path / "outputs"))
    module = _load_example("parallel_snapshots")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        assert module.main(["--run", str(run423), "--stride", "6", "--backend", "serial", "--no-plot"]) == 0
    out = tmp_path / "outputs" / "run0423"  # the directory is called run423: products go to run0423
    assert (out / "run0423_snapshot_scalars.csv").is_file()
    assert (out / "run0423_snapshot_profiles.npz").is_file()
