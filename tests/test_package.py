import ast
import re
import shlex
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import iwopy
import pytest
import yaml

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]


def _ci_workflow(filename):
    return yaml.load(
        (ROOT / ".github/workflows" / filename).read_text(), Loader=yaml.BaseLoader
    )


def test_package_and_release_versions_match_project_metadata():
    with (ROOT / "pyproject.toml").open("rb") as stream:
        expected = tomllib.load(stream)["project"]["version"]
    assert isinstance(iwopy.__version__, str)
    assert iwopy.__version__
    assert iwopy.__version__ == version("iwopy") == expected
    changelog = (ROOT / "CHANGELOG.md").read_text()
    assert re.findall(r"^## v([^\n]+)$", changelog, re.MULTILINE)[-1] == expected
    citation = yaml.load((ROOT / "CITATION.cff").read_text(), Loader=yaml.BaseLoader)
    assert citation["version"] == expected


def test_optional_interfaces_import_does_not_load_backends():
    code = """
import sys
from importlib.abc import MetaPathFinder

class BlockBackends(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"cma", "nevergrad", "optuna", "deap"}:
            raise AssertionError(f"{fullname} was imported eagerly")

sys.meta_path.insert(0, BlockBackends())
import iwopy
from iwopy.interfaces.cma import Optimizer_cma
from iwopy.interfaces.nevergrad import Optimizer_nevergrad
from iwopy.interfaces.optuna import Optimizer_optuna
from iwopy.interfaces.deap import Optimizer_deap
assert not {"cma", "nevergrad", "optuna", "deap"}.intersection(sys.modules)
assert Optimizer_cma.__name__ == "Optimizer_cma"
"""

    completed = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr


@pytest.mark.parametrize("filename", ["test.yml", "publish.yml"])
def test_ci_workflows_install_and_verify_all_solver_packages(filename):
    workflow = _ci_workflow(filename)
    steps = {
        step["name"]: step["run"]
        for step in workflow["jobs"]["tests"]["steps"]
        if "run" in step
    }
    assert steps["Install dependencies"] == "uv sync --extra test"
    preflight = shlex.split(steps["Verify optimizer dependencies"])
    assert preflight[:4] == ["uv", "run", "python", "-c"]
    imports = ast.parse(preflight[4]).body
    assert len(imports) == 1 and isinstance(imports[0], ast.Import)
    assert {name.name for name in imports[0].names} == {
        "cma",
        "deap",
        "nevergrad",
        "optuna",
        "pygmo",
        "pymoo",
        "scipy",
    }
    assert workflow["jobs"]["tests"]["strategy"]["matrix"]["python-version"] == [
        "3.10",
        "3.11",
        "3.12",
        "3.13",
        "3.14",
    ]
    for job in ("pre-commit", "tests", "notebooks"):
        install = next(
            step["run"]
            for step in workflow["jobs"][job]["steps"]
            if step["name"] == "Install dependencies"
        )
        assert "--dev" not in shlex.split(install)
        assert "--extra test" in install


@pytest.mark.parametrize("filename", ["test.yml", "publish.yml"])
def test_ci_workflows_run_all_optimizer_and_catalog_regressions(filename):
    steps = {
        step["name"]: step["run"]
        for step in _ci_workflow(filename)["jobs"]["tests"]["steps"]
        if "run" in step
    }
    focused = shlex.split(steps["Run optimizer and benchmark regressions"])
    assert focused == [
        "uv",
        "run",
        "pytest",
        "tests/interfaces",
        "tests/scipy/test_de.py",
        "tests/test_benchmarks.py",
        "tests/test_documentation.py",
    ]
    assert all((ROOT / path).exists() for path in focused[3:])
    assert steps["Run full test suite"] == "uv run pytest tests"
    names = list(steps)
    assert (
        names.index("Verify optimizer dependencies")
        < names.index("Run optimizer and benchmark regressions")
        < names.index("Run full test suite")
    )


def test_regular_ci_can_be_triggered_manually_without_publishing():
    workflow = _ci_workflow("test.yml")
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["on"]["push"]["tags-ignore"] == ["v*"]
    assert workflow["jobs"]["publish-to-testpypi"]["if"].startswith(
        "github.event_name == 'push' &&"
    )
