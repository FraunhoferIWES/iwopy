import ast
import importlib
import csv
import json
import re
import runpy
import tokenize
from io import StringIO
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
SCRIPTS = sorted((ROOT / "examples/optimizers").glob("*.py"))
BACKENDS = {
    "iwopy": "scipy",
    "scipy": "scipy",
    "cma": "cma",
    "pymoo": "pymoo",
    "pygmo": "pygmo",
    "nlopt": "pygmo",
    "nevergrad": "nevergrad",
    "optuna": "optuna",
    "deap": "deap",
}
DEMO_LIMITS = {
    "batch_size": 20,
    "max_iterations": 200,
    "max_evaluations": 500,
    "bfgs_iterations": 100,
    "pop_size": 20,
    "population_multiplier": 5,
}


def _require_backend(path):
    package = BACKENDS[path.stem.split("_", 1)[0]]
    backend = pytest.importorskip(
        package, reason=f"Install the {package} backend to run this catalog case"
    )
    if path.stem == "scipy_cobyqa":
        pytest.importorskip("scipy", minversion="1.14")
    if package == "pygmo":
        feature = (
            "ipopt"
            if path.stem == "pygmo_ipopt"
            else "nlopt"
            if path.stem.startswith("nlopt_") or path.stem == "pygmo_nlopt"
            else None
        )
        if feature is not None and not hasattr(backend, feature):
            pytest.skip(f"This PyGMO build does not include {feature}")


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.stem)
def test_direct_script_executes(path, tmp_path, monkeypatch):
    _require_backend(path)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path), run_name="__main__")
    result = namespace["result"]
    if path.stem in {"pygmo_nsga2", "pygmo_moead", "pygmo_moead_gen", "pygmo_nspso"}:
        assert np.all(np.isfinite(result.get_f()))
        assert np.all(np.isfinite(result.get_x()))
    else:
        assert np.all(np.isfinite(result.objs))
        assert np.all(np.isfinite(result.vars_float))
    if path.stem in {"pygmo_ipopt", "pygmo_nlopt"} or path.stem.startswith("nlopt_"):
        assert len(namespace["solver"].pop) == 1
        np.testing.assert_array_equal(
            namespace["solver"].pop.get_x()[0],
            namespace["problem"].initial_values_float(),
        )
    if path.stem.startswith("pymoo_nsga"):
        assert result.objs.ndim == 2 and result.objs.shape[1] == 2
        assert len(result.objs) > 0
        selected = namespace["selected_index"]
        assert selected == int(np.argmin(result.objs @ np.array([0.5, 0.5])))
        if path.stem.endswith("_constrained"):
            assert result.cons.shape == (len(result.objs), 1)
            assert np.all(result.cons <= -0.25 + 1e-12)
            assert np.all(result.vars_float.sum(axis=1) <= -0.25 + 1e-12)
        else:
            assert result.cons.shape == (len(result.objs), 0)


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.stem)
def test_direct_script_import_is_inert(path, tmp_path, monkeypatch):
    _require_backend(path)
    monkeypatch.chdir(tmp_path)
    namespace = runpy.run_path(str(path))
    assert not {"solver", "result", "problem"}.intersection(namespace)
    assert not list(tmp_path.iterdir())


def test_catalog_links_every_case_to_real_setup():
    page = ROOT / "docs/source/optimizer_catalog.rst"
    text = page.read_text()
    rows = [
        next(csv.reader([line.strip()], skipinitialspace=True))
        for line in text.splitlines()
        if line.startswith('   "')
    ]
    linked = set()
    for row in rows:
        assert len(row) == 5
        assert row[1].startswith(("Single", "Multi")), row
        assert row[2] and row[3], row
        match = re.fullmatch(r":ref:`(\w+) <(case-[\w-]+)>`", row[4])
        assert match, row
        case, anchor = match.groups()
        script = ROOT / f"examples/optimizers/{case}.py"
        assert script in SCRIPTS
        assert f".. _{anchor}:" in text
        section = text.split(f".. _{anchor}:", 1)[1].split(".. _case-", 1)[0]
        assert f".. literalinclude:: ../../examples/optimizers/{case}.py" in section
        linked.add(case)
    assert linked == {path.stem for path in SCRIPTS}
    for include in re.findall(r"\.\. literalinclude:: (.+)", text):
        assert (page.parent / include).resolve().is_file()


def test_catalog_capabilities_are_explicit_about_supported_workflows():
    text = (ROOT / "docs/source/optimizer_catalog.rst").read_text()
    rows = {
        row[0]: row
        for row in csv.reader(
            [line.strip() for line in text.splitlines() if line.startswith('   "')],
            skipinitialspace=True,
        )
    }
    assert "When to Use" in text
    assert ":class: optimizer-catalog" in text
    for name in ("pymoo / NSGA2", "pymoo / NSGA3"):
        assert rows[name][1].startswith("Multi; bounds")
        assert rows[f"{name} (constrained flow)"][1].startswith(
            "Multi; one-sided constraints"
        )
    for name in (
        "PyGMO / nsga2",
        "PyGMO / moead",
        "PyGMO / moead_gen",
        "PyGMO / nspso",
    ):
        assert "Multi (native)" in rows[name][1]
        assert "constraints" not in rows[name][1]
    for name in (
        "SciPy / differential_evolution",
        "Optuna / TPESampler",
        "DEAP / eaSimple",
    ):
        assert "mixed" in rows[name][1] and "constraints" in rows[name][1]
    assert "unconstrained" in rows["SciPy / BFGS"][1]
    assert "constraints" not in rows["SciPy / BFGS"][1]


def test_catalog_tables_group_every_optimizer_once_and_omit_empty_categories():
    text = (ROOT / "docs/source/optimizer_catalog.rst").read_text()
    tables = text.split("Availability And Batching", 1)[0]
    pattern = r"^(Single-Objective|Multi-Objective), (Unconstrained|Constrained), (Local|Global)\n~+\n"
    headings = list(re.finditer(pattern, tables, re.MULTILINE))
    assert [match.group(0).splitlines()[0] for match in headings] == [
        "Single-Objective, Unconstrained, Local",
        "Single-Objective, Unconstrained, Global",
        "Single-Objective, Constrained, Local",
        "Single-Objective, Constrained, Global",
        "Multi-Objective, Unconstrained, Global",
        "Multi-Objective, Constrained, Global",
    ]
    local_adaptation = {
        "pycma / CMA-ES",
        "pymoo / CMAES",
        "PyGMO / cmaes",
        "PyGMO / xnes",
        "PyGMO / sea",
        "PyGMO / compass_search",
        "Nevergrad / CMA",
        "Nevergrad / OnePlusOne",
        "Nevergrad / TBPSA",
    }
    seen = set()
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(tables)
        section = tables[heading.end() : end]
        assert section.count(".. csv-table::") == 1
        rows = list(
            csv.reader(
                [
                    line.strip()
                    for line in section.splitlines()
                    if line.startswith('   "')
                ],
                skipinitialspace=True,
            )
        )
        assert rows
        for row in rows:
            name, capabilities = row[:2]
            assert name not in seen
            seen.add(name)
            assert capabilities.startswith("Multi") == (
                heading.group(1) == "Multi-Objective"
            )
            constrained = (
                "constraints" in capabilities
                or "inequalities" in capabilities
                or name in {"SciPy / default", "PyGMO / nlopt"}
            )
            assert constrained == (heading.group(2) == "Constrained"), name
            local = (
                name.startswith(("iwopy /", "NLopt /"))
                or (
                    name.startswith("SciPy /")
                    and name != "SciPy / differential_evolution"
                )
                or name in local_adaptation
                or name in {"PyGMO / nlopt", "PyGMO / ipopt"}
            )
            assert local == (heading.group(3) == "Local"), name
    assert len(seen) == len(SCRIPTS)


def _factory_selectors(relative_path, variable):
    tree = ast.parse((ROOT / relative_path).read_text())
    selectors = set()
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == variable
        ):
            for value in node.comparators:
                if isinstance(value, ast.Constant) and isinstance(value.value, str):
                    selectors.add(value.value)
                elif isinstance(value, ast.Tuple):
                    selectors.update(
                        element.value
                        for element in value.elts
                        if isinstance(element, ast.Constant)
                    )
    return selectors


def _script_selectors(family):
    selectors = set()
    for path in SCRIPTS:
        for node in ast.walk(ast.parse(path.read_text())):
            if (
                not isinstance(node, ast.Call)
                or not isinstance(node.func, ast.Name)
                or node.func.id != "create_optimizer"
            ):
                continue
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            if ast.literal_eval(keywords["family"]) == family and isinstance(
                keywords["optimizer"], ast.Constant
            ):
                selectors.add(keywords["optimizer"].value)
    return selectors


def test_every_script_has_an_explicit_constructor_in_main_guard():
    for path in SCRIPTS:
        tree = ast.parse(path.read_text())
        assert any(
            isinstance(node, ast.ImportFrom) and node.module == "iwopy.benchmarks"
            for node in tree.body
        ), path
        guards = [
            node
            for node in tree.body
            if isinstance(node, ast.If)
            and ast.dump(node.test)
            == ast.dump(ast.parse('__name__ == "__main__"', mode="eval").body)
        ]
        assert len(guards) == 1, path
        assignments = [
            node
            for node in guards[0].body
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "solver"
                for target in node.targets
            )
        ]
        assert len(assignments) == 1, path
        constructor = assignments[0].value
        assert isinstance(constructor, ast.Call) and isinstance(
            constructor.func, ast.Name
        ), path
        assert constructor.func.id == "create_optimizer", path
        assert {"family", "optimizer", "problem"} <= {
            keyword.arg for keyword in constructor.keywords
        }, path
        if path.stem != "scipy_callable":
            assert not any(isinstance(node, ast.FunctionDef) for node in tree.body), (
                path
            )


def test_public_benchmarks_have_no_import_time_initialization(monkeypatch):
    import iwopy

    def forbidden(*args, **kwargs):
        raise AssertionError("Import must not initialize a problem")

    monkeypatch.setattr(iwopy.SimpleProblem, "initialize", forbidden)
    runpy.run_path(str(ROOT / "iwopy/benchmarks/quadratic.py"))
    assert not (ROOT / "examples/optimizers/case_problems.py").exists()


def _setup_entries(source):
    tree = ast.parse(source)
    constructor = next(
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "solver"
            for target in node.targets
        )
    )
    pending = [
        (keyword.arg, keyword.value)
        for keyword in constructor.keywords
        if keyword.arg != "problem"
    ]
    entries = []
    while pending:
        name, value = pending.pop()
        if isinstance(value, ast.Dict):
            pending.extend(
                (key.value, entry) for key, entry in zip(value.keys, value.values)
            )
        else:
            entries.append((name, value))
    return constructor, entries


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.stem)
def test_setup_comments_only_annotate_individual_values(path):
    source = path.read_text()
    constructor, entries = _setup_entries(source)
    comments = {
        token.start[0]: token.string
        for token in tokenize.generate_tokens(StringIO(source).readline)
        if token.type == tokenize.COMMENT
    }
    value_lines = {value.lineno for _, value in entries}
    for name, value in entries:
        assert value.lineno == value.end_lineno, path
        assert comments.get(value.lineno, "").strip("# "), path
        if name in DEMO_LIMITS and name not in {
            "batch_size",
            "population_multiplier",
            "pop_size",
        }:
            assert "production" in comments[value.lineno].lower(), path
    for line in range(constructor.lineno, constructor.end_lineno + 1):
        if line not in value_lines:
            assert line not in comments, (path, line)


def _check_demo_limits(entries):
    limits_found = set()
    for name, value in entries:
        if name in DEMO_LIMITS:
            assert isinstance(value, ast.Constant), name
            assert type(value.value) is int, name
            assert 0 < value.value <= DEMO_LIMITS[name], name
            if name not in {"pop_size", "population_multiplier", "batch_size"}:
                limits_found.add(name)
    assert limits_found, "Each example needs an explicit short run limit"


@pytest.mark.parametrize("path", SCRIPTS, ids=lambda path: path.stem)
def test_every_example_has_short_explicit_run_limits(path):
    _, entries = _setup_entries(path.read_text())
    _check_demo_limits(entries)


@pytest.mark.parametrize("budget", [0, -1, 10000, True])
def test_demo_limits_reject_invalid_or_long_budgets(budget):
    _, entries = _setup_entries(f"solver = Optimizer_demo(max_evaluations={budget})")
    with pytest.raises(AssertionError, match="max_evaluations"):
        _check_demo_limits(entries)


def test_demo_limits_require_a_stopping_condition():
    _, entries = _setup_entries("solver = Optimizer_demo(seed=42)")
    with pytest.raises(AssertionError, match="explicit short run limit"):
        _check_demo_limits(entries)


def test_iwopy_source_labels_replace_native_example_labels():
    names = {path.stem for path in SCRIPTS}
    assert {"iwopy_gg", "iwopy_slsqp"} <= names
    assert not {"native_gg", "native_slsqp"}.intersection(names)
    catalog = (ROOT / "docs/source/optimizer_catalog.rst").read_text()
    assert '"iwopy / GG"' in catalog and '"iwopy / SLSQP"' in catalog
    assert '"Native /' not in catalog


def test_callable_method_forwards_visible_stopping_controls(monkeypatch):
    namespace = runpy.run_path(str(ROOT / "examples/optimizers/scipy_callable.py"))
    method = namespace["custom_method"]
    observed = {}
    expected = object()

    def capture(*args, **kwargs):
        observed.update(kwargs)
        return expected

    monkeypatch.setitem(method.__globals__, "minimize", capture)
    assert (
        method(lambda point: 0.0, np.zeros(2), bfgs_iterations=7, gtol=1e-5) is expected
    )
    assert observed["options"] == {"maxiter": 7, "gtol": 1e-5}
    method(lambda point: 0.0, np.zeros(2))
    assert observed["options"] == {"maxiter": 100, "gtol": 1e-6}


@pytest.mark.parametrize(
    "package", ["scipy", "pymoo", "pygmo", "nevergrad", "optuna", "deap"]
)
def test_catalog_covers_all_public_selectors(package):
    backend = pytest.importorskip(package)
    from iwopy.interfaces.deap import Optimizer_deap
    from iwopy.interfaces.nevergrad import Optimizer_nevergrad
    from iwopy.interfaces.optuna import Optimizer_optuna

    if package == "scipy":
        from scipy.optimize._minimize import MINIMIZE_METHODS

        assert set(MINIMIZE_METHODS) <= {
            method.lower()
            for method in _script_selectors("scipy")
            if isinstance(method, str)
        }
    elif package == "pymoo":
        assert _factory_selectors("iwopy/interfaces/pymoo/factory.py", "typ") - {
            "iwopy",
            "default",
            "iterations",
            "evaluations",
        } == _script_selectors("pymoo")
    elif package == "pygmo":
        assert _factory_selectors("iwopy/interfaces/pygmo/algos.py", "type") - {
            "nlopt"
        } == _script_selectors("pygmo")
        if not hasattr(backend, "nlopt"):
            pytest.skip("This PyGMO build does not include NLopt")
        nlopt_selectors = set(
            re.findall(r'^``"(\w+)"``\s+``NLOPT_', backend.nlopt.__doc__, re.MULTILINE)
        )
        assert nlopt_selectors
        assert nlopt_selectors == _script_selectors("nlopt")
    elif package == "nevergrad":
        assert set(Optimizer_nevergrad.algorithms) == _script_selectors("nevergrad")
    elif package == "optuna":
        assert set(Optimizer_optuna.samplers) == _script_selectors("optuna")
    else:
        assert set(Optimizer_deap.algorithms) == _script_selectors("deap")


@pytest.mark.parametrize(
    "case,package",
    [
        ("cma_cmaes", "cma"),
        ("pymoo_ga", "pymoo"),
        ("pygmo_de", "pygmo"),
        ("nevergrad_de", "nevergrad"),
        ("optuna_tpe", "optuna"),
        ("deap_simple", "deap"),
    ],
)
def test_missing_backend_skips_only_its_cases(case, package, monkeypatch):
    original = importlib.import_module

    def missing(name, *args, **kwargs):
        if name == package:
            raise ModuleNotFoundError(f"No module named '{package}'", name=package)
        return original(name, *args, **kwargs)

    monkeypatch.setattr(importlib, "import_module", missing)
    with pytest.raises(pytest.skip.Exception, match=package):
        _require_backend(ROOT / f"examples/optimizers/{case}.py")
    _require_backend(ROOT / "examples/optimizers/scipy_de.py")


def _assert_factory_calls(source, path):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and any(
            alias.name == "create_optimizer" for alias in node.names
        ):
            assert node.module == "iwopy", path
        if not isinstance(node, ast.Call):
            continue
        name = ast.unparse(node.func).split(".")[-1]
        assert not name.startswith("Optimizer_") and name not in {"GG", "SLSQP"}, path
        assert not ast.unparse(node.func).endswith("Optimizer.new"), path
        if name == "create_optimizer":
            assert not {
                "algo_pars",
                "problem_pars",
                "setup_pars",
                "term_pars",
                "maxiter",
                "max_iter",
                "maxeval",
                "maxfevals",
                "n_max_gen",
                "n_max_evals",
                "budget",
                "popsize",
                "gen",
            }.intersection(keyword.arg for keyword in node.keywords), path


@pytest.mark.parametrize(
    "source",
    [
        "from iwopy.core import create_optimizer",
        "solver = Optimizer_scipy(problem)",
        "solver = Optimizer.new(problem)",
        "solver = create_optimizer(family='scipy', optimizer='SLSQP', problem=problem, algo_pars={})",
    ],
)
def test_factory_source_guard_rejects_obsolete_creation(source):
    with pytest.raises(AssertionError):
        _assert_factory_calls(source, "synthetic.py")


def test_all_examples_use_factory_with_explicit_parameters():
    for path in (ROOT / "examples").rglob("*.py"):
        _assert_factory_calls(path.read_text(), path)


def test_all_notebooks_use_factory_with_explicit_parameters():
    for path in (ROOT / "notebooks").glob("*.ipynb"):
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                _assert_factory_calls("".join(cell["source"]), path)
