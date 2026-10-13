import pytest

from iwopy import SimpleProblem
from iwopy import create_optimizer
from iwopy.core import Optimizer
from iwopy.interfaces.scipy import Optimizer_scipy, Optimizer_scipy_de
from iwopy.optimizers import GG, SLSQP


@pytest.fixture
def problem():
    return SimpleProblem("factory", float_vars=["x"], init_values_float=[0.0])


@pytest.mark.parametrize("algorithm, expected", [("GG", GG), ("SLSQP", SLSQP)])
def test_iwopy_dispatch_without_initialization(problem, algorithm, expected):
    options = {"step_max": 1.0, "step_min": 0.01} if algorithm == "GG" else {}
    solver = create_optimizer("iwopy", algorithm, problem, **options)
    assert isinstance(solver, expected)
    assert solver.problem is problem
    assert not problem.initialized
    assert not solver.initialized
    assert not hasattr(Optimizer, "new")


@pytest.mark.parametrize("algorithm", ["SLSQP", None])
def test_scipy_selector_is_copied_and_options_forwarded(problem, algorithm):
    options = {"max_iterations": 7}
    solver = create_optimizer("scipy", algorithm, problem, memory_size=3, **options)
    assert isinstance(solver, Optimizer_scipy)
    assert solver._native_pars == {"options": {"maxiter": 7}, "method": algorithm}
    assert options == {"max_iterations": 7}
    assert solver.memory_size == 3


def test_scipy_differential_evolution_has_separate_adapter(problem):
    solver = create_optimizer(
        "scipy", "differential_evolution", problem, vectorize=False
    )
    assert isinstance(solver, Optimizer_scipy_de)
    assert solver.vectorize is False
    assert "method" not in solver._native_pars


@pytest.mark.parametrize("family, algorithm", [("unknown", "GG"), ("iwopy", "unknown")])
def test_unknown_factory_routes_raise(problem, family, algorithm):
    with pytest.raises(ValueError, match="[Uu]nknown"):
        create_optimizer(family, algorithm, problem)


def test_duplicate_scipy_method_is_rejected_without_mutation(problem):
    options = {"method": "SLSQP"}
    with pytest.raises(ValueError, match="select the method"):
        create_optimizer("scipy", "SLSQP", problem, **options)
    assert options == {"method": "SLSQP"}


def test_nlopt_family_injects_driver_and_solver_without_mutation(problem):
    from iwopy.interfaces.pygmo import Optimizer_pygmo

    options = {"max_evaluations": 25}
    solver = create_optimizer("nlopt", "lbfgs", problem, **options)
    assert isinstance(solver, Optimizer_pygmo)
    assert solver.optimizer == "nlopt"
    assert solver._algorithm_kwargs == {"optimizer": "lbfgs", "maxeval": 25}
    assert options == {"max_evaluations": 25}
    assert not solver.initialized


@pytest.mark.parametrize("options", [{"nlopt_optimizer": "lbfgs"}])
def test_nlopt_rejects_duplicate_selectors(problem, options):
    original = options.copy()
    with pytest.raises(ValueError, match="do not"):
        create_optimizer("nlopt", "lbfgs", problem, **options)
    assert options == original


def test_pygmo_nlopt_requires_the_nlopt_family(problem):
    with pytest.raises(ValueError, match="family='nlopt'"):
        create_optimizer("pygmo", "nlopt", problem)


@pytest.mark.parametrize(
    "family, algorithm, expected, options",
    [
        ("cma", "CMAES", "Optimizer_cma", {"sigma0": 0.5}),
        ("pymoo", "GA", "Optimizer_pymoo", {}),
        ("pygmo", "de", "Optimizer_pygmo", {}),
        ("nevergrad", "OnePlusOne", "Optimizer_nevergrad", {}),
        ("optuna", "RandomSampler", "Optimizer_optuna", {}),
        ("deap", "eaSimple", "Optimizer_deap", {}),
    ],
)
def test_each_backend_routes_without_initialization(
    problem, family, algorithm, expected, options
):
    solver = create_optimizer(family, algorithm, problem, **options)
    assert type(solver).__name__ == expected
    assert not solver.initialized


@pytest.mark.parametrize("family", ["pygmo", "nlopt"])
def test_pygmo_dependency_is_loaded_at_construction(problem, family, monkeypatch):
    from iwopy.interfaces.pygmo import imports

    def missing(*args, **kwargs):
        raise ImportError("Install the pygmo extra")

    monkeypatch.setattr(imports, "load", missing)
    with pytest.raises(ImportError, match="pygmo extra"):
        create_optimizer(family, "de" if family == "pygmo" else "lbfgs", problem)


def test_unselected_optional_backends_are_not_imported(problem, monkeypatch):
    import builtins

    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.split(".", 1)[0] in {
            "cma",
            "pymoo",
            "pygmo",
            "nevergrad",
            "optuna",
            "deap",
        }:
            raise ImportError("Optional backend was imported")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    for family, algorithm, options in [
        ("cma", "CMAES", {"sigma0": 0.5}),
        ("pymoo", "GA", {}),
        ("nevergrad", "OnePlusOne", {}),
        ("optuna", "RandomSampler", {}),
        ("deap", "eaSimple", {}),
        ("scipy", "SLSQP", {}),
    ]:
        assert not create_optimizer(family, algorithm, problem, **options).initialized


@pytest.mark.parametrize("family", ["pymoo", "pygmo"])
def test_algorithm_keywords_are_copied(problem, family):
    algorithm = "GA" if family == "pymoo" else "de"
    options = {"pop_size": 20} if family == "pymoo" else {"max_iterations": 3}
    solver = create_optimizer(family, algorithm, problem, **options)
    assert solver._algorithm_kwargs == (
        {"pop_size": 20} if family == "pymoo" else {"gen": 3}
    )
    assert "type" not in options


@pytest.mark.parametrize(
    "family, algorithm, options",
    [
        ("pymoo", "GA", {"type": "GA"}),
        ("pygmo", "de", {"type": "de"}),
        ("nevergrad", "NGOpt", {"algorithm": "NGOpt"}),
        ("optuna", "TPESampler", {"sampler": "TPESampler"}),
        ("deap", "eaSimple", {"algorithm": "eaSimple"}),
    ],
)
def test_factory_owned_fields_cannot_be_repeated(problem, family, algorithm, options):
    with pytest.raises((ValueError, TypeError)):
        create_optimizer(family, algorithm, problem, **options)


@pytest.mark.parametrize(
    "family, algorithm",
    [("scipy", 3), ("nlopt", None), ("pymoo", None), ("cma", "wrong")],
)
def test_invalid_selector_forms_are_contextual(problem, family, algorithm):
    with pytest.raises(ValueError, match=family):
        create_optimizer(family, algorithm, problem)


def test_scipy_callable_is_forwarded_without_alias(problem):
    def method(*args, **kwargs):
        raise AssertionError("Construction must not call the selected method")

    solver = create_optimizer("scipy", method, problem)
    assert solver._native_pars["method"] is method


def test_cma_requires_its_initial_step_size(problem):
    with pytest.raises(TypeError, match="sigma0"):
        create_optimizer("cma", "CMAES", problem)


def test_missing_selected_adapter_preserves_import_error(problem, monkeypatch):
    from iwopy.core import optimizer_factory

    def missing(module):
        assert module == "iwopy.interfaces.cma"
        raise ImportError("selected adapter is unavailable")

    monkeypatch.setattr(optimizer_factory, "import_module", missing)
    with pytest.raises(ImportError, match="selected adapter is unavailable"):
        create_optimizer("cma", "CMAES", problem, sigma0=0.5)


@pytest.mark.parametrize(
    "family, algorithm",
    [
        ("scipy", "SLSQP"),
        ("pymoo", "GA"),
        ("pygmo", "de"),
        ("nevergrad", "NGOpt"),
        ("optuna", "TPESampler"),
        ("nlopt", "lbfgs"),
    ],
)
@pytest.mark.parametrize(
    "keyword", ["algo_pars", "problem_pars", "setup_pars", "term_pars"]
)
def test_removed_option_containers_are_rejected(problem, family, algorithm, keyword):
    with pytest.raises(TypeError, match=keyword):
        create_optimizer(family, algorithm, problem, **{keyword: {}})


@pytest.mark.parametrize(
    "adapter, options",
    [
        (GG, {"step_max": 1.0, "step_min": 0.01, "max_iterations": 3}),
        (SLSQP, {"max_iterations": 3}),
        (Optimizer_scipy, {"optimizer": "SLSQP", "max_iterations": 3}),
    ],
)
def test_explicit_local_parameters_solve(adapter, options):
    from iwopy.benchmarks import QuadraticProblem

    solver = adapter(QuadraticProblem(), vectorize=False, memory_size=7, **options)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0) is not None
    assert not solver.vectorize


@pytest.mark.parametrize("family", ["scipy_de", "cma", "nevergrad", "optuna", "deap"])
def test_explicit_population_parameters_solve(family):
    from importlib import import_module
    from iwopy.benchmarks import QuadraticProblem

    package = "scipy" if family == "scipy_de" else family
    adapter = getattr(
        import_module(f"iwopy.interfaces.{package}"), f"Optimizer_{family}"
    )
    options = {"seed": 42}
    if family == "cma":
        options.update(sigma0=0.5, pop_size=4, max_iterations=1)
    elif family == "scipy_de":
        options.update(population_multiplier=4, max_iterations=1)
    else:
        options.update(max_evaluations=4)
        options["pop_size" if family == "deap" else "batch_size"] = 2
    solver = adapter(QuadraticProblem(), **options)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0) is not None


@pytest.mark.parametrize(
    "family, optimizer, options",
    [
        (
            "pymoo",
            "GA",
            {"termination": "iterations", "max_iterations": 2, "pop_size": 5},
        ),
        ("pygmo", "de", {"max_iterations": 2, "pop_size": 5, "seed": 42}),
        ("nlopt", "lbfgs", {"max_evaluations": 10}),
    ],
)
def test_explicit_factory_population_routes_solve(family, optimizer, options):
    from iwopy.benchmarks import QuadraticProblem

    solver = create_optimizer(family, optimizer, QuadraticProblem(), **options)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0) is not None


@pytest.mark.parametrize(
    "method, native_key",
    [
        ("Nelder-Mead", "maxfev"),
        ("Powell", "maxfev"),
        ("L-BFGS-B", "maxfun"),
        ("TNC", "maxfun"),
        ("COBYLA", "maxiter"),
        ("COBYQA", "maxfev"),
    ],
)
def test_scipy_evaluation_limit_uses_method_units(problem, method, native_key):
    solver = create_optimizer("scipy", method, problem, max_evaluations=17)
    assert solver._native_pars["options"] == {native_key: 17}


@pytest.mark.parametrize(
    "family, method, keyword, options",
    [
        ("iwopy", "GG", "max_evaluations", {"step_max": 1, "step_min": 0.01}),
        ("iwopy", "SLSQP", "max_evaluations", {}),
        ("scipy", "trust-constr", "max_evaluations", {}),
        ("scipy", "differential_evolution", "max_evaluations", {}),
        ("scipy", "TNC", "max_iterations", {}),
        ("scipy", "COBYLA", "max_iterations", {}),
        ("pygmo", "ipopt", "max_evaluations", {}),
        ("pygmo", "de", "max_evaluations", {}),
        ("pygmo", "compass_search", "max_iterations", {}),
        ("nlopt", "lbfgs", "max_iterations", {}),
        ("nevergrad", "RandomSearch", "max_iterations", {}),
        ("optuna", "RandomSampler", "max_iterations", {}),
    ],
)
def test_unsupported_limits_are_rejected_at_construction(
    problem, family, method, keyword, options
):
    with pytest.raises(ValueError, match=f"{keyword} is not supported"):
        create_optimizer(family, method, problem, **{keyword: 2}, **options)


@pytest.mark.parametrize(
    "family, method, keyword, native_key, attribute, options",
    [
        ("cma", "CMAES", "max_iterations", "maxiter", "_native_pars", {"sigma0": 0.5}),
        (
            "cma",
            "CMAES",
            "max_evaluations",
            "maxfevals",
            "_native_pars",
            {"sigma0": 0.5},
        ),
        ("pygmo", "de", "max_iterations", "gen", "_algorithm_kwargs", {}),
        ("pygmo", "ipopt", "max_iterations", "max_iter", "_algorithm_kwargs", {}),
        (
            "pygmo",
            "compass_search",
            "max_evaluations",
            "max_fevals",
            "_algorithm_kwargs",
            {},
        ),
        ("nlopt", "lbfgs", "max_evaluations", "maxeval", "_algorithm_kwargs", {}),
        ("pymoo", "GA", "max_iterations", "n_max_gen", "_termination_kwargs", {}),
        ("pymoo", "GA", "max_evaluations", "n_max_evals", "_termination_kwargs", {}),
    ],
)
def test_native_limits_are_translated_without_mutation(
    problem, family, method, keyword, native_key, attribute, options
):
    import numpy as np

    controls = {keyword: np.int64(3)}
    solver = create_optimizer(family, method, problem, **controls, **options)
    assert getattr(solver, attribute)[native_key] == 3
    assert type(getattr(solver, attribute)[native_key]) is int
    assert controls == {keyword: 3}


@pytest.mark.parametrize(
    "family, method, options",
    [
        ("iwopy", "GG", {"step_max": 1, "step_min": 0.01}),
        ("iwopy", "SLSQP", {}),
        ("scipy", "SLSQP", {}),
        ("scipy", "differential_evolution", {}),
        ("cma", "CMAES", {"sigma0": 0.5}),
        ("pymoo", "GA", {}),
        ("pygmo", "de", {}),
        ("nevergrad", "RandomSearch", {}),
        ("optuna", "RandomSampler", {}),
        ("deap", "eaSimple", {}),
    ],
)
@pytest.mark.parametrize(
    "controls",
    [
        {"max_iterations": True},
        {"max_iterations": -1},
        {"max_iterations": 1.5},
        {"max_evaluations": False},
        {"max_evaluations": 0},
        {"max_evaluations": 1.5},
        {"vectorize": "false"},
    ],
)
def test_common_controls_reject_invalid_values(
    problem, family, method, options, controls
):
    with pytest.raises(ValueError, match=next(iter(controls))):
        create_optimizer(family, method, problem, **controls, **options)


@pytest.mark.parametrize(
    "keyword",
    [
        "maxiter",
        "max_iter",
        "maxfevals",
        "maxeval",
        "gen",
        "n_max_gen",
        "n_max_evals",
        "budget",
        "maxfun",
        "maxfev",
        "max_fevals",
        "popsize",
        "pop_size",
        "batch_size",
    ],
)
def test_unowned_controls_cannot_escape_to_native_options(problem, keyword):
    with pytest.raises(TypeError, match=keyword):
        create_optimizer("scipy", "SLSQP", problem, **{keyword: 3})


@pytest.mark.parametrize(
    "family, method",
    [("nevergrad", "RandomSearch"), ("optuna", "RandomSampler"), ("deap", "eaSimple")],
)
def test_none_evaluation_limit_selects_adapter_default(problem, family, method):
    solver = create_optimizer(family, method, problem, max_evaluations=None)
    assert solver.max_evaluations == 100


def test_callable_methods_reject_portable_limits(problem):
    with pytest.raises(ValueError, match="max_iterations is not supported"):
        create_optimizer(
            "scipy", lambda *args, **kwargs: None, problem, max_iterations=2
        )
