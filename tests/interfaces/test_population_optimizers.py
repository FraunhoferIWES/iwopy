import importlib

import numpy as np
import pytest

import iwopy
from iwopy.core import OptimizationHistory


class Quadratic(iwopy.SimpleObjective):
    def f(self, var_int, var_x, var_y):
        return (var_int - 2) ** 2 + (var_x - 0.25) ** 2 + (var_y + 0.5) ** 2


class Bounds(iwopy.SimpleConstraint):
    def f(self, var_int, var_x, var_y):
        return [var_x, var_y, var_x + var_y, var_int, var_x**2 + var_y**2]


def make_problem(maximize=False, constrained=False, impossible=False):
    problem = iwopy.SimpleProblem(
        "batch_problem",
        int_vars={"index": 0},
        float_vars={"x": 0.0, "y": 0.0},
        min_values_int=[0],
        max_values_int=[4],
        min_values_float=[-2.0, -2.0],
        max_values_float=[2.0, 2.0],
    )
    problem.add_objective(Quadratic(problem, maximize=maximize))
    if constrained or impossible:
        constraint = Bounds(
            problem,
            "bounds",
            n_components=5,
            mins=[-np.inf, -1.0, -1.5, -np.inf, -np.inf],
            maxs=[-3.0 if impossible else 0.5, np.inf, 1.5, np.inf, 1.5],
        )
        constraint.tol = 0.05
        problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    return problem


def make_solver(problem, backend, **kwargs):
    module = "scipy" if backend == "scipy_de" else backend
    cls = getattr(
        importlib.import_module(f"iwopy.interfaces.{module}"), f"Optimizer_{backend}"
    )
    if backend == "scipy_de":
        return cls(
            problem, seed=42, max_iterations=200, population_multiplier=5, **kwargs
        )
    return cls(
        problem,
        max_evaluations=17,
        seed=42,
        **{"pop_size" if backend == "deap" else "batch_size": 4},
        **kwargs,
    )


@pytest.mark.parametrize(
    "iterations, evaluations", [(0, 4), (1, 4), (2, 8), (None, 17)]
)
def test_deap_generation_limit_includes_initial_population(iterations, evaluations):
    solver = make_solver(make_problem(), "deap", max_iterations=iterations)
    solver.initialize(verbosity=0)
    history = OptimizationHistory()
    solver.solve(verbosity=0, callbacks=[history])
    assert solver.n_evaluations == evaluations
    assert len(history.states) == (5 if iterations is None else max(1, iterations))


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("vectorize", [False, True])
def test_vectorize_routes_candidates_and_handles_empty_and_singleton_batches(
    backend, vectorize, monkeypatch
):
    problem = make_problem(constrained=True)
    candidates = np.array([[0.0, 0.0, 0], [0.25, -0.5, 2], [0.5, 0.25, 4]])
    expected = problem.evaluate_population(
        candidates[:, 2:].astype(np.int32), candidates[:, :2]
    )
    solver = make_solver(problem, backend, vectorize=vectorize)
    solver.initialize(verbosity=0)
    recorded = []
    method = "evaluate_population" if vectorize else "evaluate_individual"
    evaluate = getattr(problem, method)

    def record(vars_int, vars_float):
        assert vars_int.dtype == np.int32
        recorded.append((vars_int.copy(), vars_float.copy()))
        return evaluate(vars_int, vars_float)

    def forbidden(*args, **kwargs):
        raise AssertionError("The other evaluation mode must not be used")

    monkeypatch.setattr(problem, method, record)
    monkeypatch.setattr(
        problem,
        "evaluate_individual" if vectorize else "evaluate_population",
        forbidden,
    )
    actual = solver._evaluate(candidates)
    for values, reference in zip(actual, expected):
        np.testing.assert_allclose(values, reference)
    assert len(recorded) == (1 if vectorize else 3)
    empty = solver._evaluate(np.empty((0, 3)))
    assert empty[0].shape == (0, 1) and empty[1].shape == (0, 5)
    assert len(recorded) == (1 if vectorize else 3)
    solver._evaluate(candidates[:1])
    assert len(recorded) == (2 if vectorize else 4)
    assert solver.n_evaluations == 4


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("maximize", [False, True])
def test_vectorize_false_preserves_native_search_callbacks_and_results(
    backend, maximize, monkeypatch
):
    baseline = make_solver(make_problem(maximize=maximize, constrained=True), backend)
    assert baseline.vectorize is True
    baseline.initialize(verbosity=0)
    history = OptimizationHistory()
    expected = baseline.solve(verbosity=0, callbacks=[history])
    problem = make_problem(maximize=maximize, constrained=True)
    evaluate = problem.evaluate_individual
    recorded = []

    def record(vars_int, vars_float):
        assert vars_int.dtype == np.int32 and vars_int.shape == (1,)
        assert vars_float.shape == (2,)
        recorded.append((vars_int.copy(), vars_float.copy()))
        return evaluate(vars_int, vars_float)

    def forbidden(*args, **kwargs):
        raise AssertionError("Scalar mode must not evaluate populations")

    monkeypatch.setattr(problem, "evaluate_individual", record)
    monkeypatch.setattr(problem, "evaluate_population", forbidden)
    solver = make_solver(problem, backend, vectorize=False)
    solver.initialize(verbosity=0)
    scalar_history = OptimizationHistory()
    actual = solver.solve(verbosity=0, callbacks=[scalar_history])
    np.testing.assert_allclose(actual.objs, expected.objs)
    np.testing.assert_allclose(actual.cons, expected.cons)
    np.testing.assert_allclose(actual.vars_float, expected.vars_float)
    np.testing.assert_array_equal(actual.vars_int, expected.vars_int)
    assert actual.success == expected.success
    assert solver.n_evaluations == baseline.n_evaluations == len(recorded)
    assert [state.n_evaluations for state in scalar_history.states] == [
        state.n_evaluations for state in history.states
    ]
    for scalar_state, population_state in zip(scalar_history.states, history.states):
        np.testing.assert_array_equal(scalar_state.vars_int, population_state.vars_int)
        np.testing.assert_allclose(scalar_state.vars_float, population_state.vars_float)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_vectorize_rejects_non_boolean_flags(backend, value):
    with pytest.raises(ValueError, match="vectorize must be a boolean"):
        make_solver(make_problem(), backend, vectorize=value)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("output", ["objective", "constraint"])
def test_vectorize_false_rejects_malformed_individual_outputs(
    backend, output, monkeypatch
):
    problem = make_problem(constrained=True)
    solver = make_solver(problem, backend, vectorize=False)
    solver.initialize(verbosity=0)
    evaluate = problem.evaluate_individual

    def malformed(*args):
        objs, cons = evaluate(*args)
        return (objs[0], cons) if output == "objective" else (objs, cons[:-1])

    monkeypatch.setattr(problem, "evaluate_individual", malformed)
    with pytest.raises(ValueError, match=f"Invalid {output} individual"):
        solver._evaluate(np.array([[0.0, 0.0, 0]]))
    assert solver.n_evaluations == 0


@pytest.mark.parametrize("backend", ["nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("maximize", [False, True])
def test_batches_results_callbacks_and_repeated_solves(backend, maximize, monkeypatch):
    problem = make_problem(maximize)
    recorded = []
    evaluate_population = problem.evaluate_population

    def record(vars_int, vars_float):
        assert vars_int.dtype == np.int32
        assert vars_int.shape == (len(vars_float), 1)
        assert vars_float.shape == (len(vars_int), 2)
        outputs = evaluate_population(vars_int, vars_float)
        recorded.append((vars_int.copy(), vars_float.copy(), outputs[0].copy()))
        return outputs

    def scalar_forbidden(*args, **kwargs):
        raise AssertionError("Search must never evaluate individuals")

    monkeypatch.setattr(problem, "evaluate_population", record)
    monkeypatch.setattr(problem, "evaluate_individual", scalar_forbidden)
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)
    history = OptimizationHistory()
    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert [len(values[0]) for values in recorded] == [4, 4, 4, 4, 1]
    losses = np.concatenate([values[2][:, 0] for values in recorded])
    expected = np.max(losses) if maximize else np.min(losses)
    assert result.objs[0] == pytest.approx(expected)
    assert [state.n_evaluations for state in history.states] == [4, 8, 12, 16, 17]
    for state, (vars_int, vars_float, objs) in zip(history.states, recorded):
        np.testing.assert_array_equal(state.vars_int, vars_int)
        np.testing.assert_array_equal(state.vars_float, vars_float)
        np.testing.assert_array_equal(state.objs, objs)
    repeated = solver.solve(verbosity=0, callbacks=[history])
    assert repeated.success
    assert len(history.states) == 5
    assert solver.n_evaluations == 17


@pytest.mark.parametrize("backend", ["nevergrad", "optuna", "deap"])
@pytest.mark.parametrize(
    "option,value",
    [
        ("max_evaluations", 0),
        ("batch_size", 0),
        ("max_evaluations", True),
        ("batch_size", 1.5),
    ],
)
def test_invalid_budget_options(backend, option, value):
    solver = make_solver(make_problem(), backend)
    setattr(
        solver,
        "pop_size" if backend == "deap" and option == "batch_size" else option,
        value,
    )
    with pytest.raises(ValueError, match="positive integer"):
        solver.initialize(verbosity=0)


@pytest.mark.parametrize(
    "algorithm",
    ["NGOpt", "CMA", "DE", "TwoPointsDE", "PSO", "OnePlusOne", "RandomSearch", "TBPSA"],
)
def test_nevergrad_selectors(algorithm):
    solver = make_solver(make_problem(), "nevergrad", optimizer=algorithm)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0).success


@pytest.mark.parametrize("sampler", ["TPESampler", "RandomSampler"])
def test_optuna_selectors(sampler):
    solver = make_solver(make_problem(), "optuna", optimizer=sampler)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0).success
    assert len(solver.study.trials) == 17
    assert all(trial.state.name == "COMPLETE" for trial in solver.study.trials)


@pytest.mark.parametrize("algorithm", ["eaSimple", "eaMuPlusLambda", "eaMuCommaLambda"])
def test_deap_selectors_and_random_state(algorithm):
    import random

    state = random.getstate()
    solver = make_solver(make_problem(), "deap", optimizer=algorithm)
    solver.initialize(verbosity=0)
    assert solver.solve(verbosity=0).success
    assert random.getstate() == state


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("impossible", [False, True])
def test_general_constraints_and_infeasible_results(backend, impossible):
    problem = make_problem(constrained=True, impossible=impossible)
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    assert result.success is not impossible
    assert np.all(np.isfinite(result.objs))
    assert np.all(np.isfinite(result.cons))
    assert (
        bool(np.all(problem.check_constraints_individual(result.cons, verbosity=0)))
        is not impossible
    )
    assert solver._residuals(np.zeros((3, 5))).shape == (3, 5)
    np.testing.assert_allclose(
        solver._cons_lower, [-np.inf, -1.05, -1.55, -np.inf, -np.inf]
    )
    np.testing.assert_allclose(
        solver._cons_upper, [-2.95 if impossible else 0.55, np.inf, 1.55, np.inf, 1.55]
    )


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("output", ["objective", "constraint"])
@pytest.mark.parametrize("vectorize", [False, True])
def test_nonfinite_outputs_propagate_and_cleanup(
    backend, output, vectorize, monkeypatch
):
    import random

    problem = make_problem(constrained=True)
    solver = make_solver(problem, backend, vectorize=vectorize)
    solver.initialize(verbosity=0)
    method = "evaluate_population" if vectorize else "evaluate_individual"
    evaluate = getattr(problem, method)

    def nonfinite(vars_int, vars_float):
        objs, cons = evaluate(vars_int, vars_float)
        (objs if output == "objective" else cons)[:] = np.nan
        return objs, cons

    monkeypatch.setattr(problem, method, nonfinite)
    state = random.getstate()
    with pytest.raises(ValueError, match=f"nonfinite {output}"):
        solver.solve(verbosity=0)
    assert random.getstate() == state
    if backend == "optuna":
        assert all(trial.state.name == "FAIL" for trial in solver.study.trials)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_no_callback_snapshots_when_disabled(backend, monkeypatch):
    import iwopy.interfaces._population as population

    def forbidden(*args, **kwargs):
        raise AssertionError("No callbacks should allocate no snapshots")

    monkeypatch.setattr(population, "OptimizerCallbackData", forbidden)
    solver = make_solver(make_problem(), backend)
    solver.initialize(verbosity=0)
    solver.solve(verbosity=0)


@pytest.mark.parametrize("backend", ["nevergrad", "optuna", "deap"])
def test_missing_optional_dependency_guidance(backend, monkeypatch):
    module = importlib.import_module(f"iwopy.interfaces.{backend}.optimizer")
    original = module.import_module

    def missing(name, hint):
        assert hint == f"uv sync --extra {backend}"
        return original(f"not_installed_iwopy_{backend}", hint=hint)

    monkeypatch.setattr(module, "import_module", missing)
    with pytest.raises(ModuleNotFoundError, match=f"uv sync --extra {backend}"):
        make_solver(make_problem(), backend).initialize(verbosity=0)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_unbounded_variables_rejected(backend, monkeypatch):
    problem = make_problem()
    monkeypatch.setattr(problem, "min_values_float", lambda: np.array([-np.inf, -2.0]))
    with pytest.raises(ValueError, match="finite"):
        make_solver(problem, backend).initialize(verbosity=0)


@pytest.mark.parametrize("backend", ["nevergrad", "optuna", "deap"])
def test_unknown_selectors(backend):
    with pytest.raises(ValueError, match="Unknown"):
        make_solver(make_problem(), backend, optimizer="unknown").initialize(
            verbosity=0
        )


@pytest.mark.parametrize(
    "backend,key",
    [
        ("nevergrad", "num_workers"),
        ("nevergrad", "parametrization"),
        ("optuna", "constraints_func"),
    ],
)
def test_managed_native_options(backend, key):
    with pytest.raises(ValueError, match="Internally managed"):
        make_solver(make_problem(), backend, **{key: None}).initialize(verbosity=0)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("tolerance", [-1.0, np.inf, np.nan])
def test_invalid_constraint_tolerances(backend, tolerance):
    problem = make_problem(constrained=True)
    problem.cons.functions[0].tol = tolerance
    with pytest.raises(ValueError, match="Tolerance"):
        make_solver(problem, backend).initialize(verbosity=0)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_exact_equality_requires_positive_tolerance(backend):
    problem = make_problem(constrained=True)
    constraint = problem.cons.functions[0]
    constraint._mins[0] = constraint._maxs[0] = 0.0
    constraint.tol = 0.0
    with pytest.raises(ValueError, match="Exact equality"):
        make_solver(problem, backend).initialize(verbosity=0)
    constraint.tol = 0.1
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)
    assert solver._cons_lower[0] == -0.1
    assert solver._cons_upper[0] == 0.1


@pytest.mark.parametrize("backend", ["nevergrad", "optuna", "deap"])
def test_single_candidate_budget(backend):
    solver = make_solver(make_problem(), backend)
    solver.max_evaluations = 1
    solver.initialize(verbosity=0)
    history = OptimizationHistory()
    assert solver.solve(verbosity=0, callbacks=[history]).success
    assert solver.n_evaluations == 1
    assert len(history.states) == 1
    assert history.states[0].vars_int.shape == (1, 1)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_callback_exception_propagates_without_result_finalization(
    backend, monkeypatch
):
    import random

    solver = make_solver(make_problem(), backend)
    solver.initialize(verbosity=0)
    finalized = []

    def fail(data):
        raise RuntimeError("callback failed")

    monkeypatch.setattr(solver, "_notify_callbacks", fail)
    monkeypatch.setattr(
        solver, "_finalize_callbacks", lambda result: finalized.append(result)
    )
    state = random.getstate()
    with pytest.raises(RuntimeError, match="callback failed"):
        solver.solve(verbosity=0, callbacks=[OptimizationHistory()])
    assert not finalized
    assert random.getstate() == state


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
@pytest.mark.parametrize("output", ["objective", "constraint"])
def test_wrong_population_output_shapes(backend, output, monkeypatch):
    problem = make_problem()
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)

    def malformed(vars_int, vars_float):
        count = len(vars_float)
        return (
            (np.zeros((count, 2)), np.empty((count, 0)))
            if output == "objective"
            else (np.zeros((count, 1)), np.zeros((count, 1)))
        )

    monkeypatch.setattr(problem, "evaluate_population", malformed)
    with pytest.raises(ValueError, match=f"{output} population"):
        solver.solve(verbosity=0)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_pure_integer_population_and_finalization_payload(backend, monkeypatch):
    class IntegerObjective(iwopy.SimpleObjective):
        def f(self, var_int):
            return var_int**2

    problem = iwopy.SimpleProblem(
        "integer", int_vars={"index": 1}, min_values_int=[-2], max_values_int=[2]
    )
    problem.add_objective(IntegerObjective(problem))
    problem.initialize(verbosity=0)
    evaluate = problem.evaluate_population
    finalize = problem.finalize_individual
    payload = {"synthetic": True}

    def record(vars_int, vars_float):
        assert vars_int.shape == (len(vars_float), 1)
        assert vars_int.dtype == np.int32
        assert vars_float.shape == (len(vars_int), 0)
        return evaluate(vars_int, vars_float)

    def finalized(vars_int, vars_float, verbosity):
        _, objs, cons = finalize(vars_int, vars_float, verbosity=verbosity)
        return payload, objs, cons

    monkeypatch.setattr(problem, "evaluate_population", record)
    monkeypatch.setattr(problem, "finalize_individual", finalized)
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    assert result.success
    assert result.problem_results is payload
    assert result.vars_float.shape == (0,)


@pytest.mark.parametrize(
    "options",
    [{"pop_size": 1}, {"cxpb": np.nan}, {"mutpb": -1}, {"cxpb": 0.8, "mutpb": 0.8}],
)
def test_invalid_deap_variation_options(options):
    solver = make_solver(make_problem(), "deap")
    for name, value in options.items():
        setattr(solver, name, value)
    with pytest.raises(ValueError, match="DEAP"):
        solver.initialize(verbosity=0)


def test_deap_comma_scheme_selects_only_evaluated_offspring(monkeypatch):
    solver = make_solver(make_problem(), "deap", optimizer="eaMuCommaLambda")
    solver.initialize(verbosity=0)
    select = solver._tools.selTournament
    selected_sizes = []

    def record(individuals, count, tournsize):
        assert all(individual.fitness.valid for individual in individuals)
        selected_sizes.append((len(individuals), count))
        return select(individuals, count, tournsize=tournsize)

    monkeypatch.setattr(solver._tools, "selTournament", record)
    assert solver.solve(verbosity=0).success
    assert selected_sizes == [(4, 4), (4, 4), (4, 4), (1, 1)]


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_integer_unbounded_sentinels_rejected(backend, monkeypatch):
    problem = make_problem()
    monkeypatch.setattr(
        problem, "max_values_int", lambda: np.array([problem.INT_INF], dtype=np.int32)
    )
    with pytest.raises(ValueError, match="finite integer"):
        make_solver(problem, backend).initialize(verbosity=0)


@pytest.mark.parametrize("backend", ["scipy_de", "nevergrad", "optuna", "deap"])
def test_feasible_roundoff_residuals_do_not_override_objective_selection(
    backend, monkeypatch
):
    problem = make_problem(constrained=True)
    solver = make_solver(problem, backend)
    solver.initialize(verbosity=0)
    values = np.array([[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]])
    cons = np.zeros((2, 5))
    cons[1, 0] = np.nextafter(solver._cons_upper[0], np.inf)
    assert np.all(problem.check_constraints_population(cons, verbosity=0))
    monkeypatch.setattr(
        problem, "evaluate_population", lambda *args: (np.array([[9.0], [1.0]]), cons)
    )
    solver._evaluate(values)
    np.testing.assert_array_equal(solver._best_values, values[1])
