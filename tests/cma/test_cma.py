import numpy as np
import pytest

import iwopy
from iwopy.core import OptimizationHistory, OptimizerCallback, SingleObjOptResults
from iwopy.interfaces.cma import Optimizer_cma


class Quadratic(iwopy.SimpleObjective):
    def __init__(self, problem, maximize=False, name="f"):
        super().__init__(problem, name=name, maximize=maximize)
        self.sign = -1.0 if maximize else 1.0

    def f(self, var_x, var_y):
        return self.sign * ((var_x - 1.0) ** 2 + (var_y + 1.0) ** 2)


class BoundedCoordinate(iwopy.SimpleConstraint):
    def __init__(self, problem, name="bound", tol=1e-5, **kwargs):
        super().__init__(problem, name, **kwargs)
        self.tol = tol

    def f(self, var_x, var_y):
        return var_x


class SquaredNorm(iwopy.SimpleConstraint):
    def f(self, var_x, var_y):
        return var_x**2 + var_y**2


class GeneralBounds(iwopy.SimpleConstraint):
    def __init__(self, problem):
        super().__init__(
            problem,
            "general",
            n_components=4,
            mins=[1.5, -np.inf, -1.2, -np.inf],
            maxs=[np.inf, 0.5, -0.8, np.inf],
        )
        self.tol = 0.1

    def f(self, var_x, var_y):
        return [var_x, var_y, var_y, var_x + var_y]


def make_problem(maximize=False, initialize=True, **kwargs):
    parameters = {
        "float_vars": {"x": 3.0, "y": 2.0},
        "min_values_float": [-4.0, -4.0],
        "max_values_float": [4.0, 4.0],
    }
    parameters.update(kwargs)
    problem = iwopy.SimpleProblem("quadratic", **parameters)
    problem.add_objective(Quadratic(problem, maximize=maximize))
    if initialize:
        problem.initialize(verbosity=0)
    return problem


def make_solver(problem=None, **cma_pars):
    options = {"seed": 42, "popsize": 8, "ftarget": 1e-9, "maxiter": 200}
    options.update(cma_pars)
    solver = Optimizer_cma(
        make_problem() if problem is None else problem,
        sigma0=0.8,
        cma_pars=options,
    )
    solver.initialize(verbosity=0)
    return solver


def test_cma_minimizes_with_population_evaluation_only(monkeypatch):
    problem = make_problem()
    populations = []
    evaluate_population = problem.evaluate_population

    def record_population(vars_int, vars_float, **kwargs):
        assert vars_int.dtype == np.int32
        assert vars_int.shape == (len(vars_float), 0)
        assert vars_float.shape == (len(vars_float), 2)
        populations.append(vars_float.copy())
        return evaluate_population(vars_int, vars_float, **kwargs)

    def unexpected_individual(*args, **kwargs):
        raise AssertionError("CMA-ES must evaluate complete populations")

    monkeypatch.setattr(problem, "evaluate_population", record_population)
    monkeypatch.setattr(problem, "evaluate_individual", unexpected_individual)
    solver = Optimizer_cma(
        problem,
        sigma0=0.8,
        cma_pars={"seed": 42, "popsize": 8, "ftarget": 1e-9, "maxiter": 200},
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert isinstance(result, SingleObjOptResults)
    assert result.success
    assert result.vars_int.shape == (0,)
    assert result.vars_float == pytest.approx([1.0, -1.0], abs=1e-4)
    assert result.objs[0] < 1e-9
    assert populations
    assert all(len(population) == 8 for population in populations)
    assert all(np.all(np.abs(population) <= 4.0) for population in populations)


def test_cma_maximizes_and_preserves_callback_objective_signs():
    solver = make_solver(make_problem(maximize=True))
    history = OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert result.vars_float == pytest.approx([1.0, -1.0], abs=1e-4)
    assert -1e-9 < result.objs[0] <= 0.0
    for state in history.states:
        expected = -np.sum((state.vars_float - [1.0, -1.0]) ** 2, axis=1)
        assert state.objs[:, 0] == pytest.approx(expected)


def test_cma_solves_general_constraints_in_populations(monkeypatch):
    problem = make_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)
    history = OptimizationHistory()

    def unexpected_individual(*args, **kwargs):
        raise AssertionError("Constrained CMA-ES must not evaluate individuals")

    monkeypatch.setattr(problem, "evaluate_individual", unexpected_individual)
    solver = make_solver(problem, ftarget=-np.inf, maxiter=400, tolfun=1e-8)

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success, solver.stop_conditions
    assert result.vars_float == pytest.approx([1.4, -1.0], abs=2e-3)
    assert np.all(problem.check_constraints_individual(result.cons))
    for state in history.states:
        assert state.cons.shape == (8, 4)
        assert state.cons == pytest.approx(
            np.column_stack(
                (
                    state.vars_float[:, 0],
                    state.vars_float[:, 1],
                    state.vars_float[:, 1],
                    state.vars_float.sum(axis=1),
                )
            )
        )


def test_cma_solves_nonlinear_constraint():
    problem = make_problem(initialize=False)
    problem.add_constraint(SquaredNorm(problem, "norm", maxs=1.0))
    problem.initialize(verbosity=0)
    solver = make_solver(problem, ftarget=-np.inf, maxiter=400, tolfun=1e-8)

    result = solver.solve(verbosity=0)

    assert result.success, solver.stop_conditions
    assert result.vars_float == pytest.approx([np.sqrt(0.5), -np.sqrt(0.5)], abs=2e-3)
    assert np.all(problem.check_constraints_individual(result.cons))


@pytest.mark.parametrize(
    "bounds",
    [
        ([-np.inf, -np.inf], [np.inf, np.inf]),
        ([-np.inf, -4.0], [4.0, np.inf]),
    ],
)
def test_cma_supports_unbounded_and_one_sided_variables(bounds):
    lower, upper = bounds
    solver = make_solver(make_problem(min_values_float=lower, max_values_float=upper))

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0, -1.0], abs=1e-4)


def test_cma_supports_one_floating_variable():
    class ScalarQuadratic(iwopy.SimpleObjective):
        def f(self, var_x):
            return (var_x - 0.5) ** 2

    problem = iwopy.SimpleProblem("scalar", float_vars={"x": 2.0})
    problem.add_objective(ScalarQuadratic(problem))
    problem.initialize(verbosity=0)

    result = make_solver(problem).solve(verbosity=0)

    assert result.success
    assert result.vars_float.shape == (1,)
    assert result.vars_float == pytest.approx([0.5], abs=1e-4)


@pytest.mark.parametrize("n_pop", [1, 3])
def test_cma_constraint_translation_retains_component_order_and_tolerances(n_pop):
    problem = make_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.add_constraint(
        BoundedCoordinate(
            problem,
            "equality",
            mins=2.0,
            maxs=2.0,
            tol=0.02,
        )
    )
    problem.initialize(verbosity=0)
    solver = make_solver(problem)
    cons = np.tile([1.6, 0.4, -1.0, 99.0, 2.01], (n_pop, 1))

    translated = solver._constraint_values(cons)

    expected = np.tile([-0.2, -0.3, -0.03, -0.2, -0.3, -0.01], (n_pop, 1))
    assert translated.shape == (n_pop, 6)
    assert translated == pytest.approx(expected)
    assert np.all(translated <= 0.0)


def test_cma_tell_receives_matching_fitness_and_constraints(monkeypatch):
    import cma

    problem = make_problem(maximize=True, initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)
    tell = cma.CMAEvolutionStrategy.tell
    received = []

    def record_tell(strategy, solutions, values, constraints_values=None):
        population = np.asarray(solutions)
        expected = np.sum((population - [1.0, -1.0]) ** 2, axis=1)
        assert values == pytest.approx(expected)
        expected_cons = np.column_stack(
            (
                1.4 - population[:, 0],
                -1.3 - population[:, 1],
                population[:, 1] - 0.6,
                population[:, 1] + 0.7,
            )
        )
        assert constraints_values == pytest.approx(expected_cons)
        received.append(len(solutions))
        return tell(strategy, solutions, values, constraints_values=constraints_values)

    monkeypatch.setattr(cma.CMAEvolutionStrategy, "tell", record_tell)
    make_solver(problem, maxiter=2).solve(verbosity=0)

    assert received == [8, 8]


def test_cma_finalizes_selected_solution_and_keeps_problem_results(monkeypatch):
    solver = make_solver()
    finalized = []
    finalization = solver.problem.finalize_individual
    marker = object()

    def record_finalization(vars_int, vars_float, verbosity):
        finalized.append(vars_float.copy())
        _, objs, cons = finalization(vars_int, vars_float, verbosity)
        return marker, objs, cons

    monkeypatch.setattr(solver.problem, "finalize_individual", record_finalization)

    result = solver.solve(verbosity=0)

    assert len(finalized) == 1
    assert result.vars_float == pytest.approx(finalized[0])
    assert result.problem_results is marker
    assert result.cons.shape == (0,)


class RecordingCallback(OptimizerCallback):
    def __init__(self, name, calls):
        super().__init__()
        self.name = name
        self.calls = calls

    def initialize(self, optimizer):
        super().initialize(optimizer)
        self.calls.append((self.name, "initialize"))

    def notify(self, data):
        self.calls.append((self.name, data.iteration))

    def finalize(self, results):
        self.calls.append((self.name, results))


def test_cma_callback_lifecycle_shapes_counts_and_order():
    solver = make_solver(maxiter=3, ftarget=-np.inf)
    calls = []
    callbacks = [RecordingCallback("first", calls), RecordingCallback("second", calls)]
    history = OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[*callbacks, history])

    assert calls == [
        ("first", "initialize"),
        ("second", "initialize"),
        ("first", 1),
        ("second", 1),
        ("first", 2),
        ("second", 2),
        ("first", 3),
        ("second", 3),
        ("first", result),
        ("second", result),
    ]
    assert all(callback.optimizer is solver for callback in callbacks)
    assert [state.n_evaluations for state in history.states] == [8, 16, 24]
    for state in history.states:
        assert state.event == "iteration"
        assert state.vars_int.shape == (8, 0)
        assert state.vars_float.shape == (8, 2)
        assert state.objs.shape == (8, 1)
        assert state.cons.shape == (8, 0)
        assert not state.vars_float.flags.writeable


def test_cma_skips_callback_snapshots_when_unused(monkeypatch):
    solver = make_solver(maxiter=1)

    def unexpected_snapshot(*args, **kwargs):
        raise AssertionError("No snapshot should be made without callbacks")

    monkeypatch.setattr(
        "iwopy.interfaces.cma.optimizer.OptimizerCallbackData",
        unexpected_snapshot,
    )

    solver.solve(verbosity=0)


def test_cma_repeated_solves_start_fresh_and_reset_history():
    solver = make_solver(maxiter=3)
    history = OptimizationHistory()

    first = solver.solve(verbosity=0, callbacks=[history])
    solver.stop_conditions["stale"] = True
    second = solver.solve(verbosity=0, callbacks=[history])

    assert second.vars_float == pytest.approx(first.vars_float)
    assert [state.iteration for state in history.states] == [1, 2, 3]
    assert "stale" not in solver.stop_conditions


@pytest.mark.parametrize(
    "options, reason",
    [
        ({"maxiter": 1}, "maxiter"),
        ({"maxfevals": 1}, "maxfevals"),
        ({"maxiter": 0}, "maxiter"),
        ({"tolfacupx": 1e-12}, "tolfacupx"),
    ],
)
def test_cma_nonconvergence_termination_is_unsuccessful(options, reason):
    solver = make_solver(**options)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert reason in solver.stop_conditions
    assert np.all(np.isfinite(result.vars_float))


def test_cma_no_feasible_solution_is_unsuccessful():
    problem = make_problem(initialize=False)
    problem.add_constraint(
        BoundedCoordinate(problem, "impossible", mins=10.0, maxs=np.inf)
    )
    problem.initialize(verbosity=0)
    solver = make_solver(problem, maxiter=3)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert not np.all(problem.check_constraints_individual(result.cons))
    assert np.all(np.isfinite(result.vars_float))


def test_cma_tolerance_band_equality_is_supported():
    problem = make_problem(initialize=False)
    problem.add_constraint(
        BoundedCoordinate(problem, "equality", mins=2.0, maxs=2.0, tol=0.05)
    )
    problem.initialize(verbosity=0)
    solver = make_solver(problem, ftarget=-np.inf, maxiter=400, tolfun=1e-8)

    result = solver.solve(verbosity=0)

    assert result.success, solver.stop_conditions
    assert result.vars_float == pytest.approx([1.95, -1.0], abs=2e-3)


def test_cma_silent_run_creates_no_backend_files(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    problem = make_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)

    make_solver(problem, maxiter=2).solve(verbosity=0)

    assert not capsys.readouterr().out
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_cma_verbose_run_does_not_enable_constraint_logs(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    problem = make_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)

    make_solver(problem, maxiter=2).solve(verbosity=1)

    assert capsys.readouterr().out
    assert not any(path.is_file() for path in tmp_path.rglob("*"))


def test_cma_zero_generation_constrained_result_is_unsuccessful():
    problem = make_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)
    solver = make_solver(problem, maxiter=0)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert "maxiter" in solver.stop_conditions
    assert np.all(np.isfinite(result.vars_float))


@pytest.mark.parametrize("sigma0", [0.0, -1.0, np.nan, np.inf])
def test_cma_rejects_invalid_sigma(sigma0):
    solver = Optimizer_cma(make_problem(), sigma0=sigma0)

    with pytest.raises(ValueError, match="sigma0 must be positive and finite"):
        solver.initialize(verbosity=0)


@pytest.mark.parametrize(
    "name",
    [
        "bounds",
        "BoundaryHandler",
        "integer_variables",
        "fixed_variables",
        "transformation",
        "termination_callback",
        "verbose",
        "verb_disp",
        "verb_log",
    ],
)
def test_cma_rejects_managed_backend_options(name):
    solver = Optimizer_cma(make_problem(), sigma0=0.5, cma_pars={name: None})

    with pytest.raises(ValueError, match="Internally managed cma_pars"):
        solver.initialize(verbosity=0)


@pytest.mark.parametrize(
    "variables",
    [
        {"int_vars": {"i": 0}},
        {"int_vars": {"i": 0}, "float_vars": {}},
        {"float_vars": {}},
    ],
)
def test_cma_rejects_integer_or_empty_variables(variables):
    problem = make_problem(**variables)

    with pytest.raises(ValueError, match="floating variables and no integer"):
        Optimizer_cma(problem, sigma0=0.5).initialize(verbosity=0)


def test_cma_rejects_multiple_objectives():
    problem = make_problem(initialize=False)
    problem.add_objective(Quadratic(problem, name="other"))
    problem.initialize(verbosity=0)

    with pytest.raises(ValueError, match="exactly one objective"):
        Optimizer_cma(problem, sigma0=0.5).initialize(verbosity=0)


def test_cma_requires_problem_and_optimizer_initialization():
    problem = make_problem(initialize=False)
    solver = Optimizer_cma(problem, sigma0=0.5)

    with pytest.raises(ValueError, match="Initialize the problem first"):
        solver.initialize(verbosity=0)
    with pytest.raises(ValueError, match="before problem initialization"):
        solver.solve(verbosity=0)
    problem.initialize(verbosity=0)
    with pytest.raises(ValueError, match="before solver initialization"):
        solver.solve(verbosity=0)
    solver.initialize(verbosity=0)
    solver.finalize(verbosity=0)
    with pytest.raises(ValueError, match="before solver initialization"):
        solver.solve(verbosity=0)


@pytest.mark.parametrize(
    "parameters, match",
    [
        ({"min_values_float": [4.0, -4.0]}, "Variable bounds"),
        ({"min_values_float": [np.nan, -4.0]}, "Variable bounds"),
        ({"float_vars": {"x": 5.0, "y": 2.0}}, "Initial values"),
        ({"float_vars": {"x": np.nan, "y": 2.0}}, "Initial values"),
    ],
)
def test_cma_rejects_invalid_variable_bounds_or_initial_values(parameters, match):
    with pytest.raises(ValueError, match=match):
        Optimizer_cma(make_problem(**parameters), sigma0=0.5).initialize(verbosity=0)


@pytest.mark.parametrize(
    "parameters, match",
    [
        ({"mins": 2.0, "maxs": 1.0}, "Invalid bounds"),
        ({"mins": np.inf, "maxs": np.inf}, "Invalid bounds"),
        ({"mins": -np.inf, "maxs": -np.inf}, "Invalid bounds"),
        ({"mins": np.nan}, "Invalid bounds"),
        ({"tol": -0.1}, "Tolerance must"),
        ({"tol": np.inf}, "Tolerance must"),
        ({"mins": 1.0, "maxs": 1.0, "tol": 0.0}, "Exact equality"),
    ],
)
def test_cma_rejects_invalid_constraint_bounds_or_tolerances(parameters, match):
    problem = make_problem(initialize=False)
    problem.add_constraint(BoundedCoordinate(problem, "invalid", **parameters))
    problem.initialize(verbosity=0)

    with pytest.raises(ValueError, match=match):
        Optimizer_cma(problem, sigma0=0.5).initialize(verbosity=0)


@pytest.mark.parametrize(
    "family, value",
    [
        ("objectives", np.nan),
        ("objectives", np.inf),
        ("constraints", np.nan),
        ("constraints", np.inf),
        ("objectives", "wrong-shape"),
        ("constraints", "wrong-shape"),
    ],
)
def test_cma_rejects_nonfinite_or_malformed_population_values(
    monkeypatch, family, value
):
    problem = make_problem(initialize=False)
    problem.add_constraint(BoundedCoordinate(problem, "bound"))
    problem.initialize(verbosity=0)
    evaluate_population = problem.evaluate_population

    def invalid_population(vars_int, vars_float):
        objs, cons = evaluate_population(vars_int, vars_float)
        values = objs if family == "objectives" else cons
        if value == "wrong-shape":
            return (objs[:, 0], cons) if family == "objectives" else (objs, cons[:, 0])
        values[0, 0] = value
        return objs, cons

    monkeypatch.setattr(problem, "evaluate_population", invalid_population)

    with pytest.raises(
        ValueError, match=f"Population {family} must be finite with shape"
    ):
        make_solver(problem).solve(verbosity=0)


def test_cma_missing_dependency_has_installation_guidance(monkeypatch):
    def unavailable(name, hint):
        assert name == "cma"
        raise ModuleNotFoundError(f"Module '{name}' not found, maybe try '{hint}'")

    monkeypatch.setattr("iwopy.interfaces.cma.optimizer.import_module", unavailable)

    with pytest.raises(ModuleNotFoundError, match="uv sync --extra cma"):
        Optimizer_cma(make_problem(), sigma0=0.5).initialize(verbosity=0)
