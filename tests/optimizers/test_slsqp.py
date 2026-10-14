from time import perf_counter, sleep

import numpy as np
import pytest
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse import isspmatrix_csr

import iwopy
from iwopy.optimizers import SLSQP
from iwopy.wrappers import LocalFD


class Quadratic(iwopy.SimpleObjective):
    def __init__(
        self,
        problem,
        name="f",
        target=0.0,
        maximize=False,
        ana_deriv=True,
    ):
        super().__init__(
            problem,
            name=name,
            maximize=maximize,
            has_ana_derivs=ana_deriv,
        )
        self.target = target

    def f(self, x):
        return (x - self.target) ** 2

    def g(self, var, x, components):
        return 2.0 * (x - self.target)


class LinearConstraint(iwopy.SimpleConstraint):
    def f(self, x):
        return x

    def g(self, var, x, components):
        return 1.0


class CountingLinearConstraint(LinearConstraint):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.population_calls = 0

    def calc_population(self, *args, **kwargs):
        self.population_calls += 1
        return super().calc_population(*args, **kwargs)


class GuardedLinearConstraint(LinearConstraint):
    def active_components(self, vars_float, lower, upper):
        del vars_float, lower
        _, maximum = self.get_bounds()
        return (
            np.array([0], dtype=int)
            if upper[0] > maximum[0]
            else np.array([], dtype=int)
        )


class UnitScaledLimit(LinearConstraint):
    def __init__(self, problem, name, factor, maximum):
        super().__init__(problem, name, maxs=factor * maximum)
        self.factor = factor
        self.tol = 0.0

    def f(self, coordinate):
        return self.factor * coordinate

    def g(self, var, coordinate, components):
        return self.factor


class NumericalYLimit(iwopy.SimpleConstraint):
    def f(self, horizontal, vertical):
        return vertical

    def vardeps_float(self):
        return np.array([[False, True]])


class NonFiniteQuadratic(Quadratic):
    def g(self, var, x, components):
        return np.nan


class SlowQuadratic(iwopy.SimpleObjective):
    def __init__(self, problem, delay):
        super().__init__(problem, has_ana_derivs=False)
        self.delay = delay
        self.calls = 0

    def f(self, *values):
        self.calls += 1
        sleep(self.delay)
        return np.sum(np.asarray(values) ** 2, axis=0)


class SmallGradientQuadratic(iwopy.SimpleObjective):
    def f(self, x):
        return 1e-8 * (x - 500.0) ** 2

    def g(self, var, x, components):
        return 2e-8 * (x - 500.0)


def make_problem(initial=0.0, target=0.0, ana_deriv=True, maximize=False, **kwargs):
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[initial],
        **kwargs,
    )
    problem.add_objective(
        Quadratic(problem, target=target, ana_deriv=ana_deriv, maximize=maximize)
    )
    return problem


def test_slsqp_solves_analytical_quadratic():
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0], abs=1e-7)
    assert result.objs == pytest.approx([0.0], abs=1e-12)
    assert solver._jacobian_mem
    assert all(isspmatrix_csr(jacobian) for jacobian in solver._jacobian_mem.values())


def test_slsqp_skips_callback_data_without_callbacks(monkeypatch):
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    def unexpected_callback_data(*args, **kwargs):
        raise AssertionError("callback data created without callbacks")

    monkeypatch.setattr(
        "iwopy.optimizers.slsqp.OptimizerCallbackData",
        unexpected_callback_data,
    )

    solver.solve(verbosity=0)


def test_slsqp_is_available_from_optimizer_factory():
    problem = make_problem()

    solver = iwopy.core.create_optimizer("iwopy", "SLSQP", problem)

    assert isinstance(solver, SLSQP)


def test_slsqp_print_info_includes_problem_name(capsys):
    problem = make_problem()
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    solver.print_info()

    output = capsys.readouterr().out
    assert "Problem:" in output
    assert "  name         : quadratic" in output
    assert "  n_vars_int   : 0" in output
    assert "  n_vars_float : 1" in output
    assert "  n_objectives : 1" in output
    assert "  n_obj_cmptns : 1" in output
    assert "  n_constraints: 0" in output
    assert "  n_con_cmptns : 0" in output
    assert "Scipy parameters:" in output
    assert "  tol: 1e-10" in output


def test_problem_initialize_does_not_print_optimizer_summary(capsys):
    problem = make_problem()

    problem.initialize(verbosity=1)

    output = capsys.readouterr().out
    assert "n_vars_int" not in output
    assert "n_vars_float" not in output
    assert "n_objectives" not in output
    assert "n_obj_cmptns" not in output
    assert "n_constraints" not in output
    assert "n_con_cmptns" not in output


def test_slsqp_reports_progress_without_extra_evaluations(capsys, monkeypatch):
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)
    evaluations = 0
    original = problem.evaluate_individual

    def count_evaluations(*args, **kwargs):
        nonlocal evaluations
        evaluations += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(problem, "evaluate_individual", count_evaluations)
    history = iwopy.OptimizationHistory()
    result = solver.solve(verbosity=1, callbacks=[history])
    output = capsys.readouterr().out

    assert result.success
    assert "Running SLSQP" in output
    assert "objective" in output
    assert "max constraint" in output
    assert solver.n_iterations == solver.scipy_results.nit
    assert len(history.states) == solver.n_iterations
    assert [state.iteration for state in history.states] == list(
        range(1, solver.n_iterations + 1)
    )
    assert all(state.event == "iteration" for state in history.states)
    assert history.states[-1].vars_float[0] == pytest.approx(result.vars_float)
    assert history.states[-1].objs[0] == pytest.approx(result.objs)
    assert evaluations == solver.scipy_results.nfev


def test_slsqp_reports_returned_point_as_last_iteration(monkeypatch):
    problem = make_problem(initial=0.0, target=1.0)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem=problem,
    )
    solver.initialize(verbosity=0)

    def return_newer_point(fun, x0, callback, **kwargs):
        del x0, kwargs
        callback_point = np.array([2.0])
        fun(callback_point)
        callback(callback_point)
        final_point = np.array([1.0])
        final_objective = fun(final_point)
        return OptimizeResult(
            x=final_point,
            fun=final_objective,
            success=True,
            nit=1,
            nfev=2,
        )

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", return_newer_point)
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert len(history.states) == 1
    assert history.states[0].iteration == 1
    assert history.states[0].vars_float[0] == pytest.approx(result.vars_float)
    assert history.states[0].objs[0] == pytest.approx(result.objs)


def test_slsqp_keeps_best_feasible_iterate_on_infeasible_exit(monkeypatch):
    problem = make_problem(initial=0.0, target=2.0)
    constraint = LinearConstraint(problem, "bounded", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem)
    solver.initialize(verbosity=0)

    def infeasible_exit(fun, initial, callback, **parameters):
        del initial, parameters
        feasible = np.array([0.5])
        fun(feasible)
        callback(feasible)
        endpoint = np.array([2.0])
        return OptimizeResult(
            x=endpoint, fun=fun(endpoint), success=False, status=9, nit=1, nfev=2
        )

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", infeasible_exit)
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert not result.success
    assert result.vars_float == pytest.approx([0.5])
    assert result.objs == pytest.approx([2.25])
    assert np.all(problem.check_constraints_individual(result.cons))
    assert solver.scipy_results.x == pytest.approx([2.0])
    assert history.states[-1].vars_float[0] == pytest.approx([0.5])


def test_slsqp_progress_is_silent_at_zero_verbosity(capsys):
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem=problem,
    )
    solver.initialize(verbosity=0)

    history = iwopy.OptimizationHistory()
    solver.solve(verbosity=0, callbacks=[history])

    assert capsys.readouterr().out == ""
    assert len(history.states) == solver.scipy_results.nit


def test_slsqp_inward_margin_preserves_public_tolerance():
    problem = make_problem(initial=0.0)
    constraint = LinearConstraint(problem, "lower", mins=1.0, maxs=np.inf)
    constraint.tol = 0.5
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, feasibility_margin=0.01)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([0.51])
    assert constraint.tol == 0.5


@pytest.mark.parametrize("maximize", [False, True])
def test_slsqp_stops_without_feasible_progress(maximize, monkeypatch):
    problem = make_problem(initial=0.0, target=2.0, maximize=maximize)
    constraint = LinearConstraint(problem, "lower", maxs=0.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, stall_iterations=2)
    solver.initialize(verbosity=0)

    def stalled(fun, initial, callback, **parameters):
        del initial, parameters
        for iteration in range(10):
            point = np.array([1.0 + iteration * 0.01])
            fun(point)
            callback(point)
        raise AssertionError("stagnation did not stop the solver")

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", stalled)
    result = solver.solve(verbosity=0)

    assert not result.success
    assert solver.termination_reason == "feasible_stagnation"
    assert solver.n_iterations == 2
    assert result.vars_float == pytest.approx([0.0])


def test_slsqp_restores_infeasible_endpoint(monkeypatch):
    problem = make_problem(initial=0.0, target=2.0)
    constraint = LinearConstraint(problem, "upper", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, restoration_iterations=5, restoration_scales=1.0)
    solver.initialize(verbosity=0)
    original = __import__("scipy.optimize", fromlist=["minimize"]).minimize

    def failed_primary(fun, initial, **parameters):
        if fun.__name__ == "_restoration_objective":
            return original(fun, initial, **parameters)
        point = np.array([2.0])
        return OptimizeResult(x=point, fun=fun(point), success=False, nit=0)

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", failed_primary)
    result = solver.solve(verbosity=0)

    assert not result.success
    assert result.vars_float == pytest.approx([1.0])
    assert np.all(problem.check_constraints_individual(result.cons))
    assert solver.restoration_results is not None


@pytest.mark.parametrize("fd_order", [1, 2])
def test_slsqp_reduces_jacobian_with_full_feasibility_checks(fd_order):
    problem = make_problem(initial=0.0, target=1.0, ana_deriv=False)
    for name, maximum in (("far", 5.0), ("near", 0.25)):
        constraint = GuardedLinearConstraint(problem, name, maxs=maximum)
        constraint.tol = 0.0
        problem.add_constraint(constraint)
    problem = LocalFD(problem, deltas=1e-5, fd_order=fd_order)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, reduce_constraints=True, move_limit=0.5)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([0.25])
    assert result.cons.shape == (2,)
    assert solver.active_constraint_components.tolist() == [1]
    assert solver._get_jacobian(result.vars_float).shape == (2, 1)
    assert np.all(problem.check_constraints_individual(result.cons))


@pytest.mark.parametrize("maximize", [False, True])
def test_slsqp_accumulates_small_feasible_improvements(maximize, monkeypatch):
    problem = make_problem(initial=2.0, maximize=maximize)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, stall_iterations=3, progress_tol=0.3)
    solver.initialize(verbosity=0)

    def improving(fun, initial, callback, **parameters):
        del initial, parameters
        direction = 1.0 if maximize else -1.0
        for iteration in range(1, 5):
            point = solver._to_scaled_vars([np.sqrt(4.0 + direction * 0.2 * iteration)])
            value = fun(point)
            callback(point)
        return OptimizeResult(x=point, fun=value, success=True, nit=4)

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", improving)
    result = solver.solve(verbosity=0)
    assert result.success
    assert solver.n_iterations == 4
    assert solver.termination_reason == "converged"
    assert result.objs == pytest.approx([4.8 if maximize else 3.2])


def test_slsqp_restores_infeasible_start():
    problem = make_problem(initial=2.0)
    constraint = LinearConstraint(problem, "upper", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, restoration_iterations=5, restoration_scales=1.0)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([0.0])
    assert solver.restoration_results is not None
    assert solver.restoration_results.x[0] <= 1.0
    assert solver.n_iterations > solver.scipy_results.nit


def test_slsqp_restoration_failure_stays_infeasible_and_bounded():
    problem = make_problem(initial=2.0, min_values_float=[2.0], max_values_float=[3.0])
    constraint = LinearConstraint(problem, "impossible", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem, restoration_iterations=1, restoration_scales=1.0, max_iterations=2
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert solver.best_feasible is None
    assert not np.all(problem.check_constraints_individual(result.cons))
    assert solver.n_iterations <= 4
    assert solver.restoration_results is not None


@pytest.mark.parametrize(
    "parameters",
    [
        {"feasibility_margin": -1.0},
        {"feasibility_margin": np.nan},
        {"progress_tol": np.inf},
        {"stall_iterations": 0},
        {"stall_iterations": True},
        {"restoration_iterations": -1},
        {"restoration_iterations": 0.5},
        {"move_limit": 0.0},
        {"move_limit": np.inf},
        {"move_limit": [1.0, 2.0]},
    ],
)
def test_slsqp_rejects_invalid_feasible_controls(parameters):
    problem = make_problem()
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, **parameters)
    with pytest.raises(ValueError):
        solver.initialize(verbosity=0)


@pytest.mark.parametrize(
    "components", [np.array([0, 0]), np.array([-1]), np.array([1]), np.array([0.0])]
)
def test_slsqp_rejects_invalid_component_selection(components, monkeypatch):
    problem = make_problem()
    constraint = LinearConstraint(problem, "bounded")
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    monkeypatch.setattr(constraint, "active_components", lambda *args: components)
    solver = SLSQP(problem, reduce_constraints=True)
    with pytest.raises(ValueError, match="Invalid active components"):
        solver.initialize(verbosity=0)


@pytest.mark.parametrize("progress_tol", [1e-8, 1e3])
def test_slsqp_restoration_is_independent_of_objective_progress_units(
    progress_tol, monkeypatch
):
    problem = make_problem(initial=2.0)
    constraint = LinearConstraint(problem, "upper", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem,
        restoration_iterations=5,
        restoration_scales=1.0,
        progress_tol=progress_tol,
    )
    solver.initialize(verbosity=0)
    tolerances = []

    def record_restoration(fun, initial, **parameters):
        if fun.__name__ == "_restoration_objective":
            tolerances.append(parameters["options"]["ftol"])
        return minimize(fun, initial, **parameters)

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", record_restoration)
    result = solver.solve(verbosity=0)

    assert result.success
    assert tolerances == [1e-12]
    assert solver.restoration_results.x[0] <= 1.0


def test_slsqp_full_checks_reject_an_incorrect_omission(monkeypatch):
    problem = make_problem(initial=0.0, target=2.0)
    constraint = LinearConstraint(problem, "bounded", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    monkeypatch.setattr(
        constraint, "active_components", lambda *args: np.array([], dtype=int)
    )
    solver = SLSQP(problem, reduce_constraints=True, move_limit=3.0)
    solver.initialize(verbosity=0)

    def invalid_endpoint(fun, initial, **parameters):
        del initial, parameters
        point = solver._to_scaled_vars([2.0])
        return OptimizeResult(x=point, fun=fun(point), success=True, nit=1)

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", invalid_endpoint)
    result = solver.solve(verbosity=0)

    assert solver.scipy_results.success
    assert not result.success
    assert result.vars_float == pytest.approx([0.0])
    assert np.all(problem.check_constraints_individual(result.cons))


def test_slsqp_repeated_solve_refreshes_guard_and_best_point(monkeypatch):
    problem = make_problem(initial=0.0, target=1.0)
    constraint = GuardedLinearConstraint(problem, "bounded", maxs=5.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, reduce_constraints=True, move_limit=0.5)
    solver.initialize(verbosity=0)

    first = solver.solve(verbosity=0)
    assert first.success
    assert first.vars_float == pytest.approx([0.5])
    assert solver.active_constraint_components.size == 0
    snapshot = solver.best_feasible
    snapshot[0][:] = 99.0
    assert solver.best_feasible[0] == pytest.approx([0.5])

    monkeypatch.setattr(problem, "initial_values_float", lambda: np.array([4.75]))
    second = solver.solve(verbosity=0)

    assert second.success
    assert second.vars_float == pytest.approx([4.25])
    assert solver.active_constraint_components.tolist() == [0]
    assert solver.best_feasible[0] == pytest.approx(second.vars_float)


@pytest.mark.parametrize("initialize_problem", [False, True])
def test_slsqp_preserves_initialization_errors(initialize_problem):
    problem = make_problem()
    if initialize_problem:
        problem.initialize(verbosity=0)
    solver = SLSQP(problem)
    with pytest.raises(ValueError, match="before .* initialization"):
        solver.solve(verbosity=0)


def test_slsqp_stagnation_counts_only_primary_iterations(monkeypatch):
    problem = make_problem(initial=2.0, target=2.0)
    constraint = LinearConstraint(problem, "upper", maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem, restoration_iterations=5, restoration_scales=1.0, stall_iterations=1
    )
    solver.initialize(verbosity=0)

    def stalled_primary(fun, initial, **parameters):
        if fun.__name__ == "_restoration_objective":
            return minimize(fun, initial, **parameters)
        point = solver._to_scaled_vars([1.0])
        for _ in range(2):
            fun(point)
            parameters["callback"](point)
        raise AssertionError("stagnation did not stop the primary phase")

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", stalled_primary)
    result = solver.solve(verbosity=0)

    assert not result.success
    assert solver.termination_reason == "feasible_stagnation"
    assert solver.scipy_results.nit == 2
    assert solver.n_iterations > solver.scipy_results.nit
    assert solver.restoration_results is not None


def test_slsqp_patience_begins_after_first_feasible_point(monkeypatch):
    problem = make_problem(initial=2.0)
    problem.add_constraint(UnitScaledLimit(problem, "upper", 1.0, 1.0))
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, stall_iterations=1)
    solver.initialize(verbosity=0)

    def improving_feasibility(fun, initial, callback, **parameters):
        for physical in (1.5, 1.25, 0.5, 0.5):
            point = solver._to_scaled_vars([physical])
            fun(point)
            callback(point)
        raise AssertionError("feasible stagnation did not stop the solver")

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", improving_feasibility)
    result = solver.solve(verbosity=0)

    assert not result.success
    assert solver.termination_reason == "feasible_stagnation"
    assert solver.n_iterations == 4
    assert result.vars_float == pytest.approx([0.5])


def test_slsqp_no_feasible_point_leaves_stopping_to_native_budget(monkeypatch):
    problem = make_problem(initial=2.0)
    problem.add_constraint(UnitScaledLimit(problem, "upper", 1.0, 1.0))
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, stall_iterations=1, max_iterations=3)
    solver.initialize(verbosity=0)

    def infeasible_progress(fun, initial, callback, **parameters):
        assert parameters["options"]["maxiter"] == 3
        for physical in (1.8, 1.6, 1.4):
            point = solver._to_scaled_vars([physical])
            value = fun(point)
            callback(point)
        return OptimizeResult(x=point, fun=value, success=False, status=9, nit=3)

    monkeypatch.setattr("iwopy.optimizers.slsqp.minimize", infeasible_progress)
    result = solver.solve(verbosity=0)

    assert not result.success
    assert solver.best_feasible is None
    assert solver.termination_reason == "scipy_failure"
    assert solver.n_iterations == 3


@pytest.mark.parametrize("factor", [1e-6, 1.0, 1e6])
def test_slsqp_restoration_is_invariant_to_constraint_units(factor):
    problem = make_problem(initial=2.0, target=-0.5)
    problem.add_constraint(UnitScaledLimit(problem, "first", factor, 1.0))
    problem.add_constraint(UnitScaledLimit(problem, "second", 1.0, 0.0))
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem, restoration_iterations=5, restoration_scales=np.array([factor, 1.0])
    )
    solver.initialize(verbosity=0)
    initial = solver._to_scaled_vars([2.0])

    assert solver._restoration_objective(initial) == pytest.approx(2.5)
    assert solver._restoration_jac(initial) == pytest.approx([6.0])
    step = 1e-6
    numerical = (
        solver._restoration_objective(initial + step)
        - solver._restoration_objective(initial - step)
    ) / (2.0 * step)
    assert solver._restoration_jac(initial)[0] == pytest.approx(numerical)

    result = solver.solve(verbosity=0)
    assert result.success
    assert result.vars_float == pytest.approx([-0.5], abs=1e-7)
    assert np.all(problem.check_constraints_individual(result.cons))


@pytest.mark.parametrize(
    "scales", [None, 0.0, -1.0, np.inf, np.nan, [1.0, 2.0], [[1.0]]]
)
def test_slsqp_rejects_missing_or_invalid_restoration_scales(scales):
    problem = make_problem(initial=2.0)
    problem.add_constraint(UnitScaledLimit(problem, "upper", 1.0, 1.0))
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, restoration_iterations=5, restoration_scales=scales)

    with pytest.raises(ValueError, match="restoration_scales"):
        solver.initialize(verbosity=0)


def test_slsqp_restoration_preserves_exact_equality():
    problem = make_problem(initial=2.0, target=1.0)
    constraint = LinearConstraint(problem, "equality", mins=1.0, maxs=1.0)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, restoration_iterations=5, restoration_scales=10.0)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0])
    assert np.all(problem.check_constraints_individual(result.cons))


@pytest.mark.parametrize("vectorize", [False, True])
def test_slsqp_scaled_restoration_with_selected_numerical_constraint(vectorize):
    base = iwopy.SimpleProblem(
        "subset_restoration",
        float_vars=["x", "y"],
        init_values_float=[3.0, 2.0],
        min_values_float=[-5.0, -5.0],
        max_values_float=[5.0, 5.0],
    )
    base.add_objective(SlowQuadratic(base, delay=0.0))
    base.add_constraint(NumericalYLimit(base, "limit", has_ana_derivs=False, maxs=1.0))
    problem = LocalFD(base, deltas=1e-5, fd_order=2)
    problem.initialize(verbosity=0)
    solver = SLSQP(
        problem,
        vectorize=vectorize,
        restoration_iterations=5,
        restoration_scales=2.0,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([0.0, 0.0], abs=1e-7)
    assert solver.restoration_results is not None
    _, restored = problem.evaluate_individual(
        np.array([], dtype=np.int32), solver.restoration_results.x
    )
    assert np.all(problem.check_constraints_individual(restored))


def test_slsqp_scales_large_physical_variable_ranges():
    problem = iwopy.SimpleProblem(
        "scaled_quadratic",
        float_vars=["x"],
        init_values_float=[0.0],
        min_values_float=[-1000.0],
        max_values_float=[1000.0],
    )
    problem.add_objective(SmallGradientQuadratic(problem))
    problem.initialize()
    solver = SLSQP(
        problem=problem,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([500.0], abs=1e-5)
    assert solver.scipy_results.x == pytest.approx([500.0], abs=1e-5)


@pytest.mark.parametrize(
    "minimum,maximum,expected",
    [
        (1.0, np.inf, 1.0),
        (-np.inf, -1.0, -1.0),
        (1.0, 2.0, 1.0),
        (1.5, 1.5, 1.5),
    ],
)
def test_slsqp_supports_constraint_bounds(minimum, maximum, expected):
    problem = make_problem(initial=0.0)
    constraint = LinearConstraint(problem, "c", mins=minimum, maxs=maximum)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([expected], abs=1e-7)
    assert np.all(problem.check_constraints_individual(result.cons))


def test_slsqp_honors_constraint_tolerance():
    problem = make_problem(initial=0.0)
    constraint = LinearConstraint(problem, "c", mins=1.0, maxs=np.inf)
    problem.add_constraint(constraint)
    constraint.tol = 0.2
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        tol=1e-10,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([0.8], abs=1e-7)
    assert np.all(problem.check_constraints_individual(result.cons))


def test_slsqp_supports_maximization_and_variable_bounds():
    problem = iwopy.SimpleProblem(
        "maximize",
        float_vars=["x"],
        init_values_float=[0.5],
        min_values_float=[-2.0],
        max_values_float=[2.0],
    )
    problem.add_objective(Quadratic(problem, target=0.0, maximize=True))
    problem.initialize()
    solver = SLSQP(
        problem=problem,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert abs(result.vars_float[0]) == pytest.approx(2.0, abs=1e-7)
    assert result.objs == pytest.approx([4.0], abs=1e-7)


def test_slsqp_uses_population_for_local_fd_jacobian(monkeypatch):
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.add_constraint(
        LinearConstraint(
            problem,
            "c",
            mins=1.5,
            maxs=np.inf,
        )
    )
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)
    population_calls = 0
    population_sizes = []
    jacobian_pop_flags = []
    original = problem.apply_population
    original_jacobian = problem.get_jacobian

    def count_population(*args, **kwargs):
        nonlocal population_calls
        population_calls += 1
        population_sizes.append(len(args[1]))
        return original(*args, **kwargs)

    def record_jacobian_mode(*args, **kwargs):
        jacobian_pop_flags.append(kwargs.get("pop"))
        return original_jacobian(*args, **kwargs)

    monkeypatch.setattr(problem, "apply_population", count_population)
    monkeypatch.setattr(problem, "get_jacobian", record_jacobian_mode)
    solver = SLSQP(
        problem=problem,
        tol=1e-8,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.5], abs=1e-4)
    assert np.all(problem.check_constraints_individual(result.cons))
    assert np.all(problem.check_constraints_population(result.cons[None, :]))
    outside_tolerance = np.array([1.49998])
    assert not np.any(problem.check_constraints_individual(outside_tolerance))
    assert not np.any(problem.check_constraints_population(outside_tolerance[None, :]))
    assert population_calls > 0
    assert all(size == 1 for size in population_sizes)
    assert jacobian_pop_flags
    assert all(jacobian_pop_flags)


def test_local_fd_rejects_wrong_center_value_shape():
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)

    with pytest.raises(ValueError, match="func_values shape"):
        problem.get_jacobian(
            np.array([], dtype=np.int32),
            np.array([2.0]),
            func_values=np.array([1.0, 2.0]),
            pop=True,
        )


def test_local_fd_population_evaluates_only_numerical_components():
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    constraint = CountingLinearConstraint(problem, "c", mins=1.5, maxs=np.inf)
    problem.add_constraint(constraint)
    problem.initialize(verbosity=0)
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([2.0]),
        pop=True,
    )

    np.testing.assert_allclose(jacobian.toarray(), [[2.0], [1.0]], atol=1e-4)
    assert constraint.population_calls == 0


@pytest.mark.parametrize(
    ("max_population_size", "expected_sizes"),
    [(2, [2, 1]), (10, [3])],
)
def test_local_fd_bounds_population_size(
    monkeypatch, max_population_size, expected_sizes
):
    problem = iwopy.SimpleProblem(
        "batched_fd",
        float_vars=["x", "y", "z"],
        init_values_float=[1.0, 2.0, 3.0],
    )
    problem.add_objective(SlowQuadratic(problem, delay=0.0))
    problem.initialize(verbosity=0)
    problem = LocalFD(
        problem,
        deltas=1e-5,
        max_population_size=max_population_size,
    )
    problem.initialize(verbosity=0)
    population_sizes = []
    apply_population = problem.apply_population

    def record_population(*args, **kwargs):
        population_sizes.append(len(args[1]))
        return apply_population(*args, **kwargs)

    monkeypatch.setattr(problem, "apply_population", record_population)
    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]),
        func_values=np.array([14.0]),
        pop=True,
    )

    np.testing.assert_allclose(jacobian.toarray(), [[2.0, 4.0, 6.0]], atol=1e-4)
    assert population_sizes == expected_sizes


@pytest.mark.parametrize(
    ("max_population_size", "error"),
    [(0, ValueError), (-1, ValueError), (1.5, TypeError)],
)
def test_local_fd_rejects_invalid_population_size(max_population_size, error):
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)

    with pytest.raises(error, match="max_population_size"):
        LocalFD(
            problem,
            deltas={"x": 1e-5},
            max_population_size=max_population_size,
        )


def test_local_fd_reuses_center_values_in_serial_mode(monkeypatch):
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)
    individual_calls = 0
    original = problem.evaluate_individual

    def count_individual(*args, **kwargs):
        nonlocal individual_calls
        individual_calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(problem, "evaluate_individual", count_individual)
    jacobian = problem.get_jacobian(
        np.array([], dtype=np.int32),
        np.array([2.0]),
        func_values=np.array([1.0]),
        pop=False,
    )

    np.testing.assert_allclose(jacobian.toarray(), [[2.0]], atol=1e-4)
    assert individual_calls == 1


def test_local_fd_second_order_lower_bounds_use_correct_stencil():
    problem = iwopy.SimpleProblem(
        "cubic_lower_bound",
        float_vars=["x"],
        init_values_float=[0.4],
        min_values_float=[0.0],
        max_values_float=[1.0],
    )

    class Cubic(iwopy.SimpleObjective):
        def __init__(self, problem):
            super().__init__(problem, has_ana_derivs=False)

        def f(self, x):
            return x**3

    problem.add_objective(Cubic(problem))
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 0.1}, fd_order={"x": 2})
    problem.initialize(verbosity=0)

    jacobian = problem.get_jacobian(np.array([], dtype=np.int32), np.array([0.4]))

    assert jacobian[0, 0] == pytest.approx(0.48, abs=1e-2)


def test_slsqp_can_disable_population_jacobian(monkeypatch):
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)
    jacobian_pop_flags = []
    original_jacobian = problem.get_jacobian

    def record_jacobian_mode(*args, **kwargs):
        jacobian_pop_flags.append(kwargs.get("pop"))
        return original_jacobian(*args, **kwargs)

    monkeypatch.setattr(problem, "get_jacobian", record_jacobian_mode)
    solver = SLSQP(
        problem=problem,
        vectorize=False,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0], abs=1e-4)
    assert jacobian_pop_flags
    assert not any(jacobian_pop_flags)


def test_local_fd_pop_true_is_faster_than_serial_jacobian_evaluation():
    n_vars = 8
    base_problem = iwopy.SimpleProblem(
        "slow_quadratic",
        float_vars=[f"x{i}" for i in range(n_vars)],
        init_values_float=np.ones(n_vars),
    )
    objective = SlowQuadratic(base_problem, delay=0.005)
    base_problem.add_objective(objective)
    base_problem.initialize()
    problem = LocalFD(base_problem, deltas=1e-4)
    problem.initialize(verbosity=0)
    vars_int = np.array([], dtype=np.int32)
    vars_float = np.ones(n_vars)

    def measure(pop):
        start = perf_counter()
        problem.get_jacobian(vars_int, vars_float, pop=pop)
        return perf_counter() - start

    serial_times = [measure(pop=False) for _ in range(3)]
    serial_calls = objective.calls
    objective.calls = 0
    population_times = [measure(pop=True) for _ in range(3)]

    assert serial_calls == 3 * (n_vars + 1)
    assert objective.calls == 3
    assert np.median(population_times) < 0.5 * np.median(serial_times)


def test_slsqp_returns_candidate_after_iteration_limit():
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        max_iterations=1,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert np.all(np.isfinite(result.vars_float))
    assert np.all(np.isfinite(result.objs))
    assert solver.scipy_results.status != 0


@pytest.mark.parametrize("memory_size", [0, -1, 1.5, True])
def test_slsqp_rejects_invalid_memory_size(memory_size):
    problem = make_problem()
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        memory_size=memory_size,
    )

    with pytest.raises(ValueError, match="memory_size"):
        solver.initialize(verbosity=0)


def test_slsqp_rejects_multiple_objectives():
    problem = make_problem()
    problem.add_objective(Quadratic(problem, name="f2"))
    problem.initialize()
    solver = SLSQP(
        problem=problem,
    )

    with pytest.raises(ValueError, match="Exactly one objective"):
        solver.initialize(verbosity=0)


def test_slsqp_rejects_integer_variables():
    problem = iwopy.SimpleProblem(
        "mixed",
        int_vars=["i"],
        float_vars=["x"],
        init_values_int=[0],
        init_values_float=[0.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.initialize()
    solver = SLSQP(
        problem=problem,
    )

    with pytest.raises(ValueError, match="Integer variables"):
        solver.initialize(verbosity=0)


def test_slsqp_rejects_nonfinite_gradients():
    problem = iwopy.SimpleProblem(
        "nonfinite",
        float_vars=["x"],
        init_values_float=[1.0],
    )
    problem.add_objective(NonFiniteQuadratic(problem))
    problem.initialize()
    solver = SLSQP(
        problem=problem,
    )
    solver.initialize(verbosity=0)

    with pytest.raises(ValueError, match="finite.*Jacobian"):
        solver.solve(verbosity=0)


def test_slsqp_rejects_reserved_scipy_parameters():
    problem = make_problem()
    problem.initialize()
    solver = SLSQP(
        problem=problem,
        jac="2-point",
    )

    with pytest.raises(ValueError, match="managed internally: jac"):
        solver.initialize(verbosity=0)
