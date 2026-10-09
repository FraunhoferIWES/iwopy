import numpy as np
import pytest

import iwopy
from iwopy.optimizers import GG


class Quadratic(iwopy.SimpleObjective):
    def f(self, x):
        return x**2

    def g(self, var, x, components):
        return 2.0 * x


class BoundedConstraint(iwopy.SimpleConstraint):
    def f(self, x):
        return x

    def g(self, var, x, components):
        return 1.0


class ShiftedQuadratic(iwopy.SimpleObjective):
    def f(self, x):
        return (x - 1.0) ** 2

    def g(self, var, x, components):
        return 2.0 * (x - 1.0)


class CurvedConstraint(iwopy.SimpleConstraint):
    def f(self, x):
        return x**2

    def g(self, var, x, components):
        return 2.0 * x


class FlatConstraint(iwopy.SimpleConstraint):
    def f(self, x):
        return np.full_like(x, 3.0)

    def g(self, var, x, components):
        return 0.0


class NonFiniteGradient(Quadratic):
    def g(self, var, x, components):
        return np.nan


def make_problem(initial=0.0):
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[initial],
    )
    problem.add_objective(Quadratic(problem))
    problem.initialize()
    return problem


def test_gg_zero_gradient_returns_zero_step():
    problem = make_problem()
    solver = GG(problem, step_max=1.0, step_min=0.1)

    step = solver._grad2deltax(np.array([0.0]), np.array([1.0]))

    assert np.array_equal(step, np.array([0.0]))


def test_gg_accepts_feasible_initial_optimum():
    problem = make_problem()
    solver = GG(problem, step_max=1.0, step_min=1.0)
    solver.initialize()

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float[0] == 0.0
    assert result.objs[0] == 0.0


@pytest.mark.parametrize("vectorized", [False, True])
@pytest.mark.parametrize("max_iterations, expected", [(1, 0.0), (2, 0.5)])
def test_gg_rejects_infeasible_trials_and_backtracks(
    vectorized, max_iterations, expected
):
    problem = iwopy.SimpleProblem(
        "curved_boundary",
        float_vars=["x"],
        init_values_float=[0.0],
    )
    problem.add_objective(ShiftedQuadratic(problem))
    constraint = CurvedConstraint(problem, "curved", maxs=0.25)
    constraint.tol = 0.0
    problem.add_constraint(constraint)
    problem.initialize()
    solver = GG(
        problem,
        step_max=1.0,
        step_min=0.1,
        n_max_steps=1,
        max_iterations=max_iterations,
        vectorized=vectorized,
    )
    solver.initialize()
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert result.vars_float[0] == pytest.approx(expected)
    assert history.states[0].vars_float[0, 0] == 0.0
    assert all(state.cons[0, 0] <= 0.25 for state in history.states)
    assert all(state.objs[0, 0] <= 1.0 for state in history.states)
    assert len(history.states) == max_iterations


def test_gg_skips_callback_data_without_callbacks(monkeypatch):
    problem = make_problem(initial=2.0)
    solver = GG(
        problem,
        step_max=0.5,
        step_min=0.01,
        max_iterations=1,
    )
    solver.initialize(verbosity=0)

    def unexpected_callback_data(*args, **kwargs):
        raise AssertionError("callback data created without callbacks")

    monkeypatch.setattr(
        "iwopy.optimizers.gg.OptimizerCallbackData",
        unexpected_callback_data,
    )

    solver.solve(verbosity=0)


@pytest.mark.parametrize("vectorized", [False, True])
def test_gg_reports_completed_iterations(vectorized, capsys):
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[1.5],
    )
    problem.add_objective(Quadratic(problem))
    problem.add_constraint(BoundedConstraint(problem, "bounded", mins=1.65, maxs=3.0))
    problem.initialize()
    solver = GG(
        problem,
        step_max=0.1,
        step_min=0.01,
        max_iterations=2,
        n_max_steps=1,
        vectorized=vectorized,
    )
    solver.initialize()
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=1, callbacks=[history])

    assert solver.n_iterations == 2
    assert [state.iteration for state in history.states] == [1, 2]
    assert all(state.event == "iteration" for state in history.states)
    assert history.states[-1].vars_int.shape == (1, 0)
    assert history.states[-1].vars_float[0] == pytest.approx(result.vars_float)
    assert history.states[-1].objs[0] == pytest.approx(result.objs)
    rows = []
    for line in capsys.readouterr().out.splitlines():
        fields = line.split("|")
        try:
            iteration = int(fields[0].strip())
        except (ValueError, IndexError):
            continue
        rows.append((iteration, float(fields[1]), int(fields[2])))
    assert [iteration for iteration, _, _ in rows] == [1, 2]
    assert [objective for _, objective, _ in rows] == pytest.approx(
        [state.objs[0, 0] for state in history.states]
    )
    assert [n_violated for _, _, n_violated in rows] == [1, 0]


def test_gg_zero_iteration_limit_has_no_intermediate_state():
    problem = make_problem(initial=1.0)
    solver = GG(
        problem,
        step_max=0.5,
        step_min=0.01,
        max_iterations=0,
    )
    solver.initialize()
    history = iwopy.OptimizationHistory()

    solver.solve(verbosity=0, callbacks=[history])

    assert solver.n_iterations == 0
    assert history.optimizer is solver
    assert history.states == []


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_div_factor": 1.0},
        {"n_max_steps": 0},
        {"n_max_steps": True},
        {"memory_size": 0},
        {"memory_size": True},
        {"step_max": 0.0},
        {"step_min": 0.0},
        {"step_max": 0.1, "step_min": 1.0},
        {"max_iterations": -1},
        {"max_iterations": True},
    ],
)
def test_gg_rejects_invalid_configuration(kwargs):
    problem = make_problem()
    parameters = {"step_max": 1.0, "step_min": 0.1}
    parameters.update(kwargs)
    solver = GG(problem, **parameters)

    with pytest.raises(ValueError):
        solver.initialize()


def test_gg_uses_constraint_bounds_and_tolerance():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[0.0],
    )
    problem.add_objective(Quadratic(problem))
    constraint = BoundedConstraint(
        problem,
        "c",
        mins=1.0,
        maxs=2.0,
    )
    constraint.tol = 0.1
    problem.add_constraint(constraint)
    problem.initialize()
    solver = GG(problem, step_max=1.0, step_min=0.1)

    assert np.array_equal(
        solver._constraint_side(0.0, 1.0, 2.0),
        np.array([-1.0, 1.0]),
    )
    assert solver._constraint_side(1.05, 1.0, 2.0)[0] == 0.0


def test_gg_recovers_to_lower_constraint_bound():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[0.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.add_constraint(BoundedConstraint(problem, "c", mins=1.0, maxs=2.0))
    problem.initialize()
    solver = GG(problem, step_max=0.5, step_min=0.01, max_iterations=100)
    solver.initialize()

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float[0] >= 1.0 - 1e-5


def test_gg_max_iterations_is_hard_limit_for_infeasible_start():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[0.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.add_constraint(BoundedConstraint(problem, "c", mins=1.0, maxs=2.0))
    problem.initialize()
    solver = GG(problem, step_max=0.5, step_min=0.01, max_iterations=0)
    solver.initialize()

    result = solver.solve(verbosity=0)

    assert not result.success
    assert result.vars_float[0] == 0.0


def test_gg_rejects_nonfinite_gradients():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[1.0],
    )
    problem.add_objective(NonFiniteGradient(problem))
    problem.initialize()
    solver = GG(problem, step_max=0.5, step_min=0.01)
    solver.initialize()

    with pytest.raises(ValueError):
        solver.solve(verbosity=0)


def test_gg_stops_on_infeasible_zero_gradient_constraint():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[0.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.add_constraint(FlatConstraint(problem, "c", mins=1.0, maxs=2.0))
    problem.initialize()
    solver = GG(problem, step_max=0.5, step_min=0.01)
    solver.initialize()
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert not result.success
    assert result.vars_float[0] == 0.0
    assert len(history.states) == 1
    assert history.states[0].vars_float[0, 0] == 0.0
