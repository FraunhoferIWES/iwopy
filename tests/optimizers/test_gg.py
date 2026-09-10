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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"step_div_factor": 1.0},
        {"n_max_steps": 0},
        {"memory_size": 0},
        {"step_max": 0.0},
        {"step_min": 0.0},
        {"step_max": 0.1, "step_min": 1.0},
        {"max_iterations": -1},
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

    result = solver.solve(verbosity=0)

    assert not result.success
    assert result.vars_float[0] == 0.0
