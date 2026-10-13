import numpy as np
import pytest

import iwopy
from iwopy.core import OptimizationHistory
from iwopy.interfaces.scipy import Optimizer_scipy_de


class Quadratic(iwopy.SimpleObjective):
    def f(self, var_int, var_x, var_y):
        return (var_int - 2) ** 2 + (var_x - 0.25) ** 2 + (var_y + 0.5) ** 2


def make_problem():
    problem = iwopy.SimpleProblem(
        "mixed_quadratic",
        int_vars={"index": 0},
        float_vars={"x": 0.0, "y": 0.0},
        min_values_int=[0],
        max_values_int=[4],
        min_values_float=[-2.0, -2.0],
        max_values_float=[2.0, 2.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.initialize(verbosity=0)
    return problem


def test_de_evaluates_mixed_populations_without_scalar_calls(monkeypatch):
    problem = make_problem()
    populations = []
    evaluate_population = problem.evaluate_population

    def record(vars_int, vars_float):
        assert vars_int.dtype == np.int32
        assert vars_int.shape == (len(vars_float), 1)
        assert vars_float.shape == (len(vars_int), 2)
        populations.append(len(vars_float))
        return evaluate_population(vars_int, vars_float)

    def scalar_forbidden(*args, **kwargs):
        raise AssertionError("DE must evaluate populations, including singleton calls")

    monkeypatch.setattr(problem, "evaluate_population", record)
    monkeypatch.setattr(problem, "evaluate_individual", scalar_forbidden)
    solver = Optimizer_scipy_de(
        problem=problem,
        seed=42,
        popsize=5,
        maxiter=150,
    )
    solver.initialize(verbosity=0)
    history = OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert result.vars_int == pytest.approx([2])
    assert result.vars_float == pytest.approx([0.25, -0.5], abs=1e-4)
    assert max(populations) > 1
    assert history.states
    assert all(state.event == "iteration" for state in history.states)


def test_constraint_and_feasible_subset_share_one_evaluation(monkeypatch):
    problem = make_problem()
    solver = Optimizer_scipy_de(
        problem=problem,
    )
    solver.initialize(verbosity=0)
    values = np.array([[0.0, 0.1, 0.2], [-0.5, 0.0, 0.5], [1.0, 2.0, 3.0]])
    solver._constraints(values)
    evaluated = solver.n_evaluations
    losses = solver._objective(values[:, [2, 0]])
    assert solver.n_evaluations == evaluated == 3
    assert losses.shape == (2,)
    assert solver._objective(values[:, :0]).shape == (0,)
    assert solver.n_evaluations == evaluated


@pytest.mark.parametrize(
    "option",
    [
        "func",
        "args",
        "polish",
        "workers",
        "vectorized",
        "updating",
        "callback",
        "integrality",
        "x0",
        "bounds",
        "constraints",
        "disp",
    ],
)
def test_internally_managed_de_options(option):
    with pytest.raises((ValueError, TypeError), match="managed|explicit keyword"):
        solver = Optimizer_scipy_de(
            problem=make_problem(),
            **{option: None},
        )
        solver.initialize(verbosity=0)
