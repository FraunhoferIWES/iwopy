import numpy as np
import pytest

import iwopy
from iwopy import SimpleConstraint
from iwopy.benchmarks.branin import BraninProblem
from iwopy.interfaces.scipy import Optimizer_scipy


class RC(SimpleConstraint):
    def __init__(self, problem, name="c", ana_deriv=False):
        super().__init__(problem, name, n_components=2, has_ana_derivs=ana_deriv)

    def f(self, x, y):
        return [(x - 1) ** 3 - y + 1, x + y - 3]

    def g(self, var, x, y, components):
        out = np.full(len(components), np.nan, dtype=np.float64)
        for i, ci in enumerate(components):
            # (x-1)**3 - y + 1
            if ci == 0:
                out[i] = 3 * (x - 1) ** 2 if var == 0 else -1

            # x + y - 3
            elif ci == 1:
                out[i] = 1

        return out


def run_branin_slsqp(init_vals, tol):
    prob = BraninProblem(initial_values=init_vals, ana_deriv=True)
    prob.initialize()

    solver = Optimizer_scipy(
        prob,
        scipy_pars={"method": "SLSQP", "tol": tol},
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    return results


def test_branin_slsqp():
    cases = (
        (
            1e-6,
            (1.0, 1.0),
            5e-6,
            (0.0008, 0.002),
        ),
        (
            1e-7,
            (1.0, 1.0),
            5e-7,
            (7e-5, 1.1e-4),
        ),
        (
            1e-6,
            (-3, 12.0),
            7e-6,
            (0.001, 0.0016),
        ),
        (
            1e-6,
            (3, 3.0),
            3e-6,
            (0.0005, 0.00015),
        ),
    )

    resf = 0.397887
    resx = np.array([(-np.pi, 12.275), (np.pi, 2.275), (9.42478, 2.475)])

    for tol, ivals, limf, limxy in cases:
        print("\nENTERING", (tol, ivals, limf, limxy), "\n")

        results = run_branin_slsqp(ivals, tol)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - resf)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float[None, :] - resx)
        delxy = np.min(delxy, axis=0)
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


class Quadratic(iwopy.SimpleObjective):
    def f(self, x):
        return (x - 1.0) ** 2


def make_quadratic_problem():
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[3.0],
    )
    problem.add_objective(Quadratic(problem))
    problem.initialize(verbosity=0)
    return problem


def test_scipy_skips_callback_data_without_callbacks(monkeypatch):
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(problem, scipy_pars={"method": "L-BFGS-B"})
    solver.initialize(verbosity=0)

    def unexpected_callback_data(*args, **kwargs):
        raise AssertionError("callback data created without callbacks")

    monkeypatch.setattr(
        "iwopy.interfaces.scipy.optimizer.OptimizerCallbackData",
        unexpected_callback_data,
    )

    solver.solve(verbosity=0)


@pytest.mark.parametrize("method", ["L-BFGS-B", "COBYLA", "trust-constr"])
def test_scipy_reports_normalized_iterations(method):
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(problem, scipy_pars={"method": method})
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()

    result = solver.solve(verbosity=0, callbacks=[history])

    assert result.success
    assert history.states
    assert [state.iteration for state in history.states] == list(
        range(1, len(history.states) + 1)
    )
    assert all(state.event == "iteration" for state in history.states)
    assert all(state.vars_int.shape == (1, 0) for state in history.states)
    assert all(state.vars_float.shape == (1, 1) for state in history.states)
    assert history.states[-1].vars_float[0] == pytest.approx(result.vars_float)


def test_scipy_rejects_native_callback_parameter():
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": "BFGS", "callback": lambda x: None},
    )

    with pytest.raises(ValueError, match="callback is managed internally"):
        solver.initialize(verbosity=0)


def test_scipy_callback_cache_miss_does_not_evaluate(monkeypatch):
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(problem, scipy_pars={"method": "BFGS"})
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()
    solver._callback_dispatcher.callbacks = [history]
    history.initialize(solver)

    def unexpected_evaluation(*args, **kwargs):
        raise AssertionError("callback triggered a problem evaluation")

    monkeypatch.setattr(problem, "evaluate_individual", unexpected_evaluation)
    solver._dispatch_callback(np.array([7.0]))

    assert history.states[0].objs is None
    assert history.states[0].cons is None


if __name__ == "__main__":
    test_branin_slsqp()
