import numpy as np
import pytest
from scipy.optimize import OptimizeResult

import iwopy
from iwopy import SimpleConstraint
from iwopy.benchmarks.branin import BraninProblem
from iwopy.interfaces.scipy import Optimizer_scipy
from iwopy.wrappers import LocalFD


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
    def __init__(self, problem, maximize=False):
        super().__init__(problem, maximize=maximize)

    def f(self, x):
        return (x - 1.0) ** 2

    def g(self, var, x, components):
        return 2.0 * (x - 1.0)


class MixedQuadratic(iwopy.SimpleObjective):
    def f(self, i, x):
        return (i - 1) ** 2 + (x - 1.0) ** 2


class FiniteDifferenceQuadratic(iwopy.SimpleObjective):
    def __init__(self, problem):
        super().__init__(problem, has_ana_derivs=False)

    def f(self, x, y):
        return (x - 1.0) ** 2 + (y + 1.0) ** 2


class MissingDerivativeQuadratic(iwopy.SimpleObjective):
    def __init__(self, problem):
        super().__init__(problem, has_ana_derivs=False)

    def f(self, x):
        return (x - 1.0) ** 2


class MutableQuadratic(iwopy.SimpleObjective):
    def __init__(self, problem, target):
        super().__init__(problem)
        self.target = target

    def f(self, x):
        return (x - self.target) ** 2

    def g(self, var, x, components):
        return 2.0 * (x - self.target)


class LowerBound(iwopy.SimpleConstraint):
    def __init__(self, problem):
        super().__init__(problem, "lower", mins=1.5, maxs=np.inf)

    def f(self, x):
        return x

    def g(self, var, x, components):
        return 1.0


class GeneralBounds(iwopy.SimpleConstraint):
    def __init__(self, problem):
        super().__init__(
            problem,
            "general",
            n_components=3,
            mins=[2.0, 0.0, -np.inf],
            maxs=[2.0, np.inf, 1.0],
        )

    def f(self, x):
        return [x, x, x]

    def g(self, var, x, components):
        return np.ones(len(components))


class FiniteDifferenceSumLower(iwopy.SimpleConstraint):
    def __init__(self, problem):
        super().__init__(
            problem,
            "sum_lower",
            mins=1.0,
            maxs=np.inf,
            has_ana_derivs=False,
        )

    def f(self, x, y):
        return x + y


def make_quadratic_problem(initialize=True):
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[3.0],
    )
    problem.add_objective(Quadratic(problem))
    if initialize:
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

    with pytest.raises(ValueError, match="parameters managed internally: callback"):
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


def test_scipy_rejects_integer_variables():
    problem = iwopy.SimpleProblem(
        "mixed",
        int_vars={"i": 1},
        float_vars={"x": 0.0},
        min_values_int={"i": 0},
        max_values_int={"i": 2},
        min_values_float={"x": -1.0},
        max_values_float={"x": 3.0},
    )
    problem.add_objective(MixedQuadratic(problem))
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(problem)

    with pytest.raises(ValueError, match="Integer variables are not supported"):
        solver.initialize(verbosity=0)


def test_scipy_cache_respects_capacity():
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(problem, mem_size=1)
    solver.initialize(verbosity=0)

    solver._get_results(np.array([1.0]))
    solver._get_results(np.array([2.0]))

    assert solver._mem is not None
    assert len(solver._mem) == 1


def test_scipy_uses_population_for_local_fd_gradients(monkeypatch):
    base_problem = iwopy.SimpleProblem(
        "quadratic_fd",
        float_vars=["x", "y"],
        init_values_float=[3.0, -3.0],
    )
    base_problem.add_objective(FiniteDifferenceQuadratic(base_problem))
    base_problem.initialize(verbosity=0)
    problem = LocalFD(base_problem, deltas=1e-5)
    problem.initialize(verbosity=0)
    population_sizes = []
    evaluate_population = problem.evaluate_population

    def record_population(*args, **kwargs):
        population_sizes.append(len(args[1]))
        return evaluate_population(*args, **kwargs)

    monkeypatch.setattr(problem, "evaluate_population", record_population)
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": "L-BFGS-B", "tol": 1e-9},
        vectorized=True,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0, -1.0], abs=1e-4)
    assert population_sizes
    assert all(size == problem.n_vars_float for size in population_sizes)


def test_scipy_supports_constraints_through_local_fd():
    base_problem = iwopy.SimpleProblem(
        "constrained_fd",
        float_vars=["x", "y"],
        init_values_float=[3.0, -3.0],
    )
    base_problem.add_objective(FiniteDifferenceQuadratic(base_problem))
    base_problem.add_constraint(FiniteDifferenceSumLower(base_problem))
    base_problem.initialize(verbosity=0)
    problem = LocalFD(base_problem, deltas=1e-5)
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": "SLSQP", "tol": 1e-9},
    )

    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.5, -0.5], abs=1e-4)
    assert np.all(problem.check_constraints_individual(result.cons))


@pytest.mark.parametrize(
    "method",
    [
        None,
        "CG",
        "BFGS",
        "Newton-CG",
        "L-BFGS-B",
        "TNC",
        "SLSQP",
        "trust-constr",
        "dogleg",
        "trust-ncg",
        "trust-exact",
        "trust-krylov",
    ],
)
def test_scipy_supplies_jacobian_to_gradient_methods(method, monkeypatch):
    problem = make_quadratic_problem()
    scipy_pars = {} if method is None else {"method": method}
    solver = Optimizer_scipy(problem, scipy_pars=scipy_pars)
    solver.initialize(verbosity=0)

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        assert kwargs["jac"] == solver._objective_jac
        np.testing.assert_allclose(kwargs["jac"](x0), [4.0])
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)


@pytest.mark.parametrize("method", ["Nelder-Mead", "Powell", "COBYLA", "COBYQA"])
def test_scipy_omits_jacobian_for_derivative_free_methods(method, monkeypatch):
    problem = make_quadratic_problem()
    solver = Optimizer_scipy(problem, scipy_pars={"method": method})
    solver.initialize(verbosity=0)

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        assert "jac" not in kwargs
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)


def test_scipy_supplies_jacobian_to_custom_method(monkeypatch):
    problem = make_quadratic_problem()

    def custom_method(fun, x0, args=(), **kwargs):
        del fun, args, kwargs
        return OptimizeResult(x=x0, success=False)

    solver = Optimizer_scipy(problem, scipy_pars={"method": custom_method})
    solver.initialize(verbosity=0)

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        assert kwargs["jac"] == solver._objective_jac
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)


@pytest.mark.parametrize("method", ["SLSQP", "trust-constr"])
def test_scipy_honors_constraint_bounds(method):
    problem = make_quadratic_problem(initialize=False)
    problem.add_constraint(LowerBound(problem))
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": method, "tol": 1e-9},
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.5], abs=2e-4)
    assert np.all(problem.check_constraints_individual(result.cons))


def test_scipy_groups_general_constraint_bounds(monkeypatch):
    problem = make_quadratic_problem(initialize=False)
    problem.add_constraint(GeneralBounds(problem))
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(problem, scipy_pars={"method": "SLSQP"})
    solver.initialize(verbosity=0)

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        constraints = kwargs["constraints"]
        assert [constraint["type"] for constraint in constraints] == [
            "eq",
            "ineq",
            "ineq",
        ]
        np.testing.assert_allclose(
            [
                constraint["fun"](x0, *constraint["args"])[0]
                for constraint in constraints
            ],
            [1.0, 3.0, -2.0],
        )
        np.testing.assert_allclose(
            [
                constraint["jac"](x0, *constraint["args"])[0, 0]
                for constraint in constraints
            ],
            [1.0, 1.0, -1.0],
        )
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)


def test_scipy_orients_maximization_objective_and_gradient():
    problem = iwopy.SimpleProblem(
        "maximize",
        float_vars=["x"],
        init_values_float=[0.0],
        min_values_float=[-2.0],
        max_values_float=[2.0],
    )
    problem.add_objective(Quadratic(problem, maximize=True))
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(problem, scipy_pars={"method": "L-BFGS-B"})
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([-2.0], abs=1e-7)
    assert result.objs == pytest.approx([9.0], abs=1e-7)


def test_scipy_can_disable_population_gradients(monkeypatch):
    problem = make_quadratic_problem()
    gradient_pop_flags = []
    get_gradients = problem.get_gradients

    def record_gradient_mode(*args, **kwargs):
        gradient_pop_flags.append(kwargs.get("pop"))
        return get_gradients(*args, **kwargs)

    monkeypatch.setattr(problem, "get_gradients", record_gradient_mode)
    solver = Optimizer_scipy(
        problem,
        scipy_pars={"method": "L-BFGS-B"},
        vectorized=False,
    )
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert gradient_pop_flags
    assert not any(gradient_pop_flags)


def test_scipy_reuses_combined_jacobian_for_constraints(monkeypatch):
    problem = make_quadratic_problem(initialize=False)
    problem.add_constraint(LowerBound(problem))
    problem.initialize(verbosity=0)
    gradient_pop_flags = []
    get_gradients = problem.get_gradients

    def record_gradient_mode(*args, **kwargs):
        gradient_pop_flags.append(kwargs.get("pop"))
        return get_gradients(*args, **kwargs)

    monkeypatch.setattr(problem, "get_gradients", record_gradient_mode)
    solver = Optimizer_scipy(problem, scipy_pars={"method": "SLSQP"})
    solver.initialize(verbosity=0)

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        np.testing.assert_allclose(kwargs["jac"](x0), [4.0])
        constraints = kwargs["constraints"]
        assert len(constraints) == 1
        np.testing.assert_allclose(
            constraints[0]["jac"](x0, *constraints[0]["args"]),
            [[1.0]],
        )
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)

    assert gradient_pop_flags == [True]


def test_scipy_gradient_method_requires_problem_derivatives():
    problem = iwopy.SimpleProblem(
        "missing_derivatives",
        float_vars=["x"],
        init_values_float=[3.0],
    )
    problem.add_objective(MissingDerivativeQuadratic(problem))
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(problem, scipy_pars={"method": "L-BFGS-B"})
    solver.initialize(verbosity=0)

    with pytest.raises(ValueError, match="Failed to determine a finite"):
        solver.solve(verbosity=0)


def test_scipy_clears_value_and_gradient_caches_between_solves(monkeypatch):
    problem = iwopy.SimpleProblem(
        "mutable",
        float_vars=["x"],
        init_values_float=[3.0],
    )
    objective = MutableQuadratic(problem, target=1.0)
    problem.add_objective(objective)
    problem.initialize(verbosity=0)
    solver = Optimizer_scipy(problem, scipy_pars={"method": "L-BFGS-B"})
    solver.initialize(verbosity=0)
    solver._objective(np.array([3.0]))
    solver._objective_jac(np.array([3.0]))
    objective.target = -1.0
    evaluated = {}

    def fake_minimize(fun, x0, *, bounds, **kwargs):
        evaluated["objective"] = fun(x0)
        evaluated["gradient"] = kwargs["jac"](x0)
        return OptimizeResult(success=False)

    monkeypatch.setattr("iwopy.interfaces.scipy.optimizer.minimize", fake_minimize)

    solver.solve(verbosity=0)

    assert evaluated["objective"] == pytest.approx(16.0)
    np.testing.assert_allclose(evaluated["gradient"], [8.0])


if __name__ == "__main__":
    test_branin_slsqp()
