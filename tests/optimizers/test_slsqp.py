from time import perf_counter, sleep

import numpy as np
import pytest
from scipy.optimize import OptimizeResult

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


def make_problem(initial=0.0, target=0.0, ana_deriv=True, **kwargs):
    problem = iwopy.SimpleProblem(
        "quadratic",
        float_vars=["x"],
        init_values_float=[initial],
        **kwargs,
    )
    problem.add_objective(Quadratic(problem, target=target, ana_deriv=ana_deriv))
    return problem


def test_slsqp_solves_analytical_quadratic():
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize()
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0], abs=1e-7)
    assert result.objs == pytest.approx([0.0], abs=1e-12)


def test_slsqp_skips_callback_data_without_callbacks(monkeypatch):
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
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

    solver = iwopy.core.Optimizer.new("SLSQP", problem)

    assert isinstance(solver, SLSQP)


def test_slsqp_print_info_includes_problem_name(capsys):
    problem = make_problem()
    problem.initialize(verbosity=0)
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
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
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
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
    solver = SLSQP(problem)
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


def test_slsqp_progress_is_silent_at_zero_verbosity(capsys):
    problem = make_problem(initial=3.0, target=1.0)
    problem.initialize(verbosity=0)
    solver = SLSQP(problem)
    solver.initialize(verbosity=0)

    history = iwopy.OptimizationHistory()
    solver.solve(verbosity=0, callbacks=[history])

    assert capsys.readouterr().out == ""
    assert len(history.states) == solver.scipy_results.nit


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
    solver = SLSQP(problem)
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
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
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
    solver = SLSQP(problem, scipy_pars={"tol": 1e-10})
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
    solver = SLSQP(problem)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert abs(result.vars_float[0]) == pytest.approx(2.0, abs=1e-7)
    assert result.objs == pytest.approx([4.0], abs=1e-7)


def test_slsqp_uses_population_for_local_fd_gradients(monkeypatch):
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
    gradient_pop_flags = []
    original = problem.apply_population
    original_gradients = problem.get_gradients

    def count_population(*args, **kwargs):
        nonlocal population_calls
        population_calls += 1
        population_sizes.append(len(args[1]))
        return original(*args, **kwargs)

    def record_gradient_mode(*args, **kwargs):
        gradient_pop_flags.append(kwargs.get("pop"))
        return original_gradients(*args, **kwargs)

    monkeypatch.setattr(problem, "apply_population", count_population)
    monkeypatch.setattr(problem, "get_gradients", record_gradient_mode)
    solver = SLSQP(problem, scipy_pars={"tol": 1e-8})
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
    assert gradient_pop_flags
    assert all(gradient_pop_flags)


def test_local_fd_rejects_wrong_center_value_shape():
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)

    with pytest.raises(ValueError, match="func_values shape"):
        problem.get_gradients(
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

    gradients = problem.get_gradients(
        np.array([], dtype=np.int32),
        np.array([2.0]),
        pop=True,
    )

    np.testing.assert_allclose(gradients, [[2.0], [1.0]], atol=1e-4)
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
    gradients = problem.get_gradients(
        np.array([], dtype=np.int32),
        np.array([1.0, 2.0, 3.0]),
        func_values=np.array([14.0]),
        pop=True,
    )

    np.testing.assert_allclose(gradients, [[2.0, 4.0, 6.0]], atol=1e-4)
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
    gradients = problem.get_gradients(
        np.array([], dtype=np.int32),
        np.array([2.0]),
        func_values=np.array([1.0]),
        pop=False,
    )

    np.testing.assert_allclose(gradients, [[2.0]], atol=1e-4)
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

    gradients = problem.get_gradients(np.array([], dtype=np.int32), np.array([0.4]))

    assert gradients[0, 0] == pytest.approx(0.48, abs=1e-2)


def test_slsqp_can_disable_population_gradients(monkeypatch):
    problem = make_problem(initial=2.0, target=1.0, ana_deriv=False)
    problem.initialize()
    problem = LocalFD(problem, deltas={"x": 1e-5})
    problem.initialize(verbosity=0)
    gradient_pop_flags = []
    original_gradients = problem.get_gradients

    def record_gradient_mode(*args, **kwargs):
        gradient_pop_flags.append(kwargs.get("pop"))
        return original_gradients(*args, **kwargs)

    monkeypatch.setattr(problem, "get_gradients", record_gradient_mode)
    solver = SLSQP(problem, vectorized=False)
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert result.success
    assert result.vars_float == pytest.approx([1.0], abs=1e-4)
    assert gradient_pop_flags
    assert not any(gradient_pop_flags)


def test_local_fd_pop_true_is_faster_than_serial_gradient_evaluation():
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
        problem.get_gradients(vars_int, vars_float, pop=pop)
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
    solver = SLSQP(problem, scipy_pars={"options": {"maxiter": 1}})
    solver.initialize(verbosity=0)

    result = solver.solve(verbosity=0)

    assert not result.success
    assert np.all(np.isfinite(result.vars_float))
    assert np.all(np.isfinite(result.objs))
    assert solver.scipy_results.status != 0


@pytest.mark.parametrize("mem_size", [0, -1, 1.5, True])
def test_slsqp_rejects_invalid_memory_size(mem_size):
    problem = make_problem()
    problem.initialize()
    solver = SLSQP(problem, mem_size=mem_size)

    with pytest.raises(ValueError, match="mem_size"):
        solver.initialize(verbosity=0)


def test_slsqp_rejects_multiple_objectives():
    problem = make_problem()
    problem.add_objective(Quadratic(problem, name="f2"))
    problem.initialize()
    solver = SLSQP(problem)

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
    solver = SLSQP(problem)

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
    solver = SLSQP(problem)
    solver.initialize(verbosity=0)

    with pytest.raises(ValueError, match="finite.*Jacobian"):
        solver.solve(verbosity=0)


def test_slsqp_rejects_reserved_scipy_parameters():
    problem = make_problem()
    problem.initialize()
    solver = SLSQP(problem, scipy_pars={"jac": "2-point"})

    with pytest.raises(ValueError, match="managed internally: jac"):
        solver.initialize(verbosity=0)
