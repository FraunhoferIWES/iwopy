import numpy as np
import pytest
import pygmo as pg

import iwopy
from iwopy.benchmarks.rosenbrock import RosenbrockProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo
from iwopy.interfaces.pygmo import optimizer as pygmo_optimizer
from iwopy.interfaces.pygmo.problem import UDP
from iwopy.wrappers import LocalFD


@pytest.mark.parametrize(
    "algorithm",
    [
        "de",
        "de1220",
        "sade",
        "cmaes",
        "gwo",
        "ihs",
        "pso_gen",
        "xnes",
        "sea",
        "compass_search",
        "moead",
        "moead_gen",
        "nspso",
    ],
)
def test_additional_pygmo_algorithms_are_available(algorithm):
    problem = RosenbrockProblem()
    problem.initialize()

    algo_pars = {"type": algorithm}
    if algorithm != "compass_search":
        algo_pars.update(gen=1, seed=42)
    solver = Optimizer_pygmo(problem, algo_pars=algo_pars)
    solver.initialize(verbosity=0)

    assert solver.algo is not None


def test_ipopt_parameters_accept_numpy_scalars_and_booleans():
    problem = RosenbrockProblem(ana_deriv=True)
    problem.initialize()

    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "ipopt",
            "tol": np.float64(1e-6),
            "max_iter": np.int64(100),
            "print_timing_statistics": np.bool_(False),
        },
    )
    solver.initialize(verbosity=0)

    uda = solver.algo.extract(pg.ipopt)
    assert uda.get_numeric_options()["tol"] == 1e-6
    assert uda.get_integer_options()["max_iter"] == 100
    assert uda.get_string_options()["print_timing_statistics"] == "no"


def test_ipopt_rejects_unsupported_parameter_types():
    problem = RosenbrockProblem(ana_deriv=True)
    problem.initialize()

    with pytest.raises(TypeError, match="unsupported type list"):
        Optimizer_pygmo(
            problem,
            algo_pars={"type": "ipopt", "tol": [1e-6]},
        ).initialize(verbosity=0)


def test_ipopt_solves_analytic_rosenbrock():
    problem = RosenbrockProblem(initial=[-1.0, 1.0], ana_deriv=True)
    problem.initialize()

    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "ipopt", "tol": 1e-8, "max_iter": 100},
    )
    solver.initialize(verbosity=0)
    results = solver.solve(verbosity=0)

    assert results.success
    assert np.allclose(results.vars_float, [1.0, 1.0], atol=1e-5)


def test_ipopt_rejects_callbacks():
    problem = RosenbrockProblem(initial=[-1.0, 1.0], ana_deriv=True)
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "ipopt", "max_iter": 1},
    )
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()

    with pytest.raises(NotImplementedError, match="exact live iteration callbacks"):
        solver.solve(verbosity=0, callbacks=[history])

    assert history.optimizer is None


def test_ipopt_validates_callbacks_before_capability_check():
    problem = RosenbrockProblem(initial=[-1.0, 1.0], ana_deriv=True)
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "ipopt", "max_iter": 1},
    )
    solver.initialize(verbosity=0)

    with pytest.raises(TypeError, match="list"):
        solver.solve(
            verbosity=0,
            callbacks=(iwopy.OptimizationHistory(),),
        )
    with pytest.raises(TypeError, match="OptimizerCallback"):
        solver.solve(verbosity=0, callbacks=[object()])


def test_pygmo_rejects_callback_mode():
    problem = RosenbrockProblem(ana_deriv=True)
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "de"},
        setup_pars={"callback_mode": "iteration"},
    )

    with pytest.raises(ValueError, match="callback_mode is not supported"):
        solver.initialize(verbosity=0)


class MaximizeX(iwopy.SimpleObjective):
    def __init__(self, problem):
        super().__init__(problem, maximize=True)

    def f(self, x):
        return x


@pytest.mark.parametrize("maximize", [False, True])
def test_pygmo_gradient_matches_fitness_direction(maximize):
    class LinearObjective(iwopy.SimpleObjective):
        def __init__(self, problem):
            super().__init__(
                problem,
                maximize=maximize,
                has_ana_derivs=False,
            )

        def f(self, x):
            return x

    problem = iwopy.SimpleProblem(
        "gradient_direction",
        float_vars=["x"],
        init_values_float=[0.0],
        min_values_float=[-1.0],
        max_values_float=[1.0],
    )
    problem.add_objective(LinearObjective(problem))
    problem.initialize(verbosity=0)
    problem = LocalFD(problem, deltas=1e-5)
    problem.initialize(verbosity=0)
    udp = UDP(problem)
    x = np.array([0.25])
    step = 1e-6

    numerical = (udp.fitness(x + step)[0] - udp.fitness(x - step)[0]) / (2 * step)

    assert udp.gradient(x)[0] == pytest.approx(numerical)


def test_pygmo_batch_callback_reports_iwopy_values():
    problem = iwopy.SimpleProblem(
        "maximize_x",
        float_vars=["x"],
        init_values_float=[0.0],
        min_values_float=[-1.0],
        max_values_float=[1.0],
    )
    problem.add_objective(MaximizeX(problem))
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        problem_pars={"pop": True},
        algo_pars={"type": "pso_gen", "gen": 1, "seed": 42},
        setup_pars={"pop_size": 8},
    )
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()

    solver.solve(verbosity=0, callbacks=[history])

    assert history.states
    assert any(len(state.vars_float) > 1 for state in history.states)
    for state in history.states:
        np.testing.assert_allclose(state.objs[:, 0], state.vars_float[:, 0])
        assert state.event == "evaluation"
        assert state.vars_int.shape == (len(state.vars_float), 0)
        assert state.cons.shape == (len(state.vars_float), 0)
    assert history.states[-1].n_evaluations == sum(
        len(state.vars_float) for state in history.states
    )


def test_pygmo_callback_sink_is_removed_after_solve():
    problem = RosenbrockProblem()
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "de", "gen": 1, "seed": 42},
        setup_pars={"pop_size": 8},
    )
    solver.initialize(verbosity=0)
    history = iwopy.OptimizationHistory()

    solver.solve(verbosity=0, callbacks=[history])
    n_states = len(history.states)
    udp = solver.pop.problem.extract(UDP)

    assert udp.callback_sink is None
    solver.pop.problem.fitness(np.zeros(problem.n_vars_float))
    assert len(history.states) == n_states


def test_pygmo_does_not_create_sink_without_callbacks(monkeypatch):
    problem = RosenbrockProblem()
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "de", "gen": 1, "seed": 42},
        setup_pars={"pop_size": 8},
    )
    solver.initialize(verbosity=0)

    def unexpected_sink(*args, **kwargs):
        raise AssertionError("callback sink created without callbacks")

    monkeypatch.setattr(
        pygmo_optimizer._PygmoCallbackSink,
        "__init__",
        unexpected_sink,
    )

    solver.solve(verbosity=0)


class RaisingCallback(iwopy.OptimizerCallback):
    def notify(self, data):
        raise RuntimeError("stop from callback")


def test_pygmo_callback_errors_propagate():
    problem = RosenbrockProblem()
    problem.initialize(verbosity=0)
    solver = Optimizer_pygmo(
        problem,
        algo_pars={"type": "de", "gen": 1, "seed": 42},
        setup_pars={"pop_size": 8},
    )
    solver.initialize(verbosity=0)

    with pytest.raises(RuntimeError, match="stop from callback"):
        solver.solve(verbosity=0, callbacks=[RaisingCallback()])

    udp = solver.pop.problem.extract(UDP)
    assert udp.callback_sink is None
    solver.pop.problem.fitness(np.zeros(problem.n_vars_float))
