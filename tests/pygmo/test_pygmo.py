import numpy as np
import pytest
import pygmo as pg

from iwopy.benchmarks.rosenbrock import RosenbrockProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo


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
