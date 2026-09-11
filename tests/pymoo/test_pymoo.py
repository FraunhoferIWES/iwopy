import numpy as np
import pytest

from iwopy import SimpleConstraint, SimpleObjective, SimpleProblem
from iwopy.benchmarks.branin import BraninProblem
from iwopy.benchmarks.rosenbrock import RosenbrockProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo


class RC(SimpleConstraint):
    def __init__(self, problem, name="c", ana_deriv=False):
        super().__init__(problem, name, n_components=2, has_ana_derivs=ana_deriv)

    def f(self, x, y):
        return [(x - 1) ** 3 - y + 1, x + y - 3]

    def g(self, var, x, y, components):
        cmpnts = [0, 1] if components is None else components
        out = np.full(len(cmpnts), np.nan, dtype=np.float64)

        for i, ci in enumerate(cmpnts):
            # (x-1)**3 - y + 1
            if ci == 0:
                out[i] = 3 * (x - 1) ** 2 if var == 0 else -1

            # x + y - 3
            elif ci == 1:
                out[i] = 1

        return out


class IntObjective(SimpleObjective):
    def f(self, *x):
        return sum((xi - 2) ** 2 for xi in x)


class RecordingIntProblem(SimpleProblem):
    def __init__(self):
        super().__init__(
            "int_problem",
            int_vars={"i0": 0, "i1": 0},
            min_values_int={"i0": 0, "i1": 0},
            max_values_int={"i0": 4, "i1": 4},
        )
        self.vars_int_dtypes = []

    def apply_individual(self, vars_int, vars_float):
        self.vars_int_dtypes.append(vars_int.dtype)

    def apply_population(self, vars_int, vars_float):
        self.vars_int_dtypes.append(vars_int.dtype)


@pytest.mark.parametrize("vectorize", [False, True])
def test_integer_ga_keeps_integer_variables(vectorize):
    prob = RecordingIntProblem()
    prob.add_objective(IntObjective(prob))
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={
            "vectorize": vectorize,
        },
        algo_pars={
            "type": "GA",
            "pop_size": 10,
            "seed": 42,
        },
        setup_pars={},
        term_pars=("n_gen", 2),
    )
    solver.initialize()
    solver.solve(verbosity=0)

    assert prob.vars_int_dtypes
    assert all(np.issubdtype(dtype, np.integer) for dtype in prob.vars_int_dtypes)


def test_pso_factory_uses_requested_sampling():
    prob = BraninProblem(initial_values=(1.0, 1.0))
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={"vectorize": False},
        algo_pars={
            "type": "PSO",
            "pop_size": 10,
            "seed": 42,
            "sampling": "float_random",
        },
        setup_pars={},
        term_pars=("n_gen", 1),
    )
    solver.initialize()

    assert hasattr(solver.algo, "initialization")
    assert type(solver.algo.initialization.sampling).__name__ == "FloatRandomSampling"
    assert type(solver.algo.output).__name__ == "SingleObjectiveOutput"


@pytest.mark.parametrize("algorithm", ["DE", "CMAES", "NSGA3"])
def test_factory_supports_additional_pymoo_algorithms(algorithm):
    prob = BraninProblem(initial_values=(1.0, 1.0))
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={"vectorize": False},
        algo_pars={"type": algorithm, "pop_size": 10, "seed": 42},
        setup_pars={},
        term_pars=("n_gen", 1),
    )
    solver.initialize()

    assert type(solver.algo).__name__ == algorithm
    if algorithm == "CMAES":
        assert solver.solve(verbosity=0).success


def run_branin_ga(type, init_vals, ngen, npop, pop):
    prob = BraninProblem(initial_values=init_vals)
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={
            "vectorize": pop,
        },
        algo_pars={
            "type": type,
            "pop_size": npop,
            "seed": 42,
        },
        setup_pars={},
        term_pars=("n_gen", ngen),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def test_branin_ga():
    cases = (
        (
            "GA",
            100,
            50,
            (1.0, 1.0),
            0.397887,
            6e-5,
            False,
        ),
        (
            "GA",
            100,
            50,
            (1.0, 1.0),
            0.397887,
            6e-5,
            True,
        ),
    )

    for typ, ngen, npop, ivals, f, limf, pop in cases:
        print("\nENTERING", (typ, ngen, npop, ivals, f, limf, pop), "\n")

        results = run_branin_ga(typ, ivals, ngen, npop, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf


def run_rosen0_ga(type, inits, ngen, npop, pop):
    prob = RosenbrockProblem(initial=inits, ana_deriv=False)
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={
            "vectorize": pop,
        },
        algo_pars={
            "type": type,
            "pop_size": npop,
            "seed": 42,
        },
        setup_pars={},
        term_pars=("n_gen", ngen),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def run_rosen_ga(type, lower, upper, inits, ngen, npop, pop):
    prob = RosenbrockProblem(lower=lower, upper=upper, initial=inits)
    prob.add_constraint(RC(prob, ana_deriv=False))
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars={
            "vectorize": pop,
        },
        algo_pars={
            "type": type,
            "pop_size": npop,
            "seed": 42,
        },
        setup_pars={},
        term_pars=("n_gen", ngen),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def test_rosen0_ga():
    cases = (
        (
            "GA",
            [0.0, 0.0],
            100,
            150,
            0.03,
            0.0,
            (1.0, 1.0),
            (0.05, 0.1),
            True,
        ),
        (
            "GA",
            [0.0, 0.0],
            200,
            200,
            1e-4,
            0.0,
            (1.0, 1.0),
            (0.03, 0.05),
            True,
        ),
    )

    for typ, inits, ngen, npop, limf, f, xy, limxy, pop in cases:
        print("\nENTERING", (typ, inits, ngen, npop, limf, f, xy, limxy, pop), "\n")

        results = run_rosen0_ga(typ, inits, ngen, npop, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float - np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


def test_rosen_ga():
    cases = (
        (
            "GA",
            (-5, -5),
            (-0.2, -0.2),
            (-3.3, -1.45),
            100,
            50,
            1e-4,
            7.2,
            (-0.2, -0.2),
            (1e-3, 1e-3),
            True,
        ),
        (
            "GA",
            (1.6, 1.3),
            (15.0, 15.0),
            (5.0, 8.0),
            500,
            200,
            5e-3,
            134.92,
            (1.6, 1.4),
            (1e-2, 1e-2),
            True,
        ),
    )

    for typ, low, up, inits, ngen, npop, limf, f, xy, limxy, pop in cases:
        print(
            "\nENTERING",
            (typ, low, up, inits, ngen, npop, limf, f, xy, limxy, pop),
            "\n",
        )

        results = run_rosen_ga(typ, low, up, inits, ngen, npop, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float - np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


if __name__ == "__main__":
    # test_branin_ga()
    test_rosen0_ga()
    # test_rosen_ga()
