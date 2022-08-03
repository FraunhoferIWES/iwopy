import numpy as np

from iwopy.interfaces.pymoo import Optimizer_pymoo
from iwopy.benchmarks.branin import BraninProblem
from iwopy.benchmarks.rosenbrock import RosenbrockProblem


def run_branin_ga(type, init_vals, ngen, npop, pop):

    prob = BraninProblem(initial_values=init_vals)
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars=dict(
            vectorize=pop,
        ),
        algo_pars=dict(
            type=type,
            pop_size=npop,
            seed=42,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=ngen,
            ftol=0,
            xtol=0,
        ),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def test_branin_ga():

    cases = (
        (
            "ga",
            100,
            50,
            (1.0, 1.0),
            0.397887,
            6e-5,
            False,
        ),
        (
            "ga",
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

    prob = RosenbrockProblem(initial=inits, constrained=False)
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars=dict(
            vectorize=pop,
        ),
        algo_pars=dict(
            type=type,
            pop_size=npop,
            seed=42,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=ngen,
            ftol=0,
            xtol=0,
        ),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def run_rosen_ga(type, cons, lower, upper, inits, ngen, npop, pop):

    prob = RosenbrockProblem(lower=lower, upper=upper, initial=inits, constrained=cons)
    prob.initialize()

    solver = Optimizer_pymoo(
        prob,
        problem_pars=dict(
            vectorize=pop,
        ),
        algo_pars=dict(
            type=type,
            pop_size=npop,
            seed=42,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=ngen,
            ftol=0,
            xtol=0,
        ),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=1)
    solver.finalize(results)

    return results


def test_rosen0_ga():

    cases = (
        (
            "ga",
            [0.0, 0.0],
            200,
            100,
            0.0,
            0.0,
            False,
        ),
    )

    for typ, inits, ngen, npop, limf, f, pop in cases:

        print("\nENTERING", (typ, inits, ngen, npop, limf, f, pop), "\n")

        results = run_rosen0_ga(typ, inits, ngen, npop, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf


if __name__ == "__main__":
    # test_branin_ga()
    test_rosen0_ga()
