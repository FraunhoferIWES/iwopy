import numpy as np

from iwopy import SimpleConstraint
from iwopy.interfaces.pymoo import Optimizer_pymoo
from iwopy.benchmarks.branin import BraninProblem
from iwopy.benchmarks.rosenbrock import RosenbrockProblem


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

    prob = RosenbrockProblem(initial=inits, ana_deriv=False)
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


def run_rosen_ga(type, lower, upper, inits, ngen, npop, pop):

    prob = RosenbrockProblem(lower=lower, upper=upper, initial=inits)
    prob.add_constraint(RC(prob, ana_deriv=False))
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
            100,
            50,
            0.03,
            0.0,
            (1.0, 1.0),
            (0.05, 0.1),
            True,
        ),
        (
            "ga",
            [0.0, 0.0],
            200,
            100,
            5e-6,
            0.0,
            (1.0, 1.0),
            (0.003, 0.005),
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
            "ga",
            (-5, -5),
            (-0.2, -0.2),
            (-3.3, -1.45),
            100,
            50,
            1e-10,
            7.2,
            (-0.2, -0.2),
            (1e-10, 1e-10),
            True,
        ),
        (
            "ga",
            (1.6, 1.3),
            (15.0, 15.0),
            (5.0, 8.0),
            500,
            100,
            1e-4,
            134.92,
            (1.6, 1.4),
            (1e-6, 1e-6),
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
    # test_rosen0_ga()
    test_rosen_ga()
