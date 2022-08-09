import numpy as np

from iwopy import DiscretizeRegGrid, SimpleConstraint
from iwopy.benchmarks.branin import BraninProblem
from iwopy.benchmarks.rosenbrock import RosenbrockProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo


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


def run_branin_ipopt(init_vals, dx, dy, tol, pop):

    prob = BraninProblem(initial_values=init_vals)
    gprob = DiscretizeRegGrid(prob, deltas={"x": dx, "y": dy})
    gprob.initialize()

    solver = Optimizer_pygmo(
        gprob,
        problem_pars=dict(pop=pop),
        algo_pars=dict(type="ipopt", tol=tol),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    return results


def test_branin_ipopt():

    cases = (
        (
            0.001,
            0.001,
            1e-6,
            (1.0, 1.0),
            0.397887,
            (9.42478, 2.475),
            5e-6,
            (0.0008, 0.002),
            False,
        ),
        (
            0.0001,
            0.0001,
            1e-7,
            (1.0, 1.0),
            0.397887,
            (9.42478, 2.475),
            5e-7,
            (7e-5, 1.1e-4),
            False,
        ),
        (
            0.001,
            0.001,
            1e-6,
            (-3, 12.0),
            0.397887,
            (-np.pi, 12.275),
            7e-6,
            (0.001, 0.0016),
            True,
        ),
        (
            0.001,
            0.001,
            1e-6,
            (3, 3.0),
            0.397887,
            (np.pi, 2.275),
            3e-6,
            (0.0005, 0.00015),
            True,
        ),
    )

    for dx, dy, tol, ivals, f, xy, limf, limxy, pop in cases:

        print("\nENTERING", (dx, dy, tol, ivals, f, xy, limf, limxy, pop), "\n")

        results = run_branin_ipopt(ivals, dx, dy, tol, pop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float - np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


def run_branin_heu(type, init_vals, ngen, npop):

    prob = BraninProblem(initial_values=init_vals)
    prob.initialize()

    solver = Optimizer_pygmo(
        prob,
        problem_pars=dict(),
        algo_pars=dict(
            type=type,
            gen=ngen,
            seed=42,
        ),
        setup_pars=dict(
            pop_size=npop,
            seed=42,
        ),
    )
    solver.initialize()
    solver.print_info()

    results = solver.solve(verbosity=0)
    solver.finalize(results)

    return results


def test_branin_sga():

    cases = (
        (
            "sga",
            500,
            50,
            (1.0, 1.0),
            0.397887,
            0.0002,
        ),
    )

    for typ, ngen, npop, ivals, f, limf in cases:

        print("\nENTERING", (typ, ngen, npop, ivals, f, limf), "\n")

        results = run_branin_heu(typ, ivals, ngen, npop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf


def test_branin_pso():

    cases = (
        (
            "pso",
            500,
            50,
            (1.0, 1.0),
            0.397887,
            5e-7,
        ),
    )

    for typ, ngen, npop, ivals, f, limf in cases:

        print("\nENTERING", (typ, ngen, npop, ivals, f, limf), "\n")

        results = run_branin_heu(typ, ivals, ngen, npop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf


def test_branin_bee():

    cases = (
        (
            "bee_colony",
            500,
            50,
            (1.0, 1.0),
            0.397887,
            5e-7,
        ),
    )

    for typ, ngen, npop, ivals, f, limf in cases:

        print("\nENTERING", (typ, ngen, npop, ivals, f, limf), "\n")

        results = run_branin_heu(typ, ivals, ngen, npop)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf


def run_rosen0_ipopt(lower, upper, inits, dx, dy, tol, pop, ana):

    prob = RosenbrockProblem(lower=lower, upper=upper, initial=inits, ana_deriv=ana)

    gprob = DiscretizeRegGrid(
        prob, deltas={"x": dx, "y": dy}, fd_order=2, fd_bounds_order=1, tol=1e-6
    )
    gprob.initialize()

    solver = Optimizer_pygmo(
        gprob,
        problem_pars=dict(
            pop=pop,
        ),
        algo_pars=dict(
            type="ipopt",
            tol=tol,
            max_iter=100,
        ),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    return results


def test_rosen0_ipopt():

    cases = (
        (
            0.01,
            0.01,
            1e-3,
            (-2.0, -2),
            (2, 2),
            (1.8, 1.9),
            0.0,
            (1.0, 1.0),
            0.00011,
            (0.011, 0.021),
            True,
            True,
        ),
        (
            0.001,
            0.001,
            1e-4,
            (-2.0, -2),
            (2, 2),
            (1.8, 1.9),
            0.0,
            (1.0, 1.0),
            7e-7,
            (0.001, 0.002),
            True,
            False,
        ),
        (
            0.0001,
            0.0001,
            1e-5,
            (-2.0, -2),
            (2, 2),
            (1.8, 1.9),
            0.0,
            (1.0, 1.0),
            1e-8,
            (1e-4, 2e-4),
            True,
            True,
        ),
    )

    for dx, dy, tol, low, up, ivals, f, xy, limf, limxy, pop, ana in cases:

        print(
            "\nENTERING",
            (dx, dy, tol, low, up, ivals, f, xy, limf, limxy, pop, ana),
            "\n",
        )

        results = run_rosen0_ipopt(low, up, ivals, dx, dy, tol, pop, ana)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float - np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


def run_rosen_ipopt(lower, upper, inits, dx, dy, tol, pop, ana):

    prob = RosenbrockProblem(lower=lower, upper=upper, initial=inits, ana_deriv=ana)
    prob.add_constraint(RC(prob, ana_deriv=ana))

    gprob = DiscretizeRegGrid(
        prob, deltas={"x": dx, "y": dy}, fd_order=2, fd_bounds_order=2, tol=1e-6
    )
    gprob.initialize(verbosity=0)

    solver = Optimizer_pygmo(
        gprob,
        problem_pars=dict(
            pop=pop,
        ),
        algo_pars=dict(
            type="ipopt",
            tol=tol,
        ),
    )
    solver.initialize(verbosity=0)

    results = solver.solve()
    solver.finalize(results)

    return results


def test_rosen_ipopt():

    cases = (
        (
            0.001,
            0.001,
            (-5, -5),
            (-0.2, -0.2),
            1e-4,
            (-3.3, -1.45),
            7.2,
            (-0.2, -0.2),
            3e-6,
            (1e-7, 1e-7),
            False,
            False,
        ),
        (
            0.001,
            0.001,
            (-5, -5),
            (-0.2, -0.2),
            1e-4,
            (-3.3, -1.45),
            7.2,
            (-0.2, -0.2),
            3e-6,
            (1e-7, 1e-7),
            True,
            True,
        ),
    )

    for dx, dy, low, up, tol, ivals, f, xy, limf, limxy, pop, ana in cases:

        print(
            "\nENTERING",
            (dx, dy, low, up, tol, ivals, f, xy, limf, limxy, pop, ana),
            "\n",
        )

        results = run_rosen_ipopt(low, up, ivals, dx, dy, tol, pop, ana)
        print("Opt vars:", results.vars_float)

        delf = np.abs(results.objs[0] - f)
        print("delf =", delf, ", lim =", limf)
        assert delf < limf

        delxy = np.abs(results.vars_float - np.array(xy))
        limxy = np.array(limxy)
        print("delxy =", delxy, ", lim =", limxy)
        assert np.all(delxy < limxy)


if __name__ == "__main__":

    #test_branin_ipopt()
    #test_branin_sga()
    #test_branin_pso()
    #test_branin_bee()

    test_rosen0_ipopt()
    #test_rosen_ipopt()
