import numpy as np

from iwopy import DiscretizeRegGrid, SimpleConstraint
from iwopy.benchmarks.branin import BraninProblem
from iwopy.benchmarks.rosenbrock import RosenbrockProblem
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

    prob = BraninProblem(initial_values=init_vals,ana_deriv=True)
    prob.initialize()

    solver = Optimizer_scipy(
        prob,
        scipy_pars=dict(method="SLSQP", tol=tol),
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


if __name__ == "__main__":

    test_branin_slsqp()

