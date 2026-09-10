import argparse

import numpy as np

import iwopy
from iwopy.optimizers import SLSQP


class Quadratic(iwopy.SimpleObjective):
    """A derivative-free quadratic objective."""

    def __init__(self, problem):
        super().__init__(problem, has_ana_derivs=False)

    def f(self, x, y):
        return x**2 + y**2


class MinimumSum(iwopy.SimpleConstraint):
    """Require the sum of both variables to be at least one."""

    def __init__(self, problem):
        super().__init__(
            problem,
            "minimum_sum",
            mins=1.0,
            maxs=np.inf,
            has_ana_derivs=False,
        )

    def f(self, x, y):
        return x + y


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-pop",
        help="Disable population-based finite-difference gradients",
        action="store_true",
    )
    args = parser.parse_args()

    base_problem = iwopy.SimpleProblem(
        "vectorized_slsqp",
        float_vars=["x", "y"],
        init_values_float=[0.0, 0.0],
        min_values_float=[-2.0, -2.0],
        max_values_float=[2.0, 2.0],
    )
    base_problem.add_objective(Quadratic(base_problem))
    base_problem.add_constraint(MinimumSum(base_problem))
    base_problem.initialize()

    problem = iwopy.LocalFD(
        base_problem,
        deltas={"x": 1e-5, "y": 1e-5},
        fd_order=2,
    )
    problem.initialize()

    solver = SLSQP(
        problem,
        scipy_pars={"tol": 1e-9, "options": {"maxiter": 100}},
        vectorized=not args.no_pop,
    )
    solver.initialize()
    results = solver.solve()
    solver.finalize(results)

    print(results)
    print(f"Vectorized gradients: pop={not args.no_pop}")
    print("Expected constrained optimum: x = y = 0.5")
