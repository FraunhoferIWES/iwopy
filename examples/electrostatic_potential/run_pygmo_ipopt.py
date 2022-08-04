import numpy as np
import argparse

from iwopy import DiscretizeRegGrid
from iwopy.interfaces.pygmo import Optimizer_pygmo
from model import Model, ModelProblem

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_points", help="The number of points", type=int, default=10)
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=10.)
    parser.add_argument("-p", "--pop", help="Calculate population in vectorized form", action="store_true")
    args = parser.parse_args()
    n = args.n_points
    r = args.radius

    xy = np.zeros((n, 2))
    xy[:, 0] = np.linspace(-r, r, n)

    model = Model(xy)
    problem = ModelProblem(model, (-r, -r), (r, r), r)

    gproblem = DiscretizeRegGrid(problem, deltas=0.001, fd_order=1, fd_bounds_order=1, tol=1e-6)
    gproblem.initialize()

    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(grad_pop=args.pop),
        algo_pars=dict(type="ipopt", tol=1e-6),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)