import numpy as np
import argparse
import matplotlib.pyplot as plt

from iwopy import DiscretizeRegGrid
from iwopy.interfaces.pygmo import Optimizer_pygmo
from model import ChargesProblem

if __name__ == "__main__":

    #np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--n_points", help="The number of points", type=int, default=5)
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=5.)
    args = parser.parse_args()
    n = args.n_points
    r = args.radius

    xy = np.random.uniform(-r/10., r/10., (n, 2))

    problem = ChargesProblem(xy, r)

    fig = problem.get_fig(xy)
    plt.show()
    plt.close(fig)

    gproblem = DiscretizeRegGrid(problem, deltas=0.001, fd_order=2, fd_bounds_order=1, tol=1e-10)
    gproblem.initialize()

    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(grad_pop=False),
        algo_pars=dict(type="ipopt", tol=1e-6),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    xy = results.problem_results
    print("\nResults:\n", xy.tolist())

    fig = problem.get_fig(xy)
    plt.show()
    plt.close(fig)
