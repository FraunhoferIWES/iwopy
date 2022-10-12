import numpy as np
import argparse
import matplotlib.pyplot as plt

from iwopy import DiscretizeRegGrid
from iwopy.optimizers import GG
from model import ChargesProblem

if __name__ == "__main__":

    # np.random.seed(42)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_points", help="The number of points", type=int, default=10
    )
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=5.0)
    parser.add_argument("-d", "--min_dist", help="The minimal charges distance", type=float, default=None)
    parser.add_argument("-o", "--order", help="Derivative order", type=int, default=1)
    parser.add_argument("-i", "--interpolation", help="The interpolation method", default=None)
    parser.add_argument("-nop", "--no_pop", help="Switch off vectorization", action="store_true")
    args = parser.parse_args()
    n = args.n_points
    r = args.radius
    d = args.min_dist

    np.random.seed(42)
    xy = np.random.uniform(-r / 10.0, r / 10.0, (n, 2))

    problem = ChargesProblem(xy, r, d, ctol=1e-2)

    fig = problem.get_fig(xy)
    plt.show()
    plt.close(fig)

    gproblem = DiscretizeRegGrid(
        problem, deltas=1e-8, fd_order=args.order, fd_bounds_order=1, tol=1e-10, interpolation=args.interpolation
    )
    gproblem.initialize()

    solver = GG(
        gproblem,
        step_max=0.1,
        step_min=1e-6,
        vectorized=not args.no_pop,
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    fig = problem.get_fig(results.problem_results)
    plt.show()
    plt.close(fig)