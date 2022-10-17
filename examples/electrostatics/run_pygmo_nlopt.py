import numpy as np
import argparse
import matplotlib.pyplot as plt

from iwopy import LocalFD
from iwopy.interfaces.pygmo import Optimizer_pygmo
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
    parser.add_argument("-a", "--opt_algo", help="The optimization algorithm", default="ccsaq")
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

    gproblem = LocalFD(problem, deltas=1e-2, fd_order=args.order)
    gproblem.initialize()

    solver = Optimizer_pygmo(
        gproblem,
        problem_pars=dict(pop=not args.no_pop),
        algo_pars=dict(type="nlopt", optimizer=args.opt_algo, ftol_abs=1e-6, ftol_rel=0, xtol_abs=1e-6, xtol_rel=0),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    fig = problem.get_fig(results.problem_results)
    plt.show()
    plt.close(fig)
