import numpy as np
import argparse
import matplotlib.pyplot as plt

from iwopy.interfaces.pymoo import Optimizer_pymoo
from model import ChargesProblem

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_points", help="The number of points", type=int, default=5
    )
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=5.0)
    parser.add_argument("--n_gen", help="The number of generations", type=int, default=200)
    parser.add_argument("--n_pop", help="The population size", type=int, default=50)
    parser.add_argument("--seed", help="The seed", type=int, default=None)
    parser.add_argument("--pop", help="Run in vectorized form", action="store_true")
    args = parser.parse_args()
    n = args.n_points
    r = args.radius

    xy = np.random.uniform(-r / 10.0, r / 10.0, (n, 2))

    problem = ChargesProblem(xy, r)
    problem.initialize()

    fig = problem.get_fig(xy)
    plt.show()
    plt.close(fig)

    solver = Optimizer_pymoo(
        problem,
        problem_pars=dict(
            vectorize=args.pop,
        ),
        algo_pars=dict(
            type="ga",
            pop_size=args.n_pop,
            seed=args.seed,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=args.n_gen,
            ftol=1e-6,
            xtol=1e-6,
        ),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    xy = results.problem_results
    print("\nResults:\n", xy.tolist())

    fig = problem.get_fig(xy)
    plt.show()
    plt.close(fig)
