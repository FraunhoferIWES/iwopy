import numpy as np
import argparse
import matplotlib.pyplot as plt

from iwopy.interfaces.pymoo import Optimizer_pymoo
from model import *

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_points", help="The number of points", type=int, default=10
    )
    parser.add_argument("-a", "--algo", help="The algorithm choice", default="NSGA2")
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=5.0)
    parser.add_argument(
        "-N", "--n_gen", help="The number of generations", type=int, default=200
    )
    parser.add_argument(
        "-P", "--n_pop", help="The population size", type=int, default=100
    )
    parser.add_argument("--seed", help="The seed", type=int, default=None)
    parser.add_argument(
        "-nop", "--no_pop", help="Switch off vectorization", action="store_true"
    )
    args = parser.parse_args()
    n_pop = args.n_pop
    n = args.n_points
    r = args.radius

    radii = np.random.uniform(r / 2.0, r, n)
    chain = ChainPopulation(n_pop, n, radii, alpha=45.0)

    problem = ChainProblem(chain)
    problem.add_constraint(NoCrossing(problem))
    problem.add_objective(
        MaxStretch(problem, direction=np.array([1.0, 0.0]), name="stretch_x")
    )
    problem.add_objective(
        MaxStretch(problem, direction=np.array([0.0, 1.0]), name="stretch_y")
    )
    problem.initialize()

    solver = Optimizer_pymoo(
        problem,
        problem_pars=dict(
            vectorize=not args.no_pop,
        ),
        algo_pars=dict(
            type=args.algo,
            pop_size=args.n_pop,
            # seed=args.seed,
        ),
        setup_pars=dict(),
        term_pars=dict(
            type="default",
            n_max_gen=args.n_gen,
            ftol=0,
            xtol=0,
        ),
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    ax = results.plot_pareto()
    plt.show()

    for w in [[1, 0], [0.5, 0.5], [0, 1]]:
        i = results.find_pareto_objmix(w, max=True)

        fig = chain.get_fig(i, title=f"Weights stretch x, y: {w}")
        plt.show()
        plt.close(fig)
