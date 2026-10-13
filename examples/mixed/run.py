import argparse

import matplotlib.pyplot as plt
from model import GridProblem, MaxN

from iwopy import create_optimizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-n", "--n_row_max", help="The max points per row", type=int, default=1000
    )
    parser.add_argument(
        "-a", "--algo", help="The algorithm choice", default="MixedVariableGA"
    )
    parser.add_argument("-r", "--radius", help="The radius", type=float, default=5.0)
    parser.add_argument(
        "-d",
        "--min_dist",
        help="The minimal distance",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--n_gen", help="The number of generations", type=int, default=200
    )
    parser.add_argument("--n_pop", help="The population size", type=int, default=50)
    parser.add_argument("--seed", help="The seed", type=int, default=None)
    parser.add_argument(
        "-pop", "--pop", help="Switch on vectorization", action="store_true"
    )
    args = parser.parse_args()

    problem = GridProblem(args.n_row_max, args.radius, args.min_dist)
    problem.add_objective(MaxN(problem))
    problem.initialize()

    fig = problem.get_fig()
    plt.show()
    plt.close(fig)

    solver = create_optimizer(
        family="pymoo",  # Optimizer family
        optimizer=args.algo,  # Algorithm to use
        problem=problem,
        vectorize=args.pop,
        pop_size=args.n_pop,
        seed=args.seed,
        termination="n_eval",
        n_max_evals=1000,
    )
    solver.initialize()

    results = solver.solve()
    solver.finalize(results)

    print()
    print(results)

    fig = problem.get_fig()
    plt.show()
    plt.close(fig)
