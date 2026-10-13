from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "nsga2",  # Non-dominated sorting genetic algorithm
            "gen": 10,  # Demo generations; production start: 1000
            "cr": 0.95,  # Crossover probability
            "m": 1.0 / problem.n_vars_float,  # Per-gene mutation probability
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Production: 100+ individuals, in multiples of four
            "seed": 42,  # Reproducible initial population
        },
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
