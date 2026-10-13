from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "sga",  # Simple genetic algorithm
            "gen": 10,  # Demo generations; production start: 1000
            "cr": 0.9,  # Crossover probability
            "m": 0.02,  # Per-gene mutation probability
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo population; production start: 100
            "seed": 42,  # Reproducible initial population
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
