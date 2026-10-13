from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "de",  # Differential evolution
            "gen": 10,  # Demo generations; production start: 1000
            "F": 0.8,  # Differential mutation scale
            "CR": 0.9,  # Crossover probability
            "ftol": 1e-6,  # Population objective-spread stopping tolerance
            "xtol": 1e-6,  # Population variable-spread stopping tolerance
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
