from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "gwo",  # Grey-wolf optimization
            "gen": 10,  # Demo generations; production start: 1000
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
