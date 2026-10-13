from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "sea",  # Simple evolutionary algorithm
            "gen": 100,  # One evaluation/iteration; production start: 10000
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo parent population; production start: 100
            "seed": 42,  # Reproducible initial population
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
