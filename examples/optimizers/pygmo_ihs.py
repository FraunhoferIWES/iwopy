from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "ihs",  # Improved harmony search
            "gen": 100,  # One evaluation/iteration; production start: 10000
            "phmcr": 0.85,  # Probability of reusing a harmony-memory value
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo harmony memory; production start: 100
            "seed": 42,  # Reproducible initial memory
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
