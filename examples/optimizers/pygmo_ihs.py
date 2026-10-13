from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="ihs",  # Improved harmony search
        problem=problem,
        phmcr=0.85,  # Probability of reusing a harmony-memory value
        seed=42,  # Reproducible evolution
        pop_size=20,  # Demo harmony memory; production start: 100
        max_iterations=100,  # One evaluation/iteration; production start: 10000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
