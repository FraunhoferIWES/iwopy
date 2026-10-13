from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="TNC",  # Bound-constrained truncated Newton optimization
        problem=problem,
        maxfun=500,  # Demo evaluations; production start: 10000
        gtol=1e-6,  # Projected-gradient convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
