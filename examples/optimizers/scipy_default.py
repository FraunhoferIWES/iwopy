from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer=None,  # Algorithm to use
        problem=problem,
        tol=1e-6,  # Method-specific convergence tolerance
        max_iterations=100,  # Demo iterations; production start: 1000
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
