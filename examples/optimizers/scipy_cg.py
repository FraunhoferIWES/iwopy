from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="CG",  # Nonlinear conjugate-gradient optimization
        problem=problem,
        max_iterations=100,  # Demo iterations; production start: 1000
        gtol=1e-6,  # Gradient convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
