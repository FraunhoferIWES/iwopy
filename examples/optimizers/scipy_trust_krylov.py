from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="trust-krylov",  # Krylov trust-region optimization
        problem=problem,
        hess="2-point",  # Estimate the Hessian from objective gradients
        max_iterations=100,  # Demo iterations; production start: 1000
        gtol=1e-6,  # Gradient convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
