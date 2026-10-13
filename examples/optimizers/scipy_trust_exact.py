import numpy as np

from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="trust-exact",  # Nearly exact trust-region optimization
        problem=problem,
        hess=lambda point: 2.0 * np.eye(len(point)),  # Replace for real problems
        maxiter=100,  # Demo iterations; production start: 1000
        gtol=1e-6,  # Gradient convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
