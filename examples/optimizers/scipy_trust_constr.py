from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="trust-constr",  # Constrained trust-region optimization
        problem=problem,
        maxiter=100,  # Demo iterations; production start: 1000
        gtol=1e-6,  # Lagrangian gradient and feasibility tolerance
        xtol=1e-8,  # Minimum trust-region radius
        barrier_tol=1e-8,  # Inequality barrier stopping tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
