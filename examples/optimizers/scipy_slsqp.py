from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="SLSQP",  # Sequential quadratic programming
        problem=problem,
        maxiter=100,  # Demo iterations; production start: 1000
        ftol=1e-8,  # Objective and optimality stopping tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
