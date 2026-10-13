from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="Newton-CG",  # Newton optimization with CG inner solves
        problem=problem,
        hess="2-point",  # Estimate the Hessian from objective gradients
        max_iterations=100,  # Demo iterations; production start: 1000
        xtol=1e-6,  # Average relative step convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
