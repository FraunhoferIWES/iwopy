from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="Powell",  # Derivative-free direction-set search
        problem=problem,
        maxiter=100,  # Demo iterations; production start: 1000
        maxfev=500,  # Demo evaluations; production start: 10000
        xtol=1e-6,  # Relative variable convergence tolerance
        ftol=1e-8,  # Relative objective convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
