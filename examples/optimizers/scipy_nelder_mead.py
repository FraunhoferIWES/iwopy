from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer="Nelder-Mead",  # Derivative-free simplex search
        problem=problem,
        adaptive=True,  # Adapt simplex coefficients to dimension
        maxiter=200,  # Demo iterations; production start: 10000
        xatol=1e-6,  # Absolute variable convergence tolerance
        fatol=1e-8,  # Absolute objective convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
