from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "L-BFGS-B",  # Bound-constrained limited-memory optimization
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "ftol": 1e-9,  # Relative objective improvement tolerance
                "gtol": 1e-6,  # Projected-gradient convergence tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
