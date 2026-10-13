from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "trust-krylov",  # Krylov trust-region optimization
            "hess": "2-point",  # Estimate the Hessian from objective gradients
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "gtol": 1e-6,  # Gradient convergence tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
