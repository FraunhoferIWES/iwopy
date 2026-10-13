from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "Newton-CG",  # Newton optimization with CG inner solves
            "hess": "2-point",  # Estimate the Hessian from objective gradients
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "xtol": 1e-6,  # Average relative step convergence tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
