from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "Powell",  # Derivative-free direction-set search
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "maxfev": 500,  # Demo evaluations; production start: 10000
                "xtol": 1e-6,  # Relative variable convergence tolerance
                "ftol": 1e-8,  # Relative objective convergence tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
