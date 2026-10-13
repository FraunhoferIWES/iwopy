from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "trust-constr",  # Constrained trust-region optimization
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "gtol": 1e-6,  # Lagrangian gradient and feasibility tolerance
                "xtol": 1e-8,  # Minimum trust-region radius
                "barrier_tol": 1e-8,  # Inequality barrier stopping tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
