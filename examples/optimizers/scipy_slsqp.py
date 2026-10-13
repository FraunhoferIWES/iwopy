from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "SLSQP",  # Sequential quadratic programming
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "ftol": 1e-8,  # Objective and optimality stopping tolerance
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
