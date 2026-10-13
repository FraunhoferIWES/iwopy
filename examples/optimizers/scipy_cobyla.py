from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "COBYLA",  # Derivative-free constrained optimization
            "options": {
                "maxiter": 500,  # Demo evaluations; production start: 10000
                "rhobeg": 1.0,  # Initial trust-region radius in variable units
                "tol": 1e-6,  # Final trust-region radius
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
