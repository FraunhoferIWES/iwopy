from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "COBYQA",  # Derivative-free quadratic-model optimization
            "options": {
                "maxiter": 100,  # Demo iterations; production start: 1000
                "maxfev": 500,  # Demo evaluations; production start: 10000
                "initial_tr_radius": 1.0,  # Initial radius in variable units
                "final_tr_radius": 1e-6,  # Final trust-region radius
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
