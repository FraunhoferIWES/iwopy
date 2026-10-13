from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(
        problem,
        scipy_pars={
            "method": "Nelder-Mead",  # Derivative-free simplex search
            "options": {
                "maxiter": 200,  # Demo iterations; production start: 10000
                "xatol": 1e-6,  # Absolute variable convergence tolerance
                "fatol": 1e-8,  # Absolute objective convergence tolerance
                "adaptive": True,  # Adapt simplex coefficients to dimension
            },
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
