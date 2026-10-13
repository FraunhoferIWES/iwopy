from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy_de

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy_de(
        problem,
        vectorize=True,  # Evaluate each generation as an iwopy population
        scipy_pars={
            "maxiter": 30,  # Demo generations; production start: 1000
            "popsize": 5,  # Demo multiplier per free variable; production: 15
            "tol": 1e-6,  # Relative population-energy convergence tolerance
            "seed": 42,  # Reproducible initialization and variation
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
