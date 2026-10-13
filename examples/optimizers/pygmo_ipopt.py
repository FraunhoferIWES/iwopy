from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "ipopt",  # Select the IPOPT backend
            "max_iter": 100,  # Demo iterations; production start: 1000
            "tol": 1e-8,  # Scaled convergence tolerance
            "constr_viol_tol": 1e-6,  # Maximum absolute constraint violation
            "hessian_approximation": "limited-memory",  # Approximate the Hessian
        },
        setup_pars={
            "pop_size": 1,  # One starting point, not a population search
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
