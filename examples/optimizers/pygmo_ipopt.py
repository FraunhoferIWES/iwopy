from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="pygmo",  # Optimizer family
        optimizer="ipopt",  # Select the IPOPT backend
        problem=problem,
        hessian_approximation="limited-memory",  # Approximate the Hessian
        pop_size=1,  # One starting point, not a population search
        max_iterations=100,  # Demo iterations; production start: 1000
        tol=1e-8,  # Scaled convergence tolerance
        constr_viol_tol=1e-6,  # Maximum absolute constraint violation
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
