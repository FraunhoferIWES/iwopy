from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pymoo(
        problem,
        problem_pars={
            "vectorize": True,  # Evaluate each generation as an iwopy population
        },
        algo_pars={
            "type": "PSO",  # Particle-swarm optimization
            "pop_size": 20,  # Demo swarm; production start: 40
            "adaptive": True,  # Adapt inertia and attraction coefficients
        },
        setup_pars={
            "seed": 42,  # Reproducible sampling and variation
        },
        term_pars={
            "n_max_gen": 10,  # Demo generations; production start: 1000
            "ftol": 1e-6,  # Objective-space convergence tolerance
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
