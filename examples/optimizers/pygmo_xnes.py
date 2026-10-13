from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "xnes",  # Exponential natural evolution strategy
            "gen": 20,  # Demo generations; production start: 1000
            "sigma0": 0.25,  # Initial search step size
            "ftol": 1e-6,  # Population objective-spread stopping tolerance
            "xtol": 1e-6,  # Search-distribution stopping tolerance
            "force_bounds": True,  # Keep sampled candidates within variable bounds
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Production: max(5, 4 + int(3 * log(n_vars)))
            "seed": 42,  # Reproducible initial population
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
