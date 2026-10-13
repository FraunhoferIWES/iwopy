from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "pso_gen",  # Generational particle-swarm optimization
            "gen": 10,  # Demo generations; production start: 1000
            "omega": 0.7298,  # Inertia or constriction coefficient
            "eta1": 2.05,  # Attraction toward personal best positions
            "eta2": 2.05,  # Attraction toward neighbourhood best positions
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo swarm; production start: 40
            "seed": 42,  # Reproducible initial swarm
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
