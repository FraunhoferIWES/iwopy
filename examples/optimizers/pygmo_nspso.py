from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "nspso",  # Non-dominated sorting particle swarm
            "gen": 10,  # Demo generations; production start: 1000
            "omega": 0.6,  # Inertia coefficient
            "diversity_mechanism": "crowding distance",  # Pareto diversity measure
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo Pareto swarm; production start: 100+
            "seed": 42,  # Reproducible initial swarm
        },
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
