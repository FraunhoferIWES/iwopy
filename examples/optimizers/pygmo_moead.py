from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = ParetoProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "moead",  # Multi-objective decomposition
            "gen": 10,  # Demo generations; production start: 1000
            "neighbours": 5,  # Production: 20 neighbours with a larger population
            "decomposition": "tchebycheff",  # Scalar subproblem formulation
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo Pareto population; production start: 100+
            "seed": 42,  # Reproducible initial population
        },
    )
    solver.initialize(verbosity=0)
    solver.algo.set_verbosity(0)
    result = solver.algo.evolve(solver.pop)
    print(result)
