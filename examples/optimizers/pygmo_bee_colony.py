from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pygmo import Optimizer_pygmo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pygmo(
        problem,
        algo_pars={
            "type": "bee_colony",  # Artificial bee colony
            "gen": 10,  # Demo generations; production start: 1000
            "limit": 20,  # Failed improvement attempts before abandoning a source
            "seed": 42,  # Reproducible evolution
        },
        setup_pars={
            "pop_size": 20,  # Demo colony; production start: 100
            "seed": 42,  # Reproducible initial colony
        },
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
