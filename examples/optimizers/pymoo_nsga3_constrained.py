from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo

if __name__ == "__main__":
    problem = ParetoProblem(constrained=True)
    solver = Optimizer_pymoo(
        problem,
        problem_pars={
            "vectorize": True,  # Evaluate each generation as an iwopy population
        },
        algo_pars={
            "type": "NSGA3",  # Reference-direction non-dominated sorting
            "n_partitions": 12,  # Production: choose partitions for 100+ directions
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
    selected_index = result.find_pareto_objmix([0.5, 0.5])
    print("Selected variables:", result.vars_float[selected_index])
    print("Selected objectives:", result.objs[selected_index])
    print("Selected constraint:", result.cons[selected_index])
