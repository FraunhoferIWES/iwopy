from iwopy.benchmarks import ParetoProblem
from iwopy import create_optimizer

if __name__ == "__main__":
    problem = ParetoProblem(constrained=True)
    solver = create_optimizer(
        family="pymoo",  # Optimizer family
        optimizer="NSGA2",  # Non-dominated sorting genetic algorithm
        problem=problem,
        vectorize=True,  # Evaluate each generation as an iwopy population
        pop_size=20,  # Demo Pareto population; production start: 100+
        seed=42,  # Reproducible sampling and variation
        max_iterations=10,  # Demo generations; production start: 1000
        ftol=1e-6,  # Objective-space convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
    selected_index = result.find_pareto_objmix([0.5, 0.5])
    print("Selected variables:", result.vars_float[selected_index])
    print("Selected objectives:", result.objs[selected_index])
    print("Selected constraint:", result.cons[selected_index])
