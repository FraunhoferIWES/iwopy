from iwopy.benchmarks import ParetoProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo

if __name__ == "__main__":
    problem = ParetoProblem(constrained=True)
    solver = Optimizer_pymoo(
        problem,
        problem_pars={"vectorize": True},
        algo_pars={"type": "NSGA2", "pop_size": 20},
        setup_pars={"seed": 42},
        term_pars=("n_gen", 5),
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
    selected_index = result.find_pareto_objmix([0.5, 0.5])
    print("Selected variables:", result.vars_float[selected_index])
    print("Selected objectives:", result.objs[selected_index])
    print("Selected constraint:", result.cons[selected_index])
