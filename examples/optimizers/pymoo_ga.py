from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.pymoo import Optimizer_pymoo

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_pymoo(
        problem,
        problem_pars={"vectorize": True},
        algo_pars={"type": "GA", "pop_size": 20},
        setup_pars={"seed": 42},
        term_pars=("n_gen", 3),
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
