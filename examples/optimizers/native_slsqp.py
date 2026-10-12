from iwopy.benchmarks import QuadraticProblem
from iwopy.optimizers import SLSQP

if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = SLSQP(problem, scipy_pars={"options": {"maxiter": 30}})
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
