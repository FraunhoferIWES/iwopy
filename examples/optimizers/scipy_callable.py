from scipy.optimize import minimize

from iwopy.benchmarks import QuadraticProblem
from iwopy.interfaces.scipy import Optimizer_scipy


def custom_method(fun, initial, args=(), jac=None, **options):
    """An example callable minimize method, delegating to BFGS."""
    return minimize(
        fun, initial, args=args, jac=jac, method="BFGS", options={"maxiter": 150}
    )


if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = Optimizer_scipy(problem, scipy_pars={"method": custom_method})
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
