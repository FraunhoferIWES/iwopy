from scipy.optimize import minimize

from iwopy.benchmarks import QuadraticProblem
from iwopy import create_optimizer


def custom_method(fun, initial, args=(), jac=None, **options):
    """Delegate a SciPy callable method to BFGS.

    Parameters
    ----------
    fun
        Objective callable supplied by SciPy.
    initial
        Initial variable vector.
    args
        Additional objective arguments.
    jac
        Objective gradient callable.
    options
        SciPy method arguments; bfgs_iterations and gtol configure the
        delegated BFGS solver. Callable methods own their stopping semantics.

    Returns
    -------
    result
        SciPy optimization result from the delegated solve.
    """
    return minimize(
        fun,
        initial,
        args=args,
        jac=jac,
        method="BFGS",
        options={
            "maxiter": options.get("bfgs_iterations", 100),  # Maximum BFGS iterations
            "gtol": options.get("gtol", 1e-6),  # Gradient convergence tolerance
        },
    )


if __name__ == "__main__":
    problem = QuadraticProblem()
    solver = create_optimizer(
        family="scipy",  # Optimizer family
        optimizer=custom_method,  # User-provided SciPy optimization method
        problem=problem,
        bfgs_iterations=100,  # Demo delegated iterations; production start: 1000
        gtol=1e-6,  # Gradient convergence tolerance
    )
    solver.initialize(verbosity=0)
    result = solver.solve(verbosity=0)
    print(result)
