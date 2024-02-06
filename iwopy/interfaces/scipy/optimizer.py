import numpy as np
from scipy.optimize import minimize

from iwopy.core import Optimizer
from iwopy.core import SingleObjOptResults


class Optimizer_scipy(Optimizer):
    """
    Interface to the scipy optimizers.

    Note that these solvers do not support
    vectorized evaluation.

    Parameters
    ----------
    problem : iwopy.Problem
        The problem to optimize
    scipy_pars : dict
        Additional parameters for
        scipy.optimze.minimize()
    mem_size : int
        The memory size, number of
        stored obj, cons evaluations
    kwargs : dict, optional
        Additional parameters for base class

    Attributes
    ----------
    scipy_pars : dict
        Additional parameters for
        scipy.optimze.minimize()
    mem_size : int
        The memory size, number of
        stored obj, cons evaluations

    """

    def __init__(self, problem, scipy_pars={}, mem_size=100, **kwargs):
        super().__init__(problem, **kwargs)
        self.scipy_pars = scipy_pars
        self.mem_size = mem_size
        self._mem = None

    def print_info(self):
        """
        Print solver info, called before solving
        """
        super().print_info()

        if len(self.scipy_pars):
            print("\nScipy parameters:")
            print("-----------------")
            for k, v in self.scipy_pars.items():
                if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    print(f"  {k}: {v}")

        print()

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """

        # Check objectives:
        if self.problem.n_objectives > 1:
            raise RuntimeError(
                "Scipy minimize does not support multi-objective optimization."
            )

        # Define constraints:
        cons = list()
        for i in range(self.problem.n_constraints):
            cons.append({"type": "ineq", "fun": self._constraints, "args": (i,)})
        self.scipy_pars["constraints"] = cons

        if verbosity:
            print(f"Using optimizer memory, size: {self.mem_size}")
        self._mem = {}

        super().initialize(verbosity)

    def _get_results(self, x):
        """
        Evaluate obj and cons

        Parameters
        ----------
        x: numpy array
            Array containing design variables

        Returns
        -------
        objs : np.array
            The objective function values, shape: (n_objectives,)
        cons : np.array
            The constraints values, shape: (n_constraints,)
        prob_results : object
            The problem results

        """
        key = tuple(x)
        if key not in self._mem:

            i0 = self.problem.n_vars_int
            vars_int = x[:i0].astype(np.int32)
            vars_float = x[i0:]

            data = self.problem.evaluate_individual(
                vars_int, vars_float, ret_prob_res=True
            )

            if len(self._mem) > self.mem_size:
                key0 = next(iter(self._mem.keys()))
                del self._mem[key0]

            self._mem[key] = data

        return self._mem[key]

    def _objective(self, x):
        """
        Function which converts array from scipy
        to readable variables for the problem and
        evaluates the objective function.

        Parameters
        ----------
        x: numpy array
            Array containing design variables

        Returns
        -------
        float:
            Current objective function value


        """
        objs, __, __ = self._get_results(x)
        return objs[0]

    def _constraints(self, x, ci):
        """
        Function which converts array from scipy
        to readable variables for the problem and
        evaluates the constraints.

        Parameters
        ----------
        x: numpy array
            Array containing design variables
        ci: int
            Index for constraint component

        Returns
        -------
        float:
            Value of constraint component

        """
        __, cons, __ = self._get_results(x)
        return cons[ci]

    def solve(self, verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.SingleObjOptResults
            The optimization results object

        """

        # check problem initialization:
        super().solve()

        # Initial values:
        x0 = np.array(self.problem.initial_values_int(), dtype=np.float64)
        x0 = np.append(
            x0, np.array(self.problem.initial_values_float(), dtype=np.float64)
        )

        # Find bounds:
        mini = [
            x if x != -self.problem.INT_INF else None
            for x in self.problem.min_values_int()
        ]
        maxi = [
            x if x != self.problem.INT_INF else None
            for x in self.problem.max_values_int()
        ]
        bounds = [(mini[i], maxi[i]) for i in range(len(mini))]
        minf = [x if x != -np.inf else None for x in self.problem.min_values_float()]
        maxf = [x if x != np.inf else None for x in self.problem.max_values_float()]
        bounds = [(minf[i], maxf[i]) for i in range(len(minf))]

        # Run minimization:
        results = minimize(self._objective, x0, bounds=bounds, **self.scipy_pars)

        # final evaluation:
        if results.success:

            x = results.x
            i0 = self.problem.n_vars_int
            vars_int = x[:i0].astype(np.int32)
            vars_float = x[i0:]
            prob_res, objs, cons = self.problem.finalize_individual(
                vars_int, vars_float, verbosity=verbosity
            )

        else:
            prob_res = None
            vars_int = None
            vars_float = None
            objs = None
            cons = None

        return SingleObjOptResults(
            self.problem, results.success, vars_int, vars_float, objs, cons, prob_res
        )
