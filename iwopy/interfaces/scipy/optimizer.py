from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)


class Optimizer_scipy(Optimizer):
    """
    Interface to the scipy optimizers.

    Note that these solvers do not support
    vectorized evaluation.

    Attributes
    ----------
    scipy_pars: dict
        Additional parameters for
        scipy.optimze.minimize()
    mem_size: int
        The memory size, number of
        stored obj, cons evaluations

    :group: interfaces.scipy

    """

    def __init__(
        self,
        problem: Problem,
        scipy_pars: dict[str, Any] | None = None,
        mem_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Constructor

        Parameters
        ----------
        problem
            The problem to optimize
        scipy_pars
            Additional parameters for
            scipy.optimze.minimize()
        mem_size
            The memory size, number of
            stored obj, cons evaluations
        kwargs
            Additional parameters for base class

        """
        if scipy_pars is None:
            scipy_pars = {}
        super().__init__(problem, **kwargs)
        self.scipy_pars: dict[str, Any] = scipy_pars.copy()
        self.mem_size = mem_size
        self._mem: (
            dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, object | None]]
            | None
        ) = None
        self._callback_iteration = 0

    def print_info(self) -> None:
        """
        Print solver info, called before solving
        """
        super().print_info()

        if len(self.scipy_pars):
            print("\nScipy parameters:")
            print("-----------------")
            for k, v in self.scipy_pars.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")

        print()

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """

        # Check objectives:
        if self.problem.n_objectives > 1:
            raise RuntimeError(
                "Scipy minimize does not support multi-objective optimization."
            )
        if "callback" in self.scipy_pars:
            raise ValueError(
                f"Optimizer '{self.name}': SciPy callback is managed internally."
            )

        # Define constraints:
        cons: list[dict[str, object]] = []
        for i in range(self.problem.n_constraints):
            cons.append({"type": "ineq", "fun": self._constraints, "args": (i,)})
        self.scipy_pars["constraints"] = cons

        if verbosity:
            print(f"Using optimizer memory, size: {self.mem_size}")
        self._mem = {}

        super().initialize(verbosity)

    def _get_results(
        self, x: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, object | None]:
        """
        Evaluate obj and cons

        Parameters
        ----------
        x
            Array containing design variables

        Returns
        -------
        objs: np.array
            The objective function values, shape: (n_objectives,)
        cons: np.array
            The constraints values, shape: (n_constraints,)
        prob_results: object
            The problem results

        """
        memory = self._mem
        assert memory is not None
        key = tuple(x)
        if key not in memory:
            i0 = self.problem.n_vars_int
            vars_int = x[:i0].astype(np.int32)
            vars_float = x[i0:]

            data = self.problem.evaluate_individual(
                vars_int, vars_float, ret_prob_res=True
            )

            if len(memory) >= self.mem_size and memory:
                key0 = next(iter(memory))
                del memory[key0]

            memory[key] = data

        return memory[key]

    def _objective(self, x: np.ndarray) -> float:
        """
        Function which converts array from scipy
        to readable variables for the problem and
        evaluates the objective function.

        Parameters
        ----------
        x
            Array containing design variables

        Returns
        -------
        float:
            Current objective function value


        """
        objs, __, __ = self._get_results(x)
        return float(objs[0])

    def _constraints(self, x: np.ndarray, ci: int) -> float:
        """
        Function which converts array from scipy
        to readable variables for the problem and
        evaluates the constraints.

        Parameters
        ----------
        x
            Array containing design variables
        ci
            Index for constraint component

        Returns
        -------
        float:
            Value of constraint component

        """
        __, cons, __ = self._get_results(x)
        return float(cons[ci])

    def _dispatch_callback(
        self, x: np.ndarray, scipy_state: object | None = None
    ) -> None:
        """Dispatch a cached SciPy iterate without evaluating the problem."""
        self._callback_iteration += 1
        x = np.asarray(x, dtype=np.float64)
        memory = self._mem
        assert memory is not None
        cached = memory.get(tuple(x))
        objs = None if cached is None else cached[0]
        cons = None if cached is None else cached[1]
        n_evaluations = getattr(scipy_state, "nfev", None)
        if n_evaluations is not None:
            n_evaluations = int(n_evaluations)
        n_vars_int = self.problem.n_vars_int
        self._notify_callbacks(
            OptimizerCallbackData(
                event="iteration",
                iteration=self._callback_iteration,
                n_evaluations=n_evaluations,
                vars_int=x[:n_vars_int].astype(np.int32),
                vars_float=x[n_vars_int:],
                objs=objs,
                cons=cons,
            )
        )

    def _callback_xk(self, xk: np.ndarray) -> None:
        """Handle SciPy methods exposing only the current coordinates."""
        self._dispatch_callback(xk)

    def _callback_intermediate(self, intermediate_result: object) -> None:
        """Handle SciPy methods exposing an intermediate result."""
        if hasattr(intermediate_result, "x"):
            self._dispatch_callback(intermediate_result.x, intermediate_result)
        else:
            self._dispatch_callback(intermediate_result)

    def _scipy_callback(self) -> Callable[..., None]:
        """Select the callback signature required by the SciPy method."""
        method = self.scipy_pars.get("method")
        if callable(method):
            return self._callback_xk
        method_name = "" if method is None else str(method).lower()
        if method_name in {"tnc", "cobyla", "cobyqa"}:
            return self._callback_xk
        return self._callback_intermediate

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults:
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for optimizer iterates

        Returns
        -------
        results: iwopy.SingleObjOptResults
            The optimization results object

        """

        # check problem initialization:
        super().solve(verbosity, callbacks)
        self._callback_iteration = 0

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
        bounds += [(minf[i], maxf[i]) for i in range(len(minf))]

        # Run minimization:
        scipy_pars = self.scipy_pars.copy()
        if self._has_callbacks:
            scipy_pars["callback"] = self._scipy_callback()
        scipy_results = minimize(self._objective, x0, bounds=bounds, **scipy_pars)

        # final evaluation:
        if scipy_results.success:
            x = scipy_results.x
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

        results = SingleObjOptResults(
            self.problem,
            scipy_results.success,
            vars_int,
            vars_float,
            objs,
            cons,
            prob_res,
        )
        return self._finalize_callbacks(results)
