from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.sparse import csr_matrix

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)
from ._options import minimize_options


_GRADIENT_METHODS = frozenset(
    {
        "bfgs",
        "cg",
        "dogleg",
        "l-bfgs-b",
        "newton-cg",
        "slsqp",
        "tnc",
        "trust-constr",
        "trust-exact",
        "trust-krylov",
        "trust-ncg",
    }
)


class Optimizer_scipy(Optimizer):
    """
    Interface to the scipy optimizers.

    Gradient-capable solvers receive objective and constraint Jacobians from
    ``Problem.get_jacobian``. By default, finite-difference points are
    evaluated through the problem's population interface.
    Jacobians are cached in CSR storage. ``trust-constr`` receives sparse
    constraint Jacobians; other methods receive dense requested rows. Scalar
    objective gradients are always dense vectors.
    """

    def __init__(
        self,
        problem: Problem,
        optimizer: str | Callable[..., Any] | None = None,
        *,
        tol: float | None = None,
        max_iterations: int | None = None,
        max_evaluations: int | None = None,
        hess: Any = None,
        hessp: Any = None,
        memory_size: int = 100,
        vectorize: bool = True,
        name: str = "Optimizer_scipy",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The problem to optimize
        optimizer
            SciPy minimize method name, callable, or None for native selection.
        tol
            Native aggregate stopping tolerance; None uses SciPy defaults.
        max_iterations
            Native iteration limit. Unsupported for TNC, COBYLA, and callable
            methods. None leaves the native default unchanged.
        max_evaluations
            Native objective-call limit for Nelder-Mead, Powell, L-BFGS-B, TNC,
            COBYLA, or COBYQA. Numerical derivatives may exceed a native limit.
            None leaves the native default unchanged.
        hess, hessp
            Native Hessian and Hessian-vector product specifications.
        memory_size
            Positive evaluation-cache capacity.
        vectorize
            Population-based Jacobian evaluation; derivative-free methods
            are unaffected.
        name
            Optimizer name.
        kwargs
            Flat native method options, such as ftol, gtol and eps.
            Bounds, callback, constraints, method and Jacobians are managed.
            For trust-constr, sparse_jacobian defaults to True; explicit native
            settings are preserved. Use False with dense-only factorizations.

        Raises
        ------
        ValueError
            If a limit is invalid or unsupported by the selected method,
            vectorize is not a boolean, or the method selector is duplicated.
        TypeError
            If a superseded or unsupported shared keyword is supplied.
        """
        super().__init__(problem, name)
        self._validate_kwargs(kwargs)
        self.optimizer = optimizer
        method = optimizer.lower() if isinstance(optimizer, str) else None
        iteration_key = (
            None if callable(optimizer) or method in {"tnc", "cobyla"} else "maxiter"
        )
        evaluation_key = {
            "nelder-mead": "maxfev",
            "powell": "maxfev",
            "l-bfgs-b": "maxfun",
            "tnc": "maxfun",
            "cobyla": "maxiter",
            "cobyqa": "maxfev",
        }.get(method or "")
        limits = self._limit_kwargs(
            max_iterations, max_evaluations, iteration_key, evaluation_key
        )
        self._native_pars = minimize_options(kwargs | limits, tol)
        if "method" in self._native_pars:
            raise ValueError("SciPy: select the method with optimizer")
        self._native_pars["method"] = optimizer
        if self._uses_sparse_constraints():
            self._native_pars["options"].setdefault("sparse_jacobian", True)
        if hess is not None:
            self._native_pars["hess"] = hess
        if hessp is not None:
            self._native_pars["hessp"] = hessp
        self.memory_size = memory_size
        self.vectorize = self._validate_vectorize(vectorize)
        self._mem: (
            dict[tuple[object, ...], tuple[np.ndarray, np.ndarray, object | None]]
            | None
        ) = None
        self._jacobian_mem: dict[tuple[object, ...], csr_matrix] | None = None
        self._constraints_scipy: list[dict[str, object]] | None = None
        self._callback_iteration = 0

    def _uses_gradients(self) -> bool:
        """Return whether the selected SciPy method accepts a Jacobian."""
        method = self._native_pars.get("method")
        return (
            method is None
            or callable(method)
            or str(method).lower() in _GRADIENT_METHODS
        )

    def _uses_sparse_constraints(self) -> bool:
        """Return whether the native method supports sparse constraint Jacobians."""
        method = self._native_pars.get("method")
        return isinstance(method, str) and method.lower() == "trust-constr"

    def print_info(self) -> None:
        """Print solver info, called before solving"""
        super().print_info()

        if len(self._native_pars):
            print("\nScipy parameters:")
            print("-----------------")
            for k, v in self._native_pars.items():
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
        if self.problem.n_objectives != 1:
            raise ValueError(
                f"Optimizer '{self.name}': Exactly one objective is required."
            )
        if self.problem.n_vars_int:
            raise ValueError(
                f"Optimizer '{self.name}': Integer variables are not supported."
            )
        if not self.problem.n_vars_float:
            raise ValueError(
                f"Optimizer '{self.name}': At least one float variable is required."
            )
        if (
            isinstance(self.memory_size, (bool, np.bool_))
            or not isinstance(self.memory_size, (int, np.integer))
            or self.memory_size < 1
        ):
            raise ValueError(
                f"Optimizer '{self.name}': memory_size must be a positive integer."
            )

        reserved = {"bounds", "callback", "constraints", "jac"}
        conflicts = reserved.intersection(self._native_pars)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"Optimizer '{self.name}': SciPy parameters managed internally: "
                f"{names}."
            )

        self._constraints_scipy = self._make_constraints()

        if verbosity:
            print(f"Using optimizer memory, size: {self.memory_size}")
        self._mem = {}
        self._jacobian_mem = {}

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
        objs
            The objective function values, shape: (n_objectives,)
        cons
            The constraints values, shape: (n_constraints,)
        prob_results
            The problem results
        """
        memory = self._mem
        assert memory is not None
        key = tuple(x)
        if key not in memory:
            data = self.problem.evaluate_individual(
                np.array([], dtype=np.int32),
                np.asarray(x, dtype=np.float64),
                ret_prob_res=True,
            )

            if len(memory) >= self.memory_size and memory:
                key0 = next(iter(memory))
                del memory[key0]

            memory[key] = data

        return memory[key]

    def _get_jacobian(self, x: np.ndarray) -> csr_matrix:
        """Return the cached objective and constraint Jacobian."""
        memory = self._jacobian_mem
        assert memory is not None
        key = tuple(x)
        if key not in memory:
            objs, cons, _ = self._get_results(x)
            try:
                jacobian = self.problem.get_jacobian(
                    np.array([], dtype=np.int32),
                    np.asarray(x, dtype=np.float64),
                    func_values=np.r_[objs, cons],
                    pop=self.vectorize,
                )
            except ValueError as error:
                raise ValueError(
                    f"Optimizer '{self.name}': Failed to determine a finite "
                    "objective and constraint Jacobian."
                ) from error
            shape = (
                self.problem.n_objectives + self.problem.n_constraints,
                self.problem.n_vars_float,
            )
            if jacobian.shape != shape:
                raise ValueError(
                    f"Optimizer '{self.name}': Expected Jacobian shape {shape}, "
                    f"received {jacobian.shape}."
                )
            if not np.all(np.isfinite(jacobian.data)):
                raise ValueError(
                    f"Optimizer '{self.name}': Non-finite objective or constraint "
                    "Jacobian entry."
                )
            if len(memory) >= self.memory_size and memory:
                del memory[next(iter(memory))]
            memory[key] = jacobian
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
        objective
            Current objective function value
        """
        objs, _, _ = self._get_results(x)
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        return float(sign * objs[0])

    def _objective_jac(self, x: np.ndarray) -> np.ndarray:
        """Return the minimization-oriented objective gradient."""
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        return sign * self._get_jacobian(x)[0].toarray().ravel()

    def _constraint_values(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> np.ndarray:
        """Return one constraint group in SciPy sign convention."""
        _, cons, _ = self._get_results(x)
        return sign * (cons[indices] - bounds)

    def _constraint_jac(
        self,
        x: np.ndarray,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> np.ndarray | csr_matrix:
        """Return one grouped constraint Jacobian."""
        del bounds
        rows = self.problem.n_objectives + indices
        jacobian = sign * self._get_jacobian(x)[rows]
        if self._uses_sparse_constraints():
            return jacobian
        return jacobian.toarray()

    def _make_constraint(
        self,
        kind: str,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> dict[str, object]:
        """Create one grouped old-style SciPy constraint."""
        constraint: dict[str, object] = {
            "type": kind,
            "fun": self._constraint_values,
            "args": (indices, bounds, sign),
        }
        if self._uses_gradients():
            constraint["jac"] = self._constraint_jac
        return constraint

    def _make_constraints(self) -> list[dict[str, object]]:
        """Translate iwopy constraint bounds to grouped SciPy constraints."""
        if not self.problem.n_constraints:
            return []

        lower = self.problem.min_values_constraints
        upper = self.problem.max_values_constraints
        if lower is None or upper is None:
            bounds = [
                constraint.get_bounds() for constraint in self.problem.cons.functions
            ]
            lower = np.concatenate([bound[0] for bound in bounds])
            upper = np.concatenate([bound[1] for bound in bounds])
        lower = np.asarray(lower, dtype=np.float64)
        upper = np.asarray(upper, dtype=np.float64)
        equal = np.isfinite(lower) & np.isfinite(upper) & (lower == upper)
        constraints: list[dict[str, object]] = []

        indices = np.flatnonzero(equal)
        if len(indices):
            constraints.append(
                self._make_constraint("eq", indices, lower[indices], 1.0)
            )

        indices = np.flatnonzero(np.isfinite(lower) & ~equal)
        if len(indices):
            constraints.append(
                self._make_constraint("ineq", indices, lower[indices], 1.0)
            )

        indices = np.flatnonzero(np.isfinite(upper) & ~equal)
        if len(indices):
            constraints.append(
                self._make_constraint("ineq", indices, upper[indices], -1.0)
            )
        return constraints

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
        method = self._native_pars.get("method")
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
        results
            The optimization results object
        """

        # check problem initialization:
        super().solve(verbosity, callbacks)
        self._callback_iteration = 0
        self._mem = {}
        self._jacobian_mem = {}

        # Initial values:
        x0 = np.asarray(self.problem.initial_values_float(), dtype=np.float64)

        # Find bounds:
        minf = [x if x != -np.inf else None for x in self.problem.min_values_float()]
        maxf = [x if x != np.inf else None for x in self.problem.max_values_float()]
        bounds = [(minf[i], maxf[i]) for i in range(len(minf))]

        # Run minimization:
        scipy_pars = self._native_pars.copy()
        constraints_scipy = self._constraints_scipy
        assert constraints_scipy is not None
        scipy_pars["constraints"] = constraints_scipy
        if self._has_callbacks:
            scipy_pars["callback"] = self._scipy_callback()
        if self._uses_gradients():
            scipy_pars["jac"] = self._objective_jac
        scipy_results = minimize(self._objective, x0, bounds=bounds, **scipy_pars)

        # final evaluation:
        if scipy_results.success:
            x = scipy_results.x
            vars_int = np.array([], dtype=np.int32)
            vars_float = x
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
