from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)


_CacheValueT = TypeVar("_CacheValueT")


class SLSQP(Optimizer):
    """
    Gradient-based SLSQP optimizer for continuous problems.

    The complete objective and constraint Jacobian is obtained through
    ``Problem.get_gradients(..., pop=True)`` at each iterate. Hence missing
    analytical derivatives can be supplied by a population-capable problem
    wrapper such as :class:`iwopy.wrappers.LocalFD`.
    """

    def __init__(
        self,
        problem: Problem,
        scipy_pars: dict[str, object] | None = None,
        mem_size: int = 100,
        vectorized: bool = True,
        name: str = "SLSQP",
    ) -> None:
        """
        Parameters
        ----------
        problem
            The continuous single-objective problem to optimize.
        scipy_pars
            Additional parameters for :func:`scipy.optimize.minimize`.
            The parameters ``method``, ``jac``, ``bounds``, and
            ``constraints`` are managed by this optimizer.
        mem_size
            Maximum number of cached value and gradient evaluations.
        vectorized
            Whether gradients use population-based function evaluation.
        name
            The optimizer name.
        """
        super().__init__(problem, name)
        self.scipy_pars = {} if scipy_pars is None else scipy_pars.copy()
        self.mem_size = mem_size
        self.vectorized = vectorized
        self.scipy_results: OptimizeResult | None = None
        self._value_mem: (
            dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] | None
        ) = None
        self._gradient_mem: dict[tuple[float, ...], np.ndarray] | None = None
        self._constraints_scipy: list[dict[str, object]] | None = None
        self._constraint_lower: np.ndarray | None = None
        self._constraint_upper: np.ndarray | None = None
        self.var_shift: np.ndarray | None = None
        self.var_scale: np.ndarray | None = None
        self.n_iterations = 0
        self._solve_verbosity = 0

    def print_info(self) -> None:
        """Print solver info, called before solving"""
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
        Initialize the optimizer.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent.
        """
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
            isinstance(self.mem_size, (bool, np.bool_))
            or not isinstance(self.mem_size, (int, np.integer))
            or self.mem_size < 1
        ):
            raise ValueError(
                f"Optimizer '{self.name}': mem_size must be a positive integer."
            )

        reserved = {"method", "jac", "bounds", "constraints", "callback"}
        conflicts = reserved.intersection(self.scipy_pars)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"Optimizer '{self.name}': SciPy parameters managed internally: {names}."
            )

        self._value_mem = {}
        self._gradient_mem = {}
        self._initialize_scaling()
        self._constraints_scipy = self._make_constraints()
        super().initialize(verbosity)

    def _initialize_scaling(self) -> None:
        """Create affine scaling from physical variable bounds."""
        initial = np.asarray(self.problem.initial_values_float(), dtype=np.float64)
        lower = np.asarray(self.problem.min_values_float(), dtype=np.float64)
        upper = np.asarray(self.problem.max_values_float(), dtype=np.float64)
        if np.any(lower > upper):
            raise ValueError(
                f"Optimizer '{self.name}': Float variable lower bounds exceed upper bounds."
            )

        both = np.isfinite(lower) & np.isfinite(upper)
        span = upper - lower
        scalable = both & (span > 0.0)
        self.var_shift = initial.copy()
        self.var_shift[scalable] = 0.5 * (lower[scalable] + upper[scalable])

        self.var_scale = np.maximum(np.abs(initial), 1.0)
        self.var_scale[scalable] = 0.5 * span[scalable]
        lower_only = np.isfinite(lower) & ~np.isfinite(upper)
        upper_only = ~np.isfinite(lower) & np.isfinite(upper)
        self.var_scale[lower_only] = np.maximum(
            np.abs(initial[lower_only] - lower[lower_only]), 1.0
        )
        self.var_scale[upper_only] = np.maximum(
            np.abs(upper[upper_only] - initial[upper_only]), 1.0
        )

    def _to_problem_vars(self, scaled: ArrayLike) -> np.ndarray:
        """Convert dimensionless optimizer variables to physical variables."""
        var_shift = self.var_shift
        var_scale = self.var_scale
        assert var_shift is not None
        assert var_scale is not None
        return var_shift + var_scale * np.asarray(scaled)

    def _to_scaled_vars(self, physical: ArrayLike) -> np.ndarray:
        """Convert physical variables to dimensionless optimizer variables."""
        var_shift = self.var_shift
        var_scale = self.var_scale
        assert var_shift is not None
        assert var_scale is not None
        return (np.asarray(physical) - var_shift) / var_scale

    def _remember(
        self,
        memory: dict[tuple[float, ...], _CacheValueT],
        key: tuple[float, ...],
        value: _CacheValueT,
    ) -> _CacheValueT:
        """Store a cache entry and evict the oldest one if required."""
        if len(memory) >= self.mem_size:
            del memory[next(iter(memory))]
        memory[key] = value
        return value

    def _get_values(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return objective and constraint values at an iterate."""
        memory = self._value_mem
        assert memory is not None
        key = tuple(float(value) for value in x)
        if key not in memory:
            values = self.problem.evaluate_individual(
                np.array([], dtype=np.int32), np.asarray(x, dtype=np.float64)
            )
            self._remember(memory, key, values)
        return memory[key]

    def _get_gradients(self, x: np.ndarray) -> np.ndarray:
        """Return the vectorized objective and constraint Jacobian."""
        memory = self._gradient_mem
        assert memory is not None
        key = tuple(float(value) for value in x)
        if key not in memory:
            try:
                objs, cons = self._get_values(x)
                gradients = self.problem.get_gradients(
                    np.array([], dtype=np.int32),
                    np.asarray(x, dtype=np.float64),
                    func_values=np.r_[objs, cons],
                    pop=self.vectorized,
                )
            except ValueError as error:
                raise ValueError(
                    f"Optimizer '{self.name}': Failed to determine a finite "
                    "objective and constraint Jacobian."
                ) from error
            shape = (1 + self.problem.n_constraints, self.problem.n_vars_float)
            if gradients.shape != shape:
                raise ValueError(
                    f"Optimizer '{self.name}': Expected gradient shape {shape}, "
                    f"received {gradients.shape}."
                )
            if not np.all(np.isfinite(gradients)):
                raise ValueError(
                    f"Optimizer '{self.name}': Non-finite objective or constraint gradient."
                )
            self._remember(memory, key, gradients)
        return memory[key]

    def _objective(self, scaled: np.ndarray) -> float:
        """Return the minimization-oriented objective value."""
        x = self._to_problem_vars(scaled)
        objs, __ = self._get_values(x)
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        return float(sign * objs[0])

    def _objective_jac(self, scaled: np.ndarray) -> np.ndarray:
        """Return the minimization-oriented objective gradient."""
        x = self._to_problem_vars(scaled)
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        var_scale = self.var_scale
        assert var_scale is not None
        return sign * self._get_gradients(x)[0] * var_scale

    def _constraint_values(
        self,
        scaled: np.ndarray,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> np.ndarray:
        """Return one group of constraints in SciPy sign convention."""
        x = self._to_problem_vars(scaled)
        __, cons = self._get_values(x)
        return sign * (cons[indices] - bounds)

    def _constraint_jac(
        self,
        scaled: np.ndarray,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> np.ndarray:
        """Return one grouped constraint Jacobian."""
        del bounds
        x = self._to_problem_vars(scaled)
        var_scale = self.var_scale
        assert var_scale is not None
        return sign * self._get_gradients(x)[1 + indices] * var_scale

    def _make_constraint(
        self,
        kind: str,
        indices: np.ndarray,
        bounds: np.ndarray,
        sign: float,
    ) -> dict[str, object]:
        """Create a grouped old-style SciPy constraint specification."""
        return {
            "type": kind,
            "fun": self._constraint_values,
            "jac": self._constraint_jac,
            "args": (indices, bounds, sign),
        }

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
        self._constraint_lower = lower
        self._constraint_upper = upper
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

    def _constraint_violation(self, cons: np.ndarray) -> float:
        """Return the maximum exact constraint-bound violation."""
        if not self.problem.n_constraints:
            return 0.0
        constraint_lower = self._constraint_lower
        constraint_upper = self._constraint_upper
        assert constraint_lower is not None
        assert constraint_upper is not None
        lower = np.maximum(constraint_lower - cons, 0.0)
        upper = np.maximum(cons - constraint_upper, 0.0)
        return float(np.max(np.maximum(lower, upper)))

    def _progress_callback(self, scaled: np.ndarray) -> None:
        """Report one accepted SLSQP iterate without new evaluations."""
        self.n_iterations += 1
        x = self._to_problem_vars(scaled)
        memory = self._value_mem
        assert memory is not None
        key = tuple(float(value) for value in x)
        values = memory.get(key)
        if values is None:
            objs = None
            cons = None
            if self._solve_verbosity:
                print(f"{self.n_iterations:>5} | {'cached values unavailable':>32}")
        else:
            objs, cons = values
            if self._solve_verbosity:
                violation = self._constraint_violation(cons)
                print(f"{self.n_iterations:>5} | {objs[0]:>14.7e} | {violation:>14.7e}")

        if self._has_callbacks:
            self._notify_callbacks(
                OptimizerCallbackData(
                    event="iteration",
                    iteration=self.n_iterations,
                    vars_int=np.array([], dtype=np.int32),
                    vars_float=x,
                    objs=objs,
                    cons=cons,
                )
            )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults:
        """
        Run the SLSQP optimizer.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent.
        callbacks
            Ordered callbacks for accepted optimizer iterates.

        Returns
        -------
        results
            The optimization results.
        """
        super().solve(verbosity, callbacks)
        self.n_iterations = 0
        self._solve_verbosity = verbosity
        x0 = np.asarray(self.problem.initial_values_float(), dtype=np.float64)
        scaled0 = self._to_scaled_vars(x0)
        lower = self._to_scaled_vars(self.problem.min_values_float())
        upper = self._to_scaled_vars(self.problem.max_values_float())
        bounds = list(
            zip(
                lower,
                upper,
            )
        )
        if verbosity:
            print("\nRunning SLSQP")
            print("--------------------+----------------+----------------")
            print("   it |      objective | max constraint")
            print("--------------------+----------------+----------------")
        report_progress = bool(verbosity) or self._has_callbacks
        constraints_scipy = self._constraints_scipy
        assert constraints_scipy is not None
        self.scipy_results = minimize(
            self._objective,
            scaled0,
            method="SLSQP",
            jac=self._objective_jac,
            bounds=bounds,
            constraints=constraints_scipy,
            callback=self._progress_callback if report_progress else None,
            **cast(dict[str, Any], self.scipy_pars),
        )
        scipy_results = self.scipy_results
        if not report_progress:
            self.n_iterations = int(scipy_results.nit)
        if verbosity:
            print("--------------------+----------------+----------------")

        vars_float = self._to_problem_vars(scipy_results.x)
        scipy_results.x = vars_float
        vars_int = np.array([], dtype=np.int32)
        problem_results, objs, cons = self.problem.finalize_individual(
            vars_int, vars_float, verbosity=verbosity
        )
        feasible = np.all(self.problem.check_constraints_individual(cons))
        success = bool(scipy_results.success and feasible)
        results = SingleObjOptResults(
            self.problem,
            success,
            vars_int,
            vars_float,
            objs,
            cons,
            problem_results,
        )
        return self._finalize_callbacks(results)
