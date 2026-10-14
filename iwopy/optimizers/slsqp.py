from typing import Any, TypeVar, cast

import numpy as np
from numpy.typing import ArrayLike
from scipy.optimize import OptimizeResult, minimize
from scipy.sparse import csr_matrix

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)
from iwopy.interfaces.scipy._options import minimize_options


_CacheValueT = TypeVar("_CacheValueT")


class _FeasibleStagnation(Exception):
    def __init__(self, variables: np.ndarray) -> None:
        self.variables = variables.copy()


class SLSQP(Optimizer):
    """Gradient-based SLSQP optimizer for continuous problems.

    The objective and selected constraint Jacobian is obtained through
    ``Problem.get_jacobian(..., pop=True)`` at each iterate. Hence missing
    analytical derivatives can be supplied by a population-capable problem
    wrapper such as :class:`iwopy.wrappers.LocalFD`.
    Jacobians are cached in CSR storage and only the rows requested by native
    SciPy callbacks are converted to dense arrays.

    Constraint bounds are relaxed by the feasibility tolerance owned by each
    registered :class:`iwopy.core.Constraint`. A small inward numerical reserve
    keeps a solver result on the accepted side of iwopy's feasibility check.

    Attributes
    ----------
    scipy_results
        Primary SciPy result with physical variables and original status.
    restoration_results
        Most recent restoration result with physical variables, or None.
    termination_reason
        Primary termination: converged, scipy_failure, feasible_stagnation,
        or None before solving. Feasibility is checked independently.
    active_constraint_components
        Sorted global constraint indices selected for the current solve.
    n_iterations
        Total accepted iterations across primary and restoration phases.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        tol: float | None = None,
        max_iterations: int | None = None,
        max_evaluations: int | None = None,
        memory_size: int = 100,
        vectorize: bool = True,
        name: str = "SLSQP",
        feasibility_margin: float = 0.0,
        stall_iterations: int | None = None,
        progress_tol: float = 1e-8,
        restoration_iterations: int = 0,
        restoration_scales: float | np.ndarray | None = None,
        move_limit: float | np.ndarray | None = None,
        reduce_constraints: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize the SLSQP optimizer.

        Parameters
        ----------
        problem
            The continuous single-objective problem to optimize.
        tol
            Native aggregate stopping tolerance; None uses SciPy defaults.
        max_iterations
            Native SLSQP iteration limit; None uses the native default.
        max_evaluations
            Unsupported by native SLSQP; must be None.
        memory_size
            Maximum number of cached value and Jacobian evaluations.
        vectorize
            Whether Jacobian evaluation uses population-based function evaluation.
        name
            The optimizer name.
        feasibility_margin
            Inward reduction of solver tolerance in each constraint's units,
            clamped at its public tolerance. Exact equalities are unchanged.
        stall_iterations
            Accepted primary iterations without significant best-feasible
            improvement after the first feasible point before stopping, or
            None to disable this stop. Infeasible starts do not consume patience.
        progress_tol
            Cumulative best-feasible improvement required to reset patience,
            in evaluated objective units before finalization rescaling.
        restoration_iterations
            Iteration budget for each pre/post feasibility-restoration phase.
            Zero disables restoration; feasibility does not imply convergence.
        restoration_scales
            Positive finite characteristic constraint values in each component's
            units, scalar or shape (n_constraints,) in full constraint order.
            Required when restoration is enabled for a constrained problem.
            Violations are divided by these scales before squaring; multiply
            scales by the same factors when changing constraint units.
        move_limit
            Positive finite displacement limit from the initial point over
            one solve, scalar or shape (n_vars_float,).
        reduce_constraints
            Use bound-certified constraint rows for the solver Jacobian.
            Full constraint evaluation and feasibility checks remain unchanged.
        kwargs
            Flat native SLSQP options, such as ftol and eps. Method,
            bounds, constraints, callbacks and Jacobians are managed by iwopy.

        Raises
        ------
        ValueError
            If vectorize is not a boolean, an iteration limit is invalid, or
            an unsupported evaluation limit is requested.
        TypeError
            If a superseded or unsupported shared keyword is supplied.
        """
        super().__init__(problem, name)
        self._validate_kwargs(kwargs)
        limits = self._limit_kwargs(max_iterations, max_evaluations, "maxiter")
        self._native_pars = minimize_options(kwargs | limits, tol)
        self.memory_size = memory_size
        self.vectorize = self._validate_vectorize(vectorize)
        self.feasibility_margin = feasibility_margin
        self.stall_iterations = stall_iterations
        self.progress_tol = progress_tol
        self.restoration_iterations = restoration_iterations
        self.restoration_scales = restoration_scales
        self._restoration_scales: np.ndarray | None = None
        self.move_limit = move_limit
        self.reduce_constraints = reduce_constraints
        self.termination_reason: str | None = None
        self.restoration_results: OptimizeResult | None = None
        self.scipy_results: OptimizeResult | None = None
        self._value_mem: (
            dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] | None
        ) = None
        self._jacobian_mem: dict[tuple[float, ...], csr_matrix] | None = None
        self._constraints_scipy: list[dict[str, object]] | None = None
        self._constraint_lower: np.ndarray | None = None
        self._constraint_upper: np.ndarray | None = None
        self.var_shift: np.ndarray | None = None
        self.var_scale: np.ndarray | None = None
        self.n_iterations = 0
        self._solve_verbosity = 0
        self._pending_progress: OptimizerCallbackData | None = None
        self._best_feasible: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
        self._progress_reference: float | None = None
        self._stale_iterations = 0
        self._restoring = False
        self._solver_lower: np.ndarray | None = None
        self._solver_upper: np.ndarray | None = None
        self.active_constraint_components = np.array([], dtype=int)
        self._variable_lower: np.ndarray | None = None
        self._variable_upper: np.ndarray | None = None

    @property
    def best_feasible(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Return copies of the best feasible variables, objectives, and constraints.

        Returns
        -------
        point
            Tuple of float variables, objectives, and full constraints in
            problem order, or None before a feasible point is evaluated.
            Shapes are (n_vars_float,), (n_objectives,), and (n_constraints,).
            Function values are evaluated values before any finalization
            rescaling by the problem.
            Feasibility does not imply solver convergence.
        """
        if self._best_feasible is None:
            return None
        variables, objectives, constraints = self._best_feasible
        return variables.copy(), objectives.copy(), constraints.copy()

    def _track_feasible(
        self, variables: np.ndarray, objectives: np.ndarray, constraints: np.ndarray
    ) -> None:
        """Retain improving finite points satisfying all bounds and constraints."""
        if (
            not np.all(np.isfinite(variables))
            or not np.all(np.isfinite(objectives))
            or not np.all(np.isfinite(constraints))
            or np.any(variables < self.problem.min_values_float())
            or np.any(variables > self.problem.max_values_float())
            or not np.all(self.problem.check_constraints_individual(constraints))
        ):
            return
        maximize = self.problem.maximize_objs[0]
        previous = self._best_feasible
        if previous is None or (
            objectives[0] > previous[1][0]
            if maximize
            else objectives[0] < previous[1][0]
        ):
            self._best_feasible = (
                variables.copy(),
                objectives.copy(),
                constraints.copy(),
            )

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
        Initialize the optimizer.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent.

        Raises
        ------
        ValueError
            If the problem is not continuous and single-objective, a control
            or movement box is invalid, a reserved SciPy parameter is supplied,
            or a constraint returns an invalid component selection.
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
            isinstance(self.memory_size, (bool, np.bool_))
            or not isinstance(self.memory_size, (int, np.integer))
            or self.memory_size < 1
        ):
            raise ValueError(
                f"Optimizer '{self.name}': memory_size must be a positive integer."
            )
        for setting in ("feasibility_margin", "progress_tol"):
            value = getattr(self, setting)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(
                    f"Optimizer '{self.name}': {setting} must be finite and non-negative."
                )
        for setting, minimum in (
            ("stall_iterations", 1),
            ("restoration_iterations", 0),
        ):
            value = getattr(self, setting)
            if value is None and setting == "stall_iterations":
                continue
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or value < minimum
            ):
                raise ValueError(f"Optimizer '{self.name}': Invalid {setting}.")

        reserved = {"method", "jac", "bounds", "constraints", "callback"}
        conflicts = reserved.intersection(self._native_pars)
        if conflicts:
            names = ", ".join(sorted(conflicts))
            raise ValueError(
                f"Optimizer '{self.name}': SciPy parameters managed internally: {names}."
            )

        self._value_mem = {}
        self._jacobian_mem = {}
        self._initialize_scaling()
        initial = np.asarray(self.problem.initial_values_float(), dtype=np.float64)
        self._prepare_constraints(initial)
        super().initialize(verbosity)

    def _prepare_constraints(self, initial: np.ndarray) -> None:
        """Rebuild the guarded box and fixed component set for one solve."""
        self._initialize_variable_bounds(initial)
        assert self._variable_lower is not None
        assert self._variable_upper is not None
        self.active_constraint_components = (
            self.problem.get_active_constraint_components(
                initial, self._variable_lower, self._variable_upper
            )
            if self.reduce_constraints
            else np.arange(self.problem.n_constraints, dtype=int)
        )
        self._constraints_scipy = self._make_constraints()
        self._initialize_restoration_scales()

    def _initialize_restoration_scales(self) -> None:
        """Validate explicit characteristic values for dimensionless restoration."""
        self._restoration_scales = None
        if self.restoration_scales is None:
            if self.restoration_iterations and self.problem.n_constraints:
                raise ValueError(
                    f"Optimizer '{self.name}': restoration_scales are required "
                    "when feasibility restoration is enabled."
                )
            return
        scales = np.asarray(self.restoration_scales, dtype=np.float64)
        if scales.ndim == 0:
            if not np.isfinite(scales) or scales <= 0.0:
                raise ValueError(
                    f"Optimizer '{self.name}': restoration_scales must be positive and finite."
                )
            scales = np.full(self.problem.n_constraints, float(scales))
        if (
            scales.shape != (self.problem.n_constraints,)
            or not np.all(np.isfinite(scales))
            or np.any(scales <= 0.0)
        ):
            raise ValueError(
                f"Optimizer '{self.name}': restoration_scales must contain positive finite "
                f"values with shape {(self.problem.n_constraints,)}."
            )
        self._restoration_scales = scales.copy()

    def _initialize_variable_bounds(self, initial: np.ndarray) -> None:
        """Intersect physical bounds with the declared local movement box."""
        lower = np.asarray(self.problem.min_values_float(), dtype=np.float64).copy()
        upper = np.asarray(self.problem.max_values_float(), dtype=np.float64).copy()
        if self.move_limit is not None:
            limits = np.asarray(self.move_limit, dtype=np.float64)
            if limits.ndim == 0:
                limits = np.full(self.problem.n_vars_float, float(limits))
            if (
                limits.shape != initial.shape
                or not np.all(np.isfinite(limits))
                or np.any(limits <= 0.0)
            ):
                raise ValueError(
                    f"Optimizer '{self.name}': move_limit must contain positive finite coordinate limits."
                )
            lower = np.maximum(lower, initial - limits)
            upper = np.minimum(upper, initial + limits)
        if np.any(lower > upper):
            raise ValueError(f"Optimizer '{self.name}': Empty guarded variable box.")
        self._variable_lower, self._variable_upper = lower, upper

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
        if len(memory) >= self.memory_size:
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
        values = memory[key]
        self._track_feasible(x, values[0], values[1])
        return values

    def _get_jacobian(self, x: np.ndarray) -> csr_matrix:
        """Return the vectorized objective and constraint Jacobian."""
        memory = self._jacobian_mem
        assert memory is not None
        key = tuple(float(value) for value in x)
        if key not in memory:
            try:
                objs, cons = self._get_values(x)
                jacobian = self.problem.get_jacobian(
                    np.array([], dtype=np.int32),
                    np.asarray(x, dtype=np.float64),
                    components=np.r_[0, 1 + self.active_constraint_components],
                    func_values=np.r_[objs, cons[self.active_constraint_components]],
                    pop=self.vectorize,
                )
            except ValueError as error:
                raise ValueError(
                    f"Optimizer '{self.name}': Failed to determine a finite "
                    "objective and constraint Jacobian."
                ) from error
            shape = (
                1 + len(self.active_constraint_components),
                self.problem.n_vars_float,
            )
            if jacobian.shape != shape:
                raise ValueError(
                    f"Optimizer '{self.name}': Expected Jacobian shape {shape}, "
                    f"received {jacobian.shape}."
                )
            if not np.all(np.isfinite(jacobian.data)):
                raise ValueError(
                    f"Optimizer '{self.name}': Non-finite objective or constraint Jacobian entry."
                )
            self._remember(memory, key, jacobian)
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
        return sign * self._get_jacobian(x)[0].toarray().ravel() * var_scale

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
        positions = np.searchsorted(self.active_constraint_components, indices)
        return sign * self._get_jacobian(x)[1 + positions].toarray() * var_scale

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
        tolerance = np.concatenate(
            [
                np.full(
                    constraint.n_components(),
                    constraint.tol,
                    dtype=np.float64,
                )
                for constraint in self.problem.cons.functions
            ]
        )
        self._constraint_lower = lower
        self._constraint_upper = upper
        reserve = np.maximum(
            np.maximum(tolerance * 1e-9, self.feasibility_margin),
            10.0 * np.finfo(np.float64).eps,
        )
        solver_tolerance = np.maximum(tolerance - reserve, 0.0)
        solver_lower = lower - solver_tolerance
        solver_upper = upper + solver_tolerance
        self._solver_lower = solver_lower
        self._solver_upper = solver_upper
        equal = (
            np.isfinite(lower)
            & np.isfinite(upper)
            & (lower == upper)
            & (tolerance == 0.0)
        )
        constraints: list[dict[str, object]] = []
        active = np.zeros(self.problem.n_constraints, dtype=bool)
        active[self.active_constraint_components] = True

        indices = np.flatnonzero(equal & active)
        if len(indices):
            constraints.append(
                self._make_constraint("eq", indices, solver_lower[indices], 1.0)
            )

        indices = np.flatnonzero(np.isfinite(solver_lower) & ~equal & active)
        if len(indices):
            constraints.append(
                self._make_constraint("ineq", indices, solver_lower[indices], 1.0)
            )

        indices = np.flatnonzero(np.isfinite(solver_upper) & ~equal & active)
        if len(indices):
            constraints.append(
                self._make_constraint("ineq", indices, solver_upper[indices], -1.0)
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

    def _emit_progress(self, data: OptimizerCallbackData) -> None:
        """Report a buffered SLSQP iterate."""
        if self._solve_verbosity:
            if data.objs is None or data.cons is None:
                print(f"{data.iteration:>5} | {'cached values unavailable':>32}")
            else:
                violation = self._constraint_violation(data.cons[0])
                print(
                    f"{data.iteration:>5} | {data.objs[0, 0]:>14.7e} | "
                    f"{violation:>14.7e}"
                )
        if self._has_callbacks:
            self._notify_callbacks(data)

    def _progress_callback(self, scaled: np.ndarray) -> None:
        """Buffer one accepted SLSQP iterate without new evaluations."""
        if self._pending_progress is not None:
            self._emit_progress(self._pending_progress)

        self.n_iterations += 1
        x = self._to_problem_vars(scaled)
        memory = self._value_mem
        assert memory is not None
        key = tuple(float(value) for value in x)
        values = memory.get(key)
        objs, cons = (None, None) if values is None else values
        self._pending_progress = OptimizerCallbackData(
            event="iteration",
            iteration=self.n_iterations,
            vars_int=np.array([], dtype=np.int32),
            vars_float=x,
            objs=objs,
            cons=cons,
        )
        if not self._restoring:
            self._check_progress(scaled)

    def _check_progress(self, scaled: np.ndarray) -> None:
        """Stop on insufficient cumulative fully feasible objective progress."""
        if self.stall_iterations is None:
            return
        best = self._best_feasible
        if best is None:
            return
        value = float(best[1][0])
        sign = 1.0 if self.problem.maximize_objs[0] else -1.0
        if (
            self._progress_reference is None
            or sign * (value - self._progress_reference) > self.progress_tol
        ):
            self._progress_reference = value
            self._stale_iterations = 0
        else:
            self._stale_iterations += 1
        if self._stale_iterations >= self.stall_iterations:
            raise _FeasibleStagnation(scaled)

    def _restoration_residual(self, constraints: np.ndarray) -> np.ndarray:
        """Dimensionless signed violations of inward solver bounds."""
        assert self._solver_lower is not None
        assert self._solver_upper is not None
        assert self._restoration_scales is not None
        return (
            np.maximum(self._solver_lower - constraints, 0.0)
            - np.maximum(constraints - self._solver_upper, 0.0)
        ) / self._restoration_scales

    def _restoration_objective(self, scaled: np.ndarray) -> float:
        variables = self._to_problem_vars(scaled)
        _, constraints = self._get_values(variables)
        residual = self._restoration_residual(constraints)
        return float(0.5 * np.dot(residual, residual))

    def _restoration_jac(self, scaled: np.ndarray) -> np.ndarray:
        variables = self._to_problem_vars(scaled)
        _, constraints = self._get_values(variables)
        residual = self._restoration_residual(constraints)
        components = np.flatnonzero(residual) + 1
        if not len(components):
            return np.zeros(self.problem.n_vars_float, dtype=np.float64)
        jacobian = self.problem.get_jacobian(
            np.array([], dtype=np.int32),
            variables,
            components=components,
            func_values=constraints[components - 1],
            pop=self.vectorize,
        )
        if jacobian.shape != (
            len(components),
            self.problem.n_vars_float,
        ) or not np.all(np.isfinite(jacobian.data)):
            raise ValueError(
                f"Optimizer '{self.name}': Invalid feasibility-restoration Jacobian."
            )
        assert self.var_scale is not None
        assert self._restoration_scales is not None
        weights = residual[components - 1] / self._restoration_scales[components - 1]
        return -np.asarray(weights @ jacobian).ravel() * self.var_scale

    def _restore_feasibility(
        self, scaled: np.ndarray, bounds: list[tuple[float, float]]
    ) -> np.ndarray:
        """Run one bounded restoration phase without objective derivatives."""
        self._restoring = True
        try:
            result = minimize(
                self._restoration_objective,
                scaled,
                method="SLSQP",
                jac=self._restoration_jac,
                bounds=bounds,
                callback=self._progress_callback,
                options={
                    "maxiter": self.restoration_iterations,
                    "ftol": 1e-12,
                },
            )
        finally:
            self._restoring = False
        restored = np.asarray(result.x, dtype=np.float64).copy()
        result.x = self._to_problem_vars(restored)
        self.restoration_results = result
        self._get_values(result.x)
        return restored

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
            Ordered callbacks for accepted primary and restoration iterates.
            The last snapshot describes the returned point with finalized
            function values, which the problem may rescale.

        Returns
        -------
        results
            The best feasible evaluated point, or the final SciPy point if
            none was feasible. Success still requires SciPy convergence
            at a feasible endpoint. The raw physical endpoint and status
            remain available through ``scipy_results``.
        """
        self._best_feasible = None
        self.termination_reason = None
        self.restoration_results = None
        self._stale_iterations = 0
        self._progress_reference = None
        super().solve(verbosity, callbacks)
        assert self._value_mem is not None
        assert self._jacobian_mem is not None
        self._value_mem.clear()
        self._jacobian_mem.clear()
        self.n_iterations = 0
        self._solve_verbosity = verbosity
        self._pending_progress = None
        x0 = np.asarray(self.problem.initial_values_float(), dtype=np.float64)
        self._prepare_constraints(x0)
        scaled0 = self._to_scaled_vars(x0)
        assert self._variable_lower is not None
        assert self._variable_upper is not None
        lower = self._to_scaled_vars(self._variable_lower)
        upper = self._to_scaled_vars(self._variable_upper)
        bounds = list(
            zip(
                lower,
                upper,
            )
        )
        _, initial_constraints = self._get_values(x0)
        if self._best_feasible is not None:
            self._progress_reference = float(self._best_feasible[1][0])
        if self.restoration_iterations and not np.all(
            self.problem.check_constraints_individual(initial_constraints)
        ):
            scaled0 = self._restore_feasibility(scaled0, bounds)
        if self._best_feasible is not None:
            self._progress_reference = float(self._best_feasible[1][0])
        if verbosity:
            print("\nRunning SLSQP")
            print("--------------------+----------------+----------------")
            print("   it |      objective | max constraint")
            print("--------------------+----------------+----------------")
        report_progress = (
            bool(verbosity)
            or self._has_callbacks
            or self.stall_iterations is not None
            or self.restoration_iterations > 0
        )
        constraints_scipy = self._constraints_scipy
        assert constraints_scipy is not None
        primary_iteration_start = self.n_iterations
        try:
            self.scipy_results = minimize(
                self._objective,
                scaled0,
                method="SLSQP",
                jac=self._objective_jac,
                bounds=bounds,
                constraints=constraints_scipy,
                callback=self._progress_callback if report_progress else None,
                **cast(dict[str, Any], self._native_pars),
            )
            self.termination_reason = (
                "converged" if self.scipy_results.success else "scipy_failure"
            )
        except _FeasibleStagnation as stopped:
            self.termination_reason = "feasible_stagnation"
            self.scipy_results = OptimizeResult(
                x=stopped.variables,
                fun=self._objective(stopped.variables),
                success=False,
                status=9,
                message="No significant feasible objective progress",
                nit=self.n_iterations - primary_iteration_start,
            )
        scipy_results = self.scipy_results
        if not report_progress:
            self.n_iterations = int(scipy_results.nit)

        vars_float = self._to_problem_vars(scipy_results.x)
        scipy_results.x = vars_float.copy()
        endpoint_values = self._get_values(vars_float)
        endpoint_feasible = np.all(
            self.problem.check_constraints_individual(endpoint_values[1])
        )
        if not endpoint_feasible and self.restoration_iterations:
            self._restore_feasibility(self._to_scaled_vars(vars_float), bounds)
        if self._best_feasible is not None:
            vars_float = self._best_feasible[0].copy()
        vars_int = np.array([], dtype=np.int32)
        problem_results, objs, cons = self.problem.finalize_individual(
            vars_int, vars_float, verbosity=verbosity
        )
        feasible = np.all(self.problem.check_constraints_individual(cons))
        success = bool(scipy_results.success and endpoint_feasible and feasible)
        results = SingleObjOptResults(
            self.problem,
            success,
            vars_int,
            vars_float,
            objs,
            cons,
            problem_results,
        )
        if report_progress and self.n_iterations:
            self._pending_progress = None
            self._emit_progress(
                OptimizerCallbackData(
                    event="iteration",
                    iteration=self.n_iterations,
                    vars_int=vars_int,
                    vars_float=vars_float,
                    objs=objs,
                    cons=cons,
                )
            )
        if verbosity:
            print("--------------------+----------------+----------------")
        return self._finalize_callbacks(results)
