import numpy as np
from scipy.optimize import minimize

from iwopy.core import Optimizer, SingleObjOptResults


class SLSQP(Optimizer):
    """
    Gradient-based SLSQP optimizer for continuous problems.

    The complete objective and constraint Jacobian is obtained through
    ``Problem.get_gradients(..., pop=True)`` at each iterate. Hence missing
    analytical derivatives can be supplied by a population-capable problem
    wrapper such as :class:`iwopy.wrappers.LocalFD`.

    Attributes
    ----------
    scipy_pars: dict
        Additional parameters for :func:`scipy.optimize.minimize`.
    mem_size: int
        Maximum number of cached value and gradient evaluations.
    vectorized: bool
        Whether gradients use population-based function evaluation.
    var_shift: numpy.ndarray
        Variable offsets used for internal dimensionless coordinates.
    var_scale: numpy.ndarray
        Variable scales used for internal dimensionless coordinates.
    scipy_results: scipy.optimize.OptimizeResult
        Results returned by SciPy after solving, or ``None`` before solving.
    n_iterations: int
        Number of completed SLSQP iterations in the current or latest solve.

    :group: optimizers

    """

    def __init__(
        self,
        problem,
        scipy_pars=None,
        mem_size=100,
        vectorized=True,
        name="SLSQP",
    ):
        """
        Constructor.

        Parameters
        ----------
        problem: iwopy.core.Problem
            The continuous single-objective problem to optimize.
        scipy_pars: dict, optional
            Additional parameters for :func:`scipy.optimize.minimize`.
            The parameters ``method``, ``jac``, ``bounds``, and
            ``constraints`` are managed by this optimizer.
        mem_size: int
            Maximum number of cached value and gradient evaluations.
        vectorized: bool
            Whether gradients use population-based function evaluation.
        name: str, optional
            The optimizer name.

        """
        super().__init__(problem, name)
        self.scipy_pars = {} if scipy_pars is None else scipy_pars.copy()
        self.mem_size = mem_size
        self.vectorized = vectorized
        self.scipy_results = None
        self._value_mem = None
        self._gradient_mem = None
        self._constraints_scipy = None
        self._constraint_lower = None
        self._constraint_upper = None
        self.var_shift = None
        self.var_scale = None
        self.n_iterations = 0

    def initialize(self, verbosity=1):
        """
        Initialize the optimizer.

        Parameters
        ----------
        verbosity: int
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
        if not isinstance(self.mem_size, (int, np.integer)) or self.mem_size < 1:
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

    def _initialize_scaling(self):
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

    def _to_problem_vars(self, scaled):
        """Convert dimensionless optimizer variables to physical variables."""
        return self.var_shift + self.var_scale * np.asarray(scaled)

    def _to_scaled_vars(self, physical):
        """Convert physical variables to dimensionless optimizer variables."""
        return (np.asarray(physical) - self.var_shift) / self.var_scale

    def _remember(self, memory, key, value):
        """Store a cache entry and evict the oldest one if required."""
        if len(memory) >= self.mem_size:
            del memory[next(iter(memory))]
        memory[key] = value
        return value

    def _get_values(self, x):
        """Return objective and constraint values at an iterate."""
        key = tuple(x)
        if key not in self._value_mem:
            values = self.problem.evaluate_individual(
                np.array([], dtype=np.int32), np.asarray(x, dtype=np.float64)
            )
            self._remember(self._value_mem, key, values)
        return self._value_mem[key]

    def _get_gradients(self, x):
        """Return the vectorized objective and constraint Jacobian."""
        key = tuple(x)
        if key not in self._gradient_mem:
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
            self._remember(self._gradient_mem, key, gradients)
        return self._gradient_mem[key]

    def _objective(self, scaled):
        """Return the minimization-oriented objective value."""
        x = self._to_problem_vars(scaled)
        objs, __ = self._get_values(x)
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        return sign * objs[0]

    def _objective_jac(self, scaled):
        """Return the minimization-oriented objective gradient."""
        x = self._to_problem_vars(scaled)
        sign = -1.0 if self.problem.maximize_objs[0] else 1.0
        return sign * self._get_gradients(x)[0] * self.var_scale

    def _constraint_values(self, scaled, indices, bounds, sign):
        """Return one group of constraints in SciPy sign convention."""
        x = self._to_problem_vars(scaled)
        __, cons = self._get_values(x)
        return sign * (cons[indices] - bounds)

    def _constraint_jac(self, scaled, indices, bounds, sign):
        """Return one grouped constraint Jacobian."""
        del bounds
        x = self._to_problem_vars(scaled)
        return sign * self._get_gradients(x)[1 + indices] * self.var_scale

    def _make_constraint(self, kind, indices, bounds, sign):
        """Create a grouped old-style SciPy constraint specification."""
        return {
            "type": kind,
            "fun": self._constraint_values,
            "jac": self._constraint_jac,
            "args": (indices, bounds, sign),
        }

    def _make_constraints(self):
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
        constraints = []

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

    def _constraint_violation(self, cons):
        """Return the maximum exact constraint-bound violation."""
        if not self.problem.n_constraints:
            return 0.0
        lower = np.maximum(self._constraint_lower - cons, 0.0)
        upper = np.maximum(cons - self._constraint_upper, 0.0)
        return float(np.max(np.maximum(lower, upper)))

    def _progress_callback(self, scaled):
        """Report one accepted SLSQP iterate without new evaluations."""
        self.n_iterations += 1
        x = self._to_problem_vars(scaled)
        values = self._value_mem.get(tuple(x))
        if values is None:
            print(f"{self.n_iterations:>5} | {'cached values unavailable':>32}")
            return
        objs, cons = values
        violation = self._constraint_violation(cons)
        print(f"{self.n_iterations:>5} | {objs[0]:>14.7e} | {violation:>14.7e}")

    def solve(self, verbosity=1):
        """
        Run the SLSQP optimizer.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent.

        Returns
        -------
        results: iwopy.core.SingleObjOptResults
            The optimization results.

        """
        super().solve(verbosity)
        self.n_iterations = 0
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
        callback = None
        if verbosity:
            print("\nRunning SLSQP")
            print("--------------------+----------------+----------------")
            print("   it |      objective | max constraint")
            print("--------------------+----------------+----------------")
            callback = self._progress_callback
        self.scipy_results = minimize(
            self._objective,
            scaled0,
            method="SLSQP",
            jac=self._objective_jac,
            bounds=bounds,
            constraints=self._constraints_scipy,
            callback=callback,
            **self.scipy_pars,
        )
        if verbosity:
            print("--------------------+----------------+----------------")

        vars_float = self._to_problem_vars(self.scipy_results.x)
        self.scipy_results.x = vars_float
        vars_int = np.array([], dtype=np.int32)
        problem_results, objs, cons = self.problem.finalize_individual(
            vars_int, vars_float, verbosity=verbosity
        )
        feasible = np.all(self.problem.check_constraints_individual(cons))
        success = bool(self.scipy_results.success and feasible)
        return SingleObjOptResults(
            self.problem,
            success,
            vars_int,
            vars_float,
            objs,
            cons,
            problem_results,
        )
