from types import ModuleType
from typing import Any

import numpy as np

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)
from iwopy.utils import import_module


class Optimizer_cma(Optimizer):
    """
    Vectorized single-objective CMA-ES interface to pycma.

    Each ask/tell generation evaluates objectives and constraints in one
    population call. Floating variables, box bounds, and nonlinear constraints
    are supported; integer variables and multiple objectives are not.
    Constraint bounds become nonpositive inequalities with the owning
    constraint's tolerance, handled by pycma's augmented Lagrangian.

    ``stop_conditions`` contains the backend termination reasons after the
    latest solve, and is cleared when a new run starts.
    """

    def __init__(
        self,
        problem: Problem,
        sigma0: float,
        *,
        pop_size: int | None = None,
        seed: int | None = None,
        maxiter: int | None = None,
        name: str = "Optimizer_cma",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with floating variables.
        sigma0
            Positive finite initial standard deviation in variable units;
            scale variables accordingly.
        pop_size
            Absolute population size; None uses pycma's dimension-based default.
        seed
            Native random seed; None uses the backend default.
        maxiter
            Generation limit; None uses the backend default.
        name
            The optimizer name.
        kwargs
            Flat native pycma options, such as ftarget, tolfun and tolx. Fitness
            targets use minimization-space values, including negation for
            maximization. Bounds, representation, callbacks and logging are managed.
        """
        super().__init__(problem, name)
        self._validate_kwargs(kwargs)
        self.sigma0 = sigma0
        self._native_pars = dict(kwargs)
        for key, value in (("popsize", pop_size), ("seed", seed), ("maxiter", maxiter)):
            if value is not None:
                if key in self._native_pars:
                    raise ValueError(f"CMAES: conflicting native option '{key}'")
                self._native_pars[key] = value
        self.stop_conditions: dict[str, Any] = {}
        self._cma: ModuleType | None = None
        self._cons_lower = np.empty(0, dtype=np.float64)
        self._cons_upper = np.empty(0, dtype=np.float64)
        self._lower_mask = np.empty(0, dtype=bool)
        self._upper_mask = np.empty(0, dtype=bool)

    def initialize(self, verbosity: int = 1) -> None:
        """
        Validate the problem and load the optional pycma dependency.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent.

        Raises
        ------
        ValueError
            If the problem is uninitialized, has unsupported variables or
            objectives, invalid bounds or initial values, an exact equality
            with zero tolerance, or conflicting backend options.
        ModuleNotFoundError
            If pycma is unavailable; install the ``cma`` extra.
        """
        self._validate_problem()
        self._validate_options()
        self._set_constraint_bounds()
        self._cma = import_module("cma", hint="uv sync --extra cma")
        super().initialize(verbosity)

    def _validate_problem(self) -> None:
        problem = self.problem
        if not problem.initialized:
            raise ValueError(f"Optimizer '{self.name}': Initialize the problem first")
        if problem.n_objectives != 1:
            raise ValueError(f"Optimizer '{self.name}' requires exactly one objective")
        if problem.n_vars_int or not problem.n_vars_float:
            raise ValueError(
                f"Optimizer '{self.name}' requires floating variables and no integer variables"
            )
        if not np.isfinite(self.sigma0) or self.sigma0 <= 0.0:
            raise ValueError(
                f"Optimizer '{self.name}': sigma0 must be positive and finite"
            )
        initial = np.asarray(problem.initial_values_float(), dtype=np.float64)
        lower = np.asarray(problem.min_values_float(), dtype=np.float64)
        upper = np.asarray(problem.max_values_float(), dtype=np.float64)
        if any(
            values.shape != (problem.n_vars_float,)
            for values in (initial, lower, upper)
        ):
            raise ValueError(
                f"Optimizer '{self.name}': Invalid floating variable shapes"
            )
        if np.any(np.isnan(lower) | np.isnan(upper) | (lower >= upper)):
            raise ValueError(
                f"Optimizer '{self.name}': Variable bounds must be ordered and non-NaN"
            )
        if not np.all(np.isfinite(initial) & (initial >= lower) & (initial <= upper)):
            raise ValueError(
                f"Optimizer '{self.name}': Initial values must be finite and within bounds"
            )

    def _validate_options(self) -> None:
        managed = {
            "bounds",
            "BoundaryHandler",
            "integer_variables",
            "fixed_variables",
            "transformation",
            "termination_callback",
            "verbose",
            "verb_disp",
            "verb_log",
        }
        conflicts = managed.intersection(self._native_pars)
        if conflicts:
            raise ValueError(
                f"Optimizer '{self.name}': Internally managed native options: {', '.join(sorted(conflicts))}"
            )

    def _set_constraint_bounds(self) -> None:
        lower_bounds, upper_bounds = [], []
        for constraint in self.problem.cons.functions:
            lower, upper = constraint.get_bounds()
            lower = np.asarray(lower, dtype=np.float64)
            upper = np.asarray(upper, dtype=np.float64)
            shape = (constraint.n_components(),)
            if lower.shape != shape or upper.shape != shape:
                raise ValueError(
                    f"Constraint '{constraint.name}': Invalid bound shapes"
                )
            if np.any(
                np.isnan(lower)
                | np.isnan(upper)
                | (lower > upper)
                | np.isposinf(lower)
                | np.isneginf(upper)
            ):
                raise ValueError(f"Constraint '{constraint.name}': Invalid bounds")
            if not np.isfinite(constraint.tol) or constraint.tol < 0.0:
                raise ValueError(
                    f"Constraint '{constraint.name}': Tolerance must be finite and nonnegative"
                )
            if np.any((lower == upper) & np.isfinite(lower)) and constraint.tol == 0.0:
                raise ValueError(
                    f"Constraint '{constraint.name}': Exact equality requires positive tolerance"
                )
            lower_bounds.append(lower - constraint.tol)
            upper_bounds.append(upper + constraint.tol)
        if lower_bounds:
            self._cons_lower = np.concatenate(lower_bounds)
            self._cons_upper = np.concatenate(upper_bounds)
        else:
            self._cons_lower = np.empty(0, dtype=np.float64)
            self._cons_upper = np.empty(0, dtype=np.float64)
        self._lower_mask = np.isfinite(self._cons_lower)
        self._upper_mask = np.isfinite(self._cons_upper)

    def _constraint_values(self, cons: np.ndarray) -> np.ndarray:
        return np.concatenate(
            (
                self._cons_lower[self._lower_mask] - cons[:, self._lower_mask],
                cons[:, self._upper_mask] - self._cons_upper[self._upper_mask],
            ),
            axis=1,
        )

    def _new_strategy(self, verbosity: int) -> Any:
        if self._cma is None:
            raise RuntimeError(f"Optimizer '{self.name}' has not loaded pycma")
        options = dict(self._native_pars)
        options.update(
            bounds=[self.problem.min_values_float(), self.problem.max_values_float()],
            verbose=-9,
            verb_disp=0,
            verb_log=0,
        )
        strategy = self._cma.CMAEvolutionStrategy(
            self.problem.initial_values_float(), self.sigma0, options
        )
        strategy.opts["verb_disp"] = 1 if verbosity else 0
        return strategy

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults:
        """
        Run a fresh vectorized ask/tell optimization.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent. No backend log files are written.
            pycma's constraint logger may create an empty ``outcmaes`` directory.
        callbacks
            Ordered callbacks receiving one ``iteration`` snapshot per
            completed generation. Values retain iwopy signs and component
            order; evaluation counts exclude finalization.

        Returns
        -------
        results
            Finalized best feasible solution, or the backend's favorite
            solution with ``success=False`` when none was feasible. Success
            requires a convergence stop and finite, feasible final values;
            budget-only termination is unsuccessful. Repeated calls start
            from the problem's initial values, not the previous run.

        Raises
        ------
        ValueError
            If problem or optimizer initialization is missing, or a
            population contains malformed or nonfinite function values.
        """
        super().solve(verbosity, callbacks)
        self.stop_conditions = {}
        strategy = self._new_strategy(verbosity)
        while not strategy.stop():
            solutions = strategy.ask()
            vars_float = np.asarray(solutions, dtype=np.float64)
            vars_int = np.empty((len(solutions), 0), dtype=np.int32)
            objs, cons = self.problem.evaluate_population(vars_int, vars_float)
            self._validate_values(objs, cons, len(solutions))
            fitness = objs[:, 0] * (-1.0 if self.problem.maximize_objs[0] else 1.0)
            inequalities = self._constraint_values(cons)
            strategy.tell(
                solutions,
                fitness,
                constraints_values=inequalities if inequalities.shape[1] else None,
            )
            if self._has_callbacks:
                self._notify_callbacks(
                    OptimizerCallbackData(
                        event="iteration",
                        vars_int=vars_int,
                        vars_float=vars_float,
                        objs=objs,
                        cons=cons,
                        iteration=int(strategy.countiter),
                        n_evaluations=int(strategy.countevals),
                    )
                )
            if verbosity:
                strategy.disp()
        return self._finalize_callbacks(self._results(strategy, verbosity))

    def _validate_values(self, objs: np.ndarray, cons: np.ndarray, n_pop: int) -> None:
        for name, values, width in (
            ("objectives", objs, 1),
            ("constraints", cons, self.problem.n_constraints),
        ):
            if values.shape != (n_pop, width) or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"Optimizer '{self.name}': Population {name} must be finite with shape {(n_pop, width)}"
                )

    def _results(self, strategy: Any, verbosity: int) -> SingleObjOptResults:
        constrained = bool(np.any(self._lower_mask) or np.any(self._upper_mask))
        if constrained:
            best = strategy.best_feasible.x if strategy.countiter else None
        else:
            best = strategy.result.xbest
        found = best is not None
        vars_float = np.asarray(
            best if found else strategy.result.xfavorite, dtype=np.float64
        ).copy()
        vars_int = np.empty(0, dtype=np.int32)
        problem_results, objs, cons = self.problem.finalize_individual(
            vars_int, vars_float, verbosity
        )
        self.stop_conditions = dict(strategy.stop())
        convergence_stops = {
            "ftarget",
            "tolx",
            "tolfun",
            "tolfunhist",
            "tolstagnation",
            "tolxstagnation",
            "tolflatfitness",
            "noeffectaxis",
            "noeffectcoord",
        }
        converged = bool(convergence_stops.intersection(self.stop_conditions))
        success = bool(
            found
            and converged
            and np.all(np.isfinite(objs))
            and np.all(np.isfinite(cons))
            and np.all(np.isfinite(vars_float))
            and np.all(vars_float >= self.problem.min_values_float())
            and np.all(vars_float <= self.problem.max_values_float())
            and np.all(self.problem.check_constraints_individual(cons))
        )
        return SingleObjOptResults(
            self.problem, success, vars_int, vars_float, objs, cons, problem_results
        )
