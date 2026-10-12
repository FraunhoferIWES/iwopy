import numpy as np

from iwopy.core import Optimizer, OptimizerCallbackData, Problem, SingleObjOptResults


class _PopulationOptimizer(Optimizer):
    def __init__(self, problem: Problem, name: str, vectorize: bool = True) -> None:
        super().__init__(problem, name)
        if not isinstance(vectorize, (bool, np.bool_)):
            raise ValueError(f"Optimizer '{name}': vectorize must be a boolean")
        self.vectorize = bool(vectorize)
        self.n_evaluations = 0
        self._best_values: np.ndarray | None = None
        self._best_rank: tuple[float, ...] = (np.inf,)

    def _validate_problem(self) -> None:
        problem = self.problem
        if not problem.initialized:
            raise ValueError(f"Optimizer '{self.name}': Initialize the problem first")
        if problem.n_objectives != 1:
            raise ValueError(f"Optimizer '{self.name}' requires exactly one objective")
        self._lower = np.concatenate(
            (problem.min_values_float(), problem.min_values_int())
        ).astype(float)
        self._upper = np.concatenate(
            (problem.max_values_float(), problem.max_values_int())
        ).astype(float)
        self._initial = np.concatenate(
            (problem.initial_values_float(), problem.initial_values_int())
        ).astype(float)
        shape = (problem.n_vars_float + problem.n_vars_int,)
        if not shape[0] or any(
            values.shape != shape
            for values in (self._lower, self._upper, self._initial)
        ):
            raise ValueError(
                f"Optimizer '{self.name}': Invalid variable shapes or no variables"
            )
        integer_start = problem.n_vars_float
        if np.any(self._lower[integer_start:] == -problem.INT_INF) or np.any(
            self._upper[integer_start:] == problem.INT_INF
        ):
            raise ValueError(
                f"Optimizer '{self.name}' requires finite integer variable bounds"
            )
        if not np.all(
            np.isfinite(self._lower)
            & np.isfinite(self._upper)
            & (self._lower < self._upper)
        ):
            raise ValueError(
                f"Optimizer '{self.name}' requires finite, strictly ordered variable bounds"
            )
        if not np.all(
            np.isfinite(self._initial)
            & (self._initial >= self._lower)
            & (self._initial <= self._upper)
        ):
            raise ValueError(
                f"Optimizer '{self.name}': Initial values must be finite and within bounds"
            )
        integer_limits = np.iinfo(np.int32)
        for values in (
            self._lower[integer_start:],
            self._upper[integer_start:],
            self._initial[integer_start:],
        ):
            if np.any(
                (values != np.rint(values))
                | (values < integer_limits.min)
                | (values > integer_limits.max)
            ):
                raise ValueError(
                    f"Optimizer '{self.name}': Integer values must be integral and fit int32"
                )
        self._cons_lower, self._cons_upper = self._constraint_bounds()

    def _constraint_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        lower_bounds, upper_bounds = [], []
        for constraint in self.problem.cons.functions:
            lower, upper = (
                np.asarray(values, dtype=float) for values in constraint.get_bounds()
            )
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
            if not np.isfinite(constraint.tol) or constraint.tol < 0:
                raise ValueError(
                    f"Constraint '{constraint.name}': Tolerance must be finite and nonnegative"
                )
            if np.any((lower == upper) & np.isfinite(lower)) and constraint.tol == 0:
                raise ValueError(
                    f"Constraint '{constraint.name}': Exact equality requires positive tolerance"
                )
            lower_bounds.append(lower - constraint.tol)
            upper_bounds.append(upper + constraint.tol)
        return (
            np.concatenate(lower_bounds) if lower_bounds else np.empty(0),
            np.concatenate(upper_bounds) if upper_bounds else np.empty(0),
        )

    def _reset(self) -> None:
        self.n_evaluations = 0
        self._best_values = None
        self._best_rank = (np.inf,)

    def _split(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        values = np.atleast_2d(np.asarray(values, dtype=float))
        shape = (len(values), len(self._lower))
        if values.shape != shape or not np.all(np.isfinite(values)):
            raise ValueError(f"Optimizer '{self.name}': Invalid candidate population")
        if np.any((values < self._lower) | (values > self._upper)):
            raise ValueError(
                f"Optimizer '{self.name}': Candidates outside variable bounds"
            )
        vars_float = values[:, : self.problem.n_vars_float]
        integer_values = values[:, self.problem.n_vars_float :]
        if np.any(integer_values != np.rint(integer_values)):
            raise ValueError(f"Optimizer '{self.name}': Nonintegral integer candidates")
        return integer_values.astype(np.int32), vars_float

    def _evaluate(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        vars_int, vars_float = self._split(values)
        n_pop = len(vars_float)
        if not n_pop:
            return np.empty((0, 1)), np.empty((0, self.problem.n_constraints))
        objs, cons = (
            self.problem.evaluate_population(vars_int, vars_float)
            if self.vectorize
            else self._evaluate_scalar_batch(vars_int, vars_float)
        )
        for label, output, width in (
            ("objective", objs, 1),
            ("constraint", cons, self.problem.n_constraints),
        ):
            if output.shape != (n_pop, width) or not np.all(np.isfinite(output)):
                raise ValueError(
                    f"Optimizer '{self.name}': Invalid or nonfinite {label} population"
                )
        self.n_evaluations += n_pop
        violations = np.maximum(self._residuals(cons), 0).sum(axis=1)
        feasible = np.all(
            self.problem.check_constraints_population(cons, verbosity=0), axis=1
        )
        for index, loss in enumerate(self._losses(objs)):
            rank = (
                float(not feasible[index]),
                0.0 if feasible[index] else float(violations[index]),
                float(loss),
            )
            if rank < self._best_rank:
                self._best_rank = rank
                self._best_values = np.concatenate(
                    (vars_float[index], vars_int[index])
                ).astype(float)
        return objs, cons

    def _evaluate_scalar_batch(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        objs = np.empty((len(vars_float), 1))
        cons = np.empty((len(vars_float), self.problem.n_constraints))
        for index in range(len(vars_float)):
            obj, con = self.problem.evaluate_individual(
                vars_int[index], vars_float[index]
            )
            for label, target, output in (
                ("objective", objs, obj),
                ("constraint", cons, con),
            ):
                values = np.asarray(output, dtype=float)
                if values.shape != target.shape[1:]:
                    raise ValueError(
                        f"Optimizer '{self.name}': Invalid {label} individual"
                    )
                target[index] = values
        return objs, cons

    def _losses(self, objs: np.ndarray) -> np.ndarray:
        return objs[:, 0] * (-1 if self.problem.maximize_objs[0] else 1)

    def _residuals(self, cons: np.ndarray) -> np.ndarray:
        lower_mask = np.isfinite(self._cons_lower)
        upper_mask = np.isfinite(self._cons_upper)
        return np.concatenate(
            (
                self._cons_lower[lower_mask] - cons[:, lower_mask],
                cons[:, upper_mask] - self._cons_upper[upper_mask],
            ),
            axis=1,
        )

    def _notify_population(
        self,
        values: np.ndarray,
        objs: np.ndarray | None,
        cons: np.ndarray | None,
        iteration: int,
    ) -> None:
        if self._has_callbacks:
            vars_int, vars_float = self._split(values)
            self._notify_callbacks(
                OptimizerCallbackData(
                    event="iteration",
                    vars_int=vars_int,
                    vars_float=vars_float,
                    objs=objs,
                    cons=cons,
                    iteration=iteration,
                    n_evaluations=self.n_evaluations,
                )
            )

    def _finish(
        self, values: np.ndarray | None = None, success: bool = True
    ) -> SingleObjOptResults:
        selected = self._best_values if values is None else values
        if selected is None:
            selected = self._initial
            success = False
        vars_int, vars_float = self._split(selected)
        problem_results, objs, cons = self.problem.finalize_individual(
            vars_int[0], vars_float[0], verbosity=0
        )
        success = bool(
            success
            and np.all(np.isfinite(objs))
            and np.all(np.isfinite(cons))
            and np.all(self.problem.check_constraints_individual(cons, verbosity=0))
        )
        result = SingleObjOptResults(
            self.problem,
            success,
            vars_int[0],
            vars_float[0],
            objs,
            cons,
            problem_results,
        )
        return self._finalize_callbacks(result)
