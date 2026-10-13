from typing import Any

import numpy as np
from scipy.optimize import NonlinearConstraint, differential_evolution

from iwopy.core import OptimizerCallback, Problem, SingleObjOptResults
from iwopy.interfaces._population import _PopulationOptimizer


class Optimizer_scipy_de(_PopulationOptimizer):
    """Vectorized SciPy differential evolution for bounded single-objective problems.

    Floating, integer, and mixed variables are supported. Objective and
    constraint calls share the last population evaluation. Scalar-shaped
    backend calls are evaluated as singleton populations by default.
    Set ``vectorize=False`` to evaluate iwopy candidates individually instead.
    """

    def __init__(
        self,
        problem: Problem,
        *,
        max_iterations: int | None = 1000,
        max_evaluations: int | None = None,
        population_multiplier: int = 15,
        seed: int | None = None,
        tol: float = 0.01,
        atol: float = 0.0,
        vectorize: bool = True,
        name: str = "Optimizer_scipy_de",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with finite, strictly ordered
            bounds. Constraint bounds include each function's tolerance.
        max_iterations
            Maximum number of generations after initialization. None uses the
            native default; zero still evaluates the initial population.
        max_evaluations
            Unsupported by native differential evolution; must be None.
        population_multiplier
            Native population multiplier per free variable, not an absolute size.
        seed
            Native random seed; None keeps SciPy's default. A native rng can
            instead be supplied through kwargs.
        tol, atol
            Relative and absolute native population-convergence tolerances.
        name
            Optimizer name.
        vectorize
            Evaluate each iwopy batch with ``evaluate_population`` by default.
            If false, call ``evaluate_individual`` for each candidate in order.
            SciPy's native deferred batch mode remains enabled; this flag does
            not change generation sizes, caching, callbacks, or candidate counts.
        kwargs
            Flat native differential-evolution options, such as strategy,
            mutation and recombination. Bounds, integrality, constraints,
            callbacks, workers, updating, polishing, x0 and display are managed.

        Raises
        ------
        ValueError
            If vectorize is not a boolean, an iteration limit is invalid, or
            an unsupported evaluation limit is requested.
        TypeError
            If a superseded or unsupported shared keyword is supplied.
        """
        super().__init__(problem, name, vectorize)
        self._validate_kwargs(kwargs)
        self._native_pars = dict(
            kwargs, popsize=population_multiplier, tol=tol, atol=atol
        )
        self._native_pars.update(
            self._limit_kwargs(max_iterations, max_evaluations, "maxiter")
        )
        if seed is not None:
            self._native_pars["seed"] = seed
        self.scipy_results: Any = None
        self._cached_rows: dict[tuple[float, ...], tuple[np.ndarray, np.ndarray]] = {}
        self._evaluation_error: Exception | None = None
        self._iteration = 0

    def initialize(self, verbosity: int = 1) -> None:
        """Validate representation and native options before initialization.

        Parameters
        ----------
        verbosity
            Verbosity level; zero is silent.

        Raises
        ------
        ValueError
            For an uninitialized or unsupported problem, invalid bounds or
            tolerances, or internally managed native options.
        """
        self._validate_problem()
        managed = {
            "func",
            "args",
            "bounds",
            "integrality",
            "constraints",
            "callback",
            "vectorized",
            "workers",
            "updating",
            "polish",
            "x0",
            "disp",
        }
        conflicts = managed.intersection(self._native_pars)
        if conflicts:
            raise ValueError(
                f"Internally managed native options: {', '.join(sorted(conflicts))}"
            )
        super().initialize(verbosity)

    def _values(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        population = np.atleast_2d(values.T) if values.ndim == 2 else values[None, :]
        if not len(population):
            return np.empty((0, 1)), np.empty((0, self.problem.n_constraints))
        keys = [tuple(row) for row in population]
        if all(key in self._cached_rows for key in keys):
            return np.stack([self._cached_rows[key][0] for key in keys]), np.stack(
                [self._cached_rows[key][1] for key in keys]
            )
        objs, cons = self._evaluate(population)
        self._cached_rows = {
            key: (objs[index], cons[index]) for index, key in enumerate(keys)
        }
        return objs, cons

    def _objective(self, values: np.ndarray) -> np.ndarray | float:
        try:
            objs, _ = self._values(values)
        except (ValueError, TypeError) as error:
            self._evaluation_error = error
            raise
        losses = self._losses(objs)
        return losses if values.ndim == 2 else float(losses[0])

    def _constraints(self, values: np.ndarray) -> np.ndarray:
        _, cons = self._values(values)
        return cons.T if values.ndim == 2 else cons[0]

    def _callback(self, intermediate_result: Any) -> None:
        self._iteration += 1
        energies = intermediate_result.population_energies
        objs = (
            energies[:, None] * (-1 if self.problem.maximize_objs[0] else 1)
            if np.all(np.isfinite(energies))
            else None
        )
        self._notify_population(
            intermediate_result.population, objs, None, self._iteration
        )

    def solve(
        self, verbosity: int = 1, callbacks: list[OptimizerCallback] | None = None
    ) -> SingleObjOptResults:
        """Run deferred, single-worker vectorized DE without scalar polishing.

        Parameters
        ----------
        verbosity
            Native progress display level; zero is silent.
        callbacks
            Ordered callbacks receiving current survivor populations after each
            generation. Constraint snapshots are unavailable. Objective snapshots
            are unavailable when native infeasible energies are infinite.

        Returns
        -------
        SingleObjOptResults
            Finalized native selection in original objective direction. Success
            requires native convergence and core constraint feasibility.

        Raises
        ------
        ValueError
            If a candidate or evaluated population is invalid or nonfinite.
        """
        super().solve(verbosity, callbacks)
        self._reset()
        self._cached_rows.clear()
        self._evaluation_error = None
        self._iteration = 0
        self.scipy_results = None
        constraints = (
            ()
            if not self.problem.n_constraints
            else NonlinearConstraint(
                self._constraints, self._cons_lower, self._cons_upper
            )
        )
        try:
            self.scipy_results = differential_evolution(
                self._objective,
                list(zip(self._lower, self._upper)),
                constraints=constraints,
                integrality=np.arange(len(self._lower)) >= self.problem.n_vars_float,
                vectorized=True,
                workers=1,
                updating="deferred",
                polish=False,
                x0=self._initial,
                callback=self._callback,
                disp=bool(verbosity),
                **self._native_pars,
            )
        except RuntimeError as error:
            if (
                self._evaluation_error is not None
                and error.__cause__ is self._evaluation_error
            ):
                raise self._evaluation_error from None
            raise
        return self._finish(self.scipy_results.x, bool(self.scipy_results.success))
