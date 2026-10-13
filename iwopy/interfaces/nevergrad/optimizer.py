from numbers import Integral
from typing import Any

import numpy as np

from iwopy.core import OptimizerCallback, Problem, SingleObjOptResults
from iwopy.interfaces._population import _PopulationOptimizer
from iwopy.utils import import_module


class Optimizer_nevergrad(_PopulationOptimizer):
    """Nevergrad ask/tell optimization with selectable iwopy batch evaluation.

    Bounded floating, integer, and mixed single-objective problems are supported.
    Constraints are passed as tolerance-adjusted violations; no scalar rejection
    constraints are registered. Selected results are always evaluated candidates.
    """

    algorithms = (
        "NGOpt",
        "CMA",
        "DE",
        "TwoPointsDE",
        "PSO",
        "OnePlusOne",
        "RandomSearch",
        "TBPSA",
    )

    def __init__(
        self,
        problem: Problem,
        optimizer: str = "NGOpt",
        *,
        budget: int = 100,
        batch_size: int = 8,
        seed: int | None = None,
        vectorize: bool = True,
        name: str = "Optimizer_nevergrad",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with finite, strictly ordered bounds.
        optimizer
            One of :attr:`algorithms`; only these batching-safe selectors are exposed.
        kwargs
            Flat native constructor options. Parameterization, budget, and workers
            are internally managed.
        batch_size
            Positive outstanding-ask/native-worker count. No threads are used.
        seed
            Seed for a fresh parameterization each solve.
        budget
            Positive candidate evaluation limit including the final partial batch.
        name
            Optimizer name.
        vectorize
            Evaluate batches with ``evaluate_population`` by default. If false,
            call ``evaluate_individual`` for each candidate in ask order, retaining
            batch sizes, callbacks, candidate budgets, and selection semantics.

        Raises
        ------
        ValueError
            If vectorize is not a boolean.
        """
        super().__init__(problem, name, vectorize)
        self._validate_kwargs(kwargs)
        self.algorithm = optimizer
        self.budget = budget
        self.batch_size = batch_size
        self.seed = seed
        self._optimizer_kwargs = dict(kwargs)
        self.optimizer: Any = None

    def initialize(self, verbosity: int = 1) -> None:
        """Validate the problem and options, then lazily load Nevergrad.

        Parameters
        ----------
        verbosity
            Verbosity level; zero is silent.

        Raises
        ------
        ValueError
            For invalid representations, bounds, budgets, tolerances, selectors,
            or internally managed constructor options.
        ModuleNotFoundError
            If Nevergrad is unavailable; install the ``nevergrad`` extra.
        """
        self._validate_problem()
        if self.algorithm not in self.algorithms:
            raise ValueError(
                f"Unknown Nevergrad algorithm '{self.algorithm}'; choose {self.algorithms}"
            )
        for label, value in (("budget", self.budget), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        conflicts = {"parametrization", "budget", "num_workers"}.intersection(
            self._optimizer_kwargs
        )
        if conflicts:
            raise ValueError(
                f"Internally managed native options: {', '.join(sorted(conflicts))}"
            )
        self._ng = import_module("nevergrad", hint="uv sync --extra nevergrad")
        super().initialize(verbosity)

    def _new_optimizer(self) -> Any:
        parameters = []
        for index, initial in enumerate(self._initial):
            parameter = self._ng.p.Scalar(init=float(initial)).set_bounds(
                float(self._lower[index]), float(self._upper[index])
            )
            if index >= self.problem.n_vars_float:
                parameter.set_integer_casting()
            parameters.append(parameter)
        parametrization = self._ng.p.Tuple(*parameters)
        if self.seed is not None:
            parametrization.random_state.seed(self.seed)
        optimizer = self._ng.optimizers.registry[self.algorithm](
            parametrization=parametrization,
            budget=self.budget,
            num_workers=self.batch_size,
            **self._optimizer_kwargs,
        )
        if optimizer.no_parallelization and self.batch_size > 1:
            raise ValueError(
                f"Nevergrad algorithm '{self.algorithm}' does not allow batched asks"
            )
        return optimizer

    def solve(
        self, verbosity: int = 1, callbacks: list[OptimizerCallback] | None = None
    ) -> SingleObjOptResults:
        """Run fresh ask/evaluate/tell batches and finalize the best evaluated point.

        Parameters
        ----------
        verbosity
            iwopy verbosity level; no native progress display is configured.
        callbacks
            Ordered callbacks receiving each completed evaluated batch in ask order.
            Iterations start at one; evaluation counts exclude finalization.

        Returns
        -------
        SingleObjOptResults
            Best feasible evaluated candidate, or least-violating evaluated candidate
            with unsuccessful status. Success means feasibility, not convergence.

        Raises
        ------
        ValueError
            If the backend forbids outstanding asks or produces invalid candidates,
            or if evaluated objectives or constraints are nonfinite or malformed.
        """
        super().solve(verbosity, callbacks)
        self._reset()
        self.optimizer = self._new_optimizer()
        iteration = 0
        while self.n_evaluations < self.budget:
            candidates = [
                self.optimizer.ask()
                for _ in range(min(self.batch_size, self.budget - self.n_evaluations))
            ]
            values = np.asarray(
                [candidate.value for candidate in candidates], dtype=float
            )
            objs, cons = self._evaluate(values)
            residuals = np.maximum(self._residuals(cons), 0)
            for index, candidate in enumerate(candidates):
                self.optimizer.tell(
                    candidate,
                    float(self._losses(objs)[index]),
                    constraint_violation=residuals[index]
                    if residuals.shape[1]
                    else None,
                )
            iteration += 1
            self._notify_population(values, objs, cons, iteration)
        return self._finish()
