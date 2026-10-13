from numbers import Integral
from typing import Any

import numpy as np

from iwopy.core import OptimizerCallback, Problem, SingleObjOptResults
from iwopy.interfaces._population import _PopulationOptimizer
from iwopy.utils import import_module


class Optimizer_optuna(_PopulationOptimizer):
    """Optuna ask/tell studies with selectable ordered iwopy batch evaluation.

    Bounded floating, integer, and mixed single-objective problems are supported.
    Every solve uses a fresh in-memory study. TPE receives constraint residuals;
    random sampling does not adapt to constraints but retains feasible selection.
    """

    samplers = ("TPESampler", "RandomSampler")

    def __init__(
        self,
        problem: Problem,
        optimizer: str = "TPESampler",
        *,
        budget: int = 100,
        batch_size: int = 8,
        seed: int | None = None,
        vectorize: bool = True,
        name: str = "Optimizer_optuna",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with finite, strictly ordered bounds.
        optimizer
            One of :attr:`samplers`. TPE uses constant-liar pending-trial handling
            by default; native sampler options can override that choice.
        kwargs
            Flat native sampler options. Seed is managed; deprecated native
            ``constraints_func`` callbacks cannot be supplied.
        batch_size
            Positive number of pending trials.
        seed
            Seed for a fresh sampler on each solve.
        budget
            Positive completed-trial limit including the final partial batch.
        name
            Optimizer and study name.
        vectorize
            Evaluate batches with ``evaluate_population`` by default. If false,
            call ``evaluate_individual`` for each trial in ask order, retaining
            batch sizes, callbacks, candidate budgets, and selection semantics.

        Raises
        ------
        ValueError
            If vectorize is not a boolean.
        """
        super().__init__(problem, name, vectorize)
        self._validate_kwargs(kwargs)
        self.sampler = optimizer
        self.budget = budget
        self.batch_size = batch_size
        self.seed = seed
        self._sampler_kwargs = dict(kwargs)
        self.study: Any = None

    def initialize(self, verbosity: int = 1) -> None:
        """Validate inputs and lazily load Optuna.

        Parameters
        ----------
        verbosity
            iwopy verbosity level; zero is silent for iwopy. Native Optuna logging
            follows the application's Optuna logger configuration.

        Raises
        ------
        ValueError
            For invalid problems, budgets, bounds, tolerances, samplers, or managed options.
        ModuleNotFoundError
            If Optuna is unavailable; install the ``optuna`` extra.
        """
        self._validate_problem()
        if self.sampler not in self.samplers:
            raise ValueError(
                f"Unknown Optuna sampler '{self.sampler}'; choose {self.samplers}"
            )
        for label, value in (("budget", self.budget), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        conflicts = {"seed", "constraints_func"}.intersection(self._sampler_kwargs)
        if conflicts:
            raise ValueError(
                f"Internally managed native options: {', '.join(sorted(conflicts))}"
            )
        self._optuna = import_module("optuna", hint="uv sync --extra optuna")
        super().initialize(verbosity)

    def _new_study(self) -> Any:
        options = dict(self._sampler_kwargs)
        if self.sampler == "TPESampler":
            options.setdefault("constant_liar", True)
        sampler = getattr(self._optuna.samplers, self.sampler)(
            seed=self.seed, **options
        )
        return self._optuna.create_study(
            sampler=sampler,
            direction="maximize" if self.problem.maximize_objs[0] else "minimize",
            study_name=self.name,
        )

    def _distributions(self) -> dict[str, Any]:
        distributions = {}
        for index, (lower, upper) in enumerate(zip(self._lower, self._upper)):
            if index < self.problem.n_vars_float:
                distributions[f"float_{index}"] = (
                    self._optuna.distributions.FloatDistribution(
                        float(lower), float(upper)
                    )
                )
            else:
                distributions[f"int_{index - self.problem.n_vars_float}"] = (
                    self._optuna.distributions.IntDistribution(int(lower), int(upper))
                )
        return distributions

    def _batch(self, trials: list[Any], names: list[str], iteration: int) -> None:
        values = np.asarray(
            [[trial.params[name] for name in names] for trial in trials], dtype=float
        )
        try:
            objs, cons = self._evaluate(values)
        except Exception:
            for trial in trials:
                self.study.tell(trial, state=self._optuna.trial.TrialState.FAIL)
            raise
        residuals = self._residuals(cons)
        for index, trial in enumerate(trials):
            for constraint_index, residual in enumerate(residuals[index]):
                trial.set_constraint(f"constraint_{constraint_index}", float(residual))
            self.study.tell(trial, float(objs[index, 0]))
        self._notify_population(values, objs, cons, iteration)

    def solve(
        self, verbosity: int = 1, callbacks: list[OptimizerCallback] | None = None
    ) -> SingleObjOptResults:
        """Run a fresh in-memory study using population ask/evaluate/tell batches.

        Parameters
        ----------
        verbosity
            iwopy verbosity level. No application-global logging is configured.
        callbacks
            Ordered callbacks receiving each completed batch in trial order.
            Iterations start at one and counters exclude selected-point finalization.

        Returns
        -------
        SingleObjOptResults
            Best feasible evaluated candidate, or least-violating unsuccessful
            candidate. Success means feasibility, not statistical convergence.

        Raises
        ------
        ValueError
            For invalid or nonfinite candidates, objective outputs, or constraints.
            Trials in a failed population evaluation are marked failed before propagation.
        """
        super().solve(verbosity, callbacks)
        self._reset()
        self.study = self._new_study()
        distributions = self._distributions()
        iteration = 0
        while self.n_evaluations < self.budget:
            trials = [
                self.study.ask(distributions)
                for _ in range(min(self.batch_size, self.budget - self.n_evaluations))
            ]
            iteration += 1
            self._batch(trials, list(distributions), iteration)
        return self._finish()
