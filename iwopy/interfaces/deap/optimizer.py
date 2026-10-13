import random
from numbers import Integral
from typing import Any

import numpy as np

from iwopy.core import OptimizerCallback, Problem, SingleObjOptResults
from iwopy.interfaces._population import _PopulationOptimizer
from iwopy.utils import import_module


class _Individual(list[float]):
    def __init__(self, values: np.ndarray, fitness_type: Any) -> None:
        super().__init__(values.tolist())
        self.fitness = fitness_type()


class Optimizer_deap(_PopulationOptimizer):
    """Bounded single-objective DEAP evolution with selectable batch evaluation.

    DEAP supplies variation, bounded operators, cloning, and tournament selection.
    Fitness compares constraint violation before objective loss. Mixed integer
    genes are rounded after variation. No global DEAP creator classes are registered.
    """

    algorithms = ("eaSimple", "eaMuPlusLambda", "eaMuCommaLambda")

    def __init__(
        self,
        problem: Problem,
        optimizer: str = "eaMuPlusLambda",
        *,
        max_evaluations: int | None = 100,
        max_iterations: int | None = None,
        pop_size: int = 20,
        seed: int | None = None,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        vectorize: bool = True,
        name: str = "Optimizer_deap",
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with finite, strictly ordered bounds.
        optimizer
            One of :attr:`algorithms`; selects the DEAP variation/replacement scheme.
        cxpb, mutpb
            Crossover and mutation probabilities in the closed interval zero
            to one; their sum cannot exceed one for mu/lambda schemes.
        pop_size
            Population/offspring size, at least two.
        seed
            Fresh-solve seed. Python's global random state is restored even
            after errors; run serially because native operators use it.
        max_evaluations
            Positive candidate evaluation limit, including initialization and
            reproduction. The final offspring population may be smaller. None
            selects the adapter default of 100.
        max_iterations
            Generation limit including the initial population batch. None
            leaves generations limited only by max_evaluations. Zero still
            evaluates the initial population.
        name
            Optimizer name.
        vectorize
            Evaluate batches with ``evaluate_population`` by default. If false,
            call ``evaluate_individual`` for each candidate in offspring order,
            retaining batch sizes, callbacks, candidate budgets, and selection.

        Raises
        ------
        ValueError
            If vectorize is not a boolean or an iteration/evaluation limit is invalid.
        """
        super().__init__(problem, name, vectorize)
        limits = self._limit_kwargs(
            max_iterations, max_evaluations, "max_iterations", "max_evaluations"
        )
        self.algorithm = optimizer
        self.max_evaluations = limits.get("max_evaluations", 100)
        self.max_iterations = limits.get("max_iterations")
        self.pop_size = pop_size
        self.seed = seed
        self.cxpb = cxpb
        self.mutpb = mutpb

    def initialize(self, verbosity: int = 1) -> None:
        """Validate settings and lazily load DEAP's base, tools, and algorithms.

        Parameters
        ----------
        verbosity
            Verbosity level; zero is silent.

        Raises
        ------
        ValueError
            For invalid problems, bounds, tolerances, sizes, probabilities, or selectors.
        ModuleNotFoundError
            If DEAP is unavailable; install the ``deap`` extra.
        """
        self._validate_problem()
        if self.algorithm not in self.algorithms:
            raise ValueError(
                f"Unknown DEAP algorithm '{self.algorithm}'; choose {self.algorithms}"
            )
        for label, value in (
            ("max_evaluations", self.max_evaluations),
            ("pop_size", self.pop_size),
        ):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.pop_size < 2:
            raise ValueError("DEAP pop_size must be at least two")
        if any(
            not np.isfinite(value) or not 0 <= value <= 1
            for value in (self.cxpb, self.mutpb)
        ):
            raise ValueError(
                "DEAP variation probabilities must be finite and between zero and one"
            )
        if self.algorithm != "eaSimple" and self.cxpb + self.mutpb > 1:
            raise ValueError(
                "DEAP mu/lambda crossover and mutation probabilities must sum to at most one"
            )
        self._base = import_module("deap.base", hint="uv sync --extra deap")
        self._tools = import_module("deap.tools", hint="uv sync --extra deap")
        self._algorithms = import_module("deap.algorithms", hint="uv sync --extra deap")
        super().initialize(verbosity)

    def _toolbox(self) -> Any:
        toolbox = self._base.Toolbox()
        toolbox.register(
            "mate",
            self._tools.cxSimulatedBinaryBounded,
            eta=20.0,
            low=self._lower.tolist(),
            up=self._upper.tolist(),
        )
        toolbox.register(
            "mutate",
            self._tools.mutPolynomialBounded,
            eta=20.0,
            low=self._lower.tolist(),
            up=self._upper.tolist(),
            indpb=1.0 / len(self._lower),
        )
        toolbox.register("select", self._tools.selTournament, tournsize=3)
        return toolbox

    def _evaluate_individuals(self, population: list[Any], iteration: int) -> None:
        values = np.clip(np.asarray(population, dtype=float), self._lower, self._upper)
        values[:, self.problem.n_vars_float :] = np.rint(
            values[:, self.problem.n_vars_float :]
        )
        objs, cons = self._evaluate(values)
        violations = np.maximum(self._residuals(cons), 0).sum(axis=1)
        feasible = np.all(
            self.problem.check_constraints_population(cons, verbosity=0), axis=1
        )
        for index, individual in enumerate(population):
            individual[:] = values[index].tolist()
            individual.fitness.values = (
                0.0 if feasible[index] else float(violations[index]),
                float(self._losses(objs)[index]),
            )
        self._notify_population(values, objs, cons, iteration)

    def _run(self) -> SingleObjOptResults:
        rng = np.random.default_rng(self.seed)
        values = rng.uniform(
            self._lower,
            self._upper,
            size=(min(self.pop_size, self.max_evaluations), len(self._lower)),
        )
        values[0] = self._initial
        fitness_type = type(
            "_Fitness", (self._base.Fitness,), {"weights": (-1.0, -1.0)}
        )
        population = [_Individual(row, fitness_type) for row in values]
        toolbox = self._toolbox()
        iteration = 1
        self._evaluate_individuals(population, iteration)
        while self.n_evaluations < self.max_evaluations and (
            self.max_iterations is None or iteration < self.max_iterations
        ):
            size = min(self.pop_size, self.max_evaluations - self.n_evaluations)
            if self.algorithm == "eaSimple":
                offspring = self._algorithms.varAnd(
                    toolbox.select(population, size), toolbox, self.cxpb, self.mutpb
                )
            else:
                offspring = self._algorithms.varOr(
                    population, toolbox, size, self.cxpb, self.mutpb
                )
            iteration += 1
            self._evaluate_individuals(offspring, iteration)
            if self.algorithm == "eaMuPlusLambda":
                population = toolbox.select(population + offspring, self.pop_size)
            elif self.algorithm == "eaMuCommaLambda":
                population = toolbox.select(
                    offspring, min(self.pop_size, len(offspring))
                )
            else:
                population = offspring
        return self._finish()

    def solve(
        self, verbosity: int = 1, callbacks: list[OptimizerCallback] | None = None
    ) -> SingleObjOptResults:
        """Run fresh DEAP evolution and restore the caller's Python random state.

        Parameters
        ----------
        verbosity
            iwopy verbosity level; native logbooks are not printed.
        callbacks
            Ordered evaluated-population snapshots, starting with the initial
            population at iteration one. Counters exclude finalization.

        Returns
        -------
        SingleObjOptResults
            Best feasible evaluated candidate, or least-violating unsuccessful
            candidate. Success means feasibility rather than convergence.

        Raises
        ------
        ValueError
            For invalid or nonfinite candidate, objective, or constraint populations.
        """
        super().solve(verbosity, callbacks)
        self._reset()
        previous_state = random.getstate()
        try:
            random.seed(self.seed)
            return self._run()
        finally:
            random.setstate(previous_state)
