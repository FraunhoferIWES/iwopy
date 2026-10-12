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
        algorithm: str = "eaMuPlusLambda",
        budget: int = 100,
        batch_size: int = 20,
        seed: int | None = None,
        cxpb: float = 0.5,
        mutpb: float = 0.2,
        name: str = "Optimizer_deap",
        vectorize: bool = True,
    ) -> None:
        """
        Parameters
        ----------
        problem
            Initialized single-objective problem with finite, strictly ordered bounds.
        algorithm
            One of :attr:`algorithms`; selects the DEAP variation/replacement scheme.
        budget
            Positive candidate evaluation budget, including initialization and
            reproduced offspring. The last offspring population may be smaller.
        batch_size
            Population and regular offspring size, at least two.
        seed
            Optional fresh-solve seed. Python's global random state is restored
            after execution, including exceptions. Run this adapter serially:
            DEAP's operators use Python's global random generator.
        cxpb
            Crossover probability in the closed interval zero to one.
        mutpb
            Mutation probability in the closed interval zero to one. For the
            mu/lambda schemes its sum with ``cxpb`` cannot exceed one.
        name
            Optimizer name.
        vectorize
            Evaluate batches with ``evaluate_population`` by default. If false,
            call ``evaluate_individual`` for each candidate in offspring order,
            retaining batch sizes, callbacks, candidate budgets, and selection.

        Raises
        ------
        ValueError
            If vectorize is not a boolean.
        """
        super().__init__(problem, name, vectorize)
        self.algorithm = algorithm
        self.budget = budget
        self.batch_size = batch_size
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
        for label, value in (("budget", self.budget), ("batch_size", self.batch_size)):
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if self.batch_size < 2:
            raise ValueError("DEAP batch_size must be at least two")
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
            size=(min(self.batch_size, self.budget), len(self._lower)),
        )
        values[0] = self._initial
        fitness_type = type(
            "_Fitness", (self._base.Fitness,), {"weights": (-1.0, -1.0)}
        )
        population = [_Individual(row, fitness_type) for row in values]
        toolbox = self._toolbox()
        iteration = 1
        self._evaluate_individuals(population, iteration)
        while self.n_evaluations < self.budget:
            size = min(self.batch_size, self.budget - self.n_evaluations)
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
                population = toolbox.select(population + offspring, self.batch_size)
            elif self.algorithm == "eaMuCommaLambda":
                population = toolbox.select(
                    offspring, min(self.batch_size, len(offspring))
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
