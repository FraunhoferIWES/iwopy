from __future__ import annotations

from typing import Any

import numpy as np

from iwopy.core import (
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)
from iwopy.utils import suppress_stdout

from . import imports
from .algos import AlgoFactory
from .problem import UDP


class _PygmoCallbackSink:
    """Copy-stable bridge from PyGMO fitness calls to iwopy callbacks."""

    def __init__(self, optimizer: Optimizer_pygmo) -> None:
        self.optimizer = optimizer
        self.n_evaluations = 0

    def __deepcopy__(self, memo: dict[int, Any]) -> _PygmoCallbackSink:
        return self

    def notify(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        objs: np.ndarray,
        cons: np.ndarray,
    ) -> None:
        """Dispatch one scalar or batch evaluation."""
        n_pop = 1 if np.asarray(vars_float).ndim == 1 else len(vars_float)
        self.n_evaluations += n_pop
        self.optimizer._notify_callbacks(
            OptimizerCallbackData(
                event="evaluation",
                iteration=None,
                n_evaluations=self.n_evaluations,
                vars_int=vars_int,
                vars_float=vars_float,
                objs=objs,
                cons=cons,
            )
        )


class Optimizer_pygmo(Optimizer):
    """
    Interface to the pygmo optimizers
    for serial runs.
    """

    def __init__(
        self,
        problem: Problem,
        optimizer: str = "de",
        *,
        pop_size: int = 1,
        seed: int | None = None,
        vectorize: bool = False,
        max_iterations: int | None = None,
        max_evaluations: int | None = None,
        norandom_first: bool | None = None,
        c_tol: float = 1e-4,
        problem_verbosity: int = 0,
        algorithm_verbosity: int = 1,
        nlopt_optimizer: str | None = None,
        name: str = "Optimizer_pygmo",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The problem to optimize
        optimizer
            Native driver name, such as de, ipopt, or nlopt.
        pop_size
            Absolute population size. Local solvers operate on one selected member.
        seed
            Seed for both the population and native drivers supporting seed.
        vectorize
            Enable batch fitness where supported by the selected driver.
        max_iterations
            IPOPT iteration or evolutionary-driver generation limit. Unsupported
            for NLopt and compass_search. None preserves the native default.
        max_evaluations
            Native driver fitness-call limit for NLopt or compass_search,
            excluding initial population construction. Unsupported for other
            drivers. None preserves the native default.
        norandom_first
            Replace the first individual with problem initial values; None
            enables this when pop_size is one.
        c_tol
            Native population constraint tolerance.
        problem_verbosity, algorithm_verbosity
            UDP diagnostics and native algorithm verbosity levels.
        nlopt_optimizer
            Native NLopt solver name when optimizer is nlopt. The standalone
            factory supplies it automatically for family nlopt.
        name
            Optimizer name.
        kwargs
            Flat native driver options, such as print_level and ftol_rel.

        Raises
        ------
        ValueError
            If limits are invalid or unsupported by the selected driver,
            vectorize is not a boolean, or driver selection is inconsistent.
        TypeError
            If a superseded or unsupported shared keyword is supplied.
        """
        super().__init__(problem, name)
        self._validate_kwargs(kwargs)
        if "type" in kwargs:
            raise ValueError("pygmo: select the algorithm with optimizer")
        self.optimizer = optimizer
        self.pop_size = pop_size
        self.seed = seed
        self.vectorize = self._validate_vectorize(vectorize)
        self.norandom_first = (
            pop_size == 1 if norandom_first is None else norandom_first
        )
        self.c_tol = c_tol
        self.problem_verbosity = problem_verbosity
        self.verbosity = algorithm_verbosity
        self._algorithm_kwargs = dict(kwargs)
        iteration_key = (
            "max_iter"
            if optimizer == "ipopt"
            else (None if optimizer in {"nlopt", "compass_search"} else "gen")
        )
        evaluation_key = {"nlopt": "maxeval", "compass_search": "max_fevals"}.get(
            optimizer
        )
        self._algorithm_kwargs.update(
            self._limit_kwargs(
                max_iterations, max_evaluations, iteration_key, evaluation_key
            )
        )
        if optimizer == "nlopt":
            if nlopt_optimizer is None:
                raise ValueError("PyGMO NLopt requires nlopt_optimizer")
            self._algorithm_kwargs["optimizer"] = nlopt_optimizer
        elif nlopt_optimizer is not None:
            raise ValueError("nlopt_optimizer applies only to the NLopt driver")
        if seed is not None and optimizer not in {"nlopt", "ipopt", "compass_search"}:
            self._algorithm_kwargs["seed"] = seed

        imports.load()

        self.udp: UDP | None = None
        self.algo: Any | None = None
        self.pop: Any | None = None

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """

        if "callback_mode" in self._algorithm_kwargs:
            raise ValueError("PyGMO callback_mode is not supported")

        # create pygmo problem:
        self.udp = UDP(
            self.problem, pop=self.vectorize, verbosity=self.problem_verbosity
        )

        # create algorithm:
        self.algo = AlgoFactory.new(
            type=self.optimizer, pop=self.vectorize, **self._algorithm_kwargs
        )

        # create population:
        pygmo = imports.pygmo
        assert pygmo is not None
        self.pop = pygmo.population(self.udp, size=self.pop_size, seed=self.seed)
        self.pop.problem.c_tol = [self.c_tol] * self.pop.problem.get_nc()

        # set first indiviual to initial values:
        if self.norandom_first:
            x = np.zeros(self.udp.n_vars_all)

            if self.problem.n_vars_float:
                initial_values_float = self.problem.initial_values_float()
                assert initial_values_float is not None
                x[: self.problem.n_vars_float] = initial_values_float
            if self.problem.n_vars_int:
                x[self.problem.n_vars_float :] = self.problem.initial_values_int()

            # xf = x[: self.problem.n_vars_float]
            # xi = x[self.problem.n_vars_float :].astype(np.int64)

            self.udp._active = True
            self.pop.set_x(0, x)

        super().initialize(verbosity)

    def print_info(self) -> None:
        """Print solver info, called before solving"""
        super().print_info()
        if self.algo is not None:
            print()
            print(self.algo)

    def _validate_callbacks(self, callbacks: list[OptimizerCallback]) -> None:
        """Reject callbacks when PyGMO cannot expose exact progress states."""
        if self.optimizer == "ipopt" and callbacks:
            raise NotImplementedError(
                "PyGMO IPOPT does not expose exact live iteration callbacks"
            )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults:
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for intermediate optimization states

        Returns
        -------
        results
            The optimization results object
        """

        super().solve(verbosity, callbacks)
        population = self.pop
        algorithm = self.algo
        udp = self.udp
        assert population is not None
        assert algorithm is not None
        assert udp is not None
        population_udp: UDP | None = None
        if self._has_callbacks:
            population_udp = population.problem.extract(UDP)
            population_udp.callback_sink = _PygmoCallbackSink(self)

        # try pygmo silencing:
        if algorithm.has_set_verbosity():
            algorithm.set_verbosity(verbosity)

        # general silencing for Python prints:
        silent = verbosity <= 0
        try:
            with suppress_stdout(silent):
                # Run solver:
                pop = algorithm.evolve(population)
        finally:
            if population_udp is not None:
                population_udp.callback_sink = None

        results = udp.finalize(pop, verbosity)
        return self._finalize_callbacks(results)
