import numpy as np

from iwopy.core import Optimizer, OptimizerCallback, OptimizerCallbackData
from iwopy.utils import suppress_stdout

from . import imports
from .algos import AlgoFactory
from .problem import UDP


class _PygmoCallbackSink:
    """Copy-stable bridge from PyGMO fitness calls to iwopy callbacks."""

    def __init__(self, optimizer) -> None:
        self.optimizer = optimizer
        self.n_evaluations = 0

    def __deepcopy__(self, memo):
        return self

    def notify(self, vars_int, vars_float, objs, cons):
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

    Attributes
    ----------
    problem_pars: dict
        Parameters for the problem
    algo_pars: dict
        Parameters for the alorithm
    setup_pars: dict
        Parameters for the calculation setup
    udp: iwopy.interfaces.imports.pygmo.UDA
        The pygmo problem
    algo: imports.pygmo.algo
        The pygmo algorithm

    :group: interfaces.pygmo

    """

    def __init__(self, problem, problem_pars=None, algo_pars=None, setup_pars=None):
        """
        Constructor

        Parameters
        ----------
        problem: iwopy.Problem
            The problem to optimize
        problem_pars: dict
            Parameters for the problem
        algo_pars: dict
            Parameters for the alorithm
        setup_pars: dict
            Parameters for the calculation setup

        """
        if setup_pars is None:
            setup_pars = {}
        if algo_pars is None:
            algo_pars = {}
        if problem_pars is None:
            problem_pars = {}
        super().__init__(problem)

        imports.load()

        self.problem_pars = problem_pars
        self.algo_pars = algo_pars
        self.setup_pars = setup_pars

        self.udp = None
        self.algo = None

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """

        if "callback_mode" in self.setup_pars:
            raise ValueError("PyGMO callback_mode is not supported")

        # create pygmo problem:
        pop = self.problem_pars.get("pop", False)
        self.udp = UDP(self.problem, **self.problem_pars)

        # create algorithm:
        self.algo = AlgoFactory.new(pop=pop, **self.algo_pars)

        # create population:
        psize = self.setup_pars.get("pop_size", 1)
        pseed = self.setup_pars.get("seed", None)
        pnrfi = self.setup_pars.get("norandom_first", psize == 1)
        self.pop = imports.pygmo.population(self.udp, size=psize, seed=pseed)
        self.pop.problem.c_tol = [
            self.setup_pars.get("c_tol", 1e-4)
        ] * self.pop.problem.get_nc()

        # memorize verbosity level:
        self.verbosity = self.setup_pars.get("verbosity", 1)

        # set first indiviual to initial values:
        if pnrfi:
            x = np.zeros(self.udp.n_vars_all)

            if self.problem.n_vars_float:
                x[: self.problem.n_vars_float] = self.problem.initial_values_float()
            if self.problem.n_vars_int:
                x[self.problem.n_vars_float :] = self.problem.initial_values_int()

            # xf = x[: self.problem.n_vars_float]
            # xi = x[self.problem.n_vars_float :].astype(np.int64)

            self.udp._active = True
            self.pop.set_x(0, x)

        super().initialize(verbosity)

    def print_info(self):
        """
        Print solver info, called before solving
        """
        super().print_info()
        if self.algo is not None:
            print()
            print(self.algo)

    def _validate_callbacks(self, callbacks: list[OptimizerCallback]) -> None:
        """Reject callbacks when PyGMO cannot expose exact progress states."""
        if self.algo_pars.get("type") == "ipopt" and callbacks:
            raise NotImplementedError(
                "PyGMO IPOPT does not expose exact live iteration callbacks"
            )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ):
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
        results: iwopy.SingleObjOptResults
            The optimization results object

        """

        super().solve(verbosity, callbacks)
        population_udp = None
        if self._has_callbacks:
            population_udp = self.pop.problem.extract(UDP)
            population_udp.callback_sink = _PygmoCallbackSink(self)

        # try pygmo silencing:
        if self.algo.has_set_verbosity():
            self.algo.set_verbosity(verbosity)

        # general silencing for Python prints:
        silent = verbosity <= 0
        try:
            with suppress_stdout(silent):
                # Run solver:
                pop = self.algo.evolve(self.pop)
        finally:
            if population_udp is not None:
                population_udp.callback_sink = None

        results = self.udp.finalize(pop, verbosity)
        return self._finalize_callbacks(results)
