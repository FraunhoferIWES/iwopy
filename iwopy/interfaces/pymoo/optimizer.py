from __future__ import annotations

from typing import Any

import numpy as np

from iwopy.core import (
    MultiObjOptResults,
    Optimizer,
    OptimizerCallback,
    OptimizerCallbackData,
    Problem,
    SingleObjOptResults,
)

from . import imports
from .factory import Factory
from .problem import MultiObjProblemTemplate, SingleObjProblemTemplate


class _PymooCallbackTemplate:
    """Template for the internal pymoo-to-iwopy callback adapter."""

    CLASS_NAME = "IwopyCallback"
    CLASS_DOC = "Internal pymoo-to-iwopy callback adapter"

    def __init__(self, optimizer: Optimizer_pymoo) -> None:
        self.optimizer = optimizer

    def __deepcopy__(self, memo: dict[int, Any]) -> _PymooCallbackTemplate:
        return self

    def notify(self, algorithm: Any) -> None:
        self.optimizer._notify_pymoo_callbacks(algorithm)

    @classmethod
    def get_class(cls) -> type[Any]:
        """Creates the class, dynamically derived from pymoo.Callback"""
        imports.load()
        attrb: dict[str, Any] = {
            v: d
            for v, d in cls.__dict__.items()
            if v not in ["get_class", "CLASS_NAME", "CLASS_DOC"]
        }
        initialize_template = cls.__init__

        def __init(self: Any, *args: Any, **kwargs: Any) -> None:
            imports.Callback.__init__(self)
            initialize_template(self, *args, **kwargs)

        attrb["__init__"] = __init
        attrb["__doc__"] = cls.CLASS_DOC
        return type(cls.CLASS_NAME, (imports.Callback,), attrb)


class Optimizer_pymoo(Optimizer):
    """Interface to the pymoo optimization solver."""

    def __init__(
        self,
        problem: Problem,
        optimizer: str = "GA",
        *,
        pop_size: int | None = None,
        seed: int | None = None,
        vectorize: bool = True,
        store_prob_res: bool = False,
        save_history: bool = False,
        termination: Any = "iwopy",
        max_iterations: int | None = None,
        max_evaluations: int | None = None,
        ftol: float | None = None,
        period: int | None = None,
        xtol: float | None = None,
        cvtol: float | None = None,
        name: str = "Optimizer_pymoo",
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The problem to optimize
        optimizer
            Native algorithm name, such as GA, NSGA2, or MixedVariableGA.
        pop_size
            Absolute population size; None uses the algorithm's native default.
        seed
            Native run seed; None keeps native randomization.
        vectorize
            Whether problem evaluation uses populations.
        store_prob_res
            Store current problem results for single-objective evaluation.
        save_history
            Retain native per-generation algorithm history.
        termination
            iwopy or default convergence, iterations or evaluations limits, or
            a native pymoo termination object. Any supplied limits are composed
            with the convergence criterion and stop when either is reached.
        max_iterations, max_evaluations
            Native generation and candidate-evaluation limits. Generation
            boundaries may exceed an evaluation limit. None keeps defaults.
        ftol, period, xtol, cvtol
            Native convergence tolerance/window settings. None keeps defaults.
        name
            Optimizer name.
        kwargs
            Flat native algorithm options, excluding its type selector.

        Raises
        ------
        ValueError
            If limits are invalid, vectorize is not a boolean, or a superseded
            termination/type selector is supplied.
        TypeError
            If a superseded or unsupported shared keyword is supplied.
        """
        super().__init__(problem, name)
        self._validate_kwargs(kwargs)
        if "type" in kwargs:
            raise ValueError("pymoo: select the algorithm with optimizer")
        if isinstance(termination, str) and termination in {"n_gen", "n_eval"}:
            raise ValueError("pymoo: use termination='iterations' or 'evaluations'")
        self.optimizer = optimizer
        self.vectorize = self._validate_vectorize(vectorize)
        self.store_prob_res = store_prob_res
        self.seed = seed
        self.save_history = save_history
        self._algorithm_kwargs = dict(kwargs)
        if pop_size is not None:
            self._algorithm_kwargs["pop_size"] = pop_size
        self._termination_kwargs = {"type": termination}
        self._termination_kwargs.update(
            self._limit_kwargs(
                max_iterations, max_evaluations, "n_max_gen", "n_max_evals"
            )
        )
        for key, value in (
            ("ftol", ftol),
            ("period", period),
            ("xtol", xtol),
            ("cvtol", cvtol),
        ):
            if value is not None:
                self._termination_kwargs[key] = value
        self.pymoo_problem: Any | None = None
        self.algo: Any | None = None
        self.term: Any | None = None
        self.results: Any | None = None

    def print_info(self) -> None:
        """Print solver info, called before solving"""
        super().print_info()

        if len(self._algorithm_kwargs):
            print("\nAlgorithm:")
            print("----------")
            for k, v in self._algorithm_kwargs.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")

        if len(self._termination_kwargs):
            print("\nTermination:")
            print("------------")
            for k, v in self._termination_kwargs.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")
        print()

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """
        if "callback" in self._algorithm_kwargs:
            raise ValueError(
                f"Optimizer '{self.name}': pymoo callback is managed internally."
            )
        if self.problem.n_objectives <= 1:
            self.pymoo_problem = SingleObjProblemTemplate.get_class()(
                self.problem,
                vectorize=self.vectorize,
                store_prob_res=self.store_prob_res,
            )
        else:
            self.pymoo_problem = MultiObjProblemTemplate.get_class()(
                self.problem, vectorize=self.vectorize
            )

        if verbosity:
            print("Initializing", type(self).__name__)

        factory = Factory(self.pymoo_problem, verbosity)
        self.algo = factory.get_algorithm(
            {"type": self.optimizer, **self._algorithm_kwargs}
        )
        self.term = factory.get_termination(self._termination_kwargs)

        super().initialize(verbosity)

    def _callback_variables(self, values: Any) -> tuple[np.ndarray, np.ndarray]:
        """Convert a pymoo population to iwopy variable arrays."""
        n_pop = len(values)
        pymoo_problem = self.pymoo_problem
        assert pymoo_problem is not None
        if pymoo_problem.is_mixed:
            vars_int = np.array(
                [
                    [entry[name] for name in self.problem.var_names_int()]
                    for entry in values
                ],
                dtype=np.int32,
            )
            vars_float = np.array(
                [
                    [entry[name] for name in self.problem.var_names_float()]
                    for entry in values
                ],
                dtype=np.float64,
            )
        elif pymoo_problem.is_intprob:
            vars_int = np.asarray(values, dtype=np.int32)
            vars_float = np.zeros((n_pop, 0), dtype=np.float64)
        else:
            vars_int = np.zeros((n_pop, 0), dtype=np.int32)
            vars_float = np.asarray(values, dtype=np.float64)
        return vars_int, vars_float

    def _callback_constraints(self, values: Any, n_pop: int) -> np.ndarray:
        """Restore iwopy constraint values from pymoo's convention."""
        if not self.problem.n_constraints:
            return np.zeros((n_pop, 0), dtype=np.float64)

        pymoo_problem = self.pymoo_problem
        assert pymoo_problem is not None
        transformed = np.asarray(values, dtype=np.float64)
        constraints = np.empty_like(transformed)
        has_upper = np.isfinite(pymoo_problem._cma)
        has_lower = np.isfinite(pymoo_problem._cmi)
        constraints[:, has_upper] = (
            transformed[:, has_upper] + pymoo_problem._cma[None, has_upper]
        )
        constraints[:, has_lower] = (
            pymoo_problem._cmi[None, has_lower] - transformed[:, has_lower]
        )
        return constraints

    def _notify_pymoo_callbacks(self, algorithm: Any) -> None:
        """Normalize and dispatch one completed pymoo generation."""
        population = algorithm.pop
        vars_int, vars_float = self._callback_variables(population.get("X"))
        objectives = np.asarray(population.get("F"), dtype=np.float64)
        objectives *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
        constraints = self._callback_constraints(population.get("G"), len(objectives))
        self._notify_callbacks(
            OptimizerCallbackData(
                event="iteration",
                iteration=int(algorithm.n_gen),
                n_evaluations=int(algorithm.evaluator.n_eval),
                vars_int=vars_int,
                vars_float=vars_float,
                objs=objectives,
                cons=constraints,
            )
        )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults | MultiObjOptResults:
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for completed pymoo generations

        Returns
        -------
        results
            The optimization results object
        """
        # check problem initialization:
        super().solve(verbosity, callbacks)

        # run pymoo solver:
        run_kwargs = {"seed": self.seed, "save_history": self.save_history}
        if self._has_callbacks:
            run_kwargs["callback"] = _PymooCallbackTemplate.get_class()(self)
        pymoo_problem = self.pymoo_problem
        assert pymoo_problem is not None
        assert self.algo is not None
        assert self.term is not None
        self.results = imports.minimize(
            pymoo_problem,
            algorithm=self.algo,
            termination=self.term,
            verbose=verbosity > 0,
            **run_kwargs,
        )

        results: SingleObjOptResults | MultiObjOptResults = pymoo_problem.finalize(
            self.results
        )
        if isinstance(results, SingleObjOptResults):
            return self._finalize_callbacks(results)
        return self._finalize_callbacks(results)
