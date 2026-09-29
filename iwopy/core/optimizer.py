from abc import ABCMeta, abstractmethod
from typing import TypeVar

import numpy as np

from iwopy.utils import new_instance

from .base import Base
from .optimizer_callback import (
    OptimizerCallback,
    OptimizerCallbackData,
    _OptimizerCallbackDispatcher,
)
from .opt_results import MultiObjOptResults, SingleObjOptResults
from .problem import Problem


_OptResultsT = TypeVar("_OptResultsT", SingleObjOptResults, MultiObjOptResults)


class Optimizer(Base, metaclass=ABCMeta):
    """Abstract base class for optimization solvers."""

    def __init__(self, problem: Problem, name: str = "optimizer") -> None:
        """
        Parameters
        ----------
        problem
            The problem to optimize
        name
            The name
        """
        super().__init__(name)
        self.problem = problem
        self.name = name
        self._callback_dispatcher = _OptimizerCallbackDispatcher(None)

    def print_info(self) -> None:
        """Print solver info, called before solving"""
        print("\nProblem:")
        print("--------")
        print(f"  name         : {self.problem.name}")
        print(f"  n_vars_int   : {self.problem.n_vars_int}")
        print(f"  n_vars_float : {self.problem.n_vars_float}")
        print(f"  n_objectives : {self.problem.objs.n_functions}")
        print(f"  n_obj_cmptns : {self.problem.n_objectives}")
        print(f"  n_constraints: {self.problem.cons.n_functions}")
        print(f"  n_con_cmptns : {self.problem.n_constraints}")

    @abstractmethod
    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ) -> SingleObjOptResults | MultiObjOptResults | None:
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for intermediate optimization states.

        Returns
        -------
        results
            The optimization results object
        """

        # check problem initialization:
        if not self.problem.initialized:
            raise ValueError(
                f"Optimizer called for problem '{self.problem.name}'"
                + " before problem initialization"
            )

        # check solver initialization:
        if not self.initialized:
            raise ValueError(
                f"Optimizer called for problem '{self.problem.name}'"
                + " before solver initialization"
            )

        callback_dispatcher = _OptimizerCallbackDispatcher(callbacks)
        self._validate_callbacks(callback_dispatcher.callbacks)
        self._callback_dispatcher = callback_dispatcher
        self._callback_dispatcher.initialize(self)
        return None

    def _validate_callbacks(self, callbacks: list[OptimizerCallback]) -> None:
        """Validate backend-specific callback capabilities."""
        del callbacks

    @property
    def _has_callbacks(self) -> bool:
        """Whether callbacks are active for the current solve."""
        return bool(self._callback_dispatcher.callbacks)

    def _notify_callbacks(self, data: OptimizerCallbackData) -> None:
        """Notify callbacks about an intermediate optimizer state."""
        self._callback_dispatcher.notify(data)

    def _finalize_callbacks(self, results: _OptResultsT) -> _OptResultsT:
        """Finalize callbacks and return the optimization results."""
        self._callback_dispatcher.finalize(results)
        return results

    def finalize(
        self,
        opt_results: SingleObjOptResults | MultiObjOptResults | int | None = None,
        verbosity: int = 1,
    ) -> None:
        """
        This function may be called after finishing
        the optimization.

        Parameters
        ----------
        opt_results
            The optimization results object
        verbosity
            The verbosity level, 0 = silent
        """
        if not isinstance(opt_results, (SingleObjOptResults, MultiObjOptResults)):
            super().finalize(
                verbosity=opt_results if isinstance(opt_results, int) else verbosity
            )
            return

        if verbosity:
            print(f"{type(self).__name__}: Optimization run finished")
            if (
                isinstance(opt_results.success, bool)
                or len(opt_results.success.flat) == 1
            ):
                print(f"  Success: {opt_results.success}")
            else:
                v = np.sum(opt_results.success) / len(opt_results.success.flat)
                print(f"  Success: {100 * v:.2f} %")

            if opt_results is not None and opt_results.objs is not None:
                if self.problem.n_objectives == 1:
                    i0 = 0
                    for o in self.problem.objs.functions:
                        n = o.n_components()
                        i1 = i0 + n
                        names = o.component_names
                        if n == 1:
                            val = opt_results.objs[i0]
                            print(f"  Best {o.name} = {val}")
                        else:
                            for i in range(n):
                                val = opt_results.objs[i0 + i]
                                print(f"  Best {names[i]} = {val}")
                        i0 = i1

                else:
                    i0 = 0
                    for o in self.problem.objs.functions:
                        n = o.n_components()
                        i1 = i0 + n
                        names = o.component_names
                        if n == 1:
                            if self.problem.maximize_objs[i0]:
                                val = np.max(opt_results.objs[:, i0])
                            else:
                                val = np.min(opt_results.objs[:, i0])
                            print(f"  Best {o.name} = {val}")
                        else:
                            for i in range(n):
                                if self.problem.maximize_objs[i0 + 1]:
                                    val = np.max(opt_results.objs[:, i0 + i])
                                else:
                                    val = np.min(opt_results.objs[:, i0 + i])
                                print(f"  Best {names[i]} = {val}")
                        i0 = i1

    @classmethod
    def new(cls, optimizer_type: str, *args: object, **kwargs: object) -> "Optimizer":
        """
        Run-time optimizer factory.

        Parameters
        ----------
        optimizer_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor
        """
        return new_instance(cls, optimizer_type, *args, **kwargs)
