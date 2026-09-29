from __future__ import annotations

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure

    from .optimizer import Optimizer

from .opt_results import MultiObjOptResults, SingleObjOptResults


def _read_only_population(
    values: np.ndarray,
    name: str,
    dtype: np.dtype,
) -> np.ndarray:
    """Create a read-only two-dimensional population array."""
    population = np.asarray(values, dtype=dtype)
    if population.ndim == 1:
        population = population[None, :]
    if population.ndim != 2:
        raise ValueError(
            f"Optimizer callback data '{name}' must be one- or two-dimensional."
        )
    return np.frombuffer(population.tobytes(), dtype=population.dtype).reshape(
        population.shape
    )


@dataclass(frozen=True)
class OptimizerCallbackData:
    """Immutable intermediate optimization data supplied to callbacks.

    Attributes
    ----------
    event
        The backend event represented by this snapshot.
    vars_int
        Integer variables for the current population.
    vars_float
        Floating-point variables for the current population.
    objs
        Objective values for the current population, if available.
    cons
        Constraint values for the current population, if available.
    iteration
        The solver iteration or generation, if available.
    n_evaluations
        The cumulative number of evaluated individuals, if available.

    :group: core

    """

    event: Literal["iteration", "evaluation"]
    vars_int: np.ndarray
    vars_float: np.ndarray
    objs: np.ndarray | None
    cons: np.ndarray | None
    iteration: int | None = None
    n_evaluations: int | None = None

    def __post_init__(self) -> None:
        """Validate and normalize callback data."""
        if self.event not in ("iteration", "evaluation"):
            raise ValueError(f"Unknown optimizer callback event '{self.event}'.")
        for name in ("iteration", "n_evaluations"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, int) or value < 0):
                raise ValueError(
                    f"Optimizer callback data '{name}' must be a non-negative integer."
                )

        arrays = {
            "vars_int": _read_only_population(
                self.vars_int, "vars_int", np.dtype(np.int32)
            ),
            "vars_float": _read_only_population(
                self.vars_float, "vars_float", np.dtype(np.float64)
            ),
            "objs": None
            if self.objs is None
            else _read_only_population(self.objs, "objs", np.dtype(np.float64)),
            "cons": None
            if self.cons is None
            else _read_only_population(self.cons, "cons", np.dtype(np.float64)),
        }
        population_sizes = {len(values) for values in arrays.values() if values is not None}
        if len(population_sizes) != 1:
            raise ValueError("Optimizer callback population arrays have inconsistent sizes.")
        for name, values in arrays.items():
            object.__setattr__(self, name, values)


class OptimizerCallback(metaclass=ABCMeta):
    """Base class for optimizer callbacks.

    :group: core

    """

    def __init__(self) -> None:
        """Initialize the callback."""
        self.optimizer: Optimizer | None = None

    def initialize(self, optimizer: Optimizer) -> None:
        """Prepare the callback for an optimization run.

        Parameters
        ----------
        optimizer
            The optimizer starting the run.

        """
        self.optimizer = optimizer

    @abstractmethod
    def notify(self, data: OptimizerCallbackData) -> None:
        """Process intermediate optimization data.

        Parameters
        ----------
        data
            The current normalized optimizer state.

        """

    def finalize(
        self,
        results: SingleObjOptResults | MultiObjOptResults,
    ) -> None:
        """Process the completed optimization results.

        Parameters
        ----------
        results
            The completed iwopy optimization results.

        """


class OptimizationHistory(OptimizerCallback):
    """Record intermediate optimization states.

    The ``states`` attribute contains the snapshots received during the
    current or latest optimization run.

    :group: core

    """

    def __init__(self) -> None:
        """Initialize the history."""
        super().__init__()
        self.states: list[OptimizerCallbackData] = []

    def initialize(self, optimizer: Optimizer) -> None:
        """Reset the history for a new optimization run."""
        super().initialize(optimizer)
        self.states.clear()

    def notify(self, data: OptimizerCallbackData) -> None:
        """Record an intermediate optimizer state."""
        self.states.append(data)

    def plot_objective(
        self,
        objective: int = 0,
        ax: Axes | None = None,
        **kwargs,
    ) -> Figure:
        """Plot the best objective value in each recorded state.

        Parameters
        ----------
        objective
            Index of the objective component to plot.
        ax
            Matplotlib axis receiving the plot, or ``None`` to create one.
        kwargs
            Additional arguments forwarded to the Matplotlib plot command.

        Returns
        -------
        matplotlib.figure.Figure
            The figure containing the objective history.

        """
        if self.optimizer is None:
            raise RuntimeError("Optimization history has not been initialized.")
        n_objectives = self.optimizer.problem.n_objectives
        if objective < 0 or objective >= n_objectives:
            raise IndexError(
                f"Objective index {objective} is outside [0, {n_objectives})."
            )

        states = [state for state in self.states if state.objs is not None]
        if not states:
            raise RuntimeError("Optimization history contains no objective values.")

        import matplotlib.pyplot as plt

        maximize = self.optimizer.problem.maximize_objs[objective]
        select = np.max if maximize else np.min
        objective_values = [select(state.objs[:, objective]) for state in states]
        steps = [
            state.iteration
            if state.iteration is not None
            else state.n_evaluations
            if state.n_evaluations is not None
            else index
            for index, state in enumerate(states, start=1)
        ]
        if ax is None:
            figure, ax = plt.subplots()
        else:
            figure = ax.figure
        objective_name = self.optimizer.problem.objs.component_names[objective]
        ax.plot(steps, objective_values, label=objective_name, **kwargs)
        ax.set_xlabel("iteration" if states[0].event == "iteration" else "evaluations")
        ax.set_ylabel(objective_name)
        return figure


class _OptimizerCallbackDispatcher:
    """Validate and dispatch an ordered optimizer callback list."""

    def __init__(self, callbacks: list[OptimizerCallback] | None) -> None:
        if callbacks is None:
            callbacks = []
        elif not isinstance(callbacks, list):
            raise TypeError("Optimizer callbacks must be supplied as a list.")
        if not all(isinstance(callback, OptimizerCallback) for callback in callbacks):
            raise TypeError("All optimizer callbacks must derive from OptimizerCallback.")
        self.callbacks = callbacks.copy()

    def initialize(self, optimizer: Optimizer) -> None:
        """Initialize callbacks for a solve."""
        for callback in self.callbacks:
            callback.initialize(optimizer)

    def notify(self, data: OptimizerCallbackData) -> None:
        """Notify callbacks in registration order."""
        for callback in self.callbacks:
            callback.notify(data)

    def finalize(
        self,
        results: SingleObjOptResults | MultiObjOptResults,
    ) -> None:
        """Finalize callbacks in registration order."""
        for callback in self.callbacks:
            callback.finalize(results)