from typing import Any

import numpy as np

from iwopy.utils import new_instance

from .function import OptFunction


class Constraint(OptFunction):
    """Abstract base class for optimization constraints.

    Each constraint owns the scalar feasibility tolerance applied to all of
    its components. Optimizer backends derive component-wise tolerance arrays
    from the registered constraint objects when they initialize.
    """

    def __init__(self, *args: Any, tol: float = 1e-5, **kwargs: Any) -> None:
        """Initialize the constraint.

        Parameters
        ----------
        args
            Positional parameters for the base class.
        tol
            Feasibility tolerance applied to every constraint component.
        kwargs
            Keyword parameters for the base class.
        """
        super().__init__(*args, **kwargs)
        self.tol = tol

    def get_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the bounds for all components.

        Non-existing bounds are expressed by np.inf.

        Returns
        -------
        min
            The lower bounds, shape: (n_components,)
        max
            The upper bounds, shape: (n_components,)
        """
        return (
            np.full(self.n_components(), -np.inf, dtype=np.float64),
            np.zeros(self.n_components(), dtype=np.float64),
        )

    def _feasibility_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return bounds expanded by feasibility tolerance and roundoff."""
        minimum, maximum = self.get_bounds()
        minimum = np.asarray(minimum, dtype=np.float64)
        maximum = np.asarray(maximum, dtype=np.float64)
        roundoff = 32.0 * np.finfo(np.float64).eps
        lower_slack = np.where(
            np.isfinite(minimum),
            roundoff * np.maximum(1.0, np.abs(minimum)),
            0.0,
        )
        upper_slack = np.where(
            np.isfinite(maximum),
            roundoff * np.maximum(1.0, np.abs(maximum)),
            0.0,
        )
        return minimum - self.tol - lower_slack, maximum + self.tol + upper_slack

    def check_individual(
        self, constraint_values: np.ndarray, verbosity: int = 0
    ) -> np.ndarray:
        """
        Check if the constraints are fullfilled for the
        given individual.

        Parameters
        ----------
        constraint_values
            The constraint values, shape: (n_components,)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The boolean result, shape: (n_components,)
        """
        vals = constraint_values
        minimum, maximum = self._feasibility_bounds()
        out = (vals >= minimum) & (vals <= maximum)

        if verbosity:
            print(f"Constraint '{self.name}': tol = {self.tol}")
            cnames = self.component_names
            for ci in range(self.n_components()):
                val = f"{cnames[ci]} = {vals[ci]:.3e}"
                suc = "OK" if out[ci] else "FAILED"
                print(f"  Constraint {val:<30} {suc}")

        return out

    def check_population(
        self, constraint_values: np.ndarray, verbosity: int = 0
    ) -> np.ndarray:
        """
        Check if the constraints are fullfilled for the
        given population.

        Parameters
        ----------
        constraint_values
            The constraint values, shape: (n_pop, n_components,)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The boolean result, shape: (n_pop, n_components)
        """
        vals = constraint_values
        minimum, maximum = self._feasibility_bounds()
        out = (vals >= minimum[None, :]) & (vals <= maximum[None, :])

        if verbosity:
            print(f"Constraint '{self.name}': tol = {self.tol}")
            cnames = self.component_names
            for ci in range(self.n_components()):
                suc = "OK" if np.all(out[ci]) else "FAILED"
                print(f"  Constraint {cnames[ci]:<20} {suc}")

        return out

    @classmethod
    def new(cls, constraint_type: str, *args: object, **kwargs: object) -> "Constraint":
        """
        Run-time constraint factory.

        Parameters
        ----------
        constraint_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor
        """
        return new_instance(cls, constraint_type, *args, **kwargs)
