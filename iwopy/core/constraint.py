from typing import Any

import numpy as np

from iwopy.utils import new_instance

from .function import OptFunction


class Constraint(OptFunction):
    """
    Abstract base class for optimization
    constraints.
    """

    def __init__(self, *args: Any, tol: float = 1e-5, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        args
            Positional parameters for the base class
        tol
            The tolerance for constraint violations
        kwargs
            Keyword parameters for the base class
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
        mi, ma = self.get_bounds()
        out = (vals + self.tol >= mi) & (vals - self.tol <= ma)

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
        mi, ma = self.get_bounds()
        mi = np.array(mi, dtype=np.float64)
        ma = np.array(ma, dtype=np.float64)

        out = (vals + self.tol >= mi[None, :]) & (vals - self.tol <= ma[None, :])

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
