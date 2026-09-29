from abc import abstractmethod

import numpy as np

from iwopy.utils import new_instance

from .function import OptFunction


class Objective(OptFunction):
    """Abstract base class for objective functions."""

    @abstractmethod
    def maximize(self) -> np.ndarray:
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags
            Bool array for component maximization,
            shape: (n_components,)
        """

    @classmethod
    def new(cls, objective_type: str, *args: object, **kwargs: object) -> "Objective":
        """
        Run-time objective function factory.

        Parameters
        ----------
        objective_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor
        """
        return new_instance(cls, objective_type, *args, **kwargs)
