from abc import abstractmethod

from iwopy.utils import new_instance
from .function import OptFunction


class Objective(OptFunction):
    """
    Abstract base class for objective functions.

    :group: core

    """

    @abstractmethod
    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags: np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        pass

    @classmethod
    def new(cls, objective_type, *args, **kwargs):
        """
        Run-time objective function factory.

        Parameters
        ----------
        objective_type: str
            The selected derived class name
        args: tuple, optional
            Additional parameters for constructor
        kwargs: dict, optional
            Additional parameters for constructor

        """
        return new_instance(cls, objective_type, *args, **kwargs)
