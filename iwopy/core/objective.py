from abc import abstractmethod

from .function import OptFunction


class Objective(OptFunction):
    """
    Abstract base class for objective functions.
    """

    @abstractmethod
    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags : np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        pass
