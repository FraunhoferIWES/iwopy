from abc import abstractmethod

from .function import OptFunction


class Objective(OptFunction):
    """
    Abstract base class for objective functions.

    Parameters
    ----------
    problem: iwopy.Problem
        The underlying optimization problem
    name: str
        The function name
    vnames_int : list of str, optional
        The integer variable names. Useful for mapping
        problem variables to function variables
    vnames_float : list of str, optional
        The float variable names. Useful for mapping
        problem variables to function variables
    cnames : list of str, optional
        The names of the components

    """

    def __init__(self, problem, name, vnames_int=None, vnames_float=None, cnames=None):
        super().__init__(problem, name, vnames_int, vnames_float, cnames)

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
