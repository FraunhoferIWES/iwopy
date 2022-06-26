import numpy as np
from abc import ABCMeta, abstractmethod

from iwopy.core.base import Base

class Problem(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization problems.

    Parameters
    ----------
    name: str
        The problem's name

    Attributes
    ----------
    objs: list:iwopy.ObjFunction
        The objective functions
    cons: list:iwopy.OptConstraint
        The constraints
    var_values_int : numpy.ndarray
        Integer array with variable values,
        shape: (n_vars_int,)
    var_values_float : numpy.ndarray
        Float array with variable values,
        shape: (n_vars_float,)

    """

    def __init__(self, name):
        super().__init__(name)
        self.objs = []
        self.cons = []
        self.var_values_int   = None
        self.var_values_float = None
    
    def var_names_int(self):
        """
        The names of integer variables.

        Returns
        -------
        names : list of str
            The names of the integer variables

        """
        return []

    def initial_values_int(self):
        """
        The initial values of the integer variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return 0

    @property
    def n_vars_int(self):
        """
        The number of int variables

        Returns
        -------
        n : int
            The number of int variables
            
        """
        return len(self.var_names_int())

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return []

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return None

    @property
    def n_vars_float(self):
        """
        The number of float variables

        Returns
        -------
        n : int
            The number of float variables
            
        """
        return len(self.var_names_float())

    def initialize(self, verbosity=0):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if verbosity:
            s = f"Problem '{self.name}' ({type(self).__name__}): Initializing"
            print(s)
            print("-"*len(s))
        
        n_int   = self.n_vars_int()
        n_float = self.n_vars_float()

        if verbosity:
            print(f"  n_vars_int  : {n_int}")
            print(f"  n_vars_float: {n_float}")

        self.var_values_int = np.zeros(n_int, dtype=np.int32)
        self.var_values_int[:] = self.initial_values_int()

        self.var_values_float = np.zeros(n_float, dtype=np.float64)
        self.var_values_float[:] = self.initial_values_float()

        super().initialize(verbosity)