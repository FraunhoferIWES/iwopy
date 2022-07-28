import numpy as np

from iwopy.core.problem import Problem


class ProblemWrapper(Problem):
    """
    Generic abstract problem wrapper class.

    Parameters
    ----------
    base_problem : iwopy.Problem
        The underlying concrete problem
    name : str
        The problem name
    kwargs : dict, optional
        Additional parameters for the Problem class

    Attributes
    ----------
    base_problem : iwopy.Problem
        The underlying concrete problem

    """

    def __init__(self, base_problem, name, **kwargs):
        super().__init__(name, **kwargs)
        self.base_problem = base_problem

    def var_names_int(self):
        """
        The names of integer variables.

        Returns
        -------
        names : list of str
            The names of the integer variables

        """
        return self.base_problem.var_names_int()

    def initial_values_int(self):
        """
        The initial values of the integer variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return self.base_problem.initial_values_int()

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return self.base_problem.min_values_int()

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return self.base_problem.max_values_int()

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return self.base_problem.var_names_float()

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return self.base_problem.initial_values_float()

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return self.base_problem.min_values_float()

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return self.base_problem.max_values_float()

    def initialize(self, verbosity=0):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if not self.base_problem.initialized:
            self.base_problem.initialize(verbosity)

        self.objs = self.base_problem.objs
        self.cons = self.base_problem.cons

        for f in self.objs.functions:
            f.problem = self
        self.objs.problem = self

        for f in self.cons.functions:
            f.problem = self
        self.cons.problem = self

        super().initialize(verbosity)

    def apply_individual(self, vars_int, vars_float):
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        return self.base_problem.apply_individual(vars_int, vars_float)

    def apply_population(self, vars_int, vars_float):
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        return self.base_problem.apply_population(vars_int, vars_float)
