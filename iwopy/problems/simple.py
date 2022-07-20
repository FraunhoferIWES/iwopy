import numpy as np

from iwopy.core import Problem


class SimpleProblem(Problem):
    """
    A simple problem which simply pipes variables to its
    objectives and constraints.

    Parameters
    ----------
    int_vars : dict or array-like
        The integer variables, either dict with name str
        to initial value mapping, or list of variable names
    float_vars : dict or array-like
        The float variables, either dict with name str
        to initial value mapping, or list of variable names

    """

    def __init__(self, name, int_vars=None, float_vars=None):
        super().__init__(name)

        if int_vars is None and float_vars is None:
            raise KeyError(
                f"Problem '{self.name}': No variables defined, please specify 'int_vars' and/or 'float_vars'"
            )

        if isinstance(int_vars, dict):
            self._ivars = int_vars
        elif int_vars is not None:
            self._ivars = {v: 0 for v in int_vars}
        else:
            self._ivars = {}

        if isinstance(float_vars, dict):
            self._fvars = float_vars
        elif float_vars is not None:
            self._fvars = {v: np.nan for v in float_vars}
        else:
            self._fvars = {}

    def var_names_int(self):
        """
        The names of integer variables.

        Returns
        -------
        names : list of str
            The names of the integer variables

        """
        return list(self._ivars.keys())

    def initial_values_int(self):
        """
        The initial values of the integer variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return np.array(list(self._ivars.values()), dtype=np.int32)

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return list(self._fvars.keys())

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return np.array(list(self._fvars.values()), dtype=np.float64)
