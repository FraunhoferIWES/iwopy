import numpy as np

from iwopy.core import Problem


class SimpleProblem(Problem):
    """
    A problem which simply pipes variables to its
    objectives and constraints.

    Parameters
    ----------
    int_vars : dict or array-like
        The integer variables, either dict with name str
        to initial value mapping, or list of variable names
    float_vars : dict or array-like
        The float variables, either dict with name str
        to initial value mapping, or list of variable names
    kwargs : dict, optional
        Additional parameters for the Problem class

    """

    def __init__(
        self,
        name,
        int_vars=None,
        float_vars=None,
        min_int_vars=None,
        max_int_vars=None,
        min_float_vars=None,
        max_float_vars=None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

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

        if isinstance(min_int_vars, dict):
            self._ivars_min = min_int_vars
        elif min_int_vars is not None:
            self._ivars_min = {v: min_int_vars for v in self._ivars}
        else:
            self._ivars_min = {v: -self.INT_INF for v in self._ivars}

        if isinstance(max_int_vars, dict):
            self._ivars_max = max_int_vars
        elif max_int_vars is not None:
            self._ivars_max = {v: max_int_vars for v in self._ivars}
        else:
            self._ivars_max = {v: self.INT_INF for v in self._ivars}

        if isinstance(min_float_vars, dict):
            self._fvars_min = min_float_vars
        elif min_float_vars is not None:
            self._fvars_min = {v: min_float_vars for v in self._fvars}
        else:
            self._fvars_min = {v: -np.inf for v in self._fvars}

        if isinstance(max_float_vars, dict):
            self._fvars_max = max_float_vars
        elif max_float_vars is not None:
            self._fvars_max = {v: max_float_vars for v in self._fvars}
        else:
            self._fvars_max = {v: np.inf for v in self._fvars}

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

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return np.array(
            [self._ivars_min[v] for v in self.var_names_int()], dtype=np.int32
        )

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return np.array(
            [self._ivars_max[v] for v in self.var_names_int()], dtype=np.int32
        )

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

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return np.array(
            [self._fvars_min[v] for v in self.var_names_float()], dtype=np.float64
        )

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return np.array(
            [self._fvars_max[v] for v in self.var_names_float()], dtype=np.float64
        )
