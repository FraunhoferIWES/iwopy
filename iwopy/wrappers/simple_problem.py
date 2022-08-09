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
    init_values_int : list of float, optional
        The initial values, in case of list type int_vars
    init_values_float : list of float, optional
        The initial values, in case of list type float_vars
    min_values_int : dict or list, optional
        The minimal values of the variables. Use `-self.INT_INF`
        for left-unbounded cases. None sets all values as such.
    max_values_int : dict or list, optional
        The maximal values of the variables. Use `self.INT_INF`
        for right-unbounded cases. None sets all values as such.
    min_values_float : dict or list, optional
        The minimal values of the variables. Use `-np.inf`
        for left-unbounded cases. None sets all values as such.
    max_values_float : dict or list, optional
        The maximal values of the variables. Use `np.inf`
        for right-unbounded cases. None sets all values as such.
    kwargs : dict, optional
        Additional parameters for the Problem class

    """

    def __init__(
        self,
        name,
        int_vars=None,
        float_vars=None,
        init_values_int=None,
        init_values_float=None,
        min_values_int=None,
        max_values_int=None,
        min_values_float=None,
        max_values_float=None,
        **kwargs,
    ):
        super().__init__(name, **kwargs)

        if int_vars is None and float_vars is None:
            raise KeyError(
                f"Problem '{self.name}': No variables defined, please specify 'int_vars' and/or 'float_vars'"
            )

        if isinstance(int_vars, dict):
            self._ivars = int_vars
            if init_values_int is not None:
                raise KeyError(
                    f"Problem '{self.name}': Unexpected parameter 'init_values_int' together with dict type 'int_vars'"
                )
        elif int_vars is not None:
            if init_values_int is None:
                raise KeyError(
                    f"Problem '{self.name}': Expecting parameter 'init_values_int' together with list type 'int_vars'"
                )
            self._ivars = {v: init_values_int[i] for i, v in enumerate(int_vars)}
        else:
            self._ivars = {}

        if isinstance(float_vars, dict):
            self._fvars = float_vars
            if init_values_float is not None:
                raise KeyError(
                    f"Problem '{self.name}': Unexpected parameter 'init_values_float' together with dict type 'float_vars'"
                )
        elif float_vars is not None:
            if init_values_float is None:
                raise KeyError(
                    f"Problem '{self.name}': Expecting parameter 'init_values_float' together with list type 'float_vars'"
                )
            self._fvars = {v: init_values_float[i] for i, v in enumerate(float_vars)}
        else:
            self._fvars = {}

        if isinstance(min_values_int, dict):
            self._ivars_min = min_values_int
        elif min_values_int is not None:
            self._ivars_min = {
                v: min_values_int[i] for i, v in enumerate(self._ivars.keys())
            }
        else:
            self._ivars_min = {v: -self.INT_INF for v in self._ivars}

        if isinstance(max_values_int, dict):
            self._ivars_max = max_values_int
        elif max_values_int is not None:
            self._ivars_max = {
                v: max_values_int[i] for i, v in enumerate(self._ivars.keys())
            }
        else:
            self._ivars_max = {v: self.INT_INF for v in self._ivars}

        if isinstance(min_values_float, dict):
            self._fvars_min = min_values_float
        elif min_values_float is not None:
            self._fvars_min = {
                v: min_values_float[i] for i, v in enumerate(self._fvars.keys())
            }
        else:
            self._fvars_min = {v: -np.inf for v in self._fvars}

        if isinstance(max_values_float, dict):
            self._fvars_max = max_values_float
        elif max_values_float is not None:
            self._fvars_max = {
                v: max_values_float[i] for i, v in enumerate(self._fvars.keys())
            }
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
