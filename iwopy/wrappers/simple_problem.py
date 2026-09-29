from collections.abc import Sequence
from typing import Any

import numpy as np

from iwopy.core import Problem


class SimpleProblem(Problem):
    """
    A problem which simply pipes variables to its
    objectives and constraints.
    """

    def __init__(
        self,
        name: str,
        int_vars: dict[str, int] | Sequence[str] | np.ndarray | None = None,
        float_vars: dict[str, float] | Sequence[str] | np.ndarray | None = None,
        init_values_int: Sequence[int] | np.ndarray | None = None,
        init_values_float: Sequence[float] | np.ndarray | None = None,
        min_values_int: dict[str, int] | Sequence[int] | np.ndarray | None = None,
        max_values_int: dict[str, int] | Sequence[int] | np.ndarray | None = None,
        min_values_float: dict[str, float] | Sequence[float] | np.ndarray | None = None,
        max_values_float: dict[str, float] | Sequence[float] | np.ndarray | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Parameters
        ----------
        name
            The problem name
        int_vars
            The integer variables, either dict with name str
            to initial value mapping, or list of variable names
        float_vars
            The float variables, either dict with name str
            to initial value mapping, or list of variable names
        init_values_int
            The initial values, in case of list type int_vars
        init_values_float
            The initial values, in case of list type float_vars
        min_values_int
            The minimal values of the variables. Use `-self.INT_INF`
            for left-unbounded cases. None sets all values as such.
        max_values_int
            The maximal values of the variables. Use `self.INT_INF`
            for right-unbounded cases. None sets all values as such.
        min_values_float
            The minimal values of the variables. Use `-np.inf`
            for left-unbounded cases. None sets all values as such.
        max_values_float
            The maximal values of the variables. Use `np.inf`
            for right-unbounded cases. None sets all values as such.
        kwargs
            Additional parameters for the Problem class
        """
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

    def var_names_int(self) -> list[str]:
        """
        The names of integer variables.

        Returns
        -------
        names
            The names of the integer variables
        """
        return list(self._ivars.keys())

    def initial_values_int(self) -> np.ndarray:
        """
        The initial values of the integer variables.

        Returns
        -------
        values
            Initial int values, shape: (n_vars_int,)
        """
        return np.array(list(self._ivars.values()), dtype=np.int32)

    def min_values_int(self) -> np.ndarray:
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values
            Minimal int values, shape: (n_vars_int,)
        """
        return np.array(
            [self._ivars_min[v] for v in self.var_names_int()], dtype=np.int32
        )

    def max_values_int(self) -> np.ndarray:
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values
            Maximal int values, shape: (n_vars_int,)
        """
        return np.array(
            [self._ivars_max[v] for v in self.var_names_int()], dtype=np.int32
        )

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables
        """
        return list(self._fvars.keys())

    def initial_values_float(self) -> np.ndarray:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)
        """
        return np.array(list(self._fvars.values()), dtype=np.float64)

    def min_values_float(self) -> np.ndarray:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)
        """
        return np.array(
            [self._fvars_min[v] for v in self.var_names_float()], dtype=np.float64
        )

    def max_values_float(self) -> np.ndarray:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)
        """
        return np.array(
            [self._fvars_max[v] for v in self.var_names_float()], dtype=np.float64
        )
