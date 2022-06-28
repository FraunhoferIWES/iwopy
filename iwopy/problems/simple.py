import numpy as np

from iwopy.core import Problem

class SimpleProblem(Problem):
    """
    A simple problem which simply
    pipes variables to its objectives and constraints.

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
            raise KeyError(f"Problem '{self.name}': No variables defined, please specify 'int_vars' and/or 'float_vars'")
    
        if isinstance(int_vars, dict):
            self._ivars = int_vars
        elif int_vars is not None:
            self._ivars = {v: 0 for v in int_vars}

        if isinstance(float_vars, dict):
            self._fvars = float_vars
        elif float_vars is not None:
            self._fvars = {v: np.nan for v in float_vars}

    TODO