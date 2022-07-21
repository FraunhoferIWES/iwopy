import numpy as np

from .problem_wrapper import ProblemWrapper

class DiscretizeRegGrid(ProblemWrapper):
    """
    A wrapper that provides finite distance
    differentiation on a regular grid for
    selected or all problem float variables.

    Parameters
    ----------
    base_problem : iwopy.Problem
        The underlying concrete problem
    deltas : dict
        The step sizes. Key: variable name str,
        Value: step size. Will be adjusted to the
        variable bounds if necessary.

    Attributes
    ----------
    base_problem : iwopy.Problem
        The underlying concrete problem
    deltas : dict
        The step sizes. Key: variable name str,
        Value: step size.

    """

    def __init__(self, base_problem, deltas):
        super().__init__(base_problem, base_problem.name + "_grid")
        self.deltas = deltas

    def initialize(self, verbosity=0):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        if verbosity:
            print(f"  Finite difference grid:")

        deltas = {}
        self._p0 = {}
        self._n = {}
        vnms = list(super().var_names_float())
        vmins = np.full(super().n_vars_float, np.nan, dtype=np.float64)
        vmins[:] = super().min_values_float()
        vmaxs = np.full(super().n_vars_float, np.nan, dtype=np.float64)
        vmaxs[:] = super().max_values_float()
        for v, d in self.deltas.items():

            if v not in vnms:
                raise KeyError(f"Problem '{self.name}': Variable '{v}' given in deltas, but not found in problem float variables {vnms}")
            
            vi = vnms.index(v)
            vmin = vmins[vi]
            vmax = vmaxs[vi]
            if np.isinf(vmin) and np.isinf(vmax):
                self._p0[v] = 0.
                deltas[v] = d
                self._n[v] = None
            elif np.isinf(vmin):
                self._p0[v] = vmax
                deltas[v] = d
                self._n[v] = None
            elif np.isinf(vmax):
                self._p0[v] = vmin
                deltas[v] = d
                self._n[v] = None
            else:
                self._p0[v] = vmin
                self._n[v] = int((vmax - vmin) / d)
                deltas[v] = (vmax - vmin) / self._n[v]
            
            if verbosity:
                s = f"    Variable {v}: p0 = {self._p0[v]:.3e}, d = {deltas[v]:.3e}, n = {self._n[v]}"
                print(s)
        if verbosity:
            print("-"*len(s))
        self.deltas = deltas