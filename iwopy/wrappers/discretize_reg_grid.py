import numpy as np

from iwopy.tools import LightRegGrid
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
    fd_order : dict or int
        Finite difference order. Either a dict with
        key: variable name str, value: order int, or
        a global integer order for all variables.
        1 = forward, -1 = backward, 2 = centre
    fd_bounds_order : dict or int
        Finite difference order of boundary points.
        Either a dict with key: variable name str,
        value: order int, or a global integer order
        for all variables. Default is same as fd_order

    Attributes
    ----------
    grid : iwopy.tools.LightRegGrid
        The discretization grid
    order : dict
        Finite difference order. Key: variable name
        str, value: 1 = forward, -1 = backward, 2 = centre
    orderb : dict or int
        Finite difference order of boundary points.
        Key: variable name str, value: order int

    """

    def __init__(self, base_problem, deltas, fd_order=1, fd_bounds_order=None):
        super().__init__(base_problem, base_problem.name + "_grid")

        self.grid = None
        self._deltas = deltas

        if isinstance(fd_order, int):
            self.order = {v: fd_order for v in deltas.keys()}
        else:
            self.order = fd_order
            for v in deltas.keys():
                if v not in self.order:
                    raise KeyError(
                        f"Problem '{self.name}': Missing fd_order entry for variable '{v}'"
                    )

        if fd_bounds_order is None:
            self.orderb = {v: abs(o) for v, o in self.order.items()}
        elif isinstance(fd_bounds_order, int):
            self.orderb = {v: fd_bounds_order for v in deltas.keys()}
        else:
            self.orderb = fd_bounds_order
            for v in deltas.keys():
                if v not in self.orderb:
                    raise KeyError(
                        f"Problem '{self.name}': Missing fd_bounds_order entry for variable '{v}'"
                    )

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

        self._vinds = []
        origin = []
        deltas = []
        nsteps = []
        self._order = []
        self._orderb = []

        vnms = list(super().var_names_float())
        vmins = np.full(super().n_vars_float, np.nan, dtype=np.float64)
        vmins[:] = super().min_values_float()
        vmaxs = np.full(super().n_vars_float, np.nan, dtype=np.float64)
        vmaxs[:] = super().max_values_float()
        for v, d in self._deltas.items():

            if v not in vnms:
                raise KeyError(
                    f"Problem '{self.name}': Variable '{v}' given in deltas, but not found in problem float variables {vnms}"
                )

            vi = vnms.index(v)
            vmin = vmins[vi]
            vmax = vmaxs[vi]
            self._vinds.append(vi)
            self._order.append(self.order[v])
            self._orderb.append(self.orderb[v])
            if np.isinf(vmin) and np.isinf(vmax):
                origin.append(0.0)
                deltas.append(d)
                nsteps.append(None)
            elif np.isinf(vmin):
                origin.append(vmax)
                deltas.append(d)
                nsteps.append(None)
            elif np.isinf(vmax):
                origin.append(vmin)
                deltas.append(d)
                nsteps.append(None)
            else:
                origin.append(vmin)
                nsteps.append(int((vmax - vmin) / d))
                deltas.append((vmax - vmin) / nsteps[-1])

        self.grid = LightRegGrid(origin, deltas, nsteps)
        if verbosity:
            self.grid.print_info(4)
            print(self._hline)

        self._order = np.array(self._order, dtype=np.int32)
        self._orderb = np.array(self._orderb, dtype=np.int32)

    def calc_gradients(
        self, 
        vars_int, 
        vars_float, 
        func, 
        ivars, 
        fvars, 
        vrs, 
        components, 
        verbosity=0
    ):
        """
        The actual gradient calculation.

        Can be overloaded in derived classes, the base class only considers
        analytic derivatives.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        func : iwopy.core.OptFunctionList, optional
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        ivars : list of int
            The indices of the function int variables in the problem
        fvars : list of int
            The indices of the function float variables in the problem
        vrs : list of int
            The function float variable indices wrt which the
            derivatives are to be calculated
        components : list of int
            The selected components of func, or None for all
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functions, shape:
            (n_func_cmpnts, n_vars)

        """
        # get analytic gradient results:
        gradients = super().calc_gradients(
            vars_int, vars_float, func, ivars, fvars, vrs, components, verbosity
        )

        # find variables and components of unsolved gradients:
        gnan = np.isnan(gradients)
        gvars = np.where(np.any(gnan, axis=0))[0]
        gvars = [vi for vi in gvars if vi in self._vinds]
        ivars = [self._vinds.index(vi) for vi in gvars]
        pvars = np.array(fvars)[gvars]
        cmpnts = np.where(np.any(gnan, axis=1))[0]
        cmpnts = [c for c in components if c in cmpnts]
        if not len(pvars) or not len(cmpnts):
            return gradients
        del gnan

        # get gradient grid points and coeffs:
        varsf = vars_float[None, gvars]
        order = self._order[ivars]
        orderb = self._orderb[ivars]
        gpts, coeffs = self.grid.grad_coeffs(varsf, gvars, order, orderb)
        print("HERE",gpts.shape,coeffs.shape)
        quit()
        return gradients
