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
    base_problem : iwopy.Problem
        The underlying concrete problem
    deltas : dict
        The step sizes. Key: variable name str,
        Value: step size.
    order : dict
        Finite difference order. Key: variable name 
        str, value: 1 = forward, -1 = backward, 2 = centre
    orderb : dict or int
        Finite difference order of boundary points. 
        Key: variable name str, value: order int

    """

    def __init__(self, base_problem, deltas, fd_order=1, fd_bounds_order=None):
        super().__init__(base_problem, base_problem.name + "_grid")

        self.deltas = deltas

        if isinstance(fd_order, int):
            self.order = {v: fd_order for v in deltas.keys()}
        else:
            self.order = fd_order
            for v in deltas.keys():
                if v not in self.order:
                    raise KeyError(f"Problem '{self.name}': Missing fd_order entry for variable '{v}'")
        
        if fd_bounds_order is None:
            self.orderb = {v: abs(o) for v, o in self.order.items()}
        elif isinstance(fd_bounds_order, int):
            self.orderb = {v: fd_bounds_order for v in deltas.keys()}
        else:
            self.orderb = fd_bounds_order
            for v in deltas.keys():
                if v not in self.orderb:
                    raise KeyError(f"Problem '{self.name}': Missing fd_bounds_order entry for variable '{v}'")

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
        self._origin = []
        self._delta = []
        self._nsteps = []
        self._order = []
        self._orderb = []

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
            self._vinds.append(vi)
            self._order.append(self.order[v])
            self._orderb.append(self.orderb[v])
            if np.isinf(vmin) and np.isinf(vmax):
                self._origin.append(0.)
                self._delta.append(d)
                self._nsteps.append(None)
            elif np.isinf(vmin):
                self._origin.append(vmax)
                self._delta.append(d)
                self._nsteps.append(None)
            elif np.isinf(vmax):
                self._origin.append(vmin)
                self._delta.append(d)
                self._nsteps.append(None)
            else:
                self._origin.append(vmin)
                self._nsteps.append(int((vmax - vmin) / d))
                self._delta.append((vmax - vmin) / self._nsteps[-1])
            
            if verbosity:
                s = f"    Variable {v}: p0 = {self._origin[-1]:.3e}, d = {self._delta[-1]:.3e}, n = {self._nsteps[-1]}, o = {self.order[v]}, ob = {self.orderb[v]}"
                print(s)
        if verbosity:
            print("-"*len(s))

        self._origin = np.array(self._origin, dtype=np.float64)
        self._delta = np.array(self._delta, dtype=np.float64)
        self._nsteps = np.array(self._nsteps, dtype=np.int)
        self._order = np.array(self._order, dtype=np.int)
        self._orderb = np.array(self._orderb, dtype=np.int)
    
    def get_grid_corner(self, p, subgrid=None):
        """
        Get the lower-left grid corner of a point in varaible space.

        Parameters
        ----------
        p : numpy.ndarray
            The point in variable space, shape: (n_p_dims,)
        subgrid : list of int, optional
            The variable space dimensions, shape: (n_gvars,)
            or None for all
        
        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner point, shape: (n_p_dims,)

        """
        if subgrid is None:
            return self._origin + ( (p - self._origin) // self._delta ) * self._delta
        else:
            o = self._origin[subgrid]
            d = self._delta[subgrid]
            return o + ( (p - o) // d ) * d

    def get_grid_corners(self, pts, subgrid=None):
        """
        Get the lower-left grid corner of points in varaible space.

        Parameters
        ----------
        pts : numpy.ndarray
            The points in variable space, shape: (n_pts, n_p_dims)
        subgrid : list of int, optional
            The variable space dimensions, shape: (n_gvars,)
            or None for all
        
        Returns
        -------
        p0 : numpy.ndarray
            The lower-left grid corner points, shape: (n_pts, n_p_dims)

        """
        if subgrid is None:
            o = self._origin[None, :]
            d = self._delta[None, :]
        else:
            o = self._origin[subgrid][None, :]
            d = self._delta[subgrid][None, :]
        return o + ( (pts - o) // d ) * d

    def get_grid_cell(self, p, subgrid=None):
        """
        Get the grid cell that contains a point in
        varaible space.

        Parameters
        ----------
        p : numpy.ndarray
            The point in variable space, shape: (n_p_dims,)
        subgrid : list of int, optional
            The variable space dimensions, shape: (n_gvars,)
            or None for all
        
        Returns
        -------
        cell : numpy.ndarray
            The lower-left grid corner point, shape: (n_cdims, n_p_dims)
        ongrid : nbool
            True if point is on grid

        """
        p0 = self.get_grid_corner(p, subgrid)

        n_dims = len(self._origin) if subgrid is None else len(subgrid)
        n_cdim = 2**n_dims
        cell = np.zeros((n_cdim, n_dims), dtype=np.float64)
        cell[:] = p0[None, :]

        d = self._delta[subgrid] if subgrid is not None else self._delta
        a = np.array([])
        
        return cell, np.all(p == p0)
        

        

    def get_grad_plan(self, vars_float, vars, origin, delta, nsteps, order, orderb):
        """
        Create the calculation plan for the gradient computation.

        Parameters
        ----------
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        vars : numpy.ndarray
            The indices of the gradient variables, shape: (n_vars,)
        origin : numpy.ndarray
            The origin value, shape: (n_vars,)
        delta : numpy.ndarray
            The step delta, shape: (n_vars,)
        nsteps : numpy.ndarray
            The number of steps, shape: (n_vars,)
        order : numpy.ndarray
            The finite difference order, shape: (n_vars,)
        orderb : numpy.ndarray
            The finite difference order of boundary points, 
            shape: (n_vars,)
        
        Returns
        -------
        plan : numpy.ndarray
            The variables for calculations, shape:
            (n_pop, n_vars_float)

        """
        TODO


    def calc_gradients(
        self, vars_int, vars_float, func, ivars, fvars, vrs, components, verbosity=0
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
        gradients = super().calc_gradients(vars_int, vars_float, func, ivars, fvars, vrs, components, verbosity)

        # find variables and components of unsolved gradients:
        gnan  = np.isnan(gradients)
        gvars = np.where(np.any(gnan, axis=0))[0]
        gvars = [vi for vi in gvars if vi in self._vinds]
        pvars = np.array(fvars)[gvars]
        cmpnts = np.where(np.any(gnan, axis=1))[0]
        cmpnts = [c for c in components if c in cmpnts]
        if not len(pvars) or not len(cmpnts):
            return gradients
        del gnan

        # find subgrid data:
        linds = [self._vinds.index(i) for i in pvars]
        origin = self._origin[linds]
        delta = self._delta[linds]
        nsteps = self._nsteps[linds]
        order = self._order[linds]
        orderb = self._orderb[linds]

        # get calculation plan:
        plan = self.get_grad_plan(vars_float, gvars, origin, delta, nsteps, order, orderb)

        return gradients