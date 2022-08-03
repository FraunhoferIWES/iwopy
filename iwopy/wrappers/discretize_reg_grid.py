import numpy as np

from iwopy.utils import RegularDiscretizationGrid
from .problem_wrapper import ProblemWrapper
from iwopy.core import Memory


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
    mem_size : int, optional
        The memory size, default no memory
    dpars : dict, optional
        Additional parameters for `RegularDiscretizationGrid`

    Attributes
    ----------
    grid : iwopy.tools.RegularDiscretizationGrid
        The discretization grid
    order : dict
        Finite difference order. Key: variable name
        str, value: 1 = forward, -1 = backward, 2 = centre
    orderb : dict or int
        Finite difference order of boundary points.
        Key: variable name str, value: order int

    """

    def __init__(
        self,
        base_problem,
        deltas,
        fd_order=1,
        fd_bounds_order=None,
        mem_size=1000,
        **dpars,
    ):
        super().__init__(base_problem, base_problem.name + "_grid")

        self.grid = None
        self._deltas = deltas
        self._msize = mem_size
        self._dpars = dpars

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

        self.grid = RegularDiscretizationGrid(origin, deltas, nsteps, **self._dpars)
        if verbosity:
            self.grid.print_info(4)
            print(self._hline)

        self._order = np.array(self._order, dtype=np.int32)
        self._orderb = np.array(self._orderb, dtype=np.int32)

        if self._msize is not None:

            def keyf(varsi, varsf):
                gpts = np.atleast_2d(varsf[self._vinds])
                li = varsi.tolist() if len(varsi) else []
                tf = tuple(tuple(v.tolist()) for v in self.grid.gpts2inds(gpts))
                return (tuple(li), tf)

            self.memory = Memory(self._msize, keyf)

    def calc_gradients(
        self,
        vars_int,
        vars_float,
        func,
        ivars,
        fvars,
        vrs,
        components,
        pop=False,
        verbosity=0,
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
        pop : bool
            Flag for vectorizing calculations via population
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functions, shape:
            (n_func_cmpnts, n_vrs)

        """
        # get analytic gradient results:
        gradients = super().calc_gradients(
            vars_int, vars_float, func, ivars, fvars, vrs, components, verbosity
        )

        # find variables and components of unsolved gradients:
        gnan = np.isnan(gradients)
        gvars = np.where(np.any(gnan, axis=0))[0]
        pvars = np.array(fvars)[gvars]
        gvars = [vi for vi in pvars if vi in self._vinds]
        ivars = [self._vinds.index(vi) for vi in gvars]
        cmpnts = np.where(np.any(gnan, axis=1))[0]
        cmpnts = [c for c in components if c in cmpnts]
        if not len(gvars) or not len(cmpnts):
            return gradients
        del gnan

        # get gradient grid points and coeffs:
        varsf = vars_float[None, gvars]
        order = self._order[ivars]
        orderb = self._orderb[ivars]
        gpts, coeffs = self.grid.grad_coeffs(varsf, gvars, order, orderb)

        # run the calculation:
        n_pop = len(gpts)
        varsf = np.full((n_pop, self.n_vars_float), np.nan, dtype=np.float64)
        varsf[:] = vars_float[None, :]
        varsf[:, gvars] = gpts
        if pop:
            varsi = np.zeros((n_pop, self.n_vars_int), dtype=np.int32)
            if self.n_vars_int:
                varsi[:] = vars_int[None, :]
            objs, cons = self.evaluate_population(varsi, varsf)
        else:
            objs = np.full((n_pop, self.n_objectives), np.nan, dtype=np.float64)
            cons = np.full((n_pop, self.n_constraints), np.nan, dtype=np.float64)
            for i, vf in enumerate(varsf):
                objs[i], cons[i] = self.evaluate_individual(vars_int, vf)

        # recombine results:
        fres = np.c_[objs, cons][:, cmpnts]
        gres = np.einsum("gc,vg->cv", fres, coeffs[0])
        if len(cmpnts) == func.n_components():
            gradients[:, gvars] = gres
        else:
            temp = gradients[:, gvars]
            temp[cmpnts] = gres
            gradients[:, gvars] = temp

        return gradients

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
        if self.grid.is_gridpoint(vars_float[self._vinds]):
            return super().apply_individual(vars_int, vars_float)

        else:
            raise NotImplementedError(
                f"Problem '{self.name}' cannot apply non-grid point {vars_float} to problem"
            )

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
        if self.grid.all_gridpoints(vars_float[:, self._vinds]):
            return super().apply_population(vars_int, vars_float)

        else:
            raise NotImplemented(
                f"Problem '{self.name}' cannot apply non-grid points to problem"
            )

    def evaluate_individual(self, vars_int, vars_float):
        """
        Evaluate a single individual of the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        objs : np.array
            The objective function values, shape: (n_objectives,)
        con : np.array
            The constraints values, shape: (n_constraints,)

        """
        varsf = vars_float[self._vinds]
        if self.grid.is_gridpoint(varsf):
            return super().evaluate_individual(vars_int, vars_float)

        else:
            gpts, coeffs = self.grid.interpolation_coeffs_point(varsf)

            n_gpts = len(gpts)
            objs = np.zeros((n_gpts, self.n_objectives), dtype=np.float64)
            cons = np.zeros((n_gpts, self.n_constraints), dtype=np.float64)

            for gi, gp in enumerate(gpts):
                varsf = vars_float.copy()
                varsf[self._vinds] = gp
                objs[gi], cons[gi] = super().evaluate_individual(vars_int, varsf)

            return np.einsum("go,g->o", objs, coeffs), np.einsum(
                "gc,g->c", cons, coeffs
            )

    def evaluate_population(self, vars_int, vars_float):
        """
        Evaluate all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        objs : np.array
            The objective function values, shape: (n_pop, n_objectives)
        cons : np.array
            The constraints values, shape: (n_pop, n_constraints)

        """
        varsf = vars_float[:, self._vinds]

        # case all points on grid:
        if self.grid.all_gridpoints(varsf):
            return super().evaluate_population(vars_int, vars_float)

        # case all vars are grid vars:
        elif self.n_vars_int == 0 and len(self._vinds) == self.n_vars_float:

            gpts, coeffs = self.grid.interpolation_coeffs_points(varsf)

            n_gpts = len(gpts)
            varsi = np.zeros((n_gpts, self.n_vars_int), dtype=np.int32)
            varsi[:] = vars_int[0, None, :]
            varsf = np.zeros((n_gpts, self.n_vars_float), dtype=np.float64)
            varsf[:] = vars_float[0, None, :]
            varsf[:, self._vinds] = gpts

            objs, cons = self.evaluate_population(varsi, varsf)

            return np.einsum("go,pg->po", objs, coeffs), np.einsum(
                "gc,pg->pc", cons, coeffs
            )

        # mixed case:
        else:

            gpts, coeffs, gmap = self.grid.interpolation_coeffs_points(
                varsf, ret_pmap=True
            )

            # each pop has n_gp grid points, this yields pop2:
            n_pop = len(vars_float)
            n_gp = gmap.shape[1]
            n_pop2 = n_pop * n_gp
            n_int = self.n_vars_int
            n_v = n_int + self.n_vars_float
            vinds = n_int + np.array(self._vinds)

            # create variables of pop2:
            apts = np.zeros((n_pop, n_gp, n_v), dtype=np.float64)
            apts[:, :, :n_int] = vars_int.astype(np.float64)[:, None, :]
            apts[:, :, n_int:] = vars_float[:, None, :]
            apts = apts.reshape(n_pop2, n_v)
            apts[:, vinds] = np.take_along_axis(gpts, gmap.reshape(n_pop2), axis=0)

            # the uniques of pop2 form pop3:
            upts, umap = np.unique(apts, axis=0, return_inverse=True)
            varsi = upts[:, :n_int].astype(np.int32)
            varsf = upts[:, n_int:]
            del apts, upts

            # calculate results for pop3:
            objs, cons = self.evaluate_population(varsi, varsf)
            del varsi, varsf

            # reconstruct results for pop2:
            objs = np.take_along_axis(objs, umap, axis=0)
            cons = np.take_along_axis(cons, umap, axis=0)

            # calculate results for pop, by applying coeffs:
            objs = objs.reshape(n_pop, n_gp, self.n_objectives)
            cons = cons.reshape(n_pop, n_gp, self.n_objectives)
            coeffs = np.take_along_axis(coeffs, gmap, axis=1)
            return (
                np.einsum("pgo,pg->po", objs, coeffs),
                np.einsum("pgc,pg->pc", cons, coeffs),
            )
