import numpy as np

from iwopy.utils import RegularDiscretizationGrid
from .problem_wrapper import ProblemWrapper
from iwopy.core import Memory, ProblemDefaultFunc


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
        fd_bounds_order=1,
        mem_size=1000,
        **dpars,
    ):
        super().__init__(base_problem, base_problem.name + "_grid")

        if isinstance(deltas, float):
            deltas = {v: deltas for v in base_problem.var_names_float()}

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

    def initialize(self, verbosity=1):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        if verbosity > 1:
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
        if verbosity > 1:
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
        components,
        ivars,
        fvars,
        vrs,
        pop=False,
        verbosity=0,
    ):
        """
        The actual gradient calculation, not to be called directly
        (call `get_gradients` instead).

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
        components : list of int, optional
            The function's component selection, or None for all
        ivars : list of int
            The indices of the function int variables in the problem
        fvars : list of int
            The indices of the function float variables in the problem
        vrs : list of int
            The function float variable indices wrt which the
            derivatives are to be calculated
        pop : bool
            Flag for vectorizing calculations via population
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functions, shape:
            (n_components, n_vrs)

        """
        # get analytic gradient results:
        gradients = super().calc_gradients(
            vars_int, vars_float, func, components, ivars, fvars, vrs, verbosity
        )

        # find variables and components of unsolved gradients:
        gnan = np.isnan(gradients)
        gvars = np.unique(np.where(np.any(gnan, axis=0))[0])
        pvars = np.array(fvars)[gvars]
        gvars = [vi for vi in pvars if vi in self._vinds]
        ivars = [self._vinds.index(vi) for vi in gvars]
        cmpnts = np.arange(func.n_components()) if components is None else np.array(components)
        cmptsi = np.unique(np.where(np.any(gnan, axis=1))[0])
        fcmpts = cmpnts[cmptsi]
        fcomps = fcmpts if components is not None else None
        if not len(gvars) or not len(fcmpts):
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
        values = np.full((n_pop, len(cmptsi)), np.nan, dtype=np.float64)
        varsf[:] = vars_float[None, :]
        varsf[:, gvars] = gpts
        if pop:
            varsi = np.zeros((n_pop, self.n_vars_int), dtype=np.int32)
            if self.n_vars_int:
                varsi[:] = vars_int[None, :]
            if isinstance(func, ProblemDefaultFunc):
                os, cs = self.evaluate_population(varsi, varsf)
                s = np.s_[:] if components is None else fcmpts
                values[:] = np.c_[os, cs][:, s]
                del os, cs, s 
            else:
                results = self.apply_population(varsi, varsf)
                values[:] = func.calc_population(varsi, varsf, results, fcomps)
                del results
        else:
            for i, vf in enumerate(varsf):
                if isinstance(func, ProblemDefaultFunc):
                    os, cs = self.evaluate_individual(vars_int, vf)
                    s = np.s_[:] if components is None else fcmpts
                    values[i] = np.r_[os, cs][s]
                    del os, cs, s 
                else:
                    results = self.apply_individual(vars_int, vf)
                    values[i] = func.calc_individual(vars_int, vf, results, fcomps)
                    del results

        # recombine results:
        gradients[:, gvars] = np.einsum("gc,vg->cv", values, coeffs[0])

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

    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int : np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_objectives,)
        cons : np.array
            The constraints values, shape: (n_constraints,)

        """
        if self._msize is not None:
            self.memory.clear()
        return self.base_problem.finalize_individual(vars_int, vars_float, verbosity)

    def finalize_population(self, vars_int, vars_float, verbosity=0):
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The final objective function values, shape: (n_pop, n_components)
        cons : np.array
            The final constraint values, shape: (n_pop, n_constraints)

        """
        if self._msize is not None:
            self.memory.clear()
        return self.base_problem.finalize_population(vars_int, vars_float, verbosity)
