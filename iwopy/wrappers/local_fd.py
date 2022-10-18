import numpy as np

from .problem_wrapper import ProblemWrapper
from iwopy.core import ProblemDefaultFunc


class LocalFD(ProblemWrapper):
    """
    A wrapper that provides finite distance
    differentiation by local stepwise evaluation.

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
    name : str, optional
        The problem name

    Attributes
    ----------
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
        name=None,
    ):
        name = base_problem.name + "_fd" if name is None else name
        super().__init__(base_problem, name)

        if isinstance(deltas, float):
            deltas = {v: deltas for v in base_problem.var_names_float()}
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

    def initialize(self, verbosity=1):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        super().initialize(verbosity)

        self._vinds = []
        self._order = []
        self._orderb = []
        self._d = []
        vnms = list(super().var_names_float())
        for v in self._deltas.keys():

            if v not in vnms:
                raise KeyError(
                    f"Problem '{self.name}': Variable '{v}' given in deltas, but not found in problem float variables {vnms}"
                )

            vi = vnms.index(v)
            self._vinds.append(vi)
            self._order.append(self.order[v])
            self._orderb.append(self.orderb[v])
            self._d.append(self._deltas[v])

        self._order = np.array(self._order, dtype=np.int32)
        self._orderb = np.array(self._orderb, dtype=np.int32)
        self._d = np.array(self._d, dtype=np.float64)

        sel = (self._order == -1) | (self._order == 1) | (self._order == 2)
        if not np.all(sel):
            raise NotImplementedError(
                f"Order(s) {list(np.unique(self._order[~sel]))} not implemented."
            )
        sel = (self._orderb == -1) | (self._orderb == 1) | (self._orderb == 2)
        if not np.all(sel):
            raise NotImplementedError(
                f"Boundary order(s) {list(np.unique(self._orderb[~sel]))} not implemented."
            )

    def _grad_coeffs(self, varsf, gvars, order, orderb):
        """
        Helper function that provides gradient coeffs
        """

        # prepare:
        n_vars = len(gvars)
        vmin = np.array(self.min_values_float(), dtype=np.float64)[gvars]
        vmax = np.array(self.max_values_float(), dtype=np.float64)[gvars]
        x0 = varsf
        d = self._d[gvars]

        xplus = np.zeros((self.n_vars_float, self.n_vars_float), dtype=np.float64)
        np.fill_diagonal(xplus, d)
        xplus += x0[None, :]
        xminus = np.zeros((self.n_vars_float, self.n_vars_float), dtype=np.float64)
        np.fill_diagonal(xminus, -d)
        xminus += x0[None, :]
        xminus2 = None
        xplus2 = None

        pts = np.zeros((n_vars, 2, self.n_vars_float), dtype=np.float64)
        cfs = np.zeros((n_vars, n_vars, 2), dtype=np.float64)
        cf0 = np.zeros(n_vars, dtype=np.float64)

        # domain, order 1:
        sel = (order == 1) & (xplus.diagonal() <= vmax)
        if np.any(sel):
            pts[sel, 0] = xplus[sel]
            cfs[sel, sel, 0] = 1 / d[sel]
            cf0[sel] = -1 / d[sel]

        # right boundary, order 1:
        sel = (orderb == 1) & (xplus.diagonal() > vmax)
        if np.any(sel):
            pts[sel, 0] = xminus[sel]
            cfs[sel, sel, 0] = -1 / d[sel]
            cf0[sel] = 1 / d[sel]

        # domain, order -1:
        sel = (order == -1) & (xminus.diagonal() >= vmin)
        if np.any(sel):
            pts[sel, 0] = xminus[sel]
            cfs[sel, sel, 0] = -1 / d[sel]
            cf0[sel] = 1 / d[sel]

        # left boundary, order -1:
        sel = (orderb == -1) & (xminus.diagonal() < vmin)
        if np.any(sel):
            pts[sel, 0] = xplus[sel]
            cfs[sel, sel, 0] = 1 / d[sel]
            cf0[sel] = -1 / d[sel]

        # domain, order 2:
        sel = (order == 2) & (xplus.diagonal() <= vmax) & (xminus.diagonal() >= vmin)
        if np.any(sel):
            pts[sel, 0] = xplus[sel]
            pts[sel, 1] = xminus[sel]
            cfs[sel, sel, 0] = 0.5 / d[sel]
            cfs[sel, sel, 1] = -0.5 / d[sel]

        # right boundary, order 2:
        sel = (orderb == 2) & (xplus.diagonal() > vmax)
        if np.any(sel):
            if xminus2 is None:
                xminus2 = np.zeros_like(xminus)
                np.fill_diagonal(xminus2, -2 * d)
                xminus2 += x0[None, :]
            pts[sel, 0] = xminus[sel]
            pts[sel, 1] = xminus2[sel]
            cf0[sel] = 1.5 / d[sel]
            cfs[sel, sel, 0] = -2 / d[sel]
            cfs[sel, sel, 1] = 0.5 / d[sel]

        # left boundary, order 2:
        sel = (orderb == 2) & (xminus.diagonal() < vmax)
        if np.any(sel):
            if xplus2 is None:
                xplus2 = np.zeros_like(xplus)
                np.fill_diagonal(xplus2, 2 * d)
                xplus2 += x0[None, :]
            pts[sel, 0] = xplus[sel]
            pts[sel, 1] = xplus2[sel]
            cf0[sel] = -1.5 / d[sel]
            cfs[sel, sel, 0] = 2 / d[sel]
            cfs[sel, sel, 1] = -0.5 / d[sel]

        # reduce and reorganize:
        sel = np.any(np.abs(cfs) > 1e-13, axis=0)
        pts = pts[sel]
        cfs = cfs[:, sel]

        # add centre point:
        sel = np.abs(cf0) > 1e-13
        if np.any(sel):
            pts = np.append(pts, x0[None, :], axis=0)
            cfs = np.append(cfs, cf0[:, None], axis=1)

        return pts, cfs

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
        cmpnts = (
            np.arange(func.n_components())
            if components is None
            else np.array(components)
        )
        cmptsi = np.unique(np.where(np.any(gnan, axis=1))[0])
        fcmpts = cmpnts[cmptsi]
        fcomps = fcmpts if components is not None else None
        if not len(gvars) or not len(fcmpts):
            return gradients
        del gnan

        # get gradient eval points and coeffs:
        varsf = vars_float[gvars]
        order = self._order[ivars]
        orderb = self._orderb[ivars]
        epts, coeffs = self._grad_coeffs(varsf, gvars, order, orderb)

        # run the calculation:
        n_pop = len(epts)
        varsf = np.full((n_pop, self.n_vars_float), np.nan, dtype=np.float64)
        values = np.full((n_pop, len(cmptsi)), np.nan, dtype=np.float64)
        varsf[:] = vars_float[None, :]
        varsf[:, gvars] = epts
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
        gradients[:, gvars] = np.einsum("pc,vp->cv", values, coeffs)

        return gradients
