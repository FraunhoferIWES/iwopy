from collections.abc import Sequence

import numpy as np

from iwopy.core import (
    OptFunction,
    Problem,
    ProblemDefaultFunc,
)

from .problem_wrapper import ProblemWrapper


class LocalFD(ProblemWrapper):
    """
    A wrapper that provides finite distance
    differentiation by local stepwise evaluation.

    When population evaluation is enabled, only function components without
    analytical derivatives are calculated at the finite-difference points.
    """

    def __init__(
        self,
        base_problem: Problem,
        deltas: float | dict[str, float],
        fd_order: int | dict[str, int] = 1,
        fd_bounds_order: int | dict[str, int] | None = None,
        max_population_size: int | None = None,
        name: str | None = None,
    ) -> None:
        """
        Parameters
        ----------
        base_problem
            The underlying concrete problem
        deltas
            The step sizes. Key: variable name str,
            Value: step size. Will be adjusted to the
            variable bounds if necessary.
        fd_order
            Finite difference order. Either a dict with
            key: variable name str, value: order int, or
            a global integer order for all variables.
            1 = forward, -1 = backward, 2 = centre
        fd_bounds_order
            Finite difference order of boundary points.
            Either a dict with key: variable name str,
            value: order int, or a global integer order
            for all variables. Default is same as fd_order
        max_population_size
            Maximum number of finite-difference points evaluated in one
            vectorized population call, or ``None`` for one unbounded call.
        name
            The problem name
        """
        name = base_problem.name + "_fd" if name is None else name
        super().__init__(base_problem, name)

        if max_population_size is not None:
            if not isinstance(max_population_size, (int, np.integer)):
                raise TypeError("max_population_size must be an integer or None")
            if max_population_size < 1:
                raise ValueError("max_population_size must be positive")
            max_population_size = int(max_population_size)
        self.max_population_size = max_population_size

        if isinstance(deltas, float):
            deltas = {v: deltas for v in base_problem.var_names_float()}
        self._deltas = deltas

        if isinstance(fd_order, int):
            self.order = {v: fd_order for v in deltas}
        else:
            self.order = fd_order
            for v in deltas:
                if v not in self.order:
                    raise KeyError(
                        f"Problem '{self.name}': Missing fd_order entry for variable '{v}'"
                    )

        if fd_bounds_order is None:
            self.orderb = {v: abs(o) for v, o in self.order.items()}
        elif isinstance(fd_bounds_order, int):
            self.orderb = {v: fd_bounds_order for v in deltas}
        else:
            self.orderb = fd_bounds_order
            for v in deltas:
                if v not in self.orderb:
                    raise KeyError(
                        f"Problem '{self.name}': Missing fd_bounds_order entry for variable '{v}'"
                    )

        self._vinds: list[int] = []
        self._order = np.empty(0, dtype=np.int32)
        self._orderb = np.empty(0, dtype=np.int32)
        self._d = np.empty(0, dtype=np.float64)

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """
        super().initialize(verbosity)

        vinds: list[int] = []
        order: list[int] = []
        orderb: list[int] = []
        deltas: list[float] = []
        vnms = list(super().var_names_float())
        for v in self._deltas:
            if v not in vnms:
                raise KeyError(
                    f"Problem '{self.name}': Variable '{v}' given in deltas, but not found in problem float variables {vnms}"
                )

            vi = vnms.index(v)
            vinds.append(vi)
            order.append(self.order[v])
            orderb.append(self.orderb[v])
            deltas.append(self._deltas[v])

        self._vinds = vinds
        self._order = np.array(order, dtype=np.int32)
        self._orderb = np.array(orderb, dtype=np.int32)
        self._d = np.array(deltas, dtype=np.float64)

        sel = (self._order == -1) | (self._order == 1) | (self._order == 2)
        if not np.all(sel):
            raise NotImplementedError(
                f"Order(s) {list(np.unique(self._order[~sel]))} not implemented."
            )

    def _calc_population_values(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        components: np.ndarray,
    ) -> np.ndarray:
        """Evaluate finite-difference points in bounded population batches."""
        values = np.full((len(vars_float), len(components)), np.nan, dtype=np.float64)
        batch_size = self.max_population_size or max(len(vars_float), 1)
        for start in range(0, len(vars_float), batch_size):
            stop = min(start + batch_size, len(vars_float))
            batch_float = vars_float[start:stop]
            batch_int = np.zeros((stop - start, self.n_vars_int), dtype=np.int32)
            if self.n_vars_int:
                batch_int[:] = vars_int[None, :]
            results = self.apply_population(batch_int, batch_float)
            if isinstance(func, ProblemDefaultFunc):
                fvarsi, fvarsf = self._find_vars(batch_int, batch_float, func)
                values[start:stop] = func.calc_population(
                    fvarsi, fvarsf, results, components=components
                )
            else:
                values[start:stop] = func.calc_population(
                    batch_int, batch_float, results, components
                )
        return values
        sel = (self._orderb == -1) | (self._orderb == 1) | (self._orderb == 2)
        if not np.all(sel):
            raise NotImplementedError(
                f"Boundary order(s) {list(np.unique(self._orderb[~sel]))} not implemented."
            )

    def _grad_coeffs(
        self,
        varsf: np.ndarray,
        gvars: Sequence[int] | np.ndarray,
        order: np.ndarray,
        orderb: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Helper function that provides gradient coeffs"""

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
        sel = (orderb == 2) & (xminus.diagonal() < vmin)
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
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        components: Sequence[int] | np.ndarray | None,
        ivars: list[int],
        fvars: list[int],
        vrs: list[int],
        pop: bool = False,
        verbosity: int = 0,
        func_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        The actual gradient calculation, not to be called directly
        (call `get_gradients` instead).

        Can be overloaded in derived classes, the base class only considers
        analytic derivatives.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        func
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        components
            The function's component selection, or None for all
        ivars
            The indices of the function int variables in the problem
        fvars
            The indices of the function float variables in the problem
        vrs
            The function float variable indices wrt which the
            derivatives are to be calculated
        func_values
            Previously calculated function values at the given variables,
            shape: (n_components,)
        pop
            Flag for vectorizing calculations via population
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        gradients
            The gradients of the functions, shape:
            (n_components, n_vrs)
        """
        # get analytic gradient results:
        gradients = super().calc_gradients(
            vars_int,
            vars_float,
            func,
            components,
            ivars,
            fvars,
            vrs,
            verbosity=verbosity,
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
        if not len(gvars) or not len(fcmpts):
            return gradients
        del gnan

        # get gradient eval points and coeffs:
        varsf = vars_float[gvars]
        order = self._order[ivars]
        orderb = self._orderb[ivars]
        epts, coeffs = self._grad_coeffs(varsf, gvars, order, orderb)

        center_coeffs = None
        if func_values is not None and len(epts) and np.array_equal(epts[-1], varsf):
            func_values = np.asarray(func_values, dtype=np.float64)
            n_cmpnts = gradients.shape[0]
            if func_values.shape != (n_cmpnts,):
                raise ValueError(
                    f"Problem '{self.name}': Expected func_values shape "
                    f"{(n_cmpnts,)}, received {func_values.shape}."
                )
            center_coeffs = coeffs[:, -1]
            epts = epts[:-1]
            coeffs = coeffs[:, :-1]

        # run the calculation:
        n_pop = len(epts)
        varsf = np.full((n_pop, self.n_vars_float), np.nan, dtype=np.float64)
        values = np.full((n_pop, len(cmptsi)), np.nan, dtype=np.float64)
        varsf[:] = vars_float[None, :]
        varsf[:, gvars] = epts
        if pop:
            values[:] = self._calc_population_values(vars_int, varsf, func, fcmpts)
        else:
            for i, vf in enumerate(varsf):
                if isinstance(func, ProblemDefaultFunc):
                    os, cs = self.evaluate_individual(vars_int, vf)
                    values[i] = np.r_[os, cs][fcmpts]
                    del os, cs
                else:
                    results = self.apply_individual(vars_int, vf)
                    values[i] = func.calc_individual(vars_int, vf, results, fcmpts)
                    del results

        # recombine results:
        gradients[np.ix_(cmptsi, gvars)] = np.einsum("pc,vp->cv", values, coeffs)
        if center_coeffs is not None:
            assert func_values is not None
            gradients[np.ix_(cmptsi, gvars)] += (
                func_values[fcmpts, None] * center_coeffs[None, :]
            )

        return gradients
