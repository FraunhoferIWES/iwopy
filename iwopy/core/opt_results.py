import numpy as np
import matplotlib.pyplot as plt


class SingleObjOptResults:
    """
    Container for optimization results for single objective
    problems.

    Parameters
    ----------
    problem : iwopy.core.Problem
        The problem
    success : bool
        Optimization success
    vars_int : np.array
        Optimal variables, shape: (n_vars_int,)
    vars_float : np.array
        Optimal variables, shape: (n_vars_float,)
    objs : float
        Optimal objective function value
    cons : np.array
        Constraint values, shape: (n_constraints,)
    problem_results : Object
        The results of the variable application to the problem

    Attributes
    ----------
    success : bool
        Optimization success
    vars_int : np.array
        Optimal variables, shape: (n_vars_int,)
    vars_float : np.array
        Optimal variables, shape: (n_vars_float,)
    objs : float
        Optimal objective function value
    cons : np.array
        Constraint values, shape: (n_constraints,)
    problem_results : Object
        The results of the variable application to the problem
    pname : str
        The problem's name
    vnames_int : list of str
        The int variable names
    vnames_float : list of str
        The float variable names
    onames : list of str
        The names of objectives
    cnames : list of str
        The names of constraints

    """

    def __init__(
        self,
        problem,
        success,
        vars_int,
        vars_float,
        objs,
        cons,
        problem_results,
    ):

        self.success = success
        self.vars_int = vars_int
        self.vars_float = vars_float
        self.objs = objs
        self.cons = cons
        self.problem_results = problem_results

        self.pname = problem.name
        self.vnames_int = problem.var_names_int()
        self.vnames_float = problem.var_names_float()
        self.onames = problem.objs.component_names
        self.cnames = problem.cons.component_names

        if problem.n_objectives > 1:
            raise ValueError(
                f"Wrong opt results class '{type(self).__name__}' for multi objective problem. Use 'MultiObjOptResults' instead."
            )

    def __str__(self):

        s = f"Results problem '{self.pname}':\n"
        hline = "-" * len(s) + "\n"
        if len(self.vnames_int):
            s += hline
            L = len(max(self.vnames_int, key=len))
            s += "  Integer variables:\n"
            for i, vname in enumerate(self.vnames_int):
                s += f"    {i}: {vname:<{L}} = {self.vars_int[i]}\n"
        if self.vars_float is not None and len(self.vnames_float):
            s += hline
            L = len(max(self.vnames_float, key=len))
            s += "  Float variables:\n"
            for i, vname in enumerate(self.vnames_float):
                s += f"    {i}: {vname:<{L}} = {self.vars_float[i]:.6e}\n"
        if self.objs is not None and len(self.onames):
            s += hline
            L = len(max(self.onames, key=len))
            s += "  Objectives:\n"
            for i, vname in enumerate(self.onames):
                s += f"    {i}: {vname:<{L}} = {self.objs[i]:.6e}\n"
        if self.cons is not None and len(self.cnames):
            s += hline
            L = len(max(self.cnames, key=len))
            s += "  Constraints:\n"
            for i, vname in enumerate(self.cnames):
                s += f"    {i}: {vname:<{L}} = {self.cons[i]:.6e}\n"
        s += hline
        s += f"  Success: {self.success}\n"
        s += hline

        return s


class MultiObjOptResults:
    """
    Container for optimization results for multi objective
    problems.

    Parameters
    ----------
    problem : iwopy.core.Problem
        The problem
    success : bool
        Optimization success
    vars_int : np.array
        Pareto-optimal variables, shape: (n_pop, n_vars_int)
    vars_float : np.array
        Pareto-optimal variables, shape: (n_pop, n_vars_float)
    objs : np.array
        Pareto front objective function values, shape: (n_pop, n_objectives)
    cons : np.array
        Parteo front Constraint values, shape: (n_pop, n_constraints)
    problem_results : Object
        The results of the variable application to the problem

    Attributes
    ----------
    success : bool
        Optimization success
    vars_int : np.array
        Pareto-optimal variables, shape: (n_pop, n_vars_int)
    vars_float : np.array
        Pareto-optimal variables, shape: (n_pop, n_vars_float)
    objs : np.array
        Pareto front objective function values, shape: (n_pop, n_objectives)
    cons : np.array
        Parteo front Constraint values, shape: (n_pop, n_constraints)
    problem_results : Object
        The results of the variable application to the problem
    pname : str
        The problem's name
    vnames_int : list of str
        The int variable names
    vnames_float : list of str
        The float variable names
    onames : list of str
        The names of objectives
    cnames : list of str
        The names of constraints

    """

    def __init__(
        self,
        problem,
        success,
        vars_int,
        vars_float,
        objs,
        cons,
        problem_results,
    ):

        self.success = success
        self.vars_int = vars_int
        self.vars_float = vars_float
        self.objs = objs
        self.cons = cons
        self.problem_results = problem_results

        self.pname = problem.name
        self.vnames_int = problem.var_names_int()
        self.vnames_float = problem.var_names_float()
        self.onames = problem.objs.component_names
        self.cnames = problem.cons.component_names

        if problem.n_objectives <= 1:
            raise ValueError(
                f"Wrong opt results class '{type(self).__name__}' for single objective problem. Use 'SingleObjOptResults' instead."
            )

    def __str__(self):

        s = f"Results problem '{self.pname}':\n"
        hline = "-" * 2 * len(s) + "\n"
        if len(self.vnames_int):
            s += hline
            L = len(max(self.vnames_int, key=len))
            s += "  Integer variables:\n"
            for i, vname in enumerate(self.vnames_int):
                v = self.vars_int[:, i]
                s += f"    {i}: {vname:<{L}}: {vname:<{L}} = {np.min(v)} --> {np.max(v)}\n"
        if self.vars_float is not None and len(self.vnames_float):
            s += hline
            L = len(max(self.vnames_float, key=len))
            s += "  Float variables:\n"
            for i, vname in enumerate(self.vnames_float):
                v = self.vars_float[:, i]
                s += f"    {i}: {vname:<{L}}: {vname:<{L}} = {np.min(v):.6e} --> {np.max(v):.6e}\n"
        if self.objs is not None and len(self.onames):
            s += hline
            L = len(max(self.onames, key=len))
            s += "  Objectives:\n"
            for i, vname in enumerate(self.onames):
                v = self.objs[:, i]
                s += f"    {i}: {vname:<{L}} = {np.min(v):.6e} --> {np.max(v):.6e}\n"
        if self.cons is not None and len(self.cnames):
            s += hline
            L = len(max(self.cnames, key=len))
            s += "  Constraints:\n"
            for i, vname in enumerate(self.cnames):
                v = self.cons[:, i]
                s += f"    {i}: {vname:<{L}} = {np.min(v):.6e} --> {np.max(v):.6e}\n"
        s += hline
        v = np.sum(self.success) / len(self.success.flat)
        s += f"  Success: {100*v:.2f} %\n"
        s += hline

        return s

    def plot_pareto(
        self,
        obj_0=0,
        obj_1=1,
        ax=None,
        figsize=(5, 5),
        s=50,
        color_val="orange",
        color_ival="red",
        title=None,
    ):
        """
        Get figure that shows the pareto front

        Parameters
        ----------
        obj_0 : int
            The objective on the x axis
        obj_1 : int
            The objective on the y axis
        ax : pyplot.Axis, optional
            The axis to plot on
        figsize : tuple
            The figure size, if ax is not given
        s : float
            Scatter point size
        color_val : str
            Color choice for valid points
        color_ival : str
            Color choice for invalid points
        title : str, optional
            The plot title

        Returns
        -------
        ax : pyplot.axis
            The plot axis

        """
        if ax is None:
            __, ax = plt.subplots(figsize=figsize)

        sel = self.success
        ax.scatter(
            self.objs[sel, obj_0],
            self.objs[sel, obj_1],
            s=s,
            c=color_val,
            label="valid",
        )

        sel = ~self.success
        ax.scatter(
            self.objs[sel, obj_0],
            self.objs[sel, obj_1],
            s=s,
            c=color_ival,
            label="invalid",
        )

        if np.any(sel):
            ax.legend(loc="best")
        ax.set_xlabel(self.onames[obj_0])
        ax.set_ylabel(self.onames[obj_1])
        ax.set_title(self.pname if title is None else title)
        ax.grid()

        return ax

    def find_pareto_objmix(self, obj_weights, max=False):
        """
        Find the point on the pareto front that
        approximates best the given weights of objectives

        Paramters
        ---------
        obj_weights : list of float
            The weights of the objectives
        max : bool
            Find the maximal value of the weighted result
            (otherwise find the minimal value)

        Returns
        -------
        index : int
            The index in the pareto front results

        """
        w = np.array(obj_weights, dtype=np.float64)
        res = np.einsum("po,o->p", self.objs, w)
        return np.argmax(res) if max else np.argmin(res)
