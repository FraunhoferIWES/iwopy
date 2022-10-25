import numpy as np


class OptResults:
    """
    Container for optimization results.

    Parameters
    ----------
    problem : iwopy.core.Problem
        The problem
    success : bool
        Optimization success
    vars_int : np.array
        Single objective case: Optimal variables, shape: (n_vars_int,)
        Multi objective case: Pareto variables, shape: (n_pop, n_vars_int)
    vars_float : np.array
        Single objective case: Optimal variables, shape: (n_vars_float,)
        Multi objective case: Pareto variables, shape: (n_pop, n_vars_float)
    objs : np.array
        Single objective case: Optimal objective function value, shape: (1,)
        Multi objective case: Pareto front objective function values,
        shape: (n_pop, n_objectives)
    cons : np.array
        Single objective case: Constraint values, shape: (1,)
        Multi objective case: Constraint values, shape: (n_pop, n_constraints)
    problem_results : Object
        The results of the variable application to the problem

    Attributes
    ----------
    success : bool
        Optimization success
    vars_int : np.array
        Single objective case: Optimal variables, shape: (n_vars_int,)
        Multi objective case: Pareto variables, shape: (n_pop, n_vars_int)
    vars_float : np.array
        Single objective case: Optimal variables, shape: (n_vars_float,)
        Multi objective case: Pareto variables, shape: (n_pop, n_vars_float)
    objs : np.array
        Single objective case: Optimal objective function value, shape: (1,)
        Multi objective case: Pareto front objective function values,
        shape: (n_pop, n_objectives)
    cons : np.array
        Single objective case: Constraint values, shape: (1,)
        Multi objective case: Constraint values, shape: (n_pop, n_constraints)
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
