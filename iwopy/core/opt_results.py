class OptResults:
    """
    Container for optimization results.

    Parameters
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

    """

    def __init__(
        self,
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
