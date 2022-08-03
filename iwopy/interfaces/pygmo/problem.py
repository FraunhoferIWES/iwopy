import numpy as np

from iwopy.core import OptResults, Problem


class UDP:
    """
    Generic Problem to Pygmo UserDefinedProblem adapter

    Parameters
    ----------
    problem : iwopy.Problem
        The problem to optimize
    c_tol : float
        Constraint tolerance
    grad_pop : bool
        Vectorized gradient computation
    verbosity : int
        The verbosity level, 0 = silent

    Attributes
    ----------
    problem : iwopy.Problem
        The problem to optimize
    n_vars_all :  int
        The sum of int and float variable counts
    n_fitness : int
        The sum of objective and constraint counts
    c_tol : list of float
        Constraint tolerances
    values : numpy.ndarray
        The function values, shape: (n_fitness,)
    grad_pop : bool
        Vectorized gradient computation
    verbosity : int
        The verbosity level, 0 = silent

    """

    def __init__(
        self,
        problem,
        c_tol=0.0,
        grad_pop=False,
        verbosity=0,
    ):

        self.problem = problem
        self.n_vars_all = problem.n_vars_float + problem.n_vars_int
        self.n_fitness = problem.n_objectives + problem.n_constraints

        self.c_tol = [c_tol] * problem.n_constraints
        self.values = np.zeros(self.n_fitness, dtype=np.float64)

        self.grad_pop = grad_pop
        self.verbosity = verbosity

    def apply(self, xi, xf):

        # apply new variables and calculate:
        objs, cons = self.problem.evaluate_individual(xi, xf)
        self.values[: self.problem.n_objectives] = objs
        self.values[self.problem.n_objectives :] = cons

    def fitness(self, dv):

        # extract variables:
        xf = dv[: self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float :].astype(np.int32)

        # apply new variables:
        self.apply(xi, xf)

        return self.values

    def get_bounds(self):

        lb = np.full(self.n_vars_all, -np.inf)
        ub = np.full(self.n_vars_all, np.inf)

        if self.problem.n_vars_float:
            lb[: self.problem.n_vars_float] = self.problem.min_values_float()
            ub[: self.problem.n_vars_float] = self.problem.max_values_float()

        if self.problem.n_vars_int:

            lbi = lb[self.problem.n_vars_float :]
            ubi = ub[self.problem.n_vars_float :]

            lbi[:] = self.problem.min_values_int()
            ubi[:] = self.problem.max_values_int()

            lbi[lbi == -Problem.INT_INF] = -np.inf
            ubi[ubi == Problem.INT_INF] = np.inf

        return (lb, ub)

    def get_nobj(self):
        return self.problem.n_objectives

    def get_nec(self):
        return 0

    def get_nic(self):
        return self.problem.n_constraints

    def get_nix(self):
        return self.problem.n_vars_int

    def has_batch_fitness(self):
        return False

    def has_gradient(self):
        return True

    def gradient(self, x):

        spars = np.array(self.gradient_sparsity())
        cmpnts = np.unique(spars[:, 0])
        vrs = np.unique(spars[:, 1])

        varsf = x[: self.problem.n_vars_float]
        varsi = x[self.problem.n_vars_float :].astype(np.int32)

        grad = self.problem.get_gradients(
            varsi,
            varsf,
            vars=vrs,
            components=cmpnts,
            verbosity=self.verbosity,
            pop=self.grad_pop,
        )

        return [grad[c, list(vrs).index(v)] for c, v in spars]

    def has_gradient_sparsity(self):
        return False

    def gradient_sparsity(self):

        out = []

        # add sparsity of objectives:
        out += np.argwhere(self.problem.objs.vardeps_float()).tolist()
        if self.problem.n_vars_int:
            depsi = np.argwhere(self.problem.objs.vardeps_int())
            depsi[:, 1] += self.problem.n_vars_float
            out += depsi.tolist()

        # add sparsity of constraints:
        if self.problem.n_constraints:
            depsf = np.argwhere(self.problem.cons.vardeps_float())
            depsf[:, 0] += self.problem.n_objectives
            out += depsf.tolist()
            if self.problem.n_vars_int:
                depsi = np.argwhere(self.problem.cons.vardeps_int())
                depsi[:, 0] += self.problem.n_objectives
                depsi[:, 1] += self.problem.n_vars_float
                out += depsi.tolist()

        return sorted(out)

    def has_hessians(self):
        return False

    # def hessians(self, dv):

    def has_hessians_sparsity(self):
        return False

    # def hessians_sparsity(self):

    def has_set_seed(self):
        return False

    # def set_seed(self, s):

    def get_name(self):
        return self.problem.name

    def get_extra_info(self):
        return ""

    def finalize(self, pygmo_pop, verbosity=1):
        """
        Finalize the problem.

        Parameters
        ----------
        pygmo_pop: pygmo.Population
            The results from the solver
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.OptResults
            The optimization results object

        """

        # extract variables:
        dv = pygmo_pop.champion_x
        xf = dv[: self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float :].astype(np.int32)

        # apply final variables:
        self.apply(xi, xf)

        # extract objs and cons:
        objs = self.values[: self.problem.n_objectives]
        cons = self.values[self.problem.n_objectives :]

        if verbosity:
            print()
        suc = np.all(self.problem.check_constraints_individual(cons, verbosity))
        if verbosity:
            print()

        return OptResults(suc, xi, xf, objs, cons, None)
