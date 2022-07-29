import numpy as np

from iwopy.core import OptResults, Problem

class UDP:
    """
    Generic OptProblem to Pygmo UserDefinedProblem adapter

    Parameters
    ----------
    problem : iwopy.Problem
        The problem to optimize
    c_tol : float
        Constraint tolerance

    Attributes
    ----------
    problem : iwopy.OptProblem
        The problem to optimize
    n_vars_all :  int
        The sum of int and float variable counts
    n_fitness : int
        The sum of objective and constraint counts
    c_tol : list of float
        Constraint tolerances
    values : numpy.ndarray
        The function values, shape: (n_fitness,)

    """

    def __init__(self, 
            problem,
            c_tol=0.,
        ):

        self.problem = problem
        self.n_vars_all = problem.n_vars_float + problem.n_vars_int
        self.n_fitness = problem.n_objectives + problem.n_constraints
        
        self.c_tol = [c_tol] * problem.n_constraints

        self.values = np.zeros(self.n_fitness, dtype=np.float64)

    def apply(self, xi, xf):

        # apply new variables and calculate:
        objs, cons = self.problem.evaluate_individual(xi, xf)
        self.values[:self.problem.n_objectives] = objs
        self.values[self.problem.n_objectives:] = cons

    def fitness(self, dv):

        # extract variables:
        xf = dv[:self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float:].astype(np.int32)

        # apply new variables:
        self.apply(xi, xf)

        return self.values

    def get_bounds(self):
    
        lb = np.full(self.n_vars_all, -np.inf)
        ub = np.full(self.n_vars_all, np.inf)

        if self.problem.n_vars_float:
            lb[:self.problem.n_vars_float] = self.problem.min_values_float()
            ub[:self.problem.n_vars_float] = self.problem.max_values_float()

        if self.problem.n_vars_int:

            lbi = lb[self.problem.n_vars_float:]
            ubi = ub[self.problem.n_vars_float:]

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

        TODO

        spars = self.gradient_sparsity()
        out   = np.zeros(len(spars))
        
        dx = np.ones(self.n_vars_all, dtype=np.float64)
        dx[:self.problem.n_vars_float()] = self.deriv_fd_step
        fdabs = np.ones(self.n_vars_all, dtype=np.float64)
        fdabs[:self.problem.n_vars_float()] = ( self.deriv_fd_calc == "abs" )
        isrel = ( fdabs == 0 )
        dx[isrel] *= x[isrel]

        v2f = [[]] * self.n_vars_all
        v2o = [[]] * self.n_vars_all
        v2c = [[]] * self.n_vars_all
        v2s = [[]] * self.n_vars_all
        for vi in range(self.n_vars_all):
            for si, s in enumerate(spars):

                if s[1] == vi:
                    v2s[vi].append(si)

                    if not s[0] in v2f[vi]:
                        v2f[vi].append(s[0])

                        if s[0] < self.problem.n_objectives:
                            v2o[vi].append(s[0])
                        else:
                            v2c[vi].append(s[0] - self.problem.n_objectives)
            
        vderivs_o = []
        vderivs_c = []
        for vi in range(self.n_vars_all):

            derivs_o = []
            derivs_c = []
            if len(v2f[vi]) > 0:
                derivs_o, derivs_c = calc_derivatives_order1(self.problem, vi, dx[vi], v2o[vi], v2c[vi])
            vderivs_o.append(derivs_o)
            vderivs_c.append(derivs_c)

        for si, s in enumerate(spars):
            vi = s[1]
            if s[0] < self.problem.n_objectives:
                fi = v2o[vi].index(s[0])
                out[si] = vderivs_o[vi][fi]
            else:
                fi = v2c[vi].index(s[0] - self.problem.n_objectives)
                out[si] = vderivs_c[vi][fi]
        
        return out    

    def has_gradient_sparsity(self):
        return False

    def gradient_sparsity(self):
        
        out = []

        # add sparsity of objectives:
        out += np.argwhere(self.problem.objs.vardeps_float).tolist()
        depsi = np.argwhere(self.problem.objs.vardeps_int)
        depsi[:, 1] += self.problem.n_vars_float
        out += depsi.tolist()

        # add sparsity of constraints:
        depsf = np.argwhere(self.problem.cons.vardeps_float)
        depsf[:, 0] += self.problem.n_objectives
        out += depsf.tolist()
        depsi = np.argwhere(self.problem.objs.vardeps_int)
        depsi[:, 0] += self.problem.n_objectives
        depsi[:, 1] += self.problem.n_vars_float
        out += depsi.tolist()

        return sorted(out)

    def has_hessians(self):
        return False

    #def hessians(self, dv):

    def has_hessians_sparsity(self):
        return False
    
    #def hessians_sparsity(self):

    def has_set_seed(self):
        return False

    #def set_seed(self, s):

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
        xf = dv[:self.problem.n_vars_float]
        xi = dv[self.problem.n_vars_float:].astype(np.int32)

        # apply final variables:
        self.apply(xi, xf)

        # extract objs and cons:
        objs = self.values[:self.problem.n_objectives]
        cons = self.values[self.problem.n_objectives:]

        if verbosity: print()
        suc = np.all(self.problem.check_constraints_individual(cons, verbosity))
        if verbosity: print()

        return OptResults(suc, xi, xf, objs, cons, self.results)
