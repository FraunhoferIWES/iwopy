import numpy as np

from iwopy.core import Optimizer
from iwopy.utils import suppress_stdout
from .udp import UDP
from .algos import AlgoFactory
from .import_lib import pygmo, check_import

class Optimizer_pygmo(Optimizer):
    """
    Interface to the pygmo optimizers
    for serial runs.

    Parameters
    ----------
    problem : iwopy.Problem
        The problem to optimize
    problem_pars : dict
        Parameters for the problem
    algo_pars : dict
        Parameters for the alorithm
    setup : dict, optional
        Parameters for the model setup

    Attributes
    ----------
    problem_pars: dict
        Parameters for the problem
    algo_pars: dict
        Parameters for the alorithm
    setup: dict
        Parameters for the model setup
    udp: iwopy.interfaces.pygmo.UDA
        The pygmo problem
    algo: pygmo.algo
        The pygmo algorithm

    """

    def __init__(self, problem, problem_pars, algo_pars, **setup):
        super().__init__(problem)

        check_import()

        self.problem_pars = problem_pars
        self.algo_pars = algo_pars
        self.setup = setup

        self.udp  = None
        self.algo = None

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        
        # create pygmo problem:
        self.udp = UDP(self.problem, **self.problem_pars)

        # create algorithm:
        self.algo = AlgoFactory.new(**self.algo_pars)

        # create population:
        psize = self.setup.get('pop_size', 1)
        pseed = self.setup.get('seed', None)
        pnrfi = self.setup.get('norandom_first', psize == 1)
        self.pop = pygmo.population(self.udp, size = psize, seed = pseed)
        self.pop.problem.c_tol = [self.setup.get('c_tol', 1e-10)] * self.pop.problem.get_nc()

        # memorize verbosity level:
        self.verbosity = self.setup.get('verbosity', 1)
        
        # set first indiviual to initial values:
        if pnrfi:

            x = np.zeros(self.udp.n_vars_all)

            if self.problem.n_vars_float():
                vnames, vmin, vmax, vini = self.problem.vars_float_info()
                x[:self.problem.n_vars_float()] = vini
            if self.problem.n_vars_int():
                vnames, vmin, vmax, vini = self.problem.vars_int_info()
                x[self.problem.n_vars_float():] = vini
            
            xf = x[:self.problem.n_vars_float()]
            xi = x[self.problem.n_vars_float():].astype(np.int64)

            self.udp._active = True 
            self.pop.set_x(0, x)
            self.udp.apply(xi, xf)

        super().initialize(verbosity)

    def print_info(self):
        """
        Print solver info, called before solving
        """
        super().print_info()
        if self.algo is not None:
            print()
            print(self.algo)

    def solve(self, verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.OptResults
            The optimization results object

        """

        # try pygmo silencing:
        if self.algo.has_set_verbosity():
            self.algo.set_verbosity(verbosity)

        # general silencing for Python prints:
        silent = verbosity==0
        with suppress_stdout(silent):
            
            # Run solver:
            pop = self.algo.evolve(self.pop)

        return self.udp.finalize(pop, verbosity)
