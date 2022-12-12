import numpy as np
import matplotlib.pyplot as plt

from iwopy.core import Optimizer
from .problem import SingleObjProblem, MultiObjProblem
from .imports import IMPORT_OK, check_import, Callback
from .factory import Factory

if IMPORT_OK:
    from pymoo.optimize import minimize


class DefaultCallback(Callback):
    def __init__(self):
        super().__init__()
        self.data["f_best"] = None
        self.data["cv_best"] = None

    def notify(self, algorithm):
        fvals = algorithm.pop.get("F")
        cvals = algorithm.pop.get("CV")
        n_obj = fvals.shape[1]
        n_con = cvals.shape[1]
        i = np.argmin(fvals, axis=0)
        if self.data["f_best"] is None:
            self.data["f_best"] = fvals[None, i, range(n_obj)]
            self.data["cv_best"] = cvals[None, i, range(n_con)]
        else:
            self.data["f_best"] = np.append(
                self.data["f_best"], fvals[None, i, range(n_obj)], axis=0
            )
            self.data["cv_best"] = np.append(
                self.data["cv_best"], cvals[None, i, range(n_con)], axis=0
            )


class Optimizer_pymoo(Optimizer):
    """
    Interface to the pymoo optimization solver.

    Parameters
    ----------
    problem : iwopy.Problem
        The problem to optimize
    problem_pars : dict
        Parameters for the problem
    algo_pars : dict
        Parameters for the alorithm
    setup_pars : dict
        Parameters for the calculation setup

    Attributes
    ----------
    problem_pars : dict
        Parameters for the problem
    algo_pars : dict
        Parameters for the alorithm
    setup_pars : dict
        Parameters for the calculation setup
    term_pars : dict
        Parameters for the termination conditions
    pymoo_problem : iwopy.interfaces.pymoo.SingleObjProblem
        The pygmo problem
    algo : pygmo.algo
        The pygmo algorithm

    """

    def __init__(self, problem, problem_pars, algo_pars, setup_pars={}, term_pars={}):
        super().__init__(problem)

        check_import()

        self.problem_pars = problem_pars
        self.algo_pars = algo_pars
        self.setup_pars = setup_pars
        self.term_pars = term_pars

        self.pymoo_problem = None
        self.algo = None

    def print_info(self):
        """
        Print solver info, called before solving
        """
        super().print_info()

        if len(self.problem_pars):
            print("\nProblem:")
            print("--------")
            for k, v in self.problem_pars.items():
                if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    print(f"  {k}: {v}")

        if len(self.algo_pars):
            print("\nAlgorithm:")
            print("----------")
            for k, v in self.algo_pars.items():
                if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    print(f"  {k}: {v}")

        if len(self.setup_pars):
            print("\nSetup:")
            print("------")
            for k, v in self.setup_pars.items():
                if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    print(f"  {k}: {v}")

        if len(self.term_pars):
            print("\nTermination:")
            print("------------")
            for k, v in self.term_pars.items():
                if isinstance(v, int) or isinstance(v, float) or isinstance(v, str):
                    print(f"  {k}: {v}")
        print()

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if self.problem.n_objectives <= 1:
            self.pymoo_problem = SingleObjProblem(self.problem, **self.problem_pars)
        else:
            self.pymoo_problem = MultiObjProblem(self.problem, **self.problem_pars)

        if verbosity:
            print("Initializing", type(self).__name__)

        factory = Factory(self.pymoo_problem, verbosity)
        self.algo = factory.get_algorithm(self.algo_pars)
        self.term = factory.get_termination(self.term_pars)

        super().initialize(verbosity)

    def solve(self, callback=DefaultCallback(), verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        callback : pymoo.Callback, optional
            The callback
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.SingleObjOptResults or iwopy.MultiObjOptResults
            The optimization results object

        """

        # check problem initialization:
        super().solve()

        # run pymoo solver:
        self.callback = callback
        self.results = minimize(
            self.pymoo_problem,
            algorithm=self.algo,
            termination=self.term,
            verbose=verbosity > 0,
            callback=self.callback,
            **self.setup_pars,
        )

        # transfer callback:
        if self.callback is not None:
            self.callback = self.results.algorithm.callback

        return self.pymoo_problem.finalize(self.results)

    def get_figure_f(self, fig=None, ax=None, valid_dict=None, **kwargs):
        """
        Create a figure that shows the
        objective function development
        during optimization.

        The kwargs are forwarded to the
        plot command.

        Parameters
        ----------
        fig: plt.Figure, optional
            The figure to which to add the plot
        ax: plt.Axis, optional
            The axis to which to add the plot
        valid_dict: dict, optional
            Settings for the point of first valid
            solution, forwarded to scatter

        Returns
        -------
        fig: plt.Figure
            The figure

        """
        if self.problem.n_objectives() == 1:

            if fig is None and ax is None:
                fig, ax = plt.subplots()
            elif ax is None:
                ax = fig.add_subplot(111)
            else:
                raise TypeError(f"Impossible fig/ax input")

            fname = self.problem.objs[0].base_name
            fvals = self.callback.data["f_best"]
            gens = range(len(fvals))

            ax.plot(gens, fvals, label=fname, **kwargs)

            if self.problem.n_constraints():
                cv = np.array(self.callback.data["cv_best"])
                sel = cv == 0.0
                if np.any(sel):
                    i = np.argwhere(sel)[0][0]
                    vdict = {"label": "first valid", "color": "red"}
                    if valid_dict is not None:
                        vdict.update(valid_dict)
                    ax.scatter(i, fvals[i], **vdict)
                    ax.legend()

            ax.set_xlabel("n_gen")
            ax.set_ylabel(fname)

            plt.tight_layout()

            return fig

        else:
            raise NotImplementedError
