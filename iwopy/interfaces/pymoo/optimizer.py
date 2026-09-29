import numpy as np

from iwopy.core import Optimizer, OptimizerCallback, OptimizerCallbackData

from . import imports
from .factory import Factory
from .problem import MultiObjProblemTemplate, SingleObjProblemTemplate


class _PymooCallbackTemplate:
    """Template for the internal pymoo-to-iwopy callback adapter."""

    CLASS_NAME = "IwopyCallback"
    CLASS_DOC = "Internal pymoo-to-iwopy callback adapter"

    def __init__(self, optimizer):
        self.optimizer = optimizer

    def __deepcopy__(self, memo):
        return self

    def notify(self, algorithm):
        self.optimizer._notify_pymoo_callbacks(algorithm)

    @classmethod
    def get_class(cls):
        """
        Creates the class, dynamically derived from pymoo.Callback
        """
        imports.load()
        attrb = {
            v: d
            for v, d in cls.__dict__.items()
            if v not in ["get_class", "CLASS_NAME", "CLASS_DOC"]
        }
        initialize_template = cls.__init__

        def __init(self, *args, **kwargs):
            imports.Callback.__init__(self)
            initialize_template(self, *args, **kwargs)

        attrb["__init__"] = __init
        attrb["__doc__"] = cls.CLASS_DOC
        return type(cls.CLASS_NAME, (imports.Callback,), attrb)


class Optimizer_pymoo(Optimizer):
    """
    Interface to the pymoo optimization solver.

    Attributes
    ----------
    problem_pars: dict
        Parameters for the problem
    algo_pars: dict
        Parameters for the alorithm
    setup_pars: dict
        Parameters for the calculation setup
    term_pars: dict
        Parameters for the termination conditions
    pymoo_problem: iwopy.interfaces.pymoo.SingleObjProblem
        The pygmo problem
    algo: pygmo.algo
        The pygmo algorithm

    :group: interfaces.pymoo

    """

    def __init__(
        self, problem, problem_pars, algo_pars, setup_pars=None, term_pars=None
    ):
        """
        Constructor

        Parameters
        ----------
        problem: iwopy.Problem
            The problem to optimize
        problem_pars: dict
            Parameters for the problem
        algo_pars: dict
            Parameters for the alorithm
        setup_pars: dict
            Parameters for the calculation setup

        """
        if term_pars is None:
            term_pars = {}
        if setup_pars is None:
            setup_pars = {}
        super().__init__(problem)

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

        for k, v in self.problem_pars.items():
            if isinstance(v, (int, float, str)):
                print(f"  {k}: {v}")

        if len(self.algo_pars):
            print("\nAlgorithm:")
            print("----------")
            for k, v in self.algo_pars.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")

        if len(self.setup_pars):
            print("\nSetup:")
            print("------")
            for k, v in self.setup_pars.items():
                if isinstance(v, (int, float, str)):
                    print(f"  {k}: {v}")

        if len(self.term_pars):
            print("\nTermination:")
            print("------------")
            if isinstance(self.term_pars, (tuple, list)):
                print(f"  {self.term_pars[0]}: {self.term_pars[1]}")
            else:
                for k, v in self.term_pars.items():
                    if isinstance(v, (int, float, str)):
                        print(f"  {k}: {v}")
        print()

    def initialize(self, verbosity=1):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity: int
            The verbosity level, 0 = silent

        """
        if "callback" in self.setup_pars:
            raise ValueError(
                f"Optimizer '{self.name}': pymoo callback is managed internally."
            )
        if self.problem.n_objectives <= 1:
            self.pymoo_problem = SingleObjProblemTemplate.get_class()(
                self.problem, **self.problem_pars
            )
        else:
            self.pymoo_problem = MultiObjProblemTemplate.get_class()(
                self.problem, **self.problem_pars
            )

        if verbosity:
            print("Initializing", type(self).__name__)

        factory = Factory(self.pymoo_problem, verbosity)
        self.algo = factory.get_algorithm(self.algo_pars)
        self.term = factory.get_termination(self.term_pars)

        super().initialize(verbosity)

    def _callback_variables(self, values):
        """Convert a pymoo population to iwopy variable arrays."""
        n_pop = len(values)
        if self.pymoo_problem.is_mixed:
            vars_int = np.array(
                [
                    [entry[name] for name in self.problem.var_names_int()]
                    for entry in values
                ],
                dtype=np.int32,
            )
            vars_float = np.array(
                [
                    [entry[name] for name in self.problem.var_names_float()]
                    for entry in values
                ],
                dtype=np.float64,
            )
        elif self.pymoo_problem.is_intprob:
            vars_int = np.asarray(values, dtype=np.int32)
            vars_float = np.zeros((n_pop, 0), dtype=np.float64)
        else:
            vars_int = np.zeros((n_pop, 0), dtype=np.int32)
            vars_float = np.asarray(values, dtype=np.float64)
        return vars_int, vars_float

    def _callback_constraints(self, values, n_pop):
        """Restore iwopy constraint values from pymoo's convention."""
        if not self.problem.n_constraints:
            return np.zeros((n_pop, 0), dtype=np.float64)

        transformed = np.asarray(values, dtype=np.float64)
        constraints = np.empty_like(transformed)
        has_upper = np.isfinite(self.pymoo_problem._cma)
        has_lower = np.isfinite(self.pymoo_problem._cmi)
        constraints[:, has_upper] = (
            transformed[:, has_upper] + self.pymoo_problem._cma[None, has_upper]
        )
        constraints[:, has_lower] = (
            self.pymoo_problem._cmi[None, has_lower] - transformed[:, has_lower]
        )
        return constraints

    def _notify_pymoo_callbacks(self, algorithm):
        """Normalize and dispatch one completed pymoo generation."""
        population = algorithm.pop
        vars_int, vars_float = self._callback_variables(population.get("X"))
        objectives = np.asarray(population.get("F"), dtype=np.float64)
        objectives *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
        constraints = self._callback_constraints(population.get("G"), len(objectives))
        self._notify_callbacks(
            OptimizerCallbackData(
                event="iteration",
                iteration=int(algorithm.n_gen),
                n_evaluations=int(algorithm.evaluator.n_eval),
                vars_int=vars_int,
                vars_float=vars_float,
                objs=objectives,
                cons=constraints,
            )
        )

    def solve(
        self,
        verbosity: int = 1,
        callbacks: list[OptimizerCallback] | None = None,
    ):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        callbacks
            Ordered callbacks for completed pymoo generations

        Returns
        -------
        results: iwopy.SingleObjOptResults or iwopy.MultiObjOptResults
            The optimization results object

        """
        # check problem initialization:
        super().solve(verbosity, callbacks)

        # run pymoo solver:
        setup_pars = self.setup_pars.copy()
        if self._has_callbacks:
            setup_pars["callback"] = _PymooCallbackTemplate.get_class()(self)
        self.results = imports.minimize(
            self.pymoo_problem,
            algorithm=self.algo,
            termination=self.term,
            verbose=verbosity > 0,
            **setup_pars,
        )

        results = self.pymoo_problem.finalize(self.results)
        return self._finalize_callbacks(results)
