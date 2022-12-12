import numpy as np

from iwopy.core import SingleObjOptResults, MultiObjOptResults
from .imports import check_import, Problem


class SingleObjProblem(Problem):
    """
    Wrapper around the pymoo problem for a single
    objective.

    At the moment this interface only supports
    pure int or pure float problems (not mixed).

    Parameters
    ----------
    problem : iwopy.core.Problem
        The iwopy problem to solve
    vectorize : bool, optional
        Switch for vectorized calculations, wrt
        population individuals

    Attributes
    ----------
    problem : iwopy.core.Problem
        The iwopy problem to solve
    vectorize : bool
        Switch for vectorized calculations, wrt
        population individuals
    is_intprob : bool
        Flag for integer problems

    """

    def __init__(self, problem, vectorize):

        check_import()

        self.problem = problem
        self.vectorize = vectorize

        if self.problem.n_vars_float > 0 and self.problem.n_vars_int == 0:

            self.is_intprob = False

            super().__init__(
                n_var=self.problem.n_vars_float,
                n_obj=self.problem.n_objectives,
                n_ieq_constr=self.problem.n_constraints,
                xl=self.problem.min_values_float(),
                xu=self.problem.max_values_float(),
                elementwise=not vectorize,
                type_var=np.float64,
            )

        elif self.problem.n_vars_float == 0 and self.problem.n_vars_int > 0:

            self.is_intprob = True

            super().__init__(
                n_var=self.problem.n_vars_int,
                n_obj=self.problem.n_objectives,
                n_ieq_constr=self.problem.n_constraints,
                xl=self.problem.min_values_int(),
                xu=self.problem.max_values_int(),
                elementwise=not vectorize,
                type_var=np.int32,
            )

        else:
            raise NotImplementedError(
                "Interface not implemented for mixed int/float problems."
            )

        if self.problem.n_constraints:

            self._cmi = self.problem.min_values_constraints
            self._cma = self.problem.max_values_constraints
            cnames = self.problem.cons.component_names

            sel = np.isinf(self._cmi) & np.isinf(self._cma)
            if np.any(sel):
                raise RuntimeError(f"Missing boundaries for constraints {cnames[sel]}")

            sel = (~np.isinf(self._cmi)) & (~np.isinf(self._cma))
            if np.any(sel):
                raise RuntimeError(
                    f"Constraints {cnames[sel]} have both lower and upper bounds"
                )

    def _evaluate(self, x, out, *args, **kwargs):
        """
        Overloading the abstract evaluation function
        of the pymoo base class.
        """

        # vectorized run:
        if self.vectorize:

            n_pop = x.shape[0]

            if self.is_intprob:
                dummies = np.zeros((n_pop, 0), dtype=np.float64)
                out["F"], out["G"] = self.problem.evaluate_population(x, dummies)
                out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
            else:
                dummies = np.zeros((n_pop, 0), dtype=np.int32)
                out["F"], out["G"] = self.problem.evaluate_population(dummies, x)
                out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]

            if self.problem.n_constraints:

                sel = ~np.isinf(self._cma)
                out["G"][:, sel] = out["G"][:, sel] - self._cma[None, sel]

                sel = ~np.isinf(self._cmi)
                out["G"][:, sel] = self._cmi[None, sel] - out["G"][:, sel]

        # individual run:
        else:

            if self.is_intprob:
                dummies = np.zeros(0, dtype=np.float64)
                for i in range(n_pop):
                    out["F"], out["G"] = self.problem.evaluate_individual(x, dummies)
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
            else:
                dummies = np.zeros(0, dtype=np.int32)
                out["F"], out["G"] = self.problem.evaluate_individual(dummies, x)
                out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)

            if self.problem.n_constraints:

                sel = ~np.isinf(self._cma)
                out["G"][sel] = out["G"][sel] - self._cma[sel]

                sel = ~np.isinf(self._cmi)
                out["G"][sel] = self._cmi[sel] - out["G"][sel]

    def finalize(self, pymoo_results, verbosity=1):
        """
        Finalize the problem.

        Parameters
        ----------
        pymoo_results: pymoo.Results
            The results from the solver
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.SingleObjOptResults
            The optimization results object

        """

        # prepare:
        r = pymoo_results
        suc = True

        # case no solution from pymoo:
        if r.X is None:
            suc = False
            xi = None
            xf = None
            res = None
            objs = None
            cons = None

        # evaluate pymoo final solution:
        else:
            if self.is_intprob:
                xi = r.X
                xf = np.zeros(0, dtype=np.float64)
            else:
                xi = np.zeros(0, dtype=int)
                xf = np.array(r.X, dtype=np.float64)

            if self.vectorize:

                n_pop = len(r.pop)
                n_vars = len(r.X)
                vars = np.zeros((n_pop, n_vars), dtype=np.float64)
                for pi, p in enumerate(r.pop):
                    vars[pi] = p.X

                if self.is_intprob:
                    dummies = np.zeros((n_pop, 0), dtype=np.int32)
                    self.problem.finalize_population(
                        vars.astype(np.int32), dummies, verbosity
                    )
                else:
                    dummies = np.zeros((n_pop, 0), dtype=np.float64)
                    self.problem.finalize_population(dummies, vars, verbosity)

            res, objs, cons = self.problem.finalize_individual(xi, xf, verbosity)

            if verbosity:
                print()
            suc = np.all(self.problem.check_constraints_individual(cons, False))
            if verbosity:
                print()

        return SingleObjOptResults(self.problem, suc, xi, xf, objs, cons, res)


class MultiObjProblem(SingleObjProblem):
    """
    Wrapper around the pymoo problem for a multiple
    objectives.

    At the moment this interface only supports
    pure int or pure float problems (not mixed).

    Parameters
    ----------
    problem : iwopy.core.Problem
        The iwopy problem to solve
    vectorize : bool, optional
        Switch for vectorized calculations, wrt
        population individuals

    Attributes
    ----------
    problem : iwopy.core.Problem
        The iwopy problem to solve
    vectorize : bool
        Switch for vectorized calculations, wrt
        population individuals
    is_intprob : bool
        Flag for integer problems

    """

    def __init__(self, problem, vectorize):
        super().__init__(problem, vectorize)

    def finalize(self, pymoo_results, verbosity=1):
        """
        Finalize the problem.

        Parameters
        ----------
        pymoo_results: pymoo.Results
            The results from the solver
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.SingleObjOptResults
            The optimization results object

        """

        # prepare:
        r = pymoo_results
        suc = True

        # case no solution from pymoo:
        if r.X is None:
            suc = False
            xi = None
            xf = None
            res = None
            objs = None
            cons = None

        # evaluate pymoo final solution:
        else:
            n_pop = len(r.pop)

            if self.is_intprob:
                xi = r.X
                xf = np.zeros((n_pop, 0), dtype=np.float64)
            else:
                xi = np.zeros((n_pop, 0), dtype=int)
                xf = r.X

            res, objs, cons = self.problem.finalize_population(xi, xf, verbosity)
            if verbosity:
                print()
            suc = np.all(self.problem.check_constraints_population(cons, False), axis=1)
            if verbosity:
                print()

        return MultiObjOptResults(self.problem, suc, xi, xf, objs, cons, res)
