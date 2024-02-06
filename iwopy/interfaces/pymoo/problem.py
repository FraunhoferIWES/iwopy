import numpy as np

from iwopy.core import SingleObjOptResults, MultiObjOptResults
from . import imports


def get_single_obj_class(verbosity=1):

    imports.load(verbosity)

    class SingleObjProblem(imports.Problem):
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
        is_mixed : bool
            Flag for mixed integer/float problems
        is_intprob : bool
            Flag for integer problems

        """

        def __init__(self, problem, vectorize):

            self.problem = problem
            self.vectorize = vectorize

            if self.problem.n_vars_float > 0 and self.problem.n_vars_int == 0:

                self.is_mixed = False
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

                self.is_mixed = False
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

                self.is_mixed = True
                self.is_intprob = False

                vars = {}

                nami = self.problem.var_names_int()
                inii = self.problem.initial_values_int()
                mini = self.problem.min_values_int()
                maxi = self.problem.max_values_int()
                for i, v in enumerate(nami):
                    vars[v] = imports.Integer(value=inii[i], bounds=(mini[i], maxi[i]))

                namf = self.problem.var_names_float()
                inif = self.problem.initial_values_float()
                minf = self.problem.min_values_float()
                maxf = self.problem.max_values_float()
                for i, v in enumerate(namf):
                    vars[v] = imports.Real(value=inif[i], bounds=(minf[i], maxf[i]))

                super().__init__(
                    vars=vars,
                    n_obj=self.problem.n_objectives,
                    n_ieq_constr=self.problem.n_constraints,
                    elementwise=not vectorize,
                )

            if self.problem.n_constraints:

                self._cmi = self.problem.min_values_constraints
                self._cma = self.problem.max_values_constraints
                cnames = self.problem.cons.component_names

                sel = np.isinf(self._cmi) & np.isinf(self._cma)
                if np.any(sel):
                    raise RuntimeError(
                        f"Missing boundaries for constraints {cnames[sel]}"
                    )

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

                if self.is_mixed:
                    xi = np.array(
                        [[dct[v] for v in self.problem.var_names_int()] for dct in x],
                        dtype=np.int32,
                    )
                    xf = np.array(
                        [[dct[v] for v in self.problem.var_names_float()] for dct in x],
                        dtype=np.float64,
                    )
                    out["F"], out["G"] = self.problem.evaluate_population(xi, xf)
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
                else:
                    n_pop = x.shape[0]
                    if self.is_intprob:
                        dummies = np.zeros((n_pop, 0), dtype=np.float64)
                        out["F"], out["G"] = self.problem.evaluate_population(
                            x, dummies
                        )
                        out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[
                            None, :
                        ]
                    else:
                        dummies = np.zeros((n_pop, 0), dtype=np.int32)
                        out["F"], out["G"] = self.problem.evaluate_population(
                            dummies, x
                        )
                        out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[
                            None, :
                        ]

                if self.problem.n_constraints:

                    sel = ~np.isinf(self._cma)
                    out["G"][:, sel] = out["G"][:, sel] - self._cma[None, sel]

                    sel = ~np.isinf(self._cmi)
                    out["G"][:, sel] = self._cmi[None, sel] - out["G"][:, sel]

            # individual run:
            else:

                if self.is_mixed:
                    xi = np.array(
                        [x[v] for v in self.problem.var_names_int()], dtype=np.int32
                    )
                    xf = np.array(
                        [x[v] for v in self.problem.var_names_float()], dtype=np.float64
                    )
                    out["F"], out["G"] = self.problem.evaluate_individual(xi, xf)
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
                else:
                    n_pop = x.shape[0]
                    if self.is_intprob:
                        dummies = np.zeros(0, dtype=np.float64)
                        out["F"], out["G"] = self.problem.evaluate_individual(
                            x, dummies
                        )
                        out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
                    else:
                        dummies = np.zeros(0, dtype=np.int32)
                        out["F"], out["G"] = self.problem.evaluate_individual(
                            dummies, x
                        )
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
                if self.is_mixed:
                    xi = np.array(
                        [r.X[v] for v in self.problem.var_names_int()], dtype=np.int32
                    )
                    xf = np.array(
                        [r.X[v] for v in self.problem.var_names_float()],
                        dtype=np.float64,
                    )
                else:
                    if self.is_intprob:
                        xi = r.X
                        xf = np.zeros(0, dtype=np.float64)
                    else:
                        xi = np.zeros(0, dtype=np.int32)
                        xf = np.array(r.X, dtype=np.float64)

                if self.vectorize:

                    if self.is_mixed:
                        pxi = np.array(
                            [
                                [p.X[v] for v in self.problem.var_names_int()]
                                for p in r.pop
                            ],
                            dtype=np.int32,
                        )
                        pxf = np.array(
                            [
                                [p.X[v] for v in self.problem.var_names_float()]
                                for p in r.pop
                            ],
                            dtype=np.float64,
                        )
                        self.problem.finalize_population(pxi, pxf, verbosity)
                        del pxi, pxf

                    else:
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

    return SingleObjProblem


def get_multi_obj_class(verbosity=1):

    class MultiObjProblem(get_single_obj_class(verbosity)):
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
                suc = np.all(
                    self.problem.check_constraints_population(cons, False), axis=1
                )
                if verbosity:
                    print()

            return MultiObjOptResults(self.problem, suc, xi, xf, objs, cons, res)

    return MultiObjProblem
