from typing import Any

import numpy as np

from iwopy.core import MultiObjOptResults, Problem, SingleObjOptResults

from . import imports


class SingleObjProblemTemplate:
    """
    Template for a wrapper around the pymoo problem
    for a single objective.

    At the moment this interface only supports
    pure int or pure float problems (not mixed).
    """

    CLASS_NAME = "SingleObjProblem"
    CLASS_DOC = "The default callback"

    def __init__(
        self,
        problem: Problem,
        vectorize: bool,
        store_prob_res: bool = False,
    ) -> None:
        """
        Parameters
        ----------
        problem
            The iwopy problem to solve
        vectorize
            Switch for vectorized calculations, wrt
            population individuals
        store_prob_res
            Whether to store current problem results
        """
        self.problem = problem
        self.vectorize = vectorize
        self.store_prob_res = store_prob_res

        self.__current_problem_results: tuple[object | None, ...] | None = None
        self._cmi = np.empty(0, dtype=np.float64)
        self._cma = np.empty(0, dtype=np.float64)

        if self.problem.n_vars_float > 0 and self.problem.n_vars_int == 0:
            self.is_mixed = False
            self.is_intprob = False

            self._pargs = {
                "n_var": self.problem.n_vars_float,
                "n_obj": self.problem.n_objectives,
                "n_ieq_constr": self.problem.n_constraints,
                "xl": self.problem.min_values_float(),
                "xu": self.problem.max_values_float(),
                "elementwise": not vectorize,
                "type_var": np.float64,
            }

        elif self.problem.n_vars_float == 0 and self.problem.n_vars_int > 0:
            self.is_mixed = False
            self.is_intprob = True

            self._pargs = {
                "n_var": self.problem.n_vars_int,
                "n_obj": self.problem.n_objectives,
                "n_ieq_constr": self.problem.n_constraints,
                "xl": self.problem.min_values_int(),
                "xu": self.problem.max_values_int(),
                "elementwise": not vectorize,
                "type_var": np.int32,
            }

        else:
            self.is_mixed = True
            self.is_intprob = False

            vars: dict[str, Any] = {}

            nami = self.problem.var_names_int()
            inii = np.asarray(self.problem.initial_values_int())
            mini = np.asarray(self.problem.min_values_int())
            maxi = np.asarray(self.problem.max_values_int())
            for i, v in enumerate(nami):
                vars[v] = imports.Integer(value=inii[i], bounds=(mini[i], maxi[i]))

            namf = self.problem.var_names_float()
            inif = np.asarray(self.problem.initial_values_float())
            minf = np.asarray(self.problem.min_values_float())
            maxf = np.asarray(self.problem.max_values_float())
            for i, v in enumerate(namf):
                vars[v] = imports.Real(value=inif[i], bounds=(minf[i], maxf[i]))

            self._pargs = {
                "vars": vars,
                "n_obj": self.problem.n_objectives,
                "n_ieq_constr": self.problem.n_constraints,
                "elementwise": not vectorize,
            }

        if self.problem.n_constraints:
            self._cmi = self.problem.min_values_constraints
            self._cma = self.problem.max_values_constraints
            cnames = np.asarray(self.problem.cons.component_names)

            sel = np.isinf(self._cmi) & np.isinf(self._cma)
            if np.any(sel):
                raise RuntimeError(f"Missing boundaries for constraints {cnames[sel]}")

            sel = (~np.isinf(self._cmi)) & (~np.isinf(self._cma))
            if np.any(sel):
                raise RuntimeError(
                    f"Constraints {cnames[sel]} have both lower and upper bounds"
                )

    @property
    def current_problem_results(self) -> tuple[object | None, ...] | None:
        """Returns the current problem results, if stored."""
        if not self.store_prob_res:
            raise RuntimeError("Current problem results are not stored.")
        return self.__current_problem_results

    def _evaluate_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, object | None]:
        if self.store_prob_res:
            return self.problem.evaluate_population(vars_int, vars_float, True)
        return self.problem.evaluate_population(vars_int, vars_float)

    def _evaluate_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, object | None]:
        if self.store_prob_res:
            return self.problem.evaluate_individual(vars_int, vars_float, True)
        return self.problem.evaluate_individual(vars_int, vars_float)

    def _evaluate(
        self,
        x: Any,
        out: dict[str, Any],
        *args: Any,
        **kwargs: Any,
    ) -> None:
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
                r = self._evaluate_population(xi, xf)
                out["F"], out["G"] = r[:2]
                out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
                if self.store_prob_res:
                    self.__current_problem_results = r[2:]
                del r
            else:
                n_pop = x.shape[0]
                if self.is_intprob:
                    dummies = np.zeros((n_pop, 0), dtype=np.float64)
                    r = self._evaluate_population(x, dummies)
                    out["F"], out["G"] = r[:2]
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
                    if self.store_prob_res:
                        self.__current_problem_results = r[2:]
                    del r
                else:
                    dummies = np.zeros((n_pop, 0), dtype=np.int32)
                    r = self._evaluate_population(dummies, x)
                    out["F"], out["G"] = r[:2]
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)[None, :]
                    if self.store_prob_res:
                        self.__current_problem_results = r[2:]
                    del r

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
                r = self._evaluate_individual(xi, xf)
                out["F"], out["G"] = r[:2]
                out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
                if self.store_prob_res:
                    self.__current_problem_results = r[2:]
                del r
            else:
                n_pop = x.shape[0]
                if self.is_intprob:
                    dummies = np.zeros(0, dtype=np.float64)
                    r = self._evaluate_individual(x, dummies)
                    out["F"], out["G"] = r[:2]
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
                    if self.store_prob_res:
                        self.__current_problem_results = r[2:]
                    del r
                else:
                    dummies = np.zeros(0, dtype=np.int32)
                    r = self._evaluate_individual(dummies, x)
                    out["F"], out["G"] = r[:2]
                    out["F"] *= np.where(self.problem.maximize_objs, -1.0, 1.0)
                    if self.store_prob_res:
                        self.__current_problem_results = r[2:]
                    del r

            if self.problem.n_constraints:
                sel = ~np.isinf(self._cma)
                out["G"][sel] = out["G"][sel] - self._cma[sel]

                sel = ~np.isinf(self._cmi)
                out["G"][sel] = self._cmi[sel] - out["G"][sel]

    def finalize(self, pymoo_results: Any, verbosity: int = 1) -> SingleObjOptResults:
        """
        Finalize the problem.

        Parameters
        ----------
        pymoo_results
            The results from the solver
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        results
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
                        [[p.X[v] for v in self.problem.var_names_int()] for p in r.pop],
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

    @classmethod
    def get_class(cls) -> type[Any]:
        """Creates the class, dynamically derived from pymoo.Problem"""
        imports.load()
        attrb: dict[str, Any] = {
            v: d
            for v, d in cls.__dict__.items()
            if v not in ["get_class", "CLASS_NAME"]
        }
        init0 = cls.__init__

        def __init(self: Any, *args: Any, **kwargs: Any) -> None:
            init0(self, *args, **kwargs)
            imports.Problem.__init__(self, **self._pargs)

        attrb["__init__"] = __init
        attrb["__doc__"] = attrb["__doc__"].replace("Template for a w", "W")
        return type(cls.CLASS_NAME, (imports.Problem,), attrb)


class MultiObjProblemTemplate:
    """
    Template for a wrapper around the pymoo problem
    for a multiple objectives problem.

    At the moment this interface only supports
    pure int or pure float problems (not mixed).
    """

    CLASS_NAME = "MultiObjProblem"

    problem: Problem
    vectorize: bool
    is_mixed: bool
    is_intprob: bool

    def __init__(self, problem: Problem, vectorize: bool) -> None:
        """
        Parameters
        ----------
        problem
            The iwopy problem to solve
        vectorize
            Switch for vectorized calculations, wrt
            population individuals
        """

    def finalize(self, pymoo_results: Any, verbosity: int = 1) -> MultiObjOptResults:
        """
        Finalize the problem.

        Parameters
        ----------
        pymoo_results
            The results from the solver
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        results
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

    @classmethod
    def get_class(cls) -> type[Any]:
        """Creates the class, dynamically derived from SingleObjProblem"""
        scls = SingleObjProblemTemplate.get_class()
        attrb: dict[str, Any] = {
            v: d
            for v, d in cls.__dict__.items()
            if v not in ["get_class", "CLASS_NAME"]
        }

        def init(self: Any, *args: Any, **kwargs: Any) -> None:
            scls.__init__(self, *args, **kwargs)

        attrb["__init__"] = init
        attrb["__doc__"] = attrb["__doc__"].replace("Template for a w", "W")
        return type(cls.CLASS_NAME, (scls,), attrb)
