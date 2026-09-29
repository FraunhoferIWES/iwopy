import fnmatch
from abc import ABCMeta
from collections.abc import Callable, Hashable, Mapping, Sequence
from typing import Literal, overload

import numpy as np
from numpy.typing import ArrayLike

from iwopy.utils import RegularDiscretizationGrid, new_instance

from .base import Base
from .constraint import Constraint
from .function import OptFunction
from .function_list import OptFunctionList
from .memory import Memory
from .objective import Objective


class Problem(Base, metaclass=ABCMeta):
    """Abstract base class for optimization problems."""

    INT_INF = RegularDiscretizationGrid.INT_INF

    def __init__(
        self,
        name: str,
        mem_size: int | None = None,
        mem_keyf: Callable[[np.ndarray, np.ndarray], Hashable] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        name
            The problem's name
        mem_size
            The memory size, default no memory
        mem_keyf
            The memory key function. Parameters:
            (vars_int, vars_float), returns key Object
        """
        super().__init__(name)

        self.objs: OptFunctionList[Objective] = OptFunctionList(self, "objs")
        self.cons: OptFunctionList[Constraint] = OptFunctionList(self, "cons")

        self.memory: Memory | None = None
        self._mem_size = mem_size
        self._mem_keyf = mem_keyf

        self._cons_mi: np.ndarray | None = None
        self._cons_ma: np.ndarray | None = None
        self._cons_tol: np.ndarray | None = None
        self._maximize: np.ndarray | None = None

    def var_names_int(self) -> list[str]:
        """
        The names of integer variables.

        Returns
        -------
        names
            The names of the integer variables
        """
        return []

    def initial_values_int(self) -> ArrayLike:
        """
        The initial values of the integer variables.

        Returns
        -------
        values
            Initial int values, shape: (n_vars_int,)
        """
        return 0

    def min_values_int(self) -> ArrayLike:
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values
            Minimal int values, shape: (n_vars_int,)
        """
        return -self.INT_INF

    def max_values_int(self) -> ArrayLike:
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values
            Maximal int values, shape: (n_vars_int,)
        """
        return self.INT_INF

    @property
    def n_vars_int(self) -> int:
        """
        The number of int variables

        Returns
        -------
        n
            The number of int variables
        """
        return len(self.var_names_int())

    def var_names_float(self) -> list[str]:
        """
        The names of float variables.

        Returns
        -------
        names
            The names of the float variables
        """
        return []

    def initial_values_float(self) -> ArrayLike | None:
        """
        The initial values of the float variables.

        Returns
        -------
        values
            Initial float values, shape: (n_vars_float,)
        """
        return None

    def min_values_float(self) -> ArrayLike:
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values
            Minimal float values, shape: (n_vars_float,)
        """
        return -np.inf

    def max_values_float(self) -> ArrayLike:
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values
            Maximal float values, shape: (n_vars_float,)
        """
        return np.inf

    @property
    def n_vars_float(self) -> int:
        """
        The number of float variables

        Returns
        -------
        n
            The number of float variables
        """
        return len(self.var_names_float())

    def _apply_varmap(
        self,
        vtype: str,
        function: OptFunction,
        function_type: str,
        varmap: Mapping[str | int, str | int | np.integer] | None,
    ) -> None:
        """
        Helper function for mapping function variables
        to problem variables
        """
        if varmap is None:
            return

        pnms = (
            list(self.var_names_int())
            if vtype == "int"
            else list(self.var_names_float())
        )

        vmap = {}
        for fv, pv in varmap.items():
            if isinstance(pv, str):
                pvl = fnmatch.filter(pnms, pv)
                if len(pvl) == 0:
                    raise ValueError(
                        f"Problem '{self.name}': {vtype} varmap rule '{fv} --> {pv}' failed for {function_type} '{function.name}', pattern '{pv}' not found among problem {vtype} variables {pnms}"
                    )
                elif len(pvl) > 1:
                    raise ValueError(
                        f"Problem '{self.name}': Require unique match of {vtype} variable '{fv}' of {function_type} '{function.name}' to problem variables, found {pvl} for pattern '{pv}'"
                    )
                else:
                    vmap[fv] = pvl[0]
            elif isinstance(pv, (int, np.integer)):
                index = int(pv)
                if index < 0 or index >= len(pnms):
                    raise ValueError(
                        f"Problem '{self.name}': varmap rule '{fv} --> {index}' cannot be applied for {len(pnms)} {vtype} variables {pnms}"
                    )
                vmap[fv] = pnms[index]
            else:
                raise ValueError(
                    f"Problem '{self.name}': varmap_{vtype} target variable in '{fv} --> {pv}' of {function_type} '{function.name}' is neither str nor int"
                )

        if vtype == "int":
            function.rename_vars_int(vmap)
        else:
            function.rename_vars_float(vmap)

    def add_objective(
        self,
        objective: Objective,
        varmap_int: Mapping[str | int, str | int | np.integer] | None = None,
        varmap_float: Mapping[str | int, str | int | np.integer] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Add an objective to the problem.

        Parameters
        ----------
        objective
            The objective
        varmap_int
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        varmap_float
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        verbosity
            The verbosity level, 0 = silent
        """
        if not objective.initialized:
            objective.initialize(verbosity)
        self._apply_varmap("int", objective, "objective", varmap_int)
        self._apply_varmap("float", objective, "objective", varmap_float)
        self.objs.append(objective)

    def add_constraint(
        self,
        constraint: Constraint,
        varmap_int: Mapping[str | int, str | int | np.integer] | None = None,
        varmap_float: Mapping[str | int, str | int | np.integer] | None = None,
        verbosity: int = 0,
    ) -> None:
        """
        Add a constraint to the problem.

        Parameters
        ----------
        constraint
            The constraint
        varmap_int
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        varmap_float
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        verbosity
            The verbosity level, 0 = silent
        """
        if not constraint.initialized:
            constraint.initialize(verbosity)
        self._apply_varmap("int", constraint, "constraint", varmap_int)
        self._apply_varmap("float", constraint, "constraint", varmap_float)
        self.cons.append(constraint)

        cmi, cma = constraint.get_bounds()
        ctol = np.zeros(constraint.n_components(), dtype=np.float64)
        ctol[:] = constraint.tol
        if self._cons_mi is None:
            self._cons_mi = cmi
            self._cons_ma = cma
            self._cons_tol = ctol
        else:
            self._cons_mi = np.append(self._cons_mi, cmi, axis=0)
            self._cons_ma = np.append(self._cons_ma, cma, axis=0)
            self._cons_tol = np.append(self._cons_tol, ctol, axis=0)

    @property
    def min_values_constraints(self) -> np.ndarray | None:
        """
        Gets the minimal values of constraints

        Returns
        -------
        cmi
            The minimal constraint values, shape: (n_constraints,)
        """
        return self._cons_mi

    @property
    def max_values_constraints(self) -> np.ndarray | None:
        """
        Gets the maximal values of constraints

        Returns
        -------
        cma
            The maximal constraint values, shape: (n_constraints,)
        """
        return self._cons_ma

    @property
    def constraints_tol(self) -> np.ndarray | None:
        """
        Gets the tolerance values of constraints

        Returns
        -------
        ctol
            The constraint tolerance values, shape: (n_constraints,)
        """
        return self._cons_tol

    @property
    def n_objectives(self) -> int:
        """
        The total number of objectives,
        i.e., the sum of all components

        Returns
        -------
        n_obj
            The total number of objective
            functions
        """
        return self.objs.n_components()

    @property
    def n_constraints(self) -> int:
        """
        The total number of constraints,
        i.e., the sum of all components

        Returns
        -------
        n_con
            The total number of constraint
            functions
        """
        return self.cons.n_components()

    @overload
    def _find_vars(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        ret_inds: Literal[False] = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def _find_vars(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        ret_inds: Literal[True],
    ) -> tuple[list[int], list[int]]: ...

    def _find_vars(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        ret_inds: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[list[int], list[int]]:
        """
        Helper function for reducing problem variables
        to function variables
        """
        vnmsi = list(self.var_names_int())
        vnmsf = list(self.var_names_float())
        ivars = []
        for v in func.var_names_int:
            if v not in vnmsi:
                raise ValueError(
                    f"Problem '{self.name}': int variable '{v}' of function '{func.name}' not among int problem variables {vnmsi}"
                )
            ivars.append(vnmsi.index(v))
        fvars = []
        for v in func.var_names_float:
            if v not in vnmsf:
                raise ValueError(
                    f"Problem '{self.name}': float variable '{v}' of function '{func.name}' not among float problem variables {vnmsf}"
                )
            fvars.append(vnmsf.index(v))

        if len(vars_float.shape) == 1:
            varsi = vars_int[ivars] if len(vars_int) else np.array([], dtype=np.float64)
            varsf = (
                vars_float[fvars] if len(vars_float) else np.array([], dtype=np.float64)
            )
        else:
            n_pop = vars_float.shape[0]
            varsi = (
                vars_int[:, ivars]
                if len(vars_int)
                else np.zeros((n_pop, 0), dtype=np.float64)
            )
            varsf = (
                vars_float[:, fvars]
                if len(vars_float)
                else np.zeros((n_pop, 0), dtype=np.float64)
            )

        if ret_inds:
            return ivars, fvars
        else:
            return varsi, varsf

    def calc_gradients(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction,
        components: Sequence[int] | np.ndarray | None,
        ivars: list[int],
        fvars: list[int],
        vrs: list[int],
        pop: bool = False,
        verbosity: int = 0,
        func_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        The actual gradient calculation, not to be called directly
        (call `get_gradients` instead).

        Can be overloaded in derived classes, the base class only considers
        analytic derivatives.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        func
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        components
            The function's component selection, or None for all
        ivars
            The indices of the function int variables in the problem
        fvars
            The indices of the function float variables in the problem
        vrs
            The function float variable indices wrt which the
            derivatives are to be calculated
        func_values
            Previously calculated function values at the given variables,
            shape: (n_components,)
        pop
            Flag for vectorizing calculations via population
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        gradients
            The gradients of the functions, shape:
            (n_components, n_vrs)
        """
        n_vars = len(vrs)
        n_cmpnts = func.n_components() if components is None else len(components)
        varsi = vars_int[ivars] if len(vars_int) else np.array([])
        varsf = vars_float[fvars] if len(vars_float) else np.array([])

        gradients = np.full((n_cmpnts, n_vars), np.nan, dtype=np.float64)
        for vi, v in enumerate(vrs):
            if v in fvars:
                gradients[:, vi] = func.ana_deriv(
                    varsi, varsf, fvars.index(v), components
                )
            else:
                gradients[:, vi] = 0

        return gradients

    def get_gradients(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        func: OptFunction | None = None,
        components: Sequence[int] | np.ndarray | None = None,
        vars: Sequence[str | int] | None = None,
        pop: bool = False,
        verbosity: int = 0,
        func_values: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Obtain gradients of a function that is linked to the
        problem.

        The func object typically is a `iwopy.core.OptFunctionList`
        object that contains a selection of objectives and/or constraints
        that were previously added to this problem. By default all
        objectives and constraints (and all their components) are
        being considered, cf. class `ProblemDefaultFunc`.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        func
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        components
            The function's component selection, or None for all
        vars
            The float variables wrt which the
            derivatives are to be calculated, or
            None for all
        func_values
            Previously calculated function values at the given variables,
            shape: (n_components,)
        verbosity
            The verbosity level, 0 = silent
        pop
            Flag for vectorizing calculations via population

        Returns
        -------
        gradients
            The gradients of the functions, shape:
            (n_components, n_vars)
        """
        # set and check func:
        if func is None:
            func = ProblemDefaultFunc(self)
        if func.problem is not self:
            raise ValueError(
                f"Problem '{self.name}': Attempt to calculate gradient for function '{func.name}' which is linked to different problem '{func.problem.name}'"
            )
        if not func.initialized:
            func.initialize(verbosity=(0 if verbosity < 2 else verbosity - 1))

        # find function variables:
        ivars, fvars = self._find_vars(vars_int, vars_float, func, ret_inds=True)

        # find names of differentiation variables:
        vnmsf = list(self.var_names_float())
        if vars is None:
            vars = vnmsf
        else:
            tvars = []
            for v in vars:
                if isinstance(v, (int, np.integer)):
                    index = int(v)
                    if index < 0 or index > len(vnmsf):
                        raise ValueError(
                            f"Problem '{self.name}': Variable index {index} exceeds problem float variables, count = {len(vnmsf)}"
                        )
                    tvars.append(vnmsf[index])
                elif isinstance(v, str):
                    vl = fnmatch.filter(vnmsf, v)
                    if not len(vl):
                        raise ValueError(
                            f"Problem '{self.name}': No match for variable pattern '{v}' among problem float variable {vnmsf}"
                        )
                    tvars += vl
                else:
                    raise TypeError(
                        f"Problem '{self.name}': Illegal type '{type(v)}', expecting str or int"
                    )
            vars = tvars

        # find variable indices among function float variables:
        vrs = []
        hvnmsf = np.array(vnmsf)[fvars].tolist()
        for v in vars:
            if v not in hvnmsf:
                raise ValueError(
                    f"Problem '{self.name}': Selected gradient variable '{v}' not in function variables '{hvnmsf}' for function '{func.name}'"
                )
            vrs.append(hvnmsf.index(v))

        # calculate gradients:
        gradients = self.calc_gradients(
            vars_int,
            vars_float,
            func,
            components,
            ivars,
            fvars,
            vrs,
            func_values=func_values,
            pop=pop,
            verbosity=verbosity,
        )

        # check success:
        nog = np.where(np.isnan(gradients))[1]
        if len(nog):
            nvrs = np.unique(np.array(vars)[nog]).tolist()
            raise ValueError(
                f"Problem '{self.name}': Failed to calculate derivatives for variables {nvrs}. Maybe wrap this problem into DiscretizeRegGrid?"
            )

        return gradients

    def initialize(self, verbosity: int = 1) -> None:
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent
        """

        if not self.objs.initialized:
            self.objs.initialize(verbosity)
        if not self.cons.initialized:
            self.cons.initialize(verbosity)

        if verbosity:
            s = f"Problem '{self.name}' ({type(self).__name__}): Initializing"
            print(s)
            self._hline = "-" * len(s)
            print(self._hline)

        if self._mem_size is not None:
            self.memory = Memory(self._mem_size, self._mem_keyf)
            if verbosity:
                print(f"  Memory size : {self.memory.size}")
                print(self._hline)

        if self.n_objectives == 0:
            raise ValueError("Problem initialized without added objectives.")

        self._maximize = np.zeros(self.n_objectives, dtype=bool)
        i0 = 0
        for f in self.objs.functions:
            i1 = i0 + f.n_components()
            self._maximize[i0:i1] = f.maximize()
            i0 = i1

        super().initialize(verbosity)

    @property
    def maximize_objs(self) -> np.ndarray:
        """
        Flags for objective maximization

        Returns
        -------
        maximize
            Boolean flag for maximization of objective,
            shape: (n_objectives,)
        """
        if self._maximize is None:
            raise RuntimeError(f"Problem '{self.name}' has not been initialized")
        return self._maximize

    def apply_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> object | None:
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem
        """
        return None

    def apply_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray
    ) -> object | None:
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem
        """
        return None

    @overload
    def evaluate_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: Literal[False] = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def evaluate_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, object | None]: ...

    def evaluate_individual(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, object | None]:
        """
        Evaluate a single individual of the problem.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_vars_int,)
        vars_float
            The float variable values, shape: (n_vars_float,)
        ret_prob_res
            Flag for additionally returning of problem results

        Returns
        -------
        objs
            The objective function values, shape: (n_objectives,)
        con
            The constraints values, shape: (n_constraints,)
        prob_res
            The problem results
        """
        objs, cons = None, None
        if not ret_prob_res and self.memory is not None:
            memres = self.memory.lookup_individual(vars_int, vars_float)
            if memres is not None:
                objs, cons = memres
                results = None

        if objs is None:
            results = self.apply_individual(vars_int, vars_float)

            varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
            objs = self.objs.calc_individual(varsi, varsf, results)

            varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
            cons = self.cons.calc_individual(varsi, varsf, results)

            if self.memory is not None:
                self.memory.store_individual(vars_int, vars_float, objs, cons)

        if ret_prob_res:
            return objs, cons, results
        else:
            return objs, cons

    @overload
    def evaluate_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: Literal[False] = False,
    ) -> tuple[np.ndarray, np.ndarray]: ...

    @overload
    def evaluate_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: Literal[True],
    ) -> tuple[np.ndarray, np.ndarray, object | None]: ...

    def evaluate_population(
        self,
        vars_int: np.ndarray,
        vars_float: np.ndarray,
        ret_prob_res: bool = False,
    ) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, object | None]:
        """
        Evaluate all individuals of a population.

        Parameters
        ----------
        vars_int
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values, shape: (n_pop, n_vars_float)
        ret_prob_res
            Flag for additionally returning of problem results

        Returns
        -------
        objs
            The objective function values, shape: (n_pop, n_objectives)
        cons
            The constraints values, shape: (n_pop, n_constraints)
        prob_res
            The problem results
        """

        from_mem = False
        if not ret_prob_res and self.memory is not None:
            memres = self.memory.lookup_population(vars_int, vars_float)
            if memres is not None:
                todo = np.any(np.isnan(memres), axis=1)
                from_mem = not np.all(todo)

        if from_mem and memres is not None:
            objs = memres[:, : self.n_objectives]
            cons = memres[:, self.n_objectives :]
            del memres

            if np.any(todo):
                vals_int = vars_int[todo]
                vals_float = vars_float[todo]

                results = self.apply_population(vals_int, vals_float)

                varsi, varsf = self._find_vars(vals_int, vals_float, self.objs)
                ores = self.objs.calc_population(varsi, varsf, results)
                objs[todo] = ores

                varsi, varsf = self._find_vars(vals_int, vals_float, self.cons)
                cres = self.cons.calc_population(varsi, varsf, results)
                cons[todo] = cres

                if self.memory is not None:
                    self.memory.store_population(vals_int, vals_float, ores, cres)

        else:
            results = self.apply_population(vars_int, vars_float)

            varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
            objs = self.objs.calc_population(varsi, varsf, results)

            varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
            cons = self.cons.calc_population(varsi, varsf, results)

            if self.memory is not None:
                self.memory.store_population(vars_int, vars_float, objs, cons)

        if ret_prob_res:
            return objs, cons, results
        else:
            return objs, cons

    def check_constraints_individual(
        self, constraint_values: np.ndarray, verbosity: int = 0
    ) -> np.ndarray:
        """
        Check if the constraints are fullfilled for the
        given individual.

        Parameters
        ----------
        constraint_values
            The constraint values, shape: (n_components,)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The boolean result, shape: (n_components,)
        """
        val = constraint_values
        out = np.zeros(self.n_constraints, dtype=bool)

        i0 = 0
        for c in self.cons.functions:
            i1 = i0 + c.n_components()
            out[i0:i1] = c.check_individual(val[i0:i1], verbosity)
            i0 = i1

        return out

    def check_constraints_population(
        self, constraint_values: np.ndarray, verbosity: int = 0
    ) -> np.ndarray:
        """
        Check if the constraints are fullfilled for the
        given population.

        Parameters
        ----------
        constraint_values
            The constraint values, shape: (n_pop, n_components)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        values
            The boolean result, shape: (n_pop, n_components)
        """
        val = constraint_values
        n_pop = val.shape[0]
        out = np.zeros((n_pop, self.n_constraints), dtype=bool)

        i0 = 0
        for c in self.cons.functions:
            i1 = i0 + c.n_components()
            out[:, i0:i1] = c.check_population(val[:, i0:i1], verbosity)
            i0 = i1

        return out

    def finalize_individual(
        self, vars_int: np.ndarray, vars_float: np.ndarray, verbosity: int = 1
    ) -> tuple[object | None, np.ndarray, np.ndarray]:
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float
            The optimal float variable values, shape: (n_vars_float,)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem
        objs
            The objective function values, shape: (n_objectives,)
        cons
            The constraints values, shape: (n_constraints,)
        """
        results = self.apply_individual(vars_int, vars_float)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
        objs = self.objs.finalize_individual(varsi, varsf, results, verbosity)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
        cons = self.cons.finalize_individual(varsi, varsf, results, verbosity)

        return results, objs, cons

    def finalize_population(
        self, vars_int: np.ndarray, vars_float: np.ndarray, verbosity: int = 0
    ) -> tuple[object | None, np.ndarray, np.ndarray]:
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        verbosity
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results
            The results of the variable application
            to the problem
        objs
            The final objective function values, shape: (n_pop, n_components)
        cons
            The final constraint values, shape: (n_pop, n_constraints)
        """
        results = self.apply_population(vars_int, vars_float)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
        objs = self.objs.finalize_population(varsi, varsf, results, verbosity)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
        cons = self.cons.finalize_population(varsi, varsf, results, verbosity)

        return results, objs, cons

    def prob_res_einsum_individual(
        self, prob_res_list: Sequence[object | None], coeffs: np.ndarray
    ) -> object | None:
        """
        Calculate the einsum of problem results

        Parameters
        ----------
        prob_res_list
            The problem results
        coeffs
            The coefficients

        Returns
        -------
        prob_res
            The weighted sum of problem results
        """
        if not len(prob_res_list) or prob_res_list[0] is None:
            return None

        raise NotImplementedError(
            f"Problem '{self.name}': Einsum not implemented for problem results type '{type(prob_res_list[0]).__name__}'"
        )

    def prob_res_einsum_population(
        self, prob_res_list: Sequence[object | None], coeffs: np.ndarray
    ) -> object | None:
        """
        Calculate the einsum of problem results

        Parameters
        ----------
        prob_res_list
            The problem results
        coeffs
            The coefficients

        Returns
        -------
        prob_res
            The weighted sum of problem results
        """
        if not len(prob_res_list) or prob_res_list[0] is None:
            return None

        raise NotImplementedError(
            f"Problem '{self.name}': Einsum not implemented for problem results type '{type(prob_res_list[0]).__name__}'"
        )

    @classmethod
    def new(cls, problem_type: str, *args: object, **kwargs: object) -> "Problem":
        """
        Run-time problem factory.

        Parameters
        ----------
        problem_type
            The selected derived class name
        args
            Additional parameters for constructor
        kwargs
            Additional parameters for constructor
        """
        return new_instance(cls, problem_type, *args, **kwargs)


class ProblemDefaultFunc(OptFunctionList[OptFunction]):
    """
    The default function of a problem
    for gradient calculations.
    """

    def __init__(self, problem: Problem) -> None:
        """
        Parameters
        ----------
        problem
            The problem
        """
        super().__init__(problem, "objs_cons")
        for objective in problem.objs.functions:
            self.append(objective)
        for constraint in problem.cons.functions:
            self.append(constraint)
