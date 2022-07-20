import numpy as np
import fnmatch
from abc import ABCMeta
from copy import deepcopy

from .base import Base
from .function_list import OptFunctionList


class Problem(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization problems.

    Parameters
    ----------
    name: str
        The problem's name

    Attributes
    ----------
    objs : iwopy.core.OptFunctionList
        The objective functions
    cons : iwopy.core.OptFunctionList
        The constraints
    varmap_int: dict
        Mapping from function variables to
        problem variables. Key: str or int,
        value: str or int
    varmap_float: dict
        Mapping from function variables to
        problem variables. Key: str or int,
        value: str or int

    """

    def __init__(self, name):
        super().__init__(name)

        self.objs = OptFunctionList(self, "objs")
        self.cons = OptFunctionList(self, "cons")
        self.varmap_int = {}
        self.varmap_float = {}

    def var_names_int(self):
        """
        The names of integer variables.

        Returns
        -------
        names : list of str
            The names of the integer variables

        """
        return []

    def initial_values_int(self):
        """
        The initial values of the integer variables.

        Returns
        -------
        values : numpy.ndarray
            Initial int values, shape: (n_vars_int,)

        """
        return 0

    @property
    def n_vars_int(self):
        """
        The number of int variables

        Returns
        -------
        n : int
            The number of int variables

        """
        return len(self.var_names_int())

    def var_names_float(self):
        """
        The names of float variables.

        Returns
        -------
        names : list of str
            The names of the float variables

        """
        return []

    def initial_values_float(self):
        """
        The initial values of the float variables.

        Returns
        -------
        values : numpy.ndarray
            Initial float values, shape: (n_vars_float,)

        """
        return None

    @property
    def n_vars_float(self):
        """
        The number of float variables

        Returns
        -------
        n : int
            The number of float variables

        """
        return len(self.var_names_float())

    def _add_to_varmap(self, vtype, f, ftype, varmap):
        """
        Helper function for adding entries to problem varmap
        """
        fnms = list(f.var_names_int) if vtype == "int" else list(f.var_names_float)
        if varmap is None:
            varmap = {vi: vi for vi in range(len(fnms))}

        vmap = self.varmap_int if vtype == "int" else self.varmap_float
        pnms = (
            list(self.var_names_int())
            if vtype == "int"
            else list(self.var_names_float())
        )
        for fv, pv in varmap.items():

            if isinstance(fv, str):
                fvl = fnmatch.filter(fnms, fv)
                if len(fvl) != 1:
                    raise KeyError(
                        f"Problem '{self.name}': varmap_{vtype} variable '{fv}' of {ftype} '{f.name}' not uniquely matched among {ftype} variables, found {fvl}"
                    )
                else:
                    fv = fvl[0]
            elif not isinstance(fv, int):
                raise ValueError(
                    f"Problem '{self.name}': varmap_{vtype} variable '{fv}' of {ftype} '{f.name}' is neither str nor int"
                )

            if fv in vmap:
                if pv != vmap[fv]:
                    raise ValueError(
                        f"Problem '{self.name}': Mismatch in varmap_{vtype} for {ftype} '{f.name}', found '{fv} --> {pv}', expected '{fv} --> {vmap[fv]}'"
                    )
            elif isinstance(pv, str):
                pvl = fnmatch.filter(pnms, pv)
                if len(pvl) != 1:
                    raise ValueError(
                        f"Problem '{self.name}': Require unique match of {vtype} variable '{fv}' of {ftype} '{f.name}' to problem variables, found {pvl} for pattern '{pv}'"
                    )
                else:
                    vmap[fv] = pvl[0]
            elif not isinstance(fv, int):
                raise ValueError(
                    f"Problem '{self.name}': varmap_{vtype} target variable in '{fv} --> {pv}' of {ftype} '{f.name}' is neither str nor int"
                )
            else:
                vmap[fv] = pv

    def add_objective(self, objective, varmap_int=None, varmap_float=None, verbosity=0):
        """
        Add an objective to the problem.

        Parameters
        ----------
        objective : iwopy.Objective
            The objective
        varmap_int: dict, optional
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        varmap_float: dict, optional
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        verbosity : int
            The verbosity level, 0 = silent

        """
        if not objective.initialized:
            objective.initialize(verbosity)
        self._add_to_varmap("int", objective, "objective", varmap_int)
        self._add_to_varmap("float", objective, "objective", varmap_float)
        self.objs.append(objective)

    def add_constraint(
        self, constraint, varmap_int=None, varmap_float=None, verbosity=0
    ):
        """
        Add a constraint to the problem.

        Parameters
        ----------
        constraint : iwopy.Constraint
            The constraint
        varmap_int: dict, optional
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        varmap_float: dict, optional
            Mapping from objective variables to
            problem variables. Key: str or int,
            value: str or int
        verbosity : int
            The verbosity level, 0 = silent

        """
        if not constraint.initialized:
            constraint.initialize(verbosity)
        self._add_to_varmap("int", constraint, "constraint", varmap_int)
        self._add_to_varmap("float", constraint, "constraint", varmap_float)
        self.cons.append(constraint)

    def _resolve_varmap(self, func, vtype, vmap, verbosity=0):
        """
        Helper function that evaluates the varmap for a given function
        """

        fvars = []
        fnms = (
            list(func.var_names_int) if vtype == "int" else list(func.var_names_float)
        )
        pnms = (
            list(self.var_names_int())
            if vtype == "int"
            else list(self.var_names_float())
        )
        for fvi, fv in enumerate(fnms):

            pv = None
            for fvm, pvm in vmap.items():
                if isinstance(fvm, int):
                    if fvm == fvi:
                        pv = pvm
                        break
                elif fnmatch.fnmatch(fv, fvm):
                    pv = pvm
                    break

            if pv is None:
                raise KeyError(
                    f"Problem '{self.name}': {vtype} variable '{fvi}: {fv}' of function '{func.name}' unmatched in varmap {sorted(list(vmap.keys()))}"
                )
            elif isinstance(pv, str):
                pvl = fnmatch.filter(pnms, pv)
                if not len(pvl):
                    raise KeyError(
                        f"Problem '{self.name}': {vtype} variable '{fv}' of function '{func.name}' mapped to '{pv}', but no match within problem {vtype} variables {sorted(pnms)}"
                    )
                elif len(pvl) > 1:
                    raise KeyError(
                        f"Problem '{self.name}': {vtype} variable '{fv}' of function '{func.name}' mapped to '{pv}', found multiple matches within problem {vtype} variables {sorted(pvl)}"
                    )
                pv = pnms.index(pvl[0])
            elif not isinstance(pv, int) or (pv < 0 or pv > len(pnms)):
                raise ValueError(
                    f"Problem '{self.name}': {vtype} variable '{fv}' of function '{func.name}' mapped to illegal variable index {pv} for {len(pnms)} available {vtype} variables in problem"
                )

            if verbosity:
                print(f"  {vtype} fvar {fvi}: {fv} --> {vtype} pvar {pv}: {pnms[pv]}")

            fvars.append(pv)

        return fvars

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        self.objs.initialize(verbosity)
        self.cons.initialize(verbosity)

        self.varmap = self.resolve_varmap(self.varmap, verbosity)

        super().initialize(verbosity)

    @property
    def n_objectives(self):
        """
        The total number of objectives,
        i.e., the sum of all components

        Returns
        -------
        n_obj : int
            The total number of objective
            functions

        """
        return self.objs.n_components()

    @property
    def n_constraints(self):
        """
        The total number of constraints,
        i.e., the sum of all components

        Returns
        -------
        n_con : int
            The total number of constraint
            functions

        """
        return self.cons.n_components()

    def calc_gradients(
        self, ivars, fvars, varsi, varsf, func, vrs, components, verbosity=0
    ):
        """
        The actual gradient calculation.

        Can be overloaded in derived classes, the base class only considers
        analytic derivatives.

        Parameters
        ----------
        ivars : list of int
            The indices of the int variables in the problem resulting
            from the varmap mapping
        fvars : list of int
            The indices of the float variables in the problem resulting
            from the varmap mapping
        varsi : np.array
            The integer variable values, shape: (n_func_vars_int,)
        varsf : np.array
            The float variable values, shape: (n_vfunc_ars_float,)
        func : iwopy.core.OptFunctionList, optional
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        vrs : list of int
            The float variable indices wrt which the
            derivatives are to be calculated
        components : list of int
            The selected components of func, or None for all
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functions, shape:
            (n_func_cmpnts, n_vars)

        """
        n_vars = len(vrs)
        gradients = np.full((func.n_components(), n_vars), np.nan, dtype=np.float64)
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
        vars_int,
        vars_float,
        func=None,
        vars=None,
        varmap_int=None,
        varmap_float=None,
        components=None,
        verbosity=0,
    ):
        """
        Obtain gradients of a function that is linked to the
        problem.

        The func object typically is a `iwopy.core.OptFunctionList`
        object that contains a selection of objectives and/or constraints
        that were previously added to this problem. By default all
        objectives and constraints (and all their components) are
        being considered.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        func : iwopy.core.OptFunctionList, optional
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        vars : list of int or str, optional
            The float variables wrt which the
            derivatives are to be calculated, or
            None for all
        varmap_int: dict
            Mapping from function variables to
            problem variables. Key: str or int,
            value: str or int
        varmap_float: dict
            Mapping from function variables to
            problem variables. Key: str or int,
            value: str or int
        components : list of int
            The selected components of func, or None for all
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functions, shape:
            (n_func_cmpnts, n_vars)

        """
        # set and check func:
        if func is None:
            func = OptFunctionList(self, "objs_cons")
            for f in self.objs.functions:
                func.append(f)
            for f in self.cons.functions:
                func.append(f)
            func.initialize(verbosity=0)
        if func.problem is not self:
            raise ValueError(
                f"Problem '{self.name}': Attempt to calculate gradient for function '{func.name}' which is linked to different problem '{func.problem.name}'"
            )

        # update varmaps:
        if varmap_int is None:
            vmapi = self.varmap_int
        else:
            vmapi = deepcopy(self.varmap_int).update(varmap_int)
        if varmap_float is None:
            vmapf = self.varmap_float
        else:
            vmapf = deepcopy(self.varmap_float).update(varmap_float)

        # find differentiation variables:
        if vars is None:
            vrs = list(range(self.n_vars_float))
        else:
            vnms = list(self.var_names_float())
            vrs = [vnms.index(v) if isinstance(v, str) else int(v) for v in vars]

        # calculate gradients:
        ivars = self._resolve_varmap(func, "int", vmapi, verbosity)
        fvars = self._resolve_varmap(func, "float", vmapf, verbosity)
        varsi = vars_int[ivars] if len(vars_int) else np.array([])
        varsf = vars_float[fvars] if len(vars_float) else np.array([])
        gradients = self.calc_gradients(
            ivars, fvars, varsi, varsf, func, vrs, components, verbosity
        )

        return gradients

    def initialize(self, verbosity=1):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """

        if not self.objs.initialized:
            self.objs.initialize(verbosity)
        if not self.cons.initialized:
            self.cons.initialize(verbosity)

        if verbosity:
            s = f"Problem '{self.name}' ({type(self).__name__}): Initializing"
            print(s)
            L = len(s)
            print("-" * L)

        n_int = self.n_vars_int
        n_float = self.n_vars_float
        if verbosity:
            print(f"  n_vars_int   : {n_int}")
            print(f"  n_vars_float : {n_float}")
            print("-" * L)

        if verbosity:
            print(f"  n_objectives : {self.objs.n_functions}")
            print(f"  n_obj_cmptns : {self.n_objectives}")
            print("-" * L)
            print(f"  n_constraints: {self.cons.n_functions}")
            print(f"  n_con_cmptns : {self.n_constraints}")
            print("-" * L)

        if self.n_objectives == 0:
            raise ValueError("Problem initialized without added objectives.")

        self._maximize = np.zeros(self.n_objectives, dtype=bool)
        i0 = 0
        for f in self.objs.functions:
            i1 = i0 + f.n_components()
            self._maximize[i0:i1] = np.array(f.maximize(), dtype=bool)
            i0 = i1

        super().initialize(verbosity)

    def apply_individual(self, vars_int, vars_float):
        """
        Apply new variables to the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        return None

    def apply_population(self, vars_int, vars_float):
        """
        Apply new variables to the problem,
        for a whole population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem

        """
        return None

    def evaluate_individual(self, vars_int, vars_float):
        """
        Evaluate a single individual of the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_objectives,)
        con : np.array
            The constraints values, shape: (n_constraints,)

        """
        objs = np.full(self.n_objectives, np.nan, dtype=np.float64)
        cons = np.full(self.n_constraints, np.nan, dtype=np.float64)

        results = self.apply_individual(vars_int, vars_float)

        i0 = 0
        for f in self.objs:
            i1 = i0 + f.n_components()
            objs[i0:i1] = f.calc_individual(vars_int, vars_float, results)
            i0 = i1

        objs[self._maximize] *= -1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[i0:i1] = c.calc_individual(vars_int, vars_float, results)
            i0 = i1

        return results, objs, cons

    def evaluate_population(self, vars_int, vars_float):
        """
        Evaluate all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_pop, n_objectives)
        cons : np.array
            The constraints values, shape: (n_pop, n_constraints)

        """

        n_pop = (
            vars_float.shape[0]
            if vars_float is not None and len(vars_float.shape)
            else vars_int.shape[0]
        )
        objs = np.full((n_pop, self.n_objectives), np.nan, dtype=np.float64)
        cons = np.full((n_pop, self.n_constraints), np.nan, dtype=np.float64)

        results = self.apply_population(vars_int, vars_float)

        i0 = 0
        for f in self.objs:
            i1 = i0 + f.n_components()
            objs[:, i0:i1] = f.calc_population(vars_int, vars_float, results)
            i0 = i1

        objs[:, self._maximize] *= -1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[:, i0:i1] = c.calc_population(vars_int, vars_float, results)
            i0 = i1

        return results, objs, cons

    def check_constraints_individual(self, constraint_values, verbosity=0):
        """
        Check if the constraints are fullfilled for the
        given individual.

        Parameters
        ----------
        constraint_values : np.array
            The constraint values, shape: (n_components,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        values : np.array
            The boolean result, shape: (n_components,)

        """
        val = constraint_values
        out = np.zeros(self.n_constraints, dtype=np.bool)

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            out[i0:i1] = c.check_individual(val[i0:i1], verbosity)
            i0 = i1

        return out

    def check_constraints_population(self, constraint_values, verbosity=0):
        """
        Check if the constraints are fullfilled for the
        given population.

        Parameters
        ----------
        constraint_values : np.array
            The constraint values, shape: (n_pop, n_components)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        values : np.array
            The boolean result, shape: (n_pop, n_components)

        """
        val = constraint_values
        n_pop = val.shape[0]
        out = np.zeros((n_pop, self.n_constraints()), dtype=np.bool)

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            out[:, i0:i1] = c.check_population(val[:, i0:i1], verbosity)
            i0 = i1

        return out

    def finalize_individual(self, vars_int, vars_float, verbosity=1):
        """
        Finalization, given the champion data.

        Parameters
        ----------
        vars_int : np.array
            The optimal integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The optimal float variable values, shape: (n_vars_float,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The objective function values, shape: (n_objectives,)
        cons : np.array
            The constraints values, shape: (n_constraints,)

        """
        objs = np.zeros(self.n_objectives, dtype=np.float64)
        cons = np.zeros(self.n_constraints, dtype=np.float64)

        results = self.apply_individual(vars_int, vars_float)

        i0 = 0
        for f in self.objs:
            i1 = i0 + f.n_components()
            objs[i0:i1] = f.finalize_individual(
                vars_int, vars_float, results, verbosity
            )
            i0 = i1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[i0:i1] = c.finalize_individual(
                vars_int, vars_float, results, verbosity
            )
            i0 = i1

        return results, objs, cons

    def finalize_population(self, vars_int, vars_float, verbosity=0):
        """
        Finalization, given the final population data.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values of the final
            generation, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values of the final
            generation, shape: (n_pop, n_vars_float)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        problem_results : Any
            The results of the variable application
            to the problem
        objs : np.array
            The final objective function values, shape: (n_pop, n_components)
        cons : np.array
            The final constraint values, shape: (n_pop, n_constraints)

        """
        n_pop = (
            vars_float.shape[0]
            if vars_float is not None and len(vars_float.shape)
            else vars_int.shape[0]
        )
        objs = np.full((n_pop, self.n_objectives), np.nan, dtype=np.float)
        cons = np.full((n_pop, self.n_constraints), np.nan, dtype=np.float)

        results = self.apply_population(vars_int, vars_float)

        i0 = 0
        for f in self.objs:
            i1 = i0 + f.n_components()
            objs[:, i0:i1] = f.finalize_population(
                vars_int, vars_float, results, verbosity
            )
            i0 = i1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[:, i0:i1] = c.finalize_population(
                vars_int, vars_float, results, verbosity
            )
            i0 = i1

        return results, objs, cons
