import numpy as np
import fnmatch
from abc import ABCMeta

from .base import Base
from .function_list import OptFunctionList
from .memory import Memory
from iwopy.utils import RegularDiscretizationGrid


class Problem(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization problems.

    Parameters
    ----------
    name: str
        The problem's name
    mem_size : int, optional
        The memory size, default no memory
    mem_keyf : Function, optional
        The memory key function. Parameters:
        (vars_int, vars_float), returns key Object

    Attributes
    ----------
    objs : iwopy.core.OptFunctionList
        The objective functions
    cons : iwopy.core.OptFunctionList
        The constraints
    memory : iwopy.core.Memory
        The memory, or None

    """

    INT_INF = RegularDiscretizationGrid.INT_INF

    def __init__(self, name, mem_size=None, mem_keyf=None):
        super().__init__(name)

        self.objs = OptFunctionList(self, "objs")
        self.cons = OptFunctionList(self, "cons")

        self.memory = None
        self._mem_size = mem_size
        self._mem_keyf = mem_keyf

        self._cons_mi = None
        self._cons_ma = None
        self._cons_tol = None

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

    def min_values_int(self):
        """
        The minimal values of the integer variables.

        Use -self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal int values, shape: (n_vars_int,)

        """
        return -self.INT_INF

    def max_values_int(self):
        """
        The maximal values of the integer variables.

        Use self.INT_INF for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal int values, shape: (n_vars_int,)

        """
        return self.INT_INF

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

    def min_values_float(self):
        """
        The minimal values of the float variables.

        Use -numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Minimal float values, shape: (n_vars_float,)

        """
        return -np.inf

    def max_values_float(self):
        """
        The maximal values of the float variables.

        Use numpy.inf for unbounded.

        Returns
        -------
        values : numpy.ndarray
            Maximal float values, shape: (n_vars_float,)

        """
        return np.inf

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

    def _apply_varmap(self, vtype, f, ftype, varmap):
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
                        f"Problem '{self.name}': {vtype} varmap rule '{fv} --> {pv}' failed for {ftype} '{f.name}', pattern '{pv}' not found among problem {vtype} variables {pnms}"
                    )
                elif len(pvl) > 1:
                    raise ValueError(
                        f"Problem '{self.name}': Require unique match of {vtype} variable '{fv}' of {ftype} '{f.name}' to problem variables, found {pvl} for pattern '{pv}'"
                    )
                else:
                    vmap[fv] = pvl[0]
            elif np.issubdtype(type(pv), np.integer):
                if pv < 0 or pv >= len(pnms):
                    raise ValueError(
                        f"Problem '{self.name}': varmap rule '{fv} --> {pv}' cannot be applied for {len(pnms)} {vtype} variables {pnms}"
                    )
                vmap[fv] = pnms[pv]
            else:
                raise ValueError(
                    f"Problem '{self.name}': varmap_{vtype} target variable in '{fv} --> {pv}' of {ftype} '{f.name}' is neither str nor int"
                )

        if vtype == "int":
            f.rename_vars_int(vmap)
        else:
            f.rename_vars_float(vmap)

    def add_objective(
        self,
        objective,
        varmap_int=None,
        varmap_float=None,
        verbosity=0,
    ):
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
        self._apply_varmap("int", objective, "objective", varmap_int)
        self._apply_varmap("float", objective, "objective", varmap_float)
        self.objs.append(objective)

    def add_constraint(
        self,
        constraint,
        varmap_int=None,
        varmap_float=None,
        verbosity=0,
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
    def min_values_constraints(self):
        """
        Gets the minimal values of constraints

        Returns
        -------
        cmi : numpy.ndarray
            The minimal constraint values, shape: (n_constraints,)

        """
        return self._cons_mi

    @property
    def max_values_constraints(self):
        """
        Gets the maximal values of constraints

        Returns
        -------
        cma : numpy.ndarray
            The maximal constraint values, shape: (n_constraints,)

        """
        return self._cons_ma

    @property
    def constraints_tol(self):
        """
        Gets the tolerance values of constraints

        Returns
        -------
        ctol : numpy.ndarray
            The constraint tolerance values, shape: (n_constraints,)

        """
        return self._cons_tol

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

    def _find_vars(self, vars_int, vars_float, func, ret_inds=False):
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
        vars_int,
        vars_float,
        func,
        components,
        ivars,
        fvars,
        vrs,
        pop=False,
        verbosity=0,
    ):
        """
        The actual gradient calculation, not to be called directly
        (call `get_gradients` instead).

        Can be overloaded in derived classes, the base class only considers
        analytic derivatives.

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
        components : list of int, optional
            The function's component selection, or None for all
        ivars : list of int
            The indices of the function int variables in the problem
        fvars : list of int
            The indices of the function float variables in the problem
        vrs : list of int
            The function float variable indices wrt which the
            derivatives are to be calculated
        pop : bool
            Flag for vectorizing calculations via population
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        gradients : numpy.ndarray
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
        vars_int,
        vars_float,
        func=None,
        components=None,
        vars=None,
        pop=False,
        verbosity=0,
    ):
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
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        func : iwopy.core.OptFunctionList, optional
            The functions to be differentiated, or None
            for a list of all objectives and all constraints
            (in that order)
        components : list of int, optional
            The function's component selection, or None for all
        vars : list of int or str, optional
            The float variables wrt which the
            derivatives are to be calculated, or
            None for all
        verbosity : int
            The verbosity level, 0 = silent
        pop : bool
            Flag for vectorizing calculations via population

        Returns
        -------
        gradients : numpy.ndarray
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
                if np.issubdtype(type(v), np.integer):
                    if v < 0 or v > len(vnmsf):
                        raise ValueError(
                            f"Problem '{self.name}': Variable index {v} exceeds problem float variables, count = {len(vnmsf)}"
                        )
                    tvars.append(vnmsf[v])
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
            self._hline = "-" * len(s)
            print(self._hline)

        if self._mem_size is not None:
            self.memory = Memory(self._mem_size, self._mem_keyf)
            if verbosity:
                print(f"  Memory size  : {self.memory.size}")
                print(self._hline)

        n_int = self.n_vars_int
        n_float = self.n_vars_float
        if verbosity:
            print(f"  n_vars_int   : {n_int}")
            print(f"  n_vars_float : {n_float}")
            print(self._hline)

        if verbosity:
            print(f"  n_objectives : {self.objs.n_functions}")
            print(f"  n_obj_cmptns : {self.n_objectives}")
            print(self._hline)
            print(f"  n_constraints: {self.cons.n_functions}")
            print(f"  n_con_cmptns : {self.n_constraints}")
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
    def maximize_objs(self):
        """
        Flags for objective maximization

        Returns
        -------
        maximize : numpy.ndarray
            Boolean flag for maximization of objective,
            shape: (n_objectives,)

        """
        return self._maximize

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

    def evaluate_individual(self, vars_int, vars_float, ret_prob_res=False):
        """
        Evaluate a single individual of the problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        ret_prob_res : bool
            Flag for additionally returning of problem results

        Returns
        -------
        objs : np.array
            The objective function values, shape: (n_objectives,)
        con : np.array
            The constraints values, shape: (n_constraints,)
        prob_res : object, optional
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

    def evaluate_population(self, vars_int, vars_float, ret_prob_res=False):
        """
        Evaluate all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        ret_prob_res : bool
            Flag for additionally returning of problem results

        Returns
        -------
        objs : np.array
            The objective function values, shape: (n_pop, n_objectives)
        cons : np.array
            The constraints values, shape: (n_pop, n_constraints)
        prob_res : object, optional
            The problem results

        """

        from_mem = False
        if not ret_prob_res and self.memory is not None:
            memres = self.memory.lookup_population(vars_int, vars_float)
            if memres is not None:
                todo = np.any(np.isnan(memres), axis=1)
                from_mem = not np.all(todo)

        if from_mem:

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
        out = np.zeros(self.n_constraints, dtype=bool)

        i0 = 0
        for c in self.cons.functions:
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
        out = np.zeros((n_pop, self.n_constraints), dtype=bool)

        i0 = 0
        for c in self.cons.functions:
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
        results = self.apply_individual(vars_int, vars_float)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
        objs = self.objs.finalize_individual(varsi, varsf, results, verbosity)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
        cons = self.cons.finalize_individual(varsi, varsf, results, verbosity)

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
        results = self.apply_population(vars_int, vars_float)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.objs)
        objs = self.objs.finalize_population(varsi, varsf, results, verbosity)

        varsi, varsf = self._find_vars(vars_int, vars_float, self.cons)
        cons = self.cons.finalize_population(varsi, varsf, results, verbosity)

        return results, objs, cons

    def prob_res_einsum_individual(self, prob_res_list, coeffs):
        """
        Calculate the einsum of problem results

        Parameters
        ----------
        prob_res_list : list
            The problem results
        coeffs : numpy.ndarray
            The coefficients

        Returns
        -------
        prob_res : object
            The weighted sum of problem results

        """
        if not len(prob_res_list) or prob_res_list[0] is None:
            return None

        raise NotImplementedError(
            f"Problem '{self.name}': Einsum not implemented for problem results type '{type(prob_res_list[0]).__name__}'"
        )

    def prob_res_einsum_population(self, prob_res_list, coeffs):
        """
        Calculate the einsum of problem results

        Parameters
        ----------
        prob_res_list : list
            The problem results
        coeffs : numpy.ndarray
            The coefficients

        Returns
        -------
        prob_res : object
            The weighted sum of problem results

        """
        if not len(prob_res_list) or prob_res_list[0] is None:
            return None

        raise NotImplementedError(
            f"Problem '{self.name}': Einsum not implemented for problem results type '{type(prob_res_list[0]).__name__}'"
        )


class ProblemDefaultFunc(OptFunctionList):
    """
    The default function of a problem
    for gradient calculations.

    Parameters
    ----------
    problem : iwopy.core.Problem
        The problem

    """

    def __init__(self, problem):
        super().__init__(problem, "objs_cons")
        for f in problem.objs.functions:
            self.append(f)
        for f in problem.cons.functions:
            self.append(f)
