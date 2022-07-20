import numpy as np
from abc import ABCMeta
from fnmatch import fnmatch

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
    varmap_int : dict
        Mapping from function int variables to
        problem int variables. Keys: function name,
        Values: dict with mapping from str (or int)
        to str (or int)
    varmap_float : dict
        Mapping from function float variables to
        problem float variables. Keys: function name,
        Values: dict with mapping from str (or int)
        to str (or int)

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

    def add_objective(self, objective, varmap_int={}, varmap_float={}):
        """
        Add an objective to the problem.

        Parameters
        ----------
        objective : iwopy.Objective
            The objective
        varmap_int : dict, optional
            Mapping from objective int variables to
            problem int variables. Key: str or int,
            value: str or int
        varmap_float : dict, optional
            Mapping from objective float variables to
            problem float variables. Key: str or int,
            value: str or int

        """

        if objective.name in self.varmap_float:
            raise KeyError(f"Problem '{self.name}': Cannot add objective '{objective.name}', since a function with that name has already been added before")
        self.varmap_int[objective.name] = varmap_int
        self.varmap_float[objective.name] = varmap_float

        self.objs.append(objective)

    def add_constraint(self, constraint, varmap_int={}, varmap_float={}):
        """
        Add a constraint to the problem.

        Parameters
        ----------
        constraint : iwopy.Constraint
            The constraint
        varmap_int : dict, optional
            Mapping from constraint int variables to
            problem int variables. Key: str or int,
            value: str or int
        varmap_float : dict, optional
            Mapping from constraint float variables to
            problem float variables. Key: str or int,
            value: str or int

        """

        if constraint.name in self.varmap_float:
            raise KeyError(f"Problem '{self.name}': Cannot add constraint '{constraint.name}', since a function with that name has already been added before")
        self.varmap_int[constraint.name] = varmap_int
        self.varmap_float[constraint.name] = varmap_float

        self.cons.append(constraint)

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

        onms = [f.name for f in self.objs.functions]
        cnms = [f.name for f in self.cons.functions]
        pnmsi = list(self.var_names_int())
        pnmsf = list(self.var_names_float())

        # resolve int variable mappings:
        vmapi = {}
        for fname, vmap in self.varmap_int.items():

            if fname in onms:
                f = self.objs.functions[onms.index(fname)]
            elif fname in cnms:
                f = self.cons.functions[cnms.index(fname)]
            else:
                raise KeyError(f"Problem '{self.name}': Function '{fname}' from varmap_int not found in problem")
            if verbosity and f.n_vars_int:
                print(f"Problem '{self.name}', function '{f.name}': Mapping integer variables")

            vmapi[f.name] = {}
            for fvi, fv in enumerate(f.var_names_int):
                pv = None
                for fvm, pvm in vmap.items():
                    if isinstance(fvm, int) and fvm == fvi:
                        pv = pvm
                        break
                    elif fnmatch(fv, fvm):
                        pv = pvm
                        break
                if pv is None:
                    raise KeyError(f"Problem '{self.name}': Int variable '{fvi}: {fv}' of function '{f.name}' unmatched in varmap_int {sorted(list(vmap.keys()))}")           
                elif isinstance(pv, str):
                    if pv not in pnmsi:
                        raise KeyError(f"Problem '{self.name}': Int variable '{fv}' of function '{f.name}' mapped to '{pv}', but not found in problem int variables {sorted(pnmsi)}")
                    pv = pnmsi.index(pv)
                elif not isinstance(pv, int) or (pv < 0 or pv > len(pnmsi)):
                    raise ValueError(f"Problem '{self.name}': Int variable '{fv}' of function '{f.name}' mapped to illegal variable index {pv} for {len(pnmsi)} available int variables in problem")
                if verbosity:
                    print(f"  fvar {fvi}: {fv} --> pvar {pv}: {pnmsi[pv]}")
                vmapi[f.name][fvi] = pv
        self.varmap_int = vmapi
            
        # resolve float variable mappings:
        vmapf = {}
        for fname, vmap in self.varmap_float.items():

            if fname in onms:
                f = self.objs.functions[onms.index(fname)]
            elif fname in cnms:
                f = self.cons.functions[cnms.index(fname)]
            else:
                raise KeyError(f"Problem '{self.name}': Function '{fname}' from varmap_float not found in problem")
            if verbosity and f.n_vars_float:
                print(f"Problem '{self.name}', function '{f.name}': Mapping float variables")

            vmapf[f.name] = {}
            for fvi, fv in enumerate(f.var_names_float):
                pv = None
                for fvm, pvm in vmap.items():
                    if isinstance(fvm, int) and fvm == fvi:
                        pv = pvm
                        break
                    elif fnmatch(fv, fvm):
                        pv = pvm
                        break
                if pv is None:
                    raise KeyError(f"Problem '{self.name}': FLoat variable '{fvi}: {fv}' of function '{f.name}' unmatched in varmap_float {sorted(list(vmap.keys()))}")           
                elif isinstance(pv, str):
                    if pv not in pnmsf:
                        raise KeyError(f"Problem '{self.name}': Float variable '{fv}' of function '{f.name}' mapped to '{pv}', but not found in problem float variables {sorted(pnmsf)}")
                    pv = pnmsf.index(pv)
                elif not isinstance(pv, int) or (pv < 0 or pv > len(pnmsf)):
                    raise ValueError(f"Problem '{self.name}': Float variable '{fv}' of function '{f.name}' mapped to illegal variable index {pv} for {len(pnmsf)} available float variables in problem")
                if verbosity:
                    print(f"  fvar {fvi}: {fv} --> pvar {pv}: {pnmsf[pv]}")
                vmapf[f.name][fvi] = pv
        self.varmap_float = vmapf

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

    def calc_gradients(self, func=None, vars=None):
        """
        Calculate gradients of a function that is linked to the
        problem.

        Usually the function is a `iwopy.core.OptFunctionList`
        object that contains objectives and/or constraints.

        Parameters
        ----------
        func : iwopy.core.OptFunctionList, optional
            The functions to be differentiated, or None
            for a list of all objectives and all constraints 
            (in that order)
        vars : list of int or str, optional
            The float variables for wrt which the
            derivatives are to be calculated, or 
            None for all
        varmap_int : dict, optional
            Mapping from func int variables to
            problem int variables. Key: str or int,
            value: str or int. Updates the stored varmap.
        varmap_float : dict, optional
            Mapping from func float variables to
            problem float variables. Key: str or int,
            value: str or int. Updates the stored varmap.

        Returns
        -------
        gradients : numpy.ndarray
            The gradients of the functio, 
            list element shapes: (n_obj_cmpnts, n_vars)

        """
        TODO
        if func.problem is not self:
            raise ValueError(f"Problem '{self.name}': Attempt to calculate gradient for function '{func.name}' which is linked to different problem '{func.problem.name}'")

        # find differentiation variables:
        if vars is None:
            vrs = list(range(self.n_vars_float))
        else:
            vnms = list(self.var_names_float())
            vrs  = [vnms.index(v) if isinstance(v, str) else int(v) for v in vars]
        n_vars = len(vrs)

        # map integer variables:



        """
        funcs = []

        if objs is not None:
            for ob in objs:
                if isinstance(ob, int):
                    o = self.objs[ob]
                elif isinstance(ob, str):
                    if ob in self.objs_names:
                        o = self.objs[self.objs_names.index(ob)]
                    else:
                        raise KeyError(
                            f"Problem '{self.name}': Objective '{ob}' not found in list {self.objs_names}"
                        )
                else:
                    raise ValueError(
                        f"Problem '{self.name}': Objective '{ob}' not int or str type"
                    )
            funcs.append((o[0], o[2]))
        else:
            funcs += [(o[0], o[2]) for o in self.objs]

        if cons is not None:
            for co in cons:
                if isinstance(co, int):
                    c = self.cons[co]
                elif isinstance(co, str):
                    if co in self.cons_names:
                        c = self.cons[self.cons_names.index(co)]
                    else:
                        raise KeyError(
                            f"Problem '{self.name}': Constraint '{co}' not found in list {self.cons_names}"
                        )
                else:
                    raise ValueError(
                        f"Problem '{self.name}': Constraint '{co}' not int or str type"
                    )
            funcs.append((c[0], c[2]))
        else:
            funcs += [(c[0], c[2]) for c in self.cons]

        return self.fd.calc_gradients(funcs=funcs, **kwargs)
        """

    def initialize(self, verbosity=0):
        """
        Initialize the problem.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if verbosity:
            s = f"Problem '{self.name}' ({type(self).__name__}): Initializing"
            print(s)
            L = len(s)
            print("-" * L)

        n_int = self.n_vars_int()
        n_float = self.n_vars_float()
        if verbosity:
            print(f"  n_vars_int   : {n_int}")
            print(f"  n_vars_float : {n_float}")
            print("-" * L)

        if verbosity:
            print(f"  n_objectives : {len(self.objs)}")
            print(f"  n_obj_cmptns : {self.n_objectives}")
            print("-" * L)
            print(f"  n_constraints: {len(self.cons)}")
            print(f"  n_con_cmptns : {self.n_constraints}")
            print("-" * L)

        if self.n_objectives() == 0:
            raise ValueError("Problem initialized without added objectives.")

        self._maximize = np.zeros(self.n_objectives, dtype=np.bool)

        i0 = 0
        for f in self.objs:
            if not f.initialized:
                f.initialize(verbosity)
            i1 = i0 + f.n_components()
            self._maximize[i0:i1] = np.array(f.maximize(), dtype=np.bool)
            i0 = i1

        for c in self.cons:
            if not c.initialized:
                c.initialize(verbosity)

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
