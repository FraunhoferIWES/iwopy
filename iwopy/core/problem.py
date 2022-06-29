from iwopy.core.finite_diff import FiniteDiff
import numpy as np
from abc import ABCMeta

from iwopy.core.base import Base

class Problem(Base, metaclass=ABCMeta):
    """
    Abstract base class for optimization problems.

    Parameters
    ----------
    name: str
        The problem's name
    deltas : dict or float, optional
        The finite different step sizes.
        If float, application to all variables.
        If dict, key: variable name str or parts
        of variable name str or variable index
        in problem's float variables. 
        Value: float, the step size

    Attributes
    ----------
    objs : list of tuple
        Tuples with three entries: 0 = iwopy.Objective, 
        the objective function, 1 = list of int, the indices
        of the integer variables on which the objective
        depends, 2 = list of int, same for float variables
    cons : list of tuple
        Tuples with three entries: 0 = iwopy.Constraint, 
        the constraint function, 1 = list of int, the indices
        of the integer variables on which the constraint
        depends, 2 = list of int, same for float variables
    osizes : list of int
        The number of compenents of the
        objectives
    csizes : list of int
        The number of compenents of the
        constraints
    fd : iwopy.core.FiniteDiff
        The finite difference object, or None

    """

    def __init__(self, name, deltas=None):
        super().__init__(name)

        self.objs   = []
        self.cons   = []
        self.osizes = []
        self.csizes = []

        self.fd = None if deltas is None else FiniteDiff(deltas)
    
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

    def add_objective(
            self, 
            objective, 
            varmap_int={},
            varmap_float={}
        ):
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
        varsi = []
        for i, v in enumerate(objective.var_names_int):
            w = varmap_int.get(i, varmap_int.get(v, v))
            if isinstance(w, int):
                j = w
            elif w in self.var_names_int():
                j = self.var_names_int().index(j)
            else:
                raise KeyError(f"Problem '{self.name}', objective '{objective.name}': Objective variable '{v}' mapped onto int problem variable '{w}', which is not in the int vars list {self.var_names_int()}")
            if j not in range(self.n_vars_int):
                raise ValueError(f"Problem '{self.name}', objective '{objective.name}': Objective variable '{v}' mapped onto problem int variable index '{j}', which exceeds existing {self.n_vars_int} int vars")
            varsi.append(j)

        varsf = []
        for i, v in enumerate(objective.var_names_float):
            w = varmap_float.get(i, varmap_float.get(v, v))
            if isinstance(w, int):
                j = w
            elif w in self.var_names_float():
                j = self.var_names_float().index(j)
            else:
                raise KeyError(f"Problem '{self.name}', objective '{objective.name}': Objective variable '{v}' mapped onto float problem variable '{w}', which is not in the float vars list {self.var_names_float()}")
            if j not in range(self.n_vars_float):
                raise ValueError(f"Problem '{self.name}', objective '{objective.name}': Objective variable '{v}' mapped onto problem float variable index '{j}', which exceeds existing {self.n_vars_float} float vars")
            varsf.append(j)
        
        self.objs.append((objective, varsi, varsf))
        self.osizes.append(objective.n_components())
    
    @property
    def objs_names(self):
        """
        The names of the objectives (not components)

        Returns
        -------
        names : list of str
            The names of the objectives 
            (not components)

        """
        return [o.name for o in self.objs]

    def add_constraint(
            self, 
            constraint, 
            varmap_int={},
            varmap_float={}
        ):
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
        varsi = []
        for i, v in enumerate(constraint.var_names_int):
            w = varmap_int.get(i, varmap_int.get(v, v))
            if isinstance(w, int):
                j = w
            elif w in self.var_names_int():
                j = self.var_names_int().index(j)
            else:
                raise KeyError(f"Problem '{self.name}', constraint '{constraint.name}': Objective variable '{v}' mapped onto int problem variable '{w}', which is not in the int vars list {self.var_names_int()}")
            if j not in range(self.n_vars_int):
                raise ValueError(f"Problem '{self.name}', constraint '{constraint.name}': Objective variable '{v}' mapped onto problem int variable index '{j}', which exceeds existing {self.n_vars_int} int vars")
            varsi.append(j)

        varsf = []
        for i, v in enumerate(constraint.var_names_float):
            w = varmap_float.get(i, varmap_float.get(v, v))
            if isinstance(w, int):
                j = w
            elif w in self.var_names_float():
                j = self.var_names_float().index(j)
            else:
                raise KeyError(f"Problem '{self.name}', constraint '{constraint.name}': Objective variable '{v}' mapped onto float problem variable '{w}', which is not in the float vars list {self.var_names_float()}")
            if j not in range(self.n_vars_float):
                raise ValueError(f"Problem '{self.name}', constraint '{constraint.name}': Objective variable '{v}' mapped onto problem float variable index '{j}', which exceeds existing {self.n_vars_float} float vars")
            varsf.append(j)
        
        self.objs.append((constraint, varsi, varsf))
        self.csizes.append(constraint.n_components())

    @property
    def cons_names(self):
        """
        The names of the constraints (not components)

        Returns
        -------
        names : list of str
            The names of the constraints 
            (not components)

        """
        return [c.name for c in self.cons]

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
        return sum(self.osizes)

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
        return sum(self.csizes)

    def calc_gradients(self, objs=None, cons=None, **kwargs):
        """
        Calculate gradients from finite differences.

        Parameters
        ----------
        objs : list of int or str, optional
            The objectives to be differentiated
            (all components), or None for all
        cons : list of int or str, optional
            The constraints to be differentiated
            (all components), or None for all
        kwargs: dict, optional
            Additional parameters forwarded to
            the `calc_gradients` function of the
            `iwopy.FiniteDiff` object

        Returns
        -------
        gradients : numpy.ndarray
            The gradients, shape: (n_funcs_cmpnts, n_problem_vars_float)

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
                        raise KeyError(f"Problem '{self.name}': Objective '{ob}' not found in list {self.objs_names}")
                else:
                    raise ValueError(f"Problem '{self.name}': Objective '{ob}' not int or str type")
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
                        raise KeyError(f"Problem '{self.name}': Constraint '{co}' not found in list {self.cons_names}")
                else:
                    raise ValueError(f"Problem '{self.name}': Constraint '{co}' not int or str type")
            funcs.append((c[0], c[2]))
        else:
            funcs += [(c[0], c[2]) for c in self.cons]

        return self.fd.calc_gradients(funcs=funcs, **kwargs)

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
            print("-"*L)
        
        n_int   = self.n_vars_int()
        n_float = self.n_vars_float()
        if verbosity:
            print(f"  n_vars_int   : {n_int}")
            print(f"  n_vars_float : {n_float}")
            print("-"*L)

        if verbosity:
            print(f"  n_objectives : {len(self.objs)}")
            print(f"  n_obj_cmptns : {self.n_objectives}")
            print("-"*L)
            print(f"  n_constraints: {len(self.cons)}")
            print(f"  n_con_cmptns : {self.n_constraints}")
            print("-"*L)

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

        n_pop = vars_float.shape[0] if vars_float is not None and len(vars_float.shape) \
                    else vars_int.shape[0]
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
        val   = constraint_values
        n_pop = val.shape[0]
        out   = np.zeros((n_pop, self.n_constraints()), dtype=np.bool)

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
            objs[i0:i1] = f.finalize_individual(vars_int, vars_float, results, verbosity)
            i0 = i1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[i0:i1] = c.finalize_individual(vars_int, vars_float, results, verbosity)
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
        n_pop = vars_float.shape[0] if vars_float is not None and len(vars_float.shape) \
                    else vars_int.shape[0]
        objs = np.full((n_pop, self.n_objectives), np.nan, dtype=np.float)
        cons = np.full((n_pop, self.n_constraints), np.nan, dtype=np.float)

        results = self.apply_population(vars_int, vars_float)

        i0 = 0
        for f in self.objs:
            i1 = i0 + f.n_components()
            objs[:, i0:i1] = f.finalize_population(vars_int, vars_float, results, verbosity)
            i0 = i1

        i0 = 0
        for c in self.cons:
            i1 = i0 + c.n_components()
            cons[:, i0:i1] = c.finalize_population(vars_int, vars_float, results, verbosity)
            i0 = i1

        return results, objs, cons
