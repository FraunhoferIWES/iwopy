import numpy as np

class FiniteDiff:
    """
    Performs finite difference calculations
    on problem functions.

    Parameters
    ----------
    deltas : dict or float
        The finite different step sizes.
        If float, application to all variables.
        If dict, key: variable name str or parts
        of variable name str or variable index
        in problem's float variables. 
        Value: float, the step size

    Arguments
    ---------
    deltas : dict or float
        The finite different step sizes.
        If float, application to all variables.
        If dict, key: variable name str or parts
        of variable name str or variable index
        in problem's float variables. 
        Value: float, the step size

    """
    def __init__(self, deltas):
        self.deltas = deltas
    
    def _find_delta(self, problem, vi):
        """
        Helper function for extracting delta data
        """
        for vi in vars:
            if isinstance(self.deltas, dict):
                if vi in self.deltas:
                    d = self.deltas[vi]
                else:
                    v = problem.var_names_float()[vi]
                    if v in self.deltas:
                        d = self.deltas[v]
                    else:
                        d = None
                        for k in self.deltas.keys():
                            for w in problem.var_names_float():
                                if k in w:
                                    d = self.deltas[w]
                                    break
                            if d is not None:
                                break
                        if d is None:
                            raise KeyError(f"Missing matching pattern for variable '{v}' or index {vi} in deltas, found {sorted(list(self.deltas.keys()))}")
            else:
                d = self.deltas
        return d

    def calc_gradients(
            self,
            funcs,
            pvars0_int, 
            pvars0_float,  
            order, 
            pop, 
            results0=None
        ):
        """
        Calculate the gradient of a problem
        function wrt the given problem variables.

        Parameters
        ----------
        funcs : list of tuple
            Tuples of length 2. First entry:
            iwopy.core.Function, the function to be
            differentiated (all components). Second 
            entry: list of int, the problem float variables 
            on which the function depends.
        pvars0_int : numpy.ndarray
            The evaluation problem function int variables,
            shape: (n_problem_vars_int,)
        pvars0_float : numpy.ndarray
            The evaluation problem function float variables,
            shape: (n_problem_vars_float,)
        order : int
            The order of the gradient, either -1 (backward),
            1 (forward) or 2
        pop : bool
            Flag for simulataneous calculation using 
            populations
        results0 : numpy.ndarray, optional
            The functions result values at pvars0
            variables, shape: (n_funcs_cmpnts,)

        Returns
        -------
        gradients : numpy.ndarray
            The gradients, shape: (n_funcs_cmpnts, n_problem_vars_float)

        """
        # check problems:
        problem = funcs[0][0].problem
        for fi in range(1, n_funcs):
            if funcs[fi][0].problem is not problem:
                raise ValueError(f"Functions '{funcs[0][0].name}' and '{funcs[fi][0].name}' have different underlying problems: '{problem.name}' vs. '{funcs[fi][0].problem.name}'")

        # find vars:
        vars = set()
        for f in funcs:
            vars.update(f[1])

        # prepare:
        n_funcs   = len(funcs)
        n_cmpnts  = sum([f[0].n_components() for f in funcs]) 
        n_vars    = len(vars)
        n_pvars   = problem.n_vars_float

        # plan calculation for order 1:
        if order == 1 or order == -1:

            i0       = 0 if results0 is not None else 1
            n_pop    = n_vars + i0
            cvars    = np.zeros((n_pop, n_pvars), dtype=np.float64)
            cvars[:] = pvars0_float[None, :]
            sgn      = 1 if order > 0 else -1
            
            for vi in vars:
                d = self._find_delta(problem, vi)
                cvars[i0 + vi][vi] += sgn * d

        # plan calculation for order 2:
        if order == 2:

            i0       = 0 if results0 is not None else 1
            n_pop    = 2 * n_vars + i0
            cvars    = np.zeros((n_pop, n_pvars), dtype=np.float64)
            cvars[:] = pvars0_float[None, :]
            
            for vi in vars:
                d = self._find_delta(problem, vi)
                cvars[i0 + 2 * vi][vi]     += sgn * d
                cvars[i0 + 2 * vi + 1][vi] -= sgn * d

        # run calculation, by individuals:
        cres = np.full((n_pop, n_cmpnts), np.nan, dtype=np.float64)
        if not pop:
            for n, pvars in enumerate(cvars):
                pres = problem.apply_individual(pvars0_int, pvars)
                i0 = 0
                for f in funcs:
                    i1 = i0 + f[0].n_components()
                    cres[n, i0:i1] = f[0].calc_individual(pvars0_int, pvars, pres)
                    i0 = i1
            
        # run calculation, by population:
        else:
            pres = problem.apply_population(pvars0_int, cvars)
            i0 = 0
            for f in funcs:
                i1 = i0 + f[0].n_components()
                cres[:, i0:i1] = f[0].calc_population(pvars0_int, pvars, pres)
                i0 = i1
        
        # cleanup:
        del pres, cvars
        
        # prepare gradients:
        gradients = np.zeros((n_pvars, n_cmpnts), dtype=np.float64)
        if results0 is None:
            res0 = cres[0]
            cres = cres[1:]
        else:
            res0 = results0

        # calc gradient order 1:
        if order == 1:
            i0 = 0
            for f in funcs:
                i1 = i0 + f[0].n_components()
                gradients[:, i0:i1][f[1]] = ( cres[:, f[1]] - res0[])
                i0 = i1
