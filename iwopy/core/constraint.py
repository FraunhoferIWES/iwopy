import numpy as np

from iwopy.core.function import Function

class Constraint(Function):
    """
    Abstract base class for optimization 
    constraints.

    Parameters
    ----------
    name: str
        The function name
    vnames_int : list of str, optional
        The integer variable names. Useful for mapping
        problem variables to function variables
    vnames_float : list of str, optional
        The float variable names. Useful for mapping
        problem variables to function variables
    cnames : list of str, optional
        The names of the components

    """

    def __init__(
            self, 
            problem, 
            name, 
            vnames_int=None, 
            vnames_float=None,
            cnames=None
        ):
        super().__init__(problem, name, vnames_int, 
                            vnames_float, cnames)
    
    def get_bounds(self):
        """
        Returns the bounds for all components.

        Express non-existing bounds using np.inf.

        Returns
        -------
        min : np.array
            The lower bounds, shape: (n_components,)
        max : np.array
            The upper bounds, shape: (n_components,)

        """
        return  np.full(self.n_components(), -np.inf, dtype=np.float64), \
                np.zeros(self.n_components(), dtype=np.float64)

    def check_individual(self, constraint_values, verbosity=0):
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
        vals   = constraint_values
        mi, ma = self.get_bounds()
        out    = ( vals >= mi ) & ( vals <= ma )
        
        if verbosity:
            cnames = self.names()
            for ci in range(self.n_components()):
                val = f"{cnames[ci]} = {vals[ci]:.3f}"
                suc = "OK" if out[ci] else "FAILED"
                print(f"Constraint {val:<30} {suc}")

        return out

    def check_population(self, constraint_values, verbosity=0):
        """
        Check if the constraints are fullfilled for the
        given population.

        Parameters
        ----------
        constraint_values : np.array
            The constraint values, shape: (n_pop, n_components,)
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        values : np.array
            The boolean result, shape: (n_pop, n_components)    

        """
        vals   = constraint_values
        mi, ma = self.get_bounds()
        
        out = ( vals >= mi[None, :] ) & ( vals <= ma[None, :] )

        if verbosity:
            cnames = self.names()
            for ci in range(self.n_components()):
                suc = "OK" if np.all(out[ci]) else "FAILED"
                print(f"Constraint {cnames[ci]:<20} {suc}")
        
        return out
