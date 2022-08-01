import numpy as np

from iwopy.core.problem import OptProblem
from iwopy.core.objective import ObjFunction
from iwopy.core.constraint import OptConstraint

class Problem(OptProblem):
    """
    Problem definition of benchmark function Rosenbrock.

    The Rosenbrock function is defined as

    f(x,y) = (a-x)^2 + b(y-x^2)^2

    Recommended values for the parameters are:
    a = 1
    b = 100

    Domain:
    x = [-inf, inf]
    y = [-inf, inf]

    The unconstraint Rosenbrock function has a global minima at

    (x,y) = (1,1)

    with a function value of

    f(x,y) = 0

    Parameters
    ----------
    x_bounds: tuple
        The (min, max) values for x
    y_bounds: tuple
        The (min, max) values for y
    name: str
        The name of the problem

    """
    def __init__(
            self, 
            x_bounds=(-np.inf, np.inf),
            y_bounds=(-np.inf, np.inf),
            name="rosenbrock"
        ):
        super().__init__(name)

        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def n_vars_int(self):
        """
        Returns the number of integer variables.

        Returns
        -------
        int:
            The number of integer variables

        """
        return 0

    def n_vars_float(self):
        """
        Return the number of float variables

        Returns
        -------
        int:
            The number of float variables
        
        """
        return 2

    def vars_float_info(self):
        """
        Get basic info about the float variables.

        Returns
        -------
        names: list
            The variable names, len = n_vars_float
        min: np.array
            The minimum float values, shape: (n_vars_float)
        max: np.array
            The maximum float values, shape: (n_vars_float)
        initial: np.array
            The initial float values, shape: (n_vars_float)

        """
        return  ['x','y'], \
                np.array([self.x_bounds[0], self.y_bounds[0]]), \
                np.array([self.x_bounds[1], self.y_bounds[1]]), \
                np.array([1., 0.])


class Objective(ObjFunction):
    """
    The objective function for the Rosenbrock problem.

    Parameters
    ----------
    problem: iwopy.OptProblem
        The underlying optimization problem
    base_name: str
        The base name of the objective functions

    """
    def __init__(self, problem, base_name="f"):
        super().__init__(problem, base_name)
    
    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return 1

    def maximize(self):
        """
        Returns flag for maximization of each component.

        Returns
        -------
        flags: np.array
            Bool array for component maximization,
            shape: (n_components)

        """
        return np.zeros(1, dtype=np.bool)

    def calc_individual(self, vars_int, vars_float, problem_results):
        """
        Calculate values for a single individual of the 
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_vars_float)
        problem_results: object
            The results of the variable application 
            to the problem  

        Returns
        -------
        values: np.array
            The component values, shape: (n_components)    

        """
        x = vars_float[0]
        y = vars_float[1]

        # Parameters of rosenbrock function
        a = 1
        b = 100

        result = (a-x)**2 + b*(y-x**2)**2

        return np.array([result])

    def calc_population(self, vars_int, vars_float, problem_results):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: object
            The results of the variable application 
            to the problem  

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_components) 

        """

        x = vars_float[:, 0]
        y = vars_float[:, 1]

        # Parameters of rosenbrock function
        a = 1
        b = 100

        result = (a-x)**2 + b*(y-x**2)**2

        return result[:, None]


class Constraints(OptConstraint):
    """
    The constraints for the Rosenbrock problem.

    The Rosenbrock function is defined as

    f(x,y) = (a-x)^2 + b(y-x^2)^2

    subjected to:
    (x-1)^3 - y + 1 <= 0 ,
    x + y - 3 <= 0

    Recommended values for the parameters are:
    a = 1
    b = 100

    Domain:
    x = [-1.5, 1.5]
    y = [-0.5, 2.5]

    The constrained Rosenbrock function has a global minima at

    (x,y) = (1.68,1.32)

    with a function value of

    f(x,y) = 229

    Parameters
    ----------
    problem: iwopy.OptProblem
        The underlying optimization problem
    base_name: str
        The base name of the objective functions

    """
    def __init__(self, problem, base_name='c'):
        super().__init__(problem, base_name)
    
    def n_components(self):
        """
        Returns the number of components of the
        function.

        Returns
        -------
        int:
            The number of components.

        """
        return 2

    def get_bounds(self):
        """
        Returns the bounds for all components.

        Express non-existing bounds using np.inf.

        Returns
        -------
        min: np.array
            The lower bounds, shape: (n_components)
        max: np.array
            The upper bounds, shape: (n_components)

        """
        return np.array([-np.inf, -np.inf]), np.array([0., 0.])

    def calc_individual(self, vars_int, vars_float, problem_results):
        """
        Calculate values for a single individual of the 
        underlying problem.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_vars_float)
        problem_results: object
            The results of the variable application 
            to the problem  

        Returns
        -------
        values: np.array
            The component values, shape: (n_components)    

        """

        x  = vars_float[0]
        y  = vars_float[1]
        c1 = (x-1)**3 - y + 1   
        c2 = x + y - 3       

        return np.array([c1, c2])

    def calc_population(self, vars_int, vars_float, problem_results):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int: np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float: np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results: object
            The results of the variable application 
            to the problem  

        Returns
        -------
        values: np.array
            The component values, shape: (n_pop, n_components) 

        """

        n_pop = vars_float.shape[0]

        x  = vars_float[:, 0]
        y  = vars_float[:, 1]

        out       = np.zeros((n_pop, self.n_components()))
        out[:, 0] = (x-1)**3 - y + 1  
        out[:, 1] = x + y - 3 

        return out