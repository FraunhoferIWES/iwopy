import numpy as np

from iwopy import SimpleProblem, Objective, Constraint

class RosenbrockObjective(Objective):
    """
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
    problem : iwopy.Problem
        The underlying optimization problem
    x_bounds : tuple
        The (min, max) values for x, 
        use np.inf for infinite
    y_bounds : tuple
        The (min, max) values for y, 
        use np.inf for infinite
    pars
    name : str
        The function name

    """
    def __init__(
            self, 
            problem,  
            x_bounds,
            y_bounds,
            pars=(1., 100.),
            name="f",
        ):
        super().__init__(problem, name, vnames_float=["x", "y"])

        # Parameters of branin function,
        # (a, b)
        self._pars = pars

    def f(self, x, y):
        """
        The Rosenbrock function f(x, y)
        """
        a, b = self._pars
        return (a - x)**2 + b * (y - x**2)**2

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
        flags : np.array
            Bool array for component maximization,
            shape: (n_components,)

        """
        return [False]
    
    def calc_individual(self, vars_int, vars_float, problem_results):
        """
        Calculate values for a single individual of the
        underlying problem.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        problem_results : Any
            The results of the variable application
            to the problem

        Returns
        -------
        values : np.array
            The component values, shape: (n_components,)

        """
        x, y = vars_float
        return np.array([self.f(x,y)])

    def calc_population(self, vars_int, vars_float, problem_results):
        """
        Calculate values for all individuals of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        problem_results : Any
            The results of the variable application
            to the problem

        Returns
        -------
        values : np.array
            The component values, shape: (n_pop, n_components,)

        """
        x = vars_float[:, 0]
        y = vars_float[:, 1]
        return self.f(x,y)[:, None]
class RosenbrockConstraints(Constraint):
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
    problem: iwopy.Problem
        The underlying optimization problem
    base_name: str
        The base name of the objective functions

    """
    def __init__(self, problem, name='c'):
        super().__init__(problem, name, vnames_float=["x", "y"])
    
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

class RosenbrockProblem(SimpleProblem):
    """
    Problem definition of benchmark function Rosenbrock.

    Parameters
    ----------
    name : str
        The name of the problem
    initial_values : list of float
        The initial values
    
    Attributes
    ----------
    initial_values : list of float
        The initial values

    """

    def __init__(self, name="branin", initial_values=[1., 1.]):
        
        super().__init__(
            name,
            float_vars={"x": initial_values[0], "y": initial_values[1]},
            min_float_vars={"x": -5., "y": 0.},
            max_float_vars={"x": 10., "y": 15},
        )

        self.add_objective(BraninObjective(self))
