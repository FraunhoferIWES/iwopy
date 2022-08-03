import numpy as np

from iwopy import SimpleProblem, Objective


class BraninObjective(Objective):
    """
    The objective function for the Branin problem.

    The Branin, or Branin-Hoo function is defined as

    f(x,y) = a(y-bx^2+cx-r)^2 + s(1-t)cos(x)+s

    Recommended values for the parameters are:
    a = 1
    b = 5.1/(4*pi^2)
    c = 5/pi
    r = 6
    s = 10
    t = 1/(8*pi)

    Domain:
    x = [-5, 10]
    y = [0, 15]

    The Branin function has three global minima at

    (x,y) = (-pi, 12.275), (pi, 2.275), (9.42478, 2.475)

    with a function value of

    f(x,y) = 0.397887

    Parameters
    ----------
    problem: iwopy.Problem
        The underlying optimization problem
    ana_deriv : bool
        Switch for analytical derivatives
    name: str
        The function name

    """

    def __init__(self, problem, ana_deriv=False, name="f"):
        super().__init__(problem, name, vnames_float=["x", "y"])

        # Parameters of branin function,
        # (a, b, c, r, s, t)
        self._pars = (
            1,
            5.1 / (4 * np.pi**2),
            5 / np.pi,
            6,
            10,
            1 / (8 * np.pi),
        )

        self._ana_deriv = ana_deriv

    def f(self, x, y):
        """
        The Branin function f(x, y)
        """
        a, b, c, r, s, t = self._pars
        return a * (y - b*x**2 + c*x - r)** 2 + s*(1 - t)*np.cos(x) + s

    def ana_deriv(self, vars_int, vars_float, var, components=None):
        """
        Calculates the analytic derivative, if possible.

        Use `numpy.nan` if analytic derivatives cannot be calculated.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        var : int
            The index of the differentiation float variable
        components : list of int
            The selected components, or None for all

        Returns
        -------
        deriv : numpy.ndarray
            The derivative values, shape: (n_sel_components,)

        """
        if not self._ana_deriv:
            return super().ana_deriv(vars_int, vars_float, var, components)

        x, y = vars_float
        a, b, c, r, s, t = self._pars

        if var == 0:
            return np.array(
                [
                    2*a*(y - b*x**2 + c*x - r)*(-2*b*x + c) -s*(1 - t)*np.sin(x)
                ], dtype=np.float64
            )

        elif var == 1:
            return np.array([2*a*(y - b*x**2 + c*x - r)], dtype=np.float64)

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
        return np.array([self.f(x, y)])

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
        return self.f(x, y)[:, None]


class BraninProblem(SimpleProblem):
    """
    Problem definition of benchmark function Branin.

    Parameters
    ----------
    name : str
        The name of the problem
    ana_deriv : bool
        Switch for analytical derivatives
    initial_values : list of float
        The initial values

    Attributes
    ----------
    initial_values : list of float
        The initial values

    """

    def __init__(self, name="branin", initial_values=[1.0, 1.0], ana_deriv=False):

        super().__init__(
            name,
            float_vars={"x": initial_values[0], "y": initial_values[1]},
            min_float_vars={"x": -5.0, "y": 0.0},
            max_float_vars={"x": 10.0, "y": 15},
        )

        self.add_objective(BraninObjective(self, ana_deriv=ana_deriv))
