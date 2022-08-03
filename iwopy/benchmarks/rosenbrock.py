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
    pars : tuple
        The a, b parameters
    ana_deriv : bool
        Switch for analytical derivatives
    name : str
        The function name

    """

    def __init__(
        self,
        problem,
        pars=(1.0, 100.0),
        ana_deriv=False,
        name="f",
    ):
        super().__init__(problem, name, vnames_float=["x", "y"])

        # Parameters of branin function,
        # (a, b)
        self._pars = pars

        self._ana_deriv = ana_deriv

    def f(self, x, y):
        """
        The Rosenbrock function f(x, y)
        """
        a, b = self._pars
        return (a - x) ** 2 + b * (y - x**2) ** 2

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
        a, b = self._pars

        if var == 0:
            return np.array(
                [-2 * (a - x) - 2 * b * (y - x**2) * 2 * x], dtype=np.float64
            )

        elif var == 1:
            return np.array([2 * b * (y - x**2)], dtype=np.float64)

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

class RosenbrockProblem(SimpleProblem):
    """
    Problem definition of benchmark function Rosenbrock.

    Parameters
    ----------
    lower : list of float
        The minimal variable values
    upper : list of float
        The maximal variable values
    initial : list of float
        The initial values
    ana_deriv : bool
        Switch for analytical derivatives
    name : str
        The name of the problem

    Attributes
    ----------
    initial_values : list of float
        The initial values

    """

    def __init__(
        self,
        lower=[-5., -5.],
        upper=[10., 10.],
        initial=[0., 0.],
        ana_deriv=False,
        name="rosenbrock",
    ):

        super().__init__(
            name,
            float_vars={"x": initial[0], "y": initial[1]},
            min_float_vars={"x": lower[0], "y": lower[1]},
            max_float_vars={"x": upper[0], "y": upper[1]},
        )

        self.add_objective(RosenbrockObjective(self, ana_deriv=ana_deriv))
