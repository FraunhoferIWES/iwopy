import numpy as np

from iwopy import SimpleProblem, SimpleObjective


class RosenbrockObjective(SimpleObjective):
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
        super().__init__(problem, name, n_components=1, has_ana_derivs=ana_deriv)

        # Parameters of branin function,
        # (a, b)
        self._pars = pars

    def f(self, x, y):
        """
        The Rosenbrock function f(x, y)
        """
        a, b = self._pars
        return (a - x) ** 2 + b * (y - x**2) ** 2

    def g(self, var, x, y, components=None):
        """
        The derivative of the Rosenbrock function
        """
        a, b = self._pars
        if var == 0:
            return -2 * (a - x) - 2 * b * (y - x**2) * 2 * x
        else:
            return 2 * b * (y - x**2)


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
        lower=[-5.0, -5.0],
        upper=[10.0, 10.0],
        initial=[0.0, 0.0],
        ana_deriv=False,
        name="rosenbrock",
    ):

        super().__init__(
            name,
            float_vars={"x": initial[0], "y": initial[1]},
            min_values_float={"x": lower[0], "y": lower[1]},
            max_values_float={"x": upper[0], "y": upper[1]},
        )

        self.add_objective(RosenbrockObjective(self, ana_deriv=ana_deriv))
