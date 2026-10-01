iwopy.interfaces.scipy
----------------------
Interface to the `scipy` package

``Optimizer_scipy`` solves continuous single-objective problems with any
algorithm exposed by :func:`scipy.optimize.minimize`. For gradient-capable
methods, it supplies the complete objective and constraint Jacobian through
``Problem.get_gradients``. Objective and constraint callbacks at the same
point share one cached Jacobian calculation.

By default, Jacobians are requested with ``pop=True``. A wrapper such as
``LocalFD`` can therefore evaluate all finite-difference points together
through the problem's population interface. Set ``vectorized=False`` to use
serial iwopy gradient evaluation. The interface does not fall back to SciPy's
scalar finite differences, so every required derivative must be analytical or
provided by a problem wrapper.

The gradient path applies to ``CG``, ``BFGS``, ``Newton-CG``, ``L-BFGS-B``,
``TNC``, ``SLSQP``, ``trust-constr``, ``dogleg``, ``trust-ncg``,
``trust-exact``, and ``trust-krylov``, as well as SciPy's automatically chosen
default and custom callable methods. Methods that require a Hessian or
Hessian-vector product still need it in ``scipy_pars``.

``Nelder-Mead``, ``Powell``, ``COBYLA``, and ``COBYQA`` are derivative-free;
SciPy offers no population-evaluation hook for their trial points, so no
Jacobian is supplied. Constraint components are grouped by equality, lower,
and upper bounds for every constrained method.

The parameters ``bounds``, ``callback``, ``constraints``, and ``jac`` are
managed by the interface. Integer variables are rejected because
``scipy.optimize.minimize`` operates on continuous variables.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/scipy/index
