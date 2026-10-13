iwopy.interfaces.scipy
----------------------
Interface to the `scipy` package

``Optimizer_scipy`` solves continuous single-objective problems with any
algorithm exposed by :func:`scipy.optimize.minimize`. For gradient-capable
methods, it supplies the complete objective and constraint Jacobian through
``Problem.get_jacobian``. Objective and constraint callbacks at the same
point share one cached CSR Jacobian calculation.

``trust-constr`` receives CSR constraint Jacobians without a dense intermediate.
Its native ``sparse_jacobian`` option defaults to ``True``; explicit values are
preserved. Set it to ``False`` when selecting a dense-only ``QRFactorization``
or ``SVDFactorization``. Other methods, including SLSQP, receive dense
constraint arrays as required by their native interface. Scalar objective
gradients remain dense vectors for every SciPy method. Sparse callbacks do not
remove dense Hessian approximations or prevent every native factorization
fallback to dense storage.

By default, Jacobians are requested with ``pop=True``. A wrapper such as
``LocalFD`` can therefore evaluate all finite-difference points together
through the problem's population interface. Set ``vectorize=False`` to use
serial iwopy Jacobian evaluation. The interface does not fall back to SciPy's
scalar finite differences, so every required derivative must be analytical or
provided by a problem wrapper.

The derivative path applies to ``CG``, ``BFGS``, ``Newton-CG``, ``L-BFGS-B``,
``TNC``, ``SLSQP``, ``trust-constr``, ``dogleg``, ``trust-ncg``,
``trust-exact``, and ``trust-krylov``, as well as SciPy's automatically chosen
default and custom callable methods. Methods that require a Hessian or
Hessian-vector product still need explicit ``hess`` or ``hessp`` parameters.

Select a method with ``optimizer`` on the class, or with
``create_optimizer(family="scipy", optimizer=..., problem=...)``. Supply
``tol``, ``memory_size``, and ``vectorize`` explicitly; flat native method
keywords such as ``maxiter`` and ``ftol`` become SciPy's method options.

``Nelder-Mead``, ``Powell``, ``COBYLA``, and ``COBYQA`` are derivative-free;
SciPy offers no population-evaluation hook for their trial points, so no
Jacobian is supplied. Constraint components are grouped by equality, lower,
and upper bounds for every constrained method.

The parameters ``bounds``, ``callback``, ``constraints``, and ``jac`` are
managed by the interface. Integer variables are rejected because
``scipy.optimize.minimize`` operates on continuous variables.

``Optimizer_scipy_de`` provides genuine population-vectorized differential
evolution for bounded single-objective floating, integer, and mixed problems.
It uses deferred updating, one worker, and no scalar polishing. See
:ref:`api-scipy-de` and its :ref:`explicit setup <case-scipy-de>` in the catalog.
Its explicit controls include ``maxiter``, ``popsize``, ``seed``, ``tol``,
``atol``, and ``vectorize``; remaining native options are flat keywords.
``popsize`` is a multiplier per free dimension, not an absolute population size.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/scipy/index
