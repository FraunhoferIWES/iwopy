iwopy.interfaces.pygmo
----------------------
Interface to the `pygmo` package

Select a driver with ``optimizer`` or use the ``pygmo`` family of
``create_optimizer``. Explicit controls include ``pop_size``, ``seed``,
``vectorize``, ``norandom_first``, ``c_tol``, ``problem_verbosity``, and
``algorithm_verbosity``. Additional native algorithm settings are flat keywords.
``norandom_first=None`` preserves a single-start initial point when
``pop_size=1``; batch fitness requires native algorithm support.

Select NLopt leaf solvers through ``create_optimizer(family="nlopt",
optimizer="lbfgs", problem=...)``. Direct class construction instead supplies
``optimizer="nlopt", nlopt_optimizer="lbfgs"``. Both require PyGMO with NLopt
support; ``nlopt`` is a family, not a new dependency. Construction loads PyGMO;
neither the class constructor nor the factory initializes or solves the optimizer.
See :doc:`api_core` and :doc:`optimizer_catalog`.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/interfaces/pygmo/index
