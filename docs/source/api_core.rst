iwopy.core
----------
Core functionality and abstract classes

Optimizer Creation
^^^^^^^^^^^^^^^^^^

Use ``from iwopy import create_optimizer`` to select an optimizer by its
case-sensitive ``family`` and ``optimizer``. The function returns an
uninitialized optimizer without initializing the problem or running a search.
Existing optimizer class names remain public for direct construction.

.. code-block:: python

    from iwopy.benchmarks import QuadraticProblem
    from iwopy import create_optimizer

    solver = create_optimizer(
        family="iwopy",
        optimizer="SLSQP",
        problem=QuadraticProblem(),
        maxiter=100,
        ftol=1e-9,
    )
    solver.initialize(verbosity=0)
    results = solver.solve(verbosity=0)

Families are ``iwopy``, ``scipy``, ``cma``, ``pymoo``, ``pygmo``, ``nlopt``,
``nevergrad``, ``optuna``, and ``deap``. iwopy selects ``GG`` or ``SLSQP``;
pycma selects ``CMAES``. SciPy accepts a native minimize method name, a callable,
or ``None`` for automatic selection. Its ``differential_evolution`` selector
chooses the separate population adapter, not ``scipy.optimize.minimize``.
Other selectors are the backend algorithm/sampler names listed in the
:doc:`optimizer_catalog`.

The ``nlopt`` family selects native solvers such as ``lbfgs`` through PyGMO;
install the ``pygmo`` extra with NLopt support. The factory supplies the PyGMO
driver and leaf selector internally. Direct ``Optimizer_pygmo`` construction
instead uses ``optimizer="nlopt", nlopt_optimizer="lbfgs"``.

Each class has explicit parameters. Additional native algorithm options, where
supported, are flat keyword arguments: SciPy minimization's ``maxiter`` and
``ftol`` become native method options, while ``tol``, ``hess``, and ``hessp``
remain top-level controls. There are no common parameter dictionaries or
``Optimizer.new`` entry point. ``vectorize`` consistently controls iwopy
population evaluation or Jacobian batching; pycma always evaluates populations.
``pop_size`` denotes an absolute population size, except that SciPy DE retains
its native ``popsize`` multiplier per free dimension. ``memory_size`` denotes
visited-point capacity for GG and evaluation-cache capacity for SLSQP and
SciPy minimization. The function is owned by core and re-exported at the root;
``iwopy.core.create_optimizer`` remains the same function object.

Optional dependencies retain their adapter lifecycle: PyGMO loads during
construction; other optional adapters load during initialization. Selecting one
family does not load another family's optional backend. Native subsolver objects
are backend configuration, not outer iwopy optimizer creation.

Jacobians
^^^^^^^^^

``Problem.get_jacobian()`` returns a SciPy CSR matrix with component rows and
selected floating-variable columns. Default rows order objectives before
constraints. ``calc_jacobian()`` is the extension point for analytical or
numerical assembly. Exact zeros are implicit; unavailable analytical partial
derivatives remain explicit ``NaN`` entries until a wrapper resolves them.
Unresolved entries are rejected before returning a complete Jacobian.

The matrix is a Jacobian even for a single component. A scalar objective's
gradient is its one-dimensional row vector. Convert only at a dense boundary:

.. code-block:: python

    import numpy as np
    from iwopy.benchmarks import QuadraticProblem

    problem = QuadraticProblem()
    jacobian = problem.get_jacobian(
        np.empty(0, dtype=np.int32),
        np.asarray(problem.initial_values_float()),
    )
    objective_gradient = jacobian.getrow(0).toarray().ravel()

``LocalFD`` fills only unavailable stored derivatives, retaining analytical
values and independent entries. The matrix APIs use singular Jacobian names
without legacy aliases. See :doc:`api_optimizers` and
:doc:`api_interfaces_scipy` for native sparse and dense boundaries.

Pipelines
^^^^^^^^^

``Pipeline.run()`` executes the half-open stage range selected by
``start_stage`` and ``end_stage``. Supply ``initial_results`` to seed the first
selected stage from an application checkpoint. For a later-stage restart, that
stage also receives the immediately preceding registered stage as
``prev_stage``; the preceding stage is not run again. Stage-reported failures
stop execution and retain their ``(False, results)`` return without leaving the
pipeline in a running state.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/core/index
