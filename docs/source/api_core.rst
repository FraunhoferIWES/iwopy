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
        max_iterations=100,
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
supported, are flat keyword arguments: SciPy minimization's ``ftol`` becomes
a native method option, while ``tol``, ``hess``, and ``hessp``
remain top-level controls. There are no common parameter dictionaries or
``Optimizer.new`` entry point. ``vectorize`` consistently controls iwopy
population evaluation or Jacobian batching. ``pop_size`` denotes an absolute
population size; SciPy DE instead exposes ``population_multiplier`` per free
dimension. ``memory_size`` denotes
visited-point capacity for GG and evaluation-cache capacity for SLSQP and
SciPy minimization. The function is owned by core and re-exported at the root;
``iwopy.core.create_optimizer`` remains the same function object.

Optional dependencies retain their adapter lifecycle: PyGMO loads during
construction; other optional adapters load during initialization. Selecting one
family does not load another family's optional backend. Native subsolver objects
are backend configuration, not outer iwopy optimizer creation.

.. _optimizer-parameters:

Shared Parameters
^^^^^^^^^^^^^^^^^

Names are shared where meanings match, not where native spellings happen to
match. Unsupported controls fail instead of passing silently to a backend.
The following names apply both to the factory and direct adapter constructors,
except that ``family`` belongs only to the factory.

.. csv-table:: Parameter meanings
    :header: "Parameter", "Meaning", "Availability / Default"
    :widths: 20, 45, 35

    "``family``", "Case-sensitive iwopy backend family", "Factory only; required"
    "``optimizer``", "Algorithm or sampler selector; a SciPy method name or callable", "All families; factory requires a selector (SciPy also permits None)"
    "``problem``", "iwopy problem defining variables and functions", "All families; required"
    "``name``", "Optimizer label used in diagnostics", "All adapters; adapter-specific label"
    "``vectorize``", "Population evaluation or Jacobian-point batching rather than ordered individual evaluation; does not enable threads", "All adapters; True except PyGMO (False); driver capabilities still apply"
    "``seed``", "Adapter/native random seed; NLopt only seeds its PyGMO population; repeatability also depends on backend and problem", "CMA / pymoo / PyGMO / NLopt / SciPy DE / Nevergrad / Optuna / DEAP; None preserves default randomness"
    "``pop_size``", "Absolute search-population size; DEAP also uses it for each offspring generation", "CMA / pymoo / PyGMO / DEAP; see adapter defaults"
    "``batch_size``", "Number of outstanding asks or pending trials evaluated before feedback; not a search-population size", "Nevergrad / Optuna; default 8"
    "``population_multiplier``", "Search-population multiplier per free dimension; initialization can adjust the actual count", "SciPy DE only; default 15"
    "``memory_size``", "Visited-point capacity (GG) or evaluation-cache capacity (SLSQP / SciPy minimize)", "GG / SLSQP / SciPy minimize; not a process-memory byte limit"
    "``max_iterations``", "Nonnegative integer iteration or generation limit; unit is adapter-specific", "See support table; None preserves native or adapter defaults"
    "``max_evaluations``", "Positive integer native objective/fitness-call or adapter candidate limit", "See support table; None preserves native or adapter defaults"
    "``verbosity``", "Lifecycle progress output; zero is silent", "initialize / solve / finalize, not a factory parameter"
    "``callbacks``", "Ordered observers of intermediate optimizer states", "solve, subject to backend callback support; not a factory parameter"

``pop_size`` and ``batch_size`` are intentionally different. A population is
algorithmic search state. A pending batch controls when ask/tell feedback is
returned and can affect search behavior without defining that population.
SciPy DE's dimension multiplier is a third unit, not an absolute size.

.. _optimizer-termination:

Termination
^^^^^^^^^^^

The limit names are standardized; their native counters are not interchangeable.
Both limits reject booleans and nonintegers. ``max_iterations=0`` is permitted,
but initialization or an initial population may still be evaluated. Unsupported
requested limits raise ``ValueError`` at construction. Do not substitute one
unit for another to evade that error.

.. csv-table:: Limit support and counting units
    :header: "Adapter / Method", "max_iterations", "max_evaluations"
    :widths: 34, 33, 33

    "GG", "GG outer iterations", "Unsupported"
    "iwopy SLSQP / SciPy SLSQP", "Native local iterations", "Unsupported"
    "SciPy Nelder-Mead / Powell / L-BFGS-B / COBYQA", "Native local iterations", "Native objective calls"
    "SciPy TNC / COBYLA", "Unsupported", "Native objective calls (COBYLA's native maxiter counts evaluations)"
    "Other named SciPy minimize methods / default selection", "Native local iterations", "Unsupported"
    "SciPy custom callable", "Unsupported; callable owns its stopping semantics", "Unsupported; callable owns its stopping semantics"
    "SciPy differential_evolution", "Native generations after initialization", "Unsupported"
    "pycma CMAES", "Native generations", "Native fitness evaluations; checked at generation boundaries"
    "pymoo", "Native generations including initial population", "Native candidate evaluations; checked at generation boundaries"
    "PyGMO evolutionary drivers", "Native driver generations/iterations; ihs and sea propose one candidate per iteration", "Unsupported"
    "PyGMO ipopt", "Native IPOPT iterations", "Unsupported"
    "PyGMO compass_search / NLopt", "Unsupported", "Native driver fitness calls; excludes initial population construction"
    "Nevergrad / Optuna", "Unsupported; pending batches do not define algorithmic iterations", "Exact adapter candidate/trial count including initialization and final partial batch"
    "DEAP", "Adapter generations including initial population; zero still evaluates that population", "Exact adapter candidate count including initialization and final partial offspring generation"

``None`` keeps the native defaults; GG then has no iteration cap. Nevergrad,
Optuna, and DEAP need a finite candidate limit and use 100 when it is None.
Native convergence criteria can end a run before a limit. DEAP stops at whichever
configured limit is reached first. pymoo supports ``termination="iwopy"`` or
``"default"`` convergence, ``"iterations"`` or ``"evaluations"`` pure limits,
and native termination objects. Explicit limits compose with a custom object
using OR semantics; neither cap is discarded. A pure-limit selector defaults its
own omitted cap to 1000.

These are native limits, not universal hard budgets enforced by intercepting
every problem evaluation. CMA and pymoo can finish a generation beyond an
evaluation cap. SciPy numerical derivative work can exceed an objective-call
cap. Cached calls, constraint evaluations, Jacobian perturbations, preprocessing,
and selected-point finalization are not universally counted the same way.
Nevergrad, Optuna, and DEAP exclude selected-point finalization from their exact
candidate counters. Compare work in the documented units, not raw parameter
values alone. Reaching a limit does not certify convergence or feasibility;
inspect the result and native termination diagnostics.

Convergence tolerances remain backend-specific: GG's ``f_tol``, SciPy's ``ftol``
or ``gtol``, CMA's ``tolfun`` / ``tolx`` / ``ftarget``, pymoo's ``ftol`` /
``xtol`` / ``cvtol`` / ``period``, and NLopt's relative/absolute tolerances do not
measure one common quantity. Variable scaling, objective normalization, window
length, and constraint tolerances remain part of each algorithm's contract.

Superseded native limit keywords (``maxiter``, ``max_iter``, ``gen``,
``n_max_gen``, ``n_max_evals``, ``budget``, ``maxeval``, ``maxfevals``,
``maxfun``, ``maxfev``, ``max_fevals``) and ``popsize`` are rejected, not aliases.
Native result/stop-reason keys retain their backend names.

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
