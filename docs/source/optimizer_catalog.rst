.. _optimizer-catalog:

Optimizer Catalog
=================

Every named optimizer selectable through iwopy is listed below, including
SciPy methods and PyGMO's NLopt subsolvers. Default selection and SciPy's callable
extension point are also shown. Arbitrary user-written callable implementations
are not a finite registry. Sampling, crossover, mutation, and termination
operators are configuration, not additional optimizers.

Each setup link shows a complete, tested Python script with an explicit
``solver = create_optimizer(...)`` call using ``family``, ``optimizer``, and
flat class/native keyword parameters. See :doc:`api_core` for selection rules.
Execution is
inside ``if __name__ == "__main__":``; importing a script does not run it.
The shared synthetic problem definitions appear below.
From the repository root, install the optional backends and run any case by its
setup name, for example:

.. code-block:: console

   uv sync --extra opt
   uv run python examples/optimizers/scipy_de.py
   uv run python examples/optimizers/optuna_tpe.py

The examples use synthetic quadratic objectives and small demonstration budgets;
they are configuration examples, not performance comparisons or convergence
guarantees. Single-objective examples minimize; Pareto examples use two objectives.
Only ``pymoo_mixed_variable_ga`` requires the example's integer/float problem.

Important parameter values each have an inline explanation. Comments beside
the short run limits and small populations suggest production starting values,
not universally optimal settings. Adapt them to dimension, evaluation cost,
constraint difficulty, and noise. Scale variables and objectives before choosing
tolerances, and keep tolerances above numerical noise. The explicit Hessians in
the dogleg and trust-exact examples belong to the benchmark objective and must
be replaced for a different problem. ``iwopy`` identifies the GG/SLSQP source;
``native`` elsewhere describes an external backend's own behavior or results.

Available Optimizers
--------------------

Groups classify objective count, nonlinear-constraint support, and search
strategy. Constraint-capable solvers also work without those constraints;
variable boxes alone remain in the unconstrained groups. ``Global`` means
exploratory search, not a global-optimum guarantee. Single-run covariance and
hill-climbing methods are grouped as local adaptation; restarts can broaden them.
Default and wrapper methods follow their supported or demonstrated workflow.
Pareto rows distinguish the demonstrated unconstrained/constrained flow;
NSGA2 and NSGA3 therefore have a setup for each.

Single-Objective, Unconstrained, Local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Local single-objective search without nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "SciPy / Nelder-Mead", "Single; bounds; derivative-free", "Local simplex search", "Small nonsmooth problems without derivatives", ":ref:`scipy_nelder_mead <case-scipy-nelder-mead>`"
   "SciPy / Powell", "Single; bounds; derivative-free", "Local direction-set search", "Derivative-free local refinement with box bounds", ":ref:`scipy_powell <case-scipy-powell>`"
   "SciPy / CG", "Single; unconstrained; gradients", "Nonlinear conjugate gradients", "Large smooth problems where memory matters", ":ref:`scipy_cg <case-scipy-cg>`"
   "SciPy / BFGS", "Single; unconstrained; gradients", "Local quasi-Newton search", "Smooth small-to-medium problems", ":ref:`scipy_bfgs <case-scipy-bfgs>`"
   "SciPy / Newton-CG", "Single; unconstrained; curvature", "Local Newton-CG; example supplies a Hessian", "Smooth problems with cheap curvature products", ":ref:`scipy_newton_cg <case-scipy-newton-cg>`"
   "SciPy / L-BFGS-B", "Single; bounds; gradients", "Limited-memory local quasi-Newton search", "Large smooth box-bounded problems", ":ref:`scipy_l_bfgs_b <case-scipy-l-bfgs-b>`"
   "SciPy / TNC", "Single; bounds; gradients", "Local truncated Newton", "Smooth box bounds with truncated-Newton steps", ":ref:`scipy_tnc <case-scipy-tnc>`"
   "SciPy / dogleg", "Single; unconstrained; Hessian", "Local dogleg trust region", "Small problems with a positive-definite Hessian", ":ref:`scipy_dogleg <case-scipy-dogleg>`"
   "SciPy / trust-ncg", "Single; unconstrained; curvature", "Local Newton-CG trust region", "Curvature products without storing a dense Hessian", ":ref:`scipy_trust_ncg <case-scipy-trust-ncg>`"
   "SciPy / trust-exact", "Single; unconstrained; Hessian", "Nearly exact local trust-region solve", "Small-to-medium problems with affordable Hessians", ":ref:`scipy_trust_exact <case-scipy-trust-exact>`"
   "SciPy / trust-krylov", "Single; unconstrained; curvature", "Local Krylov trust region", "Large smooth problems with curvature products", ":ref:`scipy_trust_krylov <case-scipy-trust-krylov>`"
   "SciPy / callable method", "Single; custom", "User method; example delegates to BFGS", "Integrate your own minimize-compatible method", ":ref:`scipy_callable <case-scipy-callable>`"
   "PyGMO / cmaes", "Single; bounds; derivative-free", "Continuous covariance adaptation", "Correlated continuous variables in a PyGMO workflow", ":ref:`pygmo_cmaes <case-pygmo-cmaes>`"
   "PyGMO / xnes", "Single; bounds; derivative-free", "Exponential natural evolution strategies", "Compare distribution-based continuous search", ":ref:`pygmo_xnes <case-pygmo-xnes>`"
   "PyGMO / sea", "Single; bounds; derivative-free", "Simple evolutionary search", "A minimal evolutionary comparison baseline", ":ref:`pygmo_sea <case-pygmo-sea>`"
   "PyGMO / compass_search", "Single; bounds; derivative-free", "Local coordinate pattern search", "Cheap derivative-free local refinement", ":ref:`pygmo_compass_search <case-pygmo-compass-search>`"
   "NLopt / bobyqa", "Single; bounds; derivative-free", "Local quadratic approximation", "Smooth box-bounded black boxes", ":ref:`nlopt_bobyqa <case-nlopt-bobyqa>`"
   "NLopt / newuoa", "Single; unconstrained; derivative-free", "Local quadratic approximation", "Smooth unconstrained black boxes", ":ref:`nlopt_newuoa <case-nlopt-newuoa>`"
   "NLopt / newuoa_bound", "Single; bounds; derivative-free", "Bound-constrained NEWUOA", "Local quadratic models with box bounds", ":ref:`nlopt_newuoa_bound <case-nlopt-newuoa-bound>`"
   "NLopt / praxis", "Single; bounds; derivative-free", "Local principal-axis search", "Derivative-free search with changing search directions", ":ref:`nlopt_praxis <case-nlopt-praxis>`"
   "NLopt / neldermead", "Single; bounds; derivative-free", "Local simplex search", "Small nonsmooth local-search problems", ":ref:`nlopt_neldermead <case-nlopt-neldermead>`"
   "NLopt / sbplx", "Single; bounds; derivative-free", "Local subspace simplex search", "Simplex search on larger or nonsmooth problems", ":ref:`nlopt_sbplx <case-nlopt-sbplx>`"
   "NLopt / lbfgs", "Single; bounds; gradients", "Local limited-memory BFGS", "Smooth problems where low memory is useful", ":ref:`nlopt_lbfgs <case-nlopt-lbfgs>`"
   "NLopt / tnewton_precond_restart", "Single; bounds; gradients", "Preconditioned restarted truncated Newton", "Large smooth problems with gradient evaluations", ":ref:`nlopt_tnewton_precond_restart <case-nlopt-tnewton-precond-restart>`"
   "NLopt / tnewton_precond", "Single; bounds; gradients", "Preconditioned truncated Newton", "Gradient-based local search with preconditioning", ":ref:`nlopt_tnewton_precond <case-nlopt-tnewton-precond>`"
   "NLopt / tnewton_restart", "Single; bounds; gradients", "Restarted truncated Newton", "Gradient-based local search with restarts", ":ref:`nlopt_tnewton_restart <case-nlopt-tnewton-restart>`"
   "NLopt / tnewton", "Single; bounds; gradients", "Local truncated Newton", "Smooth local search using gradients", ":ref:`nlopt_tnewton <case-nlopt-tnewton>`"
   "NLopt / var2", "Single; bounds; gradients", "Rank-two limited-memory variable metric", "Compare smooth variable-metric local search", ":ref:`nlopt_var2 <case-nlopt-var2>`"
   "NLopt / var1", "Single; bounds; gradients", "Rank-one limited-memory variable metric", "Compare a lighter variable-metric update", ":ref:`nlopt_var1 <case-nlopt-var1>`"

Single-Objective, Unconstrained, Global
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Exploratory single-objective search without nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "PyGMO / sga", "Single; bounds; derivative-free", "Population genetic search", "A simple box-bounded genetic baseline", ":ref:`pygmo_sga <case-pygmo-sga>`"
   "PyGMO / pso", "Single; bounds; derivative-free", "Population particle swarm", "Continuous swarm search with box bounds", ":ref:`pygmo_pso <case-pygmo-pso>`"
   "PyGMO / bee_colony", "Single; bounds; derivative-free", "Artificial bee-colony search", "An alternative continuous population heuristic", ":ref:`pygmo_bee_colony <case-pygmo-bee-colony>`"
   "PyGMO / de", "Single; bounds; derivative-free", "Population differential evolution", "A box-bounded global-search baseline", ":ref:`pygmo_de <case-pygmo-de>`"
   "PyGMO / de1220", "Single; bounds; derivative-free", "Self-adaptive DE strategy variants", "Compare DE strategies without fixing one variant", ":ref:`pygmo_de1220 <case-pygmo-de1220>`"
   "PyGMO / sade", "Single; bounds; derivative-free", "Self-adaptive differential evolution", "Reduce manual DE parameter tuning", ":ref:`pygmo_sade <case-pygmo-sade>`"
   "PyGMO / gwo", "Single; bounds; derivative-free", "Grey-wolf population search", "Compare another box-bounded population heuristic", ":ref:`pygmo_gwo <case-pygmo-gwo>`"
   "PyGMO / pso_gen", "Single; bounds; derivative-free", "Generational particle swarm", "Generational swarm search in a PyGMO workflow", ":ref:`pygmo_pso_gen <case-pygmo-pso-gen>`"

Single-Objective, Constrained, Local
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Local single-objective search supporting nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "iwopy / GG", "Single; constraints; gradients", "Local greedy-gradient search with batched trial steps", "Cheap gradients and constrained local refinement", ":ref:`iwopy_gg <case-iwopy-gg>`"
   "iwopy / SLSQP", "Single; constraints; gradients", "Scaled local SLSQP with population Jacobians", "Smooth constrained problems needing local refinement", ":ref:`iwopy_slsqp <case-iwopy-slsqp>`"
   "SciPy / default", "Single; method-dependent", "Local method chosen from bounds and constraints", "A first local solver for smooth problems", ":ref:`scipy_default <case-scipy-default>`"
   "SciPy / COBYLA", "Single; inequalities; derivative-free", "Local linear approximation", "Nonlinear inequalities without derivatives", ":ref:`scipy_cobyla <case-scipy-cobyla>`"
   "SciPy / COBYQA", "Single; constraints; derivative-free", "Local quadratic approximation; SciPy >=1.14", "Smooth constrained problems without gradients", ":ref:`scipy_cobyqa <case-scipy-cobyqa>`"
   "SciPy / SLSQP", "Single; constraints; gradients", "Local sequential quadratic programming", "Smooth equalities and inequalities", ":ref:`scipy_slsqp <case-scipy-slsqp>`"
   "SciPy / trust-constr", "Single; constraints; gradients", "Constrained local trust-region search", "Smooth constrained problems needing trust-region control", ":ref:`scipy_trust_constr <case-scipy-trust-constr>`"
   "pycma / CMA-ES", "Single; constraints; derivative-free", "Continuous population covariance adaptation", "Correlated continuous variables and nonlinear black boxes", ":ref:`cma_cmaes <case-cma-cmaes>`"
   "pymoo / CMAES", "Single; one-sided constraints; derivative-free", "Continuous covariance adaptation", "CMA-ES within an existing pymoo workflow", ":ref:`pymoo_cmaes <case-pymoo-cmaes>`"
   "PyGMO / nlopt", "Single; solver-dependent", "Optional native local-solver wrapper", "Choose a specific NLopt local method below", ":ref:`pygmo_nlopt <case-pygmo-nlopt>`"
   "PyGMO / ipopt", "Single; constraints; gradients", "Optional native interior-point local solver", "Smooth constrained problems in an IPOPT-enabled build", ":ref:`pygmo_ipopt <case-pygmo-ipopt>`"
   "NLopt / cobyla", "Single; constraints; derivative-free", "Local constrained linear approximation", "Constrained black boxes without derivatives", ":ref:`nlopt_cobyla <case-nlopt-cobyla>`"
   "NLopt / mma", "Single; inequalities; gradients", "Local method of moving asymptotes", "Smooth nonlinear inequality constraints", ":ref:`nlopt_mma <case-nlopt-mma>`"
   "NLopt / ccsaq", "Single; inequalities; gradients", "Local conservative separable approximation", "Smooth inequality-constrained local refinement", ":ref:`nlopt_ccsaq <case-nlopt-ccsaq>`"
   "NLopt / slsqp", "Single; constraints; gradients", "Local sequential quadratic programming", "Smooth equality and inequality constraints", ":ref:`nlopt_slsqp <case-nlopt-slsqp>`"
   "NLopt / auglag", "Single; constraints; gradients in setup", "Augmented Lagrangian with local L-BFGS", "General constraints with a chosen smooth local solver", ":ref:`nlopt_auglag <case-nlopt-auglag>`"
   "NLopt / auglag_eq", "Single; constraints; gradients in setup", "Equality augmented Lagrangian with local L-BFGS", "Equalities when the local solver handles remaining restrictions", ":ref:`nlopt_auglag_eq <case-nlopt-auglag-eq>`"
   "Nevergrad / CMA", "Single; mixed; constraints; derivative-free", "Population covariance adaptation", "Correlated black-box variables", ":ref:`nevergrad_cma <case-nevergrad-cma>`"
   "Nevergrad / OnePlusOne", "Single; mixed; constraints; derivative-free", "One-plus-one evolutionary search", "Small budgets or cheap incremental search", ":ref:`nevergrad_one_plus_one <case-nevergrad-one-plus-one>`"
   "Nevergrad / TBPSA", "Single; mixed; constraints; derivative-free", "Test-based population-size adaptation", "Noisy objectives with sufficient evaluations", ":ref:`nevergrad_tbpsa <case-nevergrad-tbpsa>`"

Single-Objective, Constrained, Global
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Exploratory single-objective search supporting nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "SciPy / differential_evolution", "Single; mixed; constraints; derivative-free", "Vectorized global population search", "Bounded multimodal search without derivatives", ":ref:`scipy_de <case-scipy-de>`"
   "pymoo / GA", "Single; one-sided constraints; derivative-free", "Population genetic search", "A general evolutionary baseline", ":ref:`pymoo_ga <case-pymoo-ga>`"
   "pymoo / DE", "Single; one-sided constraints; derivative-free", "Population differential evolution", "Continuous multimodal black-box search", ":ref:`pymoo_de <case-pymoo-de>`"
   "pymoo / PSO", "Single; one-sided constraints; derivative-free", "Population particle swarm", "Continuous swarm search as a comparison baseline", ":ref:`pymoo_pso <case-pymoo-pso>`"
   "pymoo / MixedVariableGA", "Single; mixed; one-sided constraints; derivative-free", "Mixed integer/float genetic search", "Genetic search with both integer and floating variables", ":ref:`pymoo_mixed_variable_ga <case-pymoo-mixed-variable-ga>`"
   "PyGMO / ihs", "Single; constraints; derivative-free", "Improved harmony search", "Constrained heuristic search without derivatives", ":ref:`pygmo_ihs <case-pygmo-ihs>`"
   "Nevergrad / NGOpt", "Single; mixed; constraints; derivative-free", "Adaptive optimizer portfolio", "A black-box starting point when strategy is uncertain", ":ref:`nevergrad_ngopt <case-nevergrad-ngopt>`"
   "Nevergrad / DE", "Single; mixed; constraints; derivative-free", "Population differential evolution", "Multimodal bounded black-box search", ":ref:`nevergrad_de <case-nevergrad-de>`"
   "Nevergrad / TwoPointsDE", "Single; mixed; constraints; derivative-free", "Two-point DE crossover", "Compare a different DE crossover strategy", ":ref:`nevergrad_two_points_de <case-nevergrad-two-points-de>`"
   "Nevergrad / PSO", "Single; mixed; constraints; derivative-free", "Population particle swarm", "A swarm baseline for bounded black boxes", ":ref:`nevergrad_pso <case-nevergrad-pso>`"
   "Nevergrad / RandomSearch", "Single; mixed; constraints; derivative-free", "Independent random search", "A reproducible baseline before adaptive search", ":ref:`nevergrad_random_search <case-nevergrad-random-search>`"
   "Optuna / TPESampler", "Single; mixed; constraints; derivative-free", "Parzen estimation with pending-trial handling", "Costly bounded integer/float tuning", ":ref:`optuna_tpe <case-optuna-tpe>`"
   "Optuna / RandomSampler", "Single; mixed; constraints; derivative-free", "Random sampling without a constraint model", "Baselines or very small evaluation budgets", ":ref:`optuna_random <case-optuna-random>`"
   "DEAP / eaSimple", "Single; mixed; constraints; derivative-free", "Tournament-based generational replacement", "Straightforward generational genetic search", ":ref:`deap_simple <case-deap-simple>`"
   "DEAP / eaMuPlusLambda", "Single; mixed; constraints; derivative-free", "Tournament selection from parents and offspring", "Genetic search that retains parent candidates", ":ref:`deap_mu_plus_lambda <case-deap-mu-plus-lambda>`"
   "DEAP / eaMuCommaLambda", "Single; mixed; constraints; derivative-free", "Offspring-only tournament replacement", "Exploratory genetic search replacing all parents", ":ref:`deap_mu_comma_lambda <case-deap-mu-comma-lambda>`"

Multi-Objective, Unconstrained, Global
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Exploratory multi-objective search without nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "pymoo / NSGA2", "Multi; bounds; derivative-free", "Unconstrained Pareto search with crowding", "Two or three competing objectives without extra restrictions", ":ref:`pymoo_nsga2 <case-pymoo-nsga2>`"
   "pymoo / NSGA3", "Multi; bounds; derivative-free", "Unconstrained reference-direction Pareto search", "Many competing objectives without extra restrictions", ":ref:`pymoo_nsga3 <case-pymoo-nsga3>`"
   "PyGMO / nsga2", "Multi (native); bounds; derivative-free", "Native Pareto nondominated sorting", "Two or three objectives when native PyGMO results suffice", ":ref:`pygmo_nsga2 <case-pygmo-nsga2>`"
   "PyGMO / moead", "Multi (native); bounds; derivative-free", "Native Pareto decomposition search", "Trade-offs organized by scalar subproblems", ":ref:`pygmo_moead <case-pygmo-moead>`"
   "PyGMO / moead_gen", "Multi (native); bounds; derivative-free", "Generational native Pareto decomposition", "A generational variant of decomposition search", ":ref:`pygmo_moead_gen <case-pygmo-moead-gen>`"
   "PyGMO / nspso", "Multi (native); bounds; derivative-free", "Native Pareto particle swarm", "Compare swarm-based trade-off search", ":ref:`pygmo_nspso <case-pygmo-nspso>`"

Multi-Objective, Constrained, Global
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. csv-table:: Exploratory multi-objective search supporting nonlinear constraints
   :header: "Interface / Selector", "Capabilities", "Characterization", "When to Use", "Minimal Python Setup"
   :widths: 20, 19, 23, 23, 15
   :class: optimizer-catalog

   "pymoo / NSGA2 (constrained flow)", "Multi; one-sided constraints; derivative-free", "Feasible Pareto search with crowding", "Competing objectives with additional restrictions", ":ref:`pymoo_nsga2_constrained <case-pymoo-nsga2-constrained>`"
   "pymoo / NSGA3 (constrained flow)", "Multi; one-sided constraints; derivative-free", "Feasible reference-direction Pareto search", "Many objectives with additional restrictions", ":ref:`pymoo_nsga3_constrained <case-pymoo-nsga3-constrained>`"

``Single`` and ``Multi`` classify the objective/result workflow; ``mixed`` means
both integer and floating variables. ``Bounds`` are variable boxes; ``constraints``
are additional nonlinear restrictions, with supported forms/tolerances in the API.
``One-sided constraints`` identifies pymoo's translation limit. ``Multi (native)``
returns a PyGMO population, not iwopy Pareto results. Other labels do not claim
every upstream capability. Curvature methods also need gradients.

Recommendations are starting points, not performance rankings. Compare suitable
solvers on your problem and budget; short setup runs do not guarantee convergence.
See :ref:`optimizer-parameters` for the parameter explanation table and
:ref:`optimizer-termination` for limit support, counting units, and convergence.

Multi-Objective Flows
-------------------------

The four pymoo Pareto setups return iwopy ``MultiObjOptResults`` and select a
compromise from the approximate front using explicit objective weights:

* Unconstrained: :ref:`case-pymoo-nsga2` and :ref:`case-pymoo-nsga3`.
* With ``x + y <= -0.25``: :ref:`case-pymoo-nsga2-constrained` and
   :ref:`case-pymoo-nsga3-constrained`.

``find_pareto_objmix([0.5, 0.5])`` returns the index minimizing that weighted sum
within the returned front. Weights assume comparable objective scales and express
preferences; they do not prove a global optimum or recover every trade-off.
The PyGMO flows below instead return native populations, as explained next.

Availability And Batching
-------------------------

PyGMO's ``ipopt`` and ``nlopt`` require builds with those optional native
features. Its four multi-objective cases initialize the iwopy wrapper and evolve
the native population directly. The existing iwopy PyGMO result finalization
uses ``champion_x`` and cannot finalize a Pareto population; these examples
therefore return a native PyGMO population, not ``MultiObjOptResults``. Use pymoo
NSGA2/NSGA3 for a complete iwopy Pareto-result workflow.

.. _api-vectorized-adapters:

Vectorized Interfaces
-------------------------

All interfaces use the same iwopy problem model, but their batching mechanisms
differ. Vectorized evaluation does not imply threads or parallel execution.

SciPy differential evolution, Nevergrad, Optuna, and DEAP expose
``vectorize=True`` on the optimizer constructor. Set ``vectorize=False`` to
evaluate candidates individually in batch order without changing native search,
batch sizes, budgets, callbacks, or selection. Their single-objective floating,
integer, and mixed problems need finite, strictly ordered bounds; exact equality
constraints need positive tolerance. Detailed contracts live in the linked API.

SciPy Minimization
~~~~~~~~~~~~~~~~~~~~

``Optimizer_scipy`` performs scalar objective search. Its default
``vectorize=True`` batches iwopy Jacobian evaluation; ``vectorize=False``
uses individual Jacobian evaluation. Unconstrained methods ignore forwarded
bounds. Curvature-based catalog cases supply the quadratic Hessian; iwopy does
not manufacture a general Hessian.

See the :doc:`Optimizer_scipy API <_autoapi/iwopy/interfaces/scipy/optimizer/index>`
for constructor parameters and methods.

.. _api-scipy-de:

SciPy Differential Evolution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``Optimizer_scipy_de`` always uses ``vectorized=True``, ``workers=1``, deferred
updating, and ``polish=False`` in SciPy. The iwopy ``vectorize`` flag controls
population versus individual problem evaluation independently of that native
batch mode. Cached rows share objective/constraint calls. Success requires
native convergence and finalized feasibility; ``scipy_results`` is the native result.

See the :doc:`Optimizer_scipy_de API <_autoapi/iwopy/interfaces/scipy/de/index>`
for constructor parameters and methods.

pycma
~~~~~~~~~~~~~~~~~~~~

``Optimizer_cma`` evaluates each asked generation as an iwopy population by
default, or ordered individuals when ``vectorize=False``.
It supports continuous single-objective search, finite bounds, and native
augmented-Lagrangian constraints using each constraint's tolerance. It has no
threaded-evaluation switch. The ``cma`` extra requires pycma >=4.4.0.

See the :doc:`Optimizer_cma API <_autoapi/iwopy/interfaces/cma/optimizer/index>`
for constructor parameters and methods.

pymoo
~~~~~~~~~~~~~~~~~~~~

``Optimizer_pymoo`` uses ``vectorize=True`` for population
evaluation; false selects elementwise evaluation. Algorithm choice determines
single-/multi-objective and mixed-variable support. iwopy translates one-sided
constraints; NSGA2/NSGA3 return iwopy Pareto results. Install the ``pymoo`` extra.

See the :doc:`Optimizer_pymoo API <_autoapi/iwopy/interfaces/pymoo/optimizer/index>`
for constructor parameters and methods.

PyGMO
~~~~~~~~~~~~~~~~~~~~

``Optimizer_pygmo`` exposes batch fitness with ``vectorize=True``.
Use an algorithm with batch-fitness support; enabling the flag alone does not
make every algorithm vectorized. The catalog uses the default scalar fitness
path. Multi-objective catalog flows return native populations. Install ``pygmo``.
Select NLopt leaf solvers with ``family="nlopt"``; they use the same PyGMO
adapter without caller-supplied nested driver settings.

See the :doc:`Optimizer_pygmo API <_autoapi/iwopy/interfaces/pygmo/optimizer/index>`
for constructor parameters and methods.

.. _api-nevergrad:

Nevergrad
~~~~~~~~~

``Optimizer_nevergrad`` asks a batch, evaluates it according to ``vectorize``,
and tells results and constraint violations in ask order. Its eight exposed
selectors permit batched asks; ``num_workers`` does not create an executor.
Install ``nevergrad``: Nevergrad >=1.0.12 and NumPy <2.4 are required for the
upstream metamodel's scalar-array conversion.

See the :doc:`Optimizer_nevergrad API <_autoapi/iwopy/interfaces/nevergrad/optimizer/index>`
for constructor parameters and methods.

.. _api-optuna:

Optuna
~~~~~~

``Optimizer_optuna`` asks ordered trial batches and evaluates them according to
``vectorize`` in a fresh in-memory study. TPE handles pending trials and constraint
residuals; random sampling retains feasible selection without learning constraints.
Failed evaluation marks the outstanding batch failed. Install ``optuna``;
Optuna >=5.0.0 is required for ``Trial.set_constraint``.

See the :doc:`Optimizer_optuna API <_autoapi/iwopy/interfaces/optuna/optimizer/index>`
for constructor parameters and methods.

.. _api-deap:

DEAP
~~~~

``Optimizer_deap`` evaluates repaired offspring batches according to ``vectorize``
using one of three native variation/replacement schemes. Fitness ranks constraint
violation before objective loss; every offspring counts toward the budget.
Run serially because native operators temporarily use Python's global random
generator. Install ``deap``; DEAP >=1.4.4 is required.

See the :doc:`Optimizer_deap API <_autoapi/iwopy/interfaces/deap/optimizer/index>`
for constructor parameters and methods.

For Nevergrad, Optuna, and DEAP, success means finalized feasibility rather than
convergence; selection prefers the best evaluated feasible candidate. Partial
final batches count toward budgets in both evaluation modes.

Shared Problem Definitions
--------------------------

The scripts instantiate ``QuadraticProblem``, ``ParetoProblem``, or
``MixedProblem`` directly. There are no optimizer setup helpers or parameter
builders. These synthetic cases are part of the installed package:

.. code-block:: python

   from iwopy.benchmarks import ParetoProblem

   problem = ParetoProblem(constrained=True)

Construction initializes the problem silently; importing the module creates no
problem instances. See the :doc:`quadratic benchmark API <_autoapi/iwopy/benchmarks/quadratic/index>`
for all three definitions, bounds, objective formulas, and lifecycle.

Minimal Setups
--------------

.. _case-iwopy-gg:

iwopy GG
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/iwopy_gg.py
   :language: python

.. _case-iwopy-slsqp:

iwopy SLSQP
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/iwopy_slsqp.py
   :language: python

.. _case-scipy-default:

SciPy Default
~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_default.py
   :language: python

.. _case-scipy-nelder-mead:

SciPy Nelder-Mead
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_nelder_mead.py
   :language: python

.. _case-scipy-powell:

SciPy Powell
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_powell.py
   :language: python

.. _case-scipy-cg:

SciPy CG
~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_cg.py
   :language: python

.. _case-scipy-bfgs:

SciPy BFGS
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_bfgs.py
   :language: python

.. _case-scipy-newton-cg:

SciPy Newton-CG
~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_newton_cg.py
   :language: python

.. _case-scipy-l-bfgs-b:

SciPy L-BFGS-B
~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_l_bfgs_b.py
   :language: python

.. _case-scipy-tnc:

SciPy TNC
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_tnc.py
   :language: python

.. _case-scipy-cobyla:

SciPy COBYLA
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_cobyla.py
   :language: python

.. _case-scipy-cobyqa:

SciPy COBYQA
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_cobyqa.py
   :language: python

.. _case-scipy-slsqp:

SciPy SLSQP
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_slsqp.py
   :language: python

.. _case-scipy-trust-constr:

SciPy Trust-Constr
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_trust_constr.py
   :language: python

.. _case-scipy-dogleg:

SciPy Dogleg
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_dogleg.py
   :language: python

.. _case-scipy-trust-ncg:

SciPy Trust-NCG
~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_trust_ncg.py
   :language: python

.. _case-scipy-trust-exact:

SciPy Trust-Exact
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_trust_exact.py
   :language: python

.. _case-scipy-trust-krylov:

SciPy Trust-Krylov
~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_trust_krylov.py
   :language: python

.. _case-scipy-callable:

SciPy Callable
~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_callable.py
   :language: python

.. _case-scipy-de:

SciPy Differential Evolution Setup
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/scipy_de.py
   :language: python

.. _case-cma-cmaes:

pycma CMA-ES
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/cma_cmaes.py
   :language: python

.. _case-pymoo-ga:

pymoo GA
~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_ga.py
   :language: python

.. _case-pymoo-de:

pymoo DE
~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_de.py
   :language: python

.. _case-pymoo-pso:

pymoo PSO
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_pso.py
   :language: python

.. _case-pymoo-nsga2:

pymoo NSGA2
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_nsga2.py
   :language: python

.. _case-pymoo-nsga3:

pymoo NSGA3
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_nsga3.py
   :language: python

.. _case-pymoo-nsga2-constrained:

pymoo NSGA2 Constrained Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_nsga2_constrained.py
   :language: python

.. _case-pymoo-nsga3-constrained:

pymoo NSGA3 Constrained Flow
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_nsga3_constrained.py
   :language: python

.. _case-pymoo-cmaes:

pymoo CMAES
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_cmaes.py
   :language: python

.. _case-pymoo-mixed-variable-ga:

pymoo MixedVariableGA
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pymoo_mixed_variable_ga.py
   :language: python

.. _case-pygmo-nlopt:

PyGMO NLopt Wrapper
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_nlopt.py
   :language: python

.. _case-pygmo-ipopt:

PyGMO IPOPT
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_ipopt.py
   :language: python

.. _case-pygmo-sga:

PyGMO SGA
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_sga.py
   :language: python

.. _case-pygmo-pso:

PyGMO PSO
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_pso.py
   :language: python

.. _case-pygmo-bee-colony:

PyGMO Bee Colony
~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_bee_colony.py
   :language: python

.. _case-pygmo-nsga2:

PyGMO NSGA2
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_nsga2.py
   :language: python

.. _case-pygmo-de:

PyGMO DE
~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_de.py
   :language: python

.. _case-pygmo-de1220:

PyGMO DE1220
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_de1220.py
   :language: python

.. _case-pygmo-sade:

PyGMO SADE
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_sade.py
   :language: python

.. _case-pygmo-cmaes:

PyGMO CMAES
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_cmaes.py
   :language: python

.. _case-pygmo-gwo:

PyGMO GWO
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_gwo.py
   :language: python

.. _case-pygmo-ihs:

PyGMO IHS
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_ihs.py
   :language: python

.. _case-pygmo-pso-gen:

PyGMO Generational PSO
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_pso_gen.py
   :language: python

.. _case-pygmo-xnes:

PyGMO XNES
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_xnes.py
   :language: python

.. _case-pygmo-sea:

PyGMO SEA
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_sea.py
   :language: python

.. _case-pygmo-compass-search:

PyGMO Compass Search
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_compass_search.py
   :language: python

.. _case-pygmo-moead:

PyGMO MOEA/D
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_moead.py
   :language: python

.. _case-pygmo-moead-gen:

PyGMO Generational MOEA/D
~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_moead_gen.py
   :language: python

.. _case-pygmo-nspso:

PyGMO NSPSO
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/pygmo_nspso.py
   :language: python

.. _case-nlopt-cobyla:

NLopt COBYLA
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_cobyla.py
   :language: python

.. _case-nlopt-bobyqa:

NLopt BOBYQA
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_bobyqa.py
   :language: python

.. _case-nlopt-newuoa:

NLopt NEWUOA
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_newuoa.py
   :language: python

.. _case-nlopt-newuoa-bound:

NLopt Bounded NEWUOA
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_newuoa_bound.py
   :language: python

.. _case-nlopt-praxis:

NLopt PRAXIS
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_praxis.py
   :language: python

.. _case-nlopt-neldermead:

NLopt Nelder-Mead
~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_neldermead.py
   :language: python

.. _case-nlopt-sbplx:

NLopt Subplex
~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_sbplx.py
   :language: python

.. _case-nlopt-mma:

NLopt MMA
~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_mma.py
   :language: python

.. _case-nlopt-ccsaq:

NLopt CCSAQ
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_ccsaq.py
   :language: python

.. _case-nlopt-slsqp:

NLopt SLSQP
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_slsqp.py
   :language: python

.. _case-nlopt-lbfgs:

NLopt L-BFGS
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_lbfgs.py
   :language: python

.. _case-nlopt-tnewton-precond-restart:

NLopt Preconditioned Restarted Newton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_tnewton_precond_restart.py
   :language: python

.. _case-nlopt-tnewton-precond:

NLopt Preconditioned Newton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_tnewton_precond.py
   :language: python

.. _case-nlopt-tnewton-restart:

NLopt Restarted Newton
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_tnewton_restart.py
   :language: python

.. _case-nlopt-tnewton:

NLopt Truncated Newton
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_tnewton.py
   :language: python

.. _case-nlopt-var2:

NLopt VAR2
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_var2.py
   :language: python

.. _case-nlopt-var1:

NLopt VAR1
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_var1.py
   :language: python

.. _case-nlopt-auglag:

NLopt Augmented Lagrangian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_auglag.py
   :language: python

.. _case-nlopt-auglag-eq:

NLopt Equality Augmented Lagrangian
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nlopt_auglag_eq.py
   :language: python

.. _case-nevergrad-ngopt:

Nevergrad NGOpt
~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_ngopt.py
   :language: python

.. _case-nevergrad-cma:

Nevergrad CMA
~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_cma.py
   :language: python

.. _case-nevergrad-de:

Nevergrad DE
~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_de.py
   :language: python

.. _case-nevergrad-two-points-de:

Nevergrad TwoPointsDE
~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_two_points_de.py
   :language: python

.. _case-nevergrad-pso:

Nevergrad PSO
~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_pso.py
   :language: python

.. _case-nevergrad-one-plus-one:

Nevergrad OnePlusOne
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_one_plus_one.py
   :language: python

.. _case-nevergrad-random-search:

Nevergrad RandomSearch
~~~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_random_search.py
   :language: python

.. _case-nevergrad-tbpsa:

Nevergrad TBPSA
~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/nevergrad_tbpsa.py
   :language: python

.. _case-optuna-tpe:

Optuna TPE
~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/optuna_tpe.py
   :language: python

.. _case-optuna-random:

Optuna Random
~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/optuna_random.py
   :language: python

.. _case-deap-simple:

DEAP Simple
~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/deap_simple.py
   :language: python

.. _case-deap-mu-plus-lambda:

DEAP Mu Plus Lambda
~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/deap_mu_plus_lambda.py
   :language: python

.. _case-deap-mu-comma-lambda:

DEAP Mu Comma Lambda
~~~~~~~~~~~~~~~~~~~~

.. literalinclude:: ../../examples/optimizers/deap_mu_comma_lambda.py
   :language: python
