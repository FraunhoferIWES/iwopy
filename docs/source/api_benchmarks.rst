iwopy.benchmarks
================

The Branin and Rosenbrock objective functions provide standard continuous
test problems for optimizer comparisons.

The optimizer catalog uses package-level ``QuadraticProblem``, ``ParetoProblem``,
and ``MixedProblem`` for minimal continuous, multi-objective, and mixed-variable
setups. They initialize silently on construction; the Pareto case optionally
adds ``x + y <= -0.25``.

.. code-block:: python

    from iwopy.benchmarks import ParetoProblem

    problem = ParetoProblem(constrained=True)

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/benchmarks/index
