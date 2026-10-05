iwopy.core
----------
Core functionality and abstract classes

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
