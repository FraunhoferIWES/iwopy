iwopy.interfaces
================
Interfaces to external optimization packages

.. csv-table::
   :header: "Package", "Description"
   :widths: auto

   ":doc:`pymoo interface <api_interfaces_pymoo>`", "Interface to the `pymoo` package"
   ":doc:`pygmo interface <api_interfaces_pygmo>`", "Interface to the `pygmo` package"
   ":doc:`scipy interface <api_interfaces_scipy>`", "Interface to the `scipy` package"
   ":doc:`pycma interface <api_interfaces_cma>`", "Population-vectorized CMA-ES with the `cma` package"
   ":doc:`SciPy differential evolution <_autoapi/iwopy/interfaces/scipy/de/index>`", "Native vectorized bounded population search"
   ":doc:`Nevergrad interface <_autoapi/iwopy/interfaces/nevergrad/index>`", "Population ask/tell with eight batching-safe selectors"
   ":doc:`Optuna interface <_autoapi/iwopy/interfaces/optuna/index>`", "Population ask/tell with TPE and random samplers"
   ":doc:`DEAP interface <_autoapi/iwopy/interfaces/deap/index>`", "Three bounded population evolution schemes"

See the :doc:`optimizer_catalog` for every user-selectable optimizer and a
direct, runnable Python constructor example for each one.

.. toctree::
   :hidden:

   api_interfaces_pymoo
   api_interfaces_pygmo
   api_interfaces_scipy
   api_interfaces_cma
   _autoapi/iwopy/interfaces/nevergrad/index
   _autoapi/iwopy/interfaces/optuna/index
   _autoapi/iwopy/interfaces/deap/index
