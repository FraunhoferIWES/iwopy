iwopy.wrappers
--------------
Wrappers that modify/simplify base class usage.

``LocalFD`` accepts per-call displacement caps through
``Problem.get_jacobian(max_deltas=...)``. Supply a positive finite array in
complete problem float-variable order, with shape ``(n_vars_float,)``.
Numerical differences use the smaller of the configured delta and its cap;
the configured deltas and analytical derivatives remain unchanged. A
second-order one-sided boundary stencil may sample at twice the capped delta.
``DiscretizeRegGrid`` retains its fixed grid spacing and stencils regardless
of these local displacement caps.

``LocalFD`` also supports numerical completion of only the missing entries in
a selected Jacobian. Component and differentiation-variable selections retain
their requested order, and analytical entries remain unchanged. Cached
``func_values`` must use selected-component order, with shape
``(n_selected_components,)``. This includes constraint-only Jacobian rows used by
SLSQP feasibility restoration.

.. toctree::
    :maxdepth: 2

    _autoapi/iwopy/wrappers/index
