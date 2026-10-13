"""Construct optimizers by backend family and algorithm without running them."""

from collections.abc import Callable
from importlib import import_module
from typing import Any, cast

from .optimizer import Optimizer
from .problem import Problem


_ADAPTERS = {
    "scipy": ("iwopy.interfaces.scipy", "Optimizer_scipy"),
    "cma": ("iwopy.interfaces.cma", "Optimizer_cma"),
    "pymoo": ("iwopy.interfaces.pymoo", "Optimizer_pymoo"),
    "pygmo": ("iwopy.interfaces.pygmo", "Optimizer_pygmo"),
    "nlopt": ("iwopy.interfaces.pygmo", "Optimizer_pygmo"),
    "nevergrad": ("iwopy.interfaces.nevergrad", "Optimizer_nevergrad"),
    "optuna": ("iwopy.interfaces.optuna", "Optimizer_optuna"),
    "deap": ("iwopy.interfaces.deap", "Optimizer_deap"),
}


def _set_selector(
    family: str, optimizer: str | Callable[..., Any] | None, kwargs: dict[str, Any]
) -> None:
    if family == "nlopt":
        if "nlopt_optimizer" in kwargs:
            raise ValueError("nlopt: do not also supply nlopt_optimizer")
        kwargs["nlopt_optimizer"] = optimizer
        kwargs["optimizer"] = "nlopt"
    else:
        kwargs["optimizer"] = optimizer


def create_optimizer(
    family: str,
    optimizer: str | Callable[..., Any] | None,
    problem: Problem,
    **kwargs: Any,
) -> Optimizer:
    """Construct an optimizer selected by backend family and algorithm.

    Parameters
    ----------
    family
        Algorithm family: ``iwopy``, ``scipy``, ``cma``, ``pymoo``, ``pygmo``, ``nlopt``,
        ``nevergrad``, ``optuna``, or ``deap``. Names are case-sensitive.
    optimizer
        Backend algorithm name. The iwopy algorithms are ``GG`` and ``SLSQP``;
        pycma uses ``CMAES``. SciPy accepts a minimize method name, a callable,
        or ``None`` for its default, and ``differential_evolution`` for its
        separate population adapter. The ``nlopt`` family accepts NLopt solver
        names such as ``lbfgs`` and uses the PyGMO adapter internally.
    problem
        Problem passed to the adapter without changing its lifecycle state.
    kwargs
        Explicit keyword parameters of the selected optimizer, plus flat native
        algorithm options accepted by that adapter. The factory does not
        initialize or solve. For NLopt it supplies the PyGMO driver and native
        solver selector internally; do not also supply nlopt_optimizer.

    Returns
    -------
    solver
        A new, uninitialized optimizer. No optimization is performed. Optional
        dependencies are loaded according to the selected adapter's lifecycle.

    Raises
    ------
    ValueError
        If the family, an iwopy/pycma selector, or the selector form is invalid,
        or a factory-owned selector is supplied twice. Backend-specific errors
        retain the selected adapter's validation and timing.
    TypeError
        If the selected adapter's constructor arguments are invalid.
    ImportError
        If a dependency needed during adapter construction is unavailable.
    """
    if family == "iwopy":
        if not isinstance(optimizer, str) or optimizer not in {"GG", "SLSQP"}:
            raise ValueError(f"iwopy: unknown optimizer {optimizer!r}; use GG or SLSQP")
        module, name = "iwopy.optimizers", optimizer
    elif family in _ADAPTERS:
        module, name = _ADAPTERS[family]
        if family == "pygmo" and optimizer == "nlopt":
            raise ValueError("pygmo: select NLopt solvers with family='nlopt'")
        if family == "scipy" and optimizer == "differential_evolution":
            name = "Optimizer_scipy_de"
        elif family == "cma":
            if optimizer != "CMAES":
                raise ValueError(f"cma: unknown optimizer {optimizer!r}; use CMAES")
        else:
            if family != "scipy" and not isinstance(optimizer, str):
                raise ValueError(
                    f"{family}: optimizer must be a backend algorithm name"
                )
            if not (
                optimizer is None or isinstance(optimizer, str) or callable(optimizer)
            ):
                raise ValueError(
                    "scipy: optimizer must be a method name, callable or None"
                )
            _set_selector(family, optimizer, kwargs)
    else:
        raise ValueError(f"Unknown optimizer family {family!r}")
    adapter = cast(type[Optimizer], getattr(import_module(module), name))
    return adapter(problem, **kwargs)
