"""Translate flat SciPy keyword arguments into native minimize options."""

from typing import Any


def minimize_options(
    kwargs: dict[str, Any], tol: float | None = None
) -> dict[str, Any]:
    managed = {"bounds", "callback", "constraints", "jac", "method"}
    options = {key: value for key, value in kwargs.items() if key not in managed}
    result = {key: value for key, value in kwargs.items() if key in managed}
    result["options"] = options
    if tol is not None:
        result["tol"] = tol
    return result
