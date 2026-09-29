import importlib
from types import ModuleType


def import_module(
    name: str, package: str | None = None, hint: str | None = None
) -> ModuleType:
    """
    Imports a module dynamically.

    Parameters
    ----------
    name
        The module name
    package
        The explicit package name, deduced from name
        if not given
    hint
        Installation advice, in case the import fails

    Returns
    -------
    mdl
        The imnported package
    """
    try:
        return importlib.import_module(name, package)
    except ModuleNotFoundError:
        mdl = name if package is None else f"{package}.{name}"
        hint = hint if hint is not None else f"pip install {name}"
        raise ModuleNotFoundError(f"Module '{mdl}' not found, maybe try '{hint}'")
