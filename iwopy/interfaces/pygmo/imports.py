from types import ModuleType

from iwopy.utils import import_module

pygmo: ModuleType | None = None
loaded: bool = False


def load(verbosity: int = 1) -> None:
    """
    Loads the pygmo package dynamically

    Parameters
    ----------
    verbosity
        The verbosity level, 0 = silent



    """

    global pygmo, loaded

    if not loaded:
        if verbosity:
            print("Loading pygmo")

        pygmo = import_module("pygmo", hint="pip install pygmo")

        loaded = True

        if verbosity:
            print("pygmo successfully loaded")
