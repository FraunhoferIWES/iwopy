try:

    from pymoo.core.callback import Callback
    from pymoo.core.problem import Problem

    IMPORT_OK = True

except ImportError:

    Problem = object

    class Callback:
        def __init__(self):
            self.data = dict()

    IMPORT_OK = False


def check_import():
    """
    Checks if library import worked,
    raises error otherwise.
    """
    if not IMPORT_OK:
        print("\n\nFailed to import pymoo. Please install, e.g. via pip:")
        print("  pip install pymoo\n\n")
        raise ImportError("Failed to import pymoo")
