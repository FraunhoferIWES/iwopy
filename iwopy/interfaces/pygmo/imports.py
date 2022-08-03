try:
    import pygmo

    IMPORT_OK = True
except ImportError:
    pygmo = None
    IMPORT_OK = False


def check_import():
    """
    Checks if library import worked,
    raises error otherwise.
    """
    if not IMPORT_OK:
        print("\n\nFailed to import pygmo. Please install, e.g. via pip:")
        print("  pip install pygmo\n\n")
        raise ImportError("Failed to import pygmo")
