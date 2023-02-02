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
        print("\n\nFailed to import pygmo. Please install, either via pip:\n")
        print("  pip install pygmo\n")
        print("or via conda:\n")
        print("  conda install -c conda-forge pygmo\n")
        raise ImportError("Failed to import pygmo")
