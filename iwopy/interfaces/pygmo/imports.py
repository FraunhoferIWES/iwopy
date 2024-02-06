from iwopy.utils import import_module

pygmo = None
loaded = False


def load(verbosity=1):

    global pygmo, loaded

    if not loaded:

        if verbosity:
            print("Loading pygmo")

        pygmo = import_module("pygmo", hint="pip install pygmo")

        loaded = True

        if verbosity:
            print("pygmo successfully loaded")
