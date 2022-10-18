import os
import sys
from contextlib import contextmanager


@contextmanager
def suppress_stdout(silent=True):
    """
    Surpresses print outputs

    Example
    -------
        >>> with suppress_stdout():
        >>>    ...

    Source:
    https://stackoverflow.com/questions/2125702/how-to-suppress-console-output-in-python

    Parameters
    ----------
    silent: bool
        Flag for the silent treatment.

    """
    with open(os.devnull, "w") as devnull:
        if silent:
            old_stdout = sys.stdout
            sys.stdout = devnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
        else:
            yield
