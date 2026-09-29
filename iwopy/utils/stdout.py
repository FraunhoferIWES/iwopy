import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager


@contextmanager
def suppress_stdout(silent: bool = True) -> Iterator[None]:
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
    silent
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
