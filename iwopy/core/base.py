from typing import overload


class Base:
    """
    Generic base for various iwopy objects.

    Attributes
    ----------
    name: str
        The name

    :group: core

    """

    def __init__(self, name: str | None) -> None:
        """
        Constructor

        Parameters
        ----------
        name
            The name

        """
        self.name = type(self).__name__ if name is None else name
        self._initialized = False

    def __str__(self) -> str:
        """
        Get info string

        Returns
        -------
        str :
            Info string

        """
        if self.name == type(self).__name__:
            return self.name
        return f"{self.name} ({type(self).__name__})"

    @property
    def initialized(self) -> bool:
        """
        Flag for finished initialization

        Returns
        -------
        bool :
            True if initialization has been done

        """
        return self._initialized

    @overload
    def initialize(self, verbosity: int = 0, /) -> None: ...

    @overload
    def initialize(self, *, verbosity: int = 0) -> None: ...

    def initialize(self, verbosity: int = 0) -> None:
        """
        Initialize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        self._initialized = True

    @overload
    def finalize(self, verbosity: int = 0, /) -> None: ...

    @overload
    def finalize(self, *, verbosity: int = 0) -> None: ...

    def finalize(self, verbosity: int = 0) -> None:
        """
        Finalize the object.

        Parameters
        ----------
        verbosity
            The verbosity level, 0 = silent

        """
        self._initialized = False
