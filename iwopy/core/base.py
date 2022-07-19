class Base:
    """
    Generic base for various iwopy objects.

    Parameters
    ----------
    name: str
        The problem's name

    Attributes
    ----------
    name: str
        The problem's name

    """

    def __init__(self, name):
        self.name = name
        self._initialized = False

    @property
    def initialized(self):
        """
        Flag for finished initialization

        Returns
        -------
        bool:
            True if initialization has been done

        """
        return self._initialized

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        self._initialized = True

    def finalize(self, verbosity=0):
        """
        Finalize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        self._initialized = False
