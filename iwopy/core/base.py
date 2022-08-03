class Base:
    """
    Generic base for various iwopy objects.

    Parameters
    ----------
    name : str
        The name

    Attributes
    ----------
    name : str
        The name

    """

    def __init__(self, name):
        self.name = name
        self._initialized = False
        if name is None:
            self.name = type(self).__name__

    def __str__(self):
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
    def initialized(self):
        """
        Flag for finished initialization

        Returns
        -------
        bool :
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
