import numpy as np


def get_default_keyf(digits=12):
    """
    Get the default key function

    Parameters
    ----------
    digits : int
        The number of digits for floats

    Returns
    -------
    Function :
        The default key function

    """

    def default_key(vars_int, vars_float):
        """
        Default key function

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        Object :
            The key

        """
        li = vars_int.tolist() if len(vars_int) else []
        tf = tuple(tuple(v.tolist()) for v in np.round(vars_float, digits))
        return (tuple(li), tf)

    return default_key


class Memory:
    """
    Storage for function results.

    Parameters
    ----------
    size : int
        The number of maximally stored results
    keyf : Function, optional
        The memory key function. Parameters:
        (vars_int, vars_float), returns key Object

    Attributes
    ----------
    max_size : int
        The number of maximally stored results
    data : dict
        The stored data. Key: keyf return type,
        Values: tuples (objs, cons)
    keyf : Function
        The memory key function. Parameters:
        (vars_int, vars_float), returns key Object

    """

    def __init__(self, size, keyf=None):
        self.max_size = size
        self.keyf = keyf if keyf is not None else get_default_keyf()
        self.data = {}

    def clear(self):
        """
        Clears the memory
        """
        self.data = {}

    @property
    def size(self):
        """
        The number of elements currently stored
        in memory

        Returns
        -------
        int :
            The number of elements currently stored
            in memory

        """
        return len(self.data)

    def found_individual(self, vars_int, vars_float):
        """
        Check if entry is found in memory.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        found : bool
            True if data is available

        """
        key = self.keyf(vars_int, vars_float)
        return key in self.data

    def found_population(self, vars_int, vars_float):
        """
        Check if entry is found in memory.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)

        Returns
        -------
        found : numpy.ndarray of bool
            True if data is available, shape: (n_pop,)

        """
        n_pop = len(vars_float)
        found = np.zeros(n_pop, dtype=bool)
        for pi in range(n_pop):
            found[pi] = self.found_individual(vars_int[pi], vars_float[pi])
        return found

    def store_individual(self, vars_int, vars_float, objs, cons):
        """
        Store objs and cons data.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)
        objs : np.array
            The objective function values, shape: (n_objectives,)
        con : np.array
            The constraints values, shape: (n_constraints,)

        """
        key = self.keyf(vars_int, vars_float)
        if key in self.data and self.size == self.max_size:
            delk = next(iter(self.dict.keys()))
            del self.dict[delk]
        self.data[key] = (objs.copy(), cons.copy())

    def store_population(self, vars_int, vars_float, objs, cons):
        """
        Store objs and cons data of a population.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        objs : np.array
            The objective function values, shape: (n_pop, n_objectives)
        con : np.array
            The constraints values, shape: (n_pop, n_constraints)

        """
        for pi in range(len(objs)):
            self.store_individual(vars_int[pi], vars_float[pi], objs[pi], cons[pi])

    def lookup_individual(self, vars_int, vars_float):
        """
        Lookup results from memory.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_vars_int,)
        vars_float : np.array
            The float variable values, shape: (n_vars_float,)

        Returns
        -------
        results : tuple or None
            The results (objs, cons) if found, None otherwise

        """
        key = self.keyf(vars_int, vars_float)
        if key not in self.data:
            return None
        objs, cons = self.data[key]
        return objs.copy(), cons.copy()

    def lookup_population(self, vars_int, vars_float, target=None):
        """
        Lookup results from memory.

        Parameters
        ----------
        vars_int : np.array
            The integer variable values, shape: (n_pop, n_vars_int)
        vars_float : np.array
            The float variable values, shape: (n_pop, n_vars_float)
        target : numpy.ndarray, optional
            The results array to write to, shape:
            (n_pop, n_objs_cmpnts + n_cons_cmpnts)

        Returns
        -------
        results : numpy.ndarray or None
            None if no results at all found, otherwise array
            with shape: (n_pop, n_objs_cmpnts + n_cons_cmpnts)

        """
        results = target
        n_pop = len(vars_float)
        for pi in range(n_pop):
            key = self.keyf(vars_int[pi], vars_float[pi])
            if key in self.data:

                objs, cons = self.data[key]

                if results is None:
                    n_o = len(objs)
                    n_c = len(cons)
                    results = np.full((n_pop, n_o + n_c), np.nan, dtype=objs.dtype)

                results[pi] = np.r_[objs, cons]

        return results
