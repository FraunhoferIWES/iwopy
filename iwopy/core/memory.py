

class Memory:
    """
    Storage for function results.

    Parameters
    ----------
    problem : iwopy.core.Problem
        The optimization problem
    size : int
        The number of maximally stored results
    keyf : Function
        The key function. Parameters: (vars_int, vars_float),
        r

    Attributes
    ----------
    problem : iwopy.core.Problem
        The optimization problem
    size : int
        The number of maximally stored results
    data : dict
        The stored data. Key: keyf return type,
        Values: numpy.ndarray, shape: 
        (n_problem_obj + n_problem_con,)


    """
    def __init__(self, problem, size):
        self.problem = problem
        self.size = size
        self.data = {}