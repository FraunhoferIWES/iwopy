import numpy as np
from scipy.spatial.distance import cdist

from iwopy import Problem, SimpleConstraint, SimpleObjective

class Model:

    def __init__(self, initial_xy):
        self.xy = initial_xy
    
    @property
    def n_points(self):
        return len(self.xy)
    
    @property
    def n_dims(self):
        return self.xy.shape[1]
    
    def calculate_potential(self):
        dists = cdist(self.xy, self.xy)
        np.fill_diagonal(dists, np.inf)
        return np.sum(1/dists)

class MinPotential(SimpleObjective):

    def __init__(self, problem):
        super().__init__(problem, "potential", has_ana_derivs=False)
        self.model = problem.model
    
    def f(self, *vars):
        return self.model.calculate_potential()

class MaxRadius(SimpleConstraint):

    def __init__(self, problem, radius):
        super().__init__(problem, "radius", n_components=problem.model.n_points, has_ana_derivs=False)
        self.model = problem.model
        self.radius = radius
    
    def f(self, *vars):
        xy = np.array(vars).reshape(self.model.n_points, self.model.n_dims)
        r = np.linalg.norm(xy, axis=-1)
        return r - self.radius

class ModelProblem(Problem):

    def __init__(self, model, mins, maxs, radius):
        super().__init__(name="my_model")
        
        self.model = model
        self.N = self.model.n_points * self.model.n_dims
        self.init_vals = self.model.xy.copy()

        self.mins = np.zeros((model.n_points, model.n_dims), dtype=np.float64)
        self.mins[:] = np.array(mins)[None, :]
        self.maxs = np.zeros((model.n_points, model.n_dims), dtype=np.float64)
        self.maxs[:] = np.array(maxs)[None, :]

        self.add_objective(MinPotential(self))
        self.add_constraint(MaxRadius(self, radius))

    def var_names_float(self):
        vnames = []
        for i in range(self.model.n_points):
            vnames += [f"x{i}", f"y{i}"]
        return vnames

    def initial_values_float(self):
        return self.init_vals.reshape(self.N)

    def min_values_float(self):
        return self.mins.reshape(self.N)

    def max_values_float(self):
        return self.maxs.reshape(self.N)
    
    def apply_individual(self, vars_int, vars_float):
        xy = vars_float.reshape(self.model.n_points, self.model.n_dims)
        self.model.xy = xy
