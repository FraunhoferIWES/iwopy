import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

from iwopy import Problem, Constraint, Objective

class ChainPopulation:

    def __init__(self, n_pop, N, radii=1., xy0=0., alpha=0.):
        self.N = N
        self.n_pop = n_pop

        self.radii = np.zeros(N)
        self.radii[:] = radii

        self.xy = np.zeros((n_pop, N, 2))
        self.xy[:, 0] = xy0

        self.alpha = np.zeros((n_pop, N-1))
        self.dists = np.zeros((n_pop, N, N))
        self.set_alpha(alpha)
    
    def set_alpha(self, alpha):

        self.alpha[:] = alpha
        arad = self.alpha*np.pi/180.
        uv = np.stack([np.cos(arad), np.sin(arad)], axis=-1)
        for i in range(1, self.N):
            self.xy[:, i] = self.xy[:, i-1] + uv[:, i-1] * (
                                self.radii[i-1] + self.radii[i])
        
        for i in range(self.N):
            d = self.xy - self.xy[:, i, None]
            self.dists[:, i] = np.linalg.norm(d, axis=-1)

    def get_fig(self, i=0):
        fig, ax = plt.subplots()
        xy = self.xy[i]
        for i, pxy in enumerate(xy):
            ax.add_patch(plt.Circle(pxy, self.radii[i], color='orange'))
        xy_imin = np.argmin(xy, axis=0)
        xy_imax = np.argmax(xy, axis=0)
        xy_min = xy[xy_imin, range(2)] - self.radii[xy_imin]
        xy_max = xy[xy_imax, range(2)] + self.radii[xy_imax]
        xy_del = xy_max - xy_min
        ax.set_xlim((xy_min[0] - 0.1*xy_del[0], xy_max[0] + 0.1*xy_del[0]))
        ax.set_ylim((xy_min[1] - 0.1*xy_del[1], xy_max[1] + 0.1*xy_del[1]))
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_title(f"N = {self.N}")
        return fig

class ChainProblem(Problem):

    def __init__(self, chain, ctol=1e-3):
        super().__init__(name="chain_problem")
        self.chain = chain

    def var_names_float(self):
        return [f"alpha_{i:04}" for i in range(self.chain.N - 1)]

    def initial_values_float(self):
        return self.chain.alpha[:-1]

    def min_values_float(self):
        return np.full(self.chain.N - 1, 0.)

    def max_values_float(self):
        return np.full(self.chain.N - 1, 360.-1e-6)

    def apply_individual(self, vars_int, vars_float):
        self.chain.set_alpha(vars_float[None, :])

    def apply_population(self, vars_int, vars_float):
        self.chain.set_alpha(vars_float)

class NoCrossing(Constraint):
    def __init__(self, problem, tol=1e-3):
        super().__init__(
            problem, "nocross", vnames_float=problem.var_names_float(), tol=tol
        )
        self.chain = problem.chain

    def n_components(self):
        N = self.chain.N
        return int((N**2 - N - 2*(N - 1))/2)

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        rmin = np.min(self.chain.radii)
        values = np.zeros(self.n_components())
        i0 = 0
        for i in range(self.chain.N - 2):
            i1 = i0 + self.chain.N - 2 - i
            meet = self.chain.dists[0, i, i+2:] - self.chain.radii[i] - self.chain.radii[i+2:]
            values[i0:i1] = 0.1 * rmin - meet
            i0 = i1
        
        return values

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        rmin = np.min(self.chain.radii)
        values = np.zeros((self.chain.n_pop, self.n_components()))
        i0 = 0
        for i in range(self.chain.N - 2):
            i1 = i0 + self.chain.N - 2 - i
            meet = self.chain.dists[:, i, i+2:] - self.chain.radii[i] - self.chain.radii[None, i+2:]
            values[:, i0:i1] = 0.1 * rmin - meet
            i0 = i1
        
        return values

class MaxStretch(Objective):
    def __init__(self, problem, direction=np.array([0., 1.]), name="stretch"):
        super().__init__(problem, name, vnames_float=problem.var_names_float())
        self.chain = problem.chain
        self.direction = direction

    def n_components(self):
        return 1

    def maximize(self):
        return [True]

    def calc_individual(self, vars_int, vars_float, problem_results, cmpnts=None):
        u = np.einsum('cd,d->c', self.chain.xy[0], self.direction)
        return np.max(u + self.chain.radii) - np.min(u + self.chain.radii)

    def calc_population(self, vars_int, vars_float, problem_results, cmpnts=None):
        u = np.einsum('pcd,d->pc', self.chain.xy, self.direction)[:, :, None]
        return np.max(u + self.chain.radii[None, :, None], axis=1) - np.min(u - self.chain.radii[None, :, None], axis=1)
