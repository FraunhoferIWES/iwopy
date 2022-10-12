import numpy as np

from iwopy.core import Optimizer, OptResults


class SCGG(Optimizer):
    """
    Simple Constrained Greedy Gradient (SCGG) optimizer

    Follows steepest decent, reducing step size
    in a finite number of steps on the way. Violated
    constaints get priority.

    Parameters
    ----------
    problem: iwopy.Problem
        The problem to optimize
    step_max : float or list or dict
        The maximal steps. Either uniform float value
        or list of floats for each problem variable,
        or dict with entry for each variable
    step_min : float or list or dict
        The minimal steps. Either uniform float value
        or list of floats for each problem variable,
        or dict with entry for each variable
    step_div_factor : float
        Step size division factor until step_min is reached
    vectorized : bool
        Flag for running in vectorized mode
    n_max_steps : int
        The maximal number of steps without fresh gradient
    memory_size : int
        The number of memorized visited points
    name: str, optional
        The name

    Attributes
    ----------
    step_max : numpy.ndarray
        Maximal step size for each problem variable,
        shape: (n_vars_float,)
    step_min : numpy.ndarray
        Minimal step size for each problem variable,
        shape: (n_vars_float,)
    step_div_factor : float
        Step size division factor until step_min is reached
    vectorized : bool
        Flag for running in vectorized mode
    n_max_steps : int
        The maximal number of steps without fresh gradient
    memory_size : int
        The number of memorized visited points
    memory : tuple
        Memorized data: (x, obj, grad, all_valid), each a
        numpy.ndarray, shapes: (memory_size, n_vars),
        (memory_size, n_vars), (memory_size,), (memory_size,)

    """

    def __init__(
            self, 
            problem, 
            step_max, 
            step_min, 
            step_div_factor=10.,
            vectorized=True, 
            n_max_steps=400,
            memory_size=500,
            name="SGG",
        ):
        super().__init__(problem, name)
        self.step_max = step_max
        self.step_min = step_min
        self.step_div_factor = step_div_factor
        self.vectorized = vectorized
        self.n_max_steps = n_max_steps
        self.memory_size = memory_size
        self.memory = None

    def initialize(self, verbosity=0):
        """
        Initialize the object.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        """
        if self.problem.n_objectives != 1:
            raise ValueError(f"Optimizer '{self.name}': Not applicable for multi-objective problems.")
        if self.problem.n_vars_int != 0:
            raise ValueError(f"Optimizer '{self.name}': Not applicable for problems with integer variables.")
        if self.problem.n_vars_float == 0:
            raise ValueError(f"Optimizer '{self.name}': Missing float variables in problem.")
        
        n_vars = self.problem.n_vars_float
        smax = np.zeros(n_vars, dtype=np.float64)
        if isinstance(self.step_max, dict):
            for i, vname in enumerate(self.problem.var_names_float()):
                if vname in self.step_max:
                    smax[i] = self.step_max[vname]
                else:
                    raise KeyError(f"Optimizer '{self.name}': Missing step_max entry for variable '{vname}'")
        elif isinstance(self.step_max, list) or isinstance(self.step_max, np.ndarray):
            if len(self.step_max) != n_vars:
                raise ValueError(f"Optimizer '{self.name}': step_max has wrong size {len(self.step_max)} for {n_vars} variables")
            smax[:] = self.step_max
        else:
            smax[:] = self.step_max
        self.step_max = smax

        smin = np.zeros(n_vars, dtype=np.float64)
        if isinstance(self.step_min, dict):
            for i, vname in enumerate(self.problem.var_names_float()):
                if vname in self.step_min:
                    smin[i] = self.step_min[vname]
                else:
                    raise KeyError(f"Optimizer '{self.name}': Missing step_min entry for variable '{vname}'")
        elif isinstance(self.step_min, list) or isinstance(self.step_min, np.ndarray):
            if len(self.step_min) != n_vars:
                raise ValueError(f"Optimizer '{self.name}': step_max has wrong size {len(self.step_min)} for {n_vars} variables")
            smin[:] = self.step_min
        else:
            smin[:] = self.step_min
        self.step_min = smin

        n_funcs = 1 + self.problem.n_constraints
        self.memory = (
            np.zeros((self.memory_size, n_vars), dtype=np.float64),
            np.zeros((self.memory_size, n_funcs, n_vars), dtype=np.float64),
            np.zeros(self.memory_size, dtype=np.float64),
            np.zeros(self.memory_size, dtype=bool)
        )

        super().initialize(verbosity)

    def print_info(self):
        """
        Print solver info, called before solving
        """
        s = f"  Optimizer '{self.name}'  "
        print(s)
        hline = "-"*len(s)
        print(hline)
        for i, vname in enumerate(self.problem.var_names_float()):
            print(f" ({i}) {vname}: step size {self.step_min:.2e} -- {self.step_max:.2e}")
        print(hline)

    def _get_newx(self, x, deltax):
        """
        Helper function for new x creation
        """
        n_vars = self.problem.n_vars_float
        newx = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
        newx[:] = x[None, :]
        for i in range(self.n_max_steps):
            newx[i] += np.sum(deltax[:i+1], axis=0)

        mi = self.problem.min_values_float()[None, :]
        ma = self.problem.max_values_float()[None, :]
        sel = np.all(newx >= mi, axis=1) & np.all(newx <= ma, axis=1)

        return newx[sel]
    
    def _grad2deltax(self, grad, step):
        """
        Helper function for deltax creation
        """
        j = np.argmax(np.abs(grad)/step)
        return grad * step[j]/np.abs(grad[j])

    def solve(self, verbosity=1):
        """
        Run the optimization solver.

        Parameters
        ----------
        verbosity : int
            The verbosity level, 0 = silent

        Returns
        -------
        results: iwopy.core.OptResults
            The optimization results object

        """
        super().solve(verbosity)

        # prepare:
        inone = np.array([], dtype=np.int32)
        n_vars = self.problem.n_vars_float   
        imem = 0
        nmem = 0

        # evaluate initial variables:     
        x = np.array(self.problem.initial_values_float(), dtype=np.float64)
        obs, cons = self.problem.evaluate_individual(inone, x)
        obs0 = obs[0]
        valid = self.problem.check_constraints_individual(cons)

        print("VARS",self.problem.var_names_float())
        
        step = self.step_max.copy()
        count = -1
        while not np.all(step < self.step_min):

            count += 1
            recover = not np.all(valid)

            #if count == 8: quit()
            print("\nCOUNT", count)
            print("X",list(x))
            print("VALID",np.sum(valid), ": RECOVER",recover)

            # check memory:
            sel = np.max(np.abs(x[None, :] - self.memory[0][:nmem]), axis=1) < 1e-13 if nmem > 0 else np.array([False])
            if np.any(sel):
                jmem = np.where(sel)[0][0]
                grads = self.memory[1][jmem]
                step /= self.step_div_factor
                print("OLD",i,list(x),self.memory[2][jmem])
            else:
                grads = self.problem.get_gradients(inone, x, pop=self.vectorized)

                # memorize:
                jmem = imem
                self.memory[0][jmem] = x
                self.memory[1][jmem] = grads
                self.memory[2][jmem] = obs[0]
                self.memory[3][jmem] = not recover
                print("NEW",jmem,obs[0],not recover)
                imem = (imem + 1) % self.memory_size
                nmem = min(nmem + 1, self.memory_size)

            # project out directions of constraint violation:
            grad = grads[0].copy()
            print("GRAD 0", list(grad))
            deltax = self._grad2deltax(-grad, step)
            ncons = cons + np.einsum('cd,d->c', grads[1:], deltax)
            nvalid = ncons <= 0 #self.problem.check_constraints_individual(ncons)
            newbad = valid & ~nvalid
            newgood = ~valid & nvalid
            cnews = newgood | newbad
            for ci in np.where(~valid | newbad )[0]:
                n = grads[1+ci] / np.linalg.norm(grads[1+ci])
                grad -= np.dot(grad, n) * n
            print("GRAD 1", list(grad))
            
            # follow grad, but move downwards along violated directions:
            deltax = np.zeros((self.n_max_steps, n_vars), dtype=np.float64)
            deltax[:] = self._grad2deltax(-grad, step)[None, :]
            for ci in np.where(~valid)[0]:
                deltax[:] += self._grad2deltax(-grads[1+ci], step)

            # linear approximation when crossing constraint bondary:
            for ci in np.where(cnews)[0]:
                m = np.linalg.norm(grads[1+ci])
                if np.abs(m) > 0:
                    deltax[0] -= grads[1+ci] * cons[ci] / m**2
            newx = self._get_newx(x, deltax)

            print("STEP",list(step))
            print("GRAD",list(grad))
            print("DELTAX 0", list(deltax[0]))
            if self.n_max_steps > 1:
                print("DELTAX 1", list(deltax[1]))
            print("NEWX", len(newx))
            if not len(newx):
                print("GOT CAUGHT, NO NEWX FOUND")
                break

            import matplotlib.pyplot as plt
            pres = self.problem.base_problem.apply_individual(inone, x)
            fig = self.problem.get_fig(pres)
            ax = fig.axes[0]
            for i, xy in enumerate(pres):
                ax.annotate(str(i), xy)
            plt.show()
            plt.close(fig)

            if self.vectorized:

                # calculate population:
                inonep = np.zeros((len(newx), 0), dtype=np.int32)
                obsp, consp = self.problem.evaluate_population(inonep, newx)
                validp = self.problem.check_constraints_population(consp)
                valc = np.all(validp, axis=1)
                print("A",n_vars,len(valc),recover,np.any(valc))

                # evaluate population results:
                if np.any(valc):
                    if recover:
                        i = np.where(valc)[0][0]
                        x = newx[i]
                        obs = obsp[i]
                        cons = consp[i]
                        valid = validp[i]
                        print("B",i,obs[0])

                    else: 

                        # find best:
                        obsp = obsp[valc]
                        if self.problem.maximize_objs[0]:
                            i = np.argmax(obsp)
                            if obsp[i][0] <= obs[0]:
                                i = -1
                        else:
                            i = np.argmin(obsp)
                            if obsp[i][0] >= obs[0]:
                                i = -1
                        print("C",obsp[i][0],i)

                        if i >= 0:
                            x = newx[valc][i]
                            obs = obsp[i]
                            cons = consp[valc][i]
                            valid = validp[valc][i]
                            print("D",i,obs[0])
                    
                else:
                    x = newx[0]
                    obs = obsp[0]
                    cons = consp[0]
                    valid = validp[0]    
                    print("E",obs[0],np.sum(~valid))
                

            # non-vectorized:
            else:
                for si in range(self.n_max_steps):

                    if si > 0:
                        newx += deltax

                    obsh, consh = self.problem.evaluate_individual(inone, newx)
                    validh = self.problem.check_constraints_individual(consh)
                    print("NEWX",list(newx))
                    print("OBSH",obsh[0])
                    quit()
                    
                    if np.all(validh):
                        if recover:
                            x = newx
                            obs = obsh
                            cons = consh
                            valid = validh
                            break

                        else:
                            if self.problem.maximize_objs[0]:
                                better = obsh[0] > obs[0]
                            else:
                                better = obsh[0] < obs[0]
                            if better:
                                x = newx
                                obs = obsh
                                consh = consh
                                valid = validh
                            else:
                                break

                    elif not recover:
                        break
        
        # final evaluation:
        pres, obs, cons = self.problem.finalize_individual(inone, x, verbosity)
        valid = self.problem.check_constraints_individual(cons, verbosity)
        if self.problem.maximize_objs[0]:
            better = obs[0] > obs0
        else:
            better = obs[0] < obs0
        success = np.all(valid) and better

        return OptResults(
            self.problem,
            success,
            inone,
            x,
            obs,
            cons,
            pres,
        )