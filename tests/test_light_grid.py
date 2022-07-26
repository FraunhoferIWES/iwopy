import numpy as np

import iwopy


def test_interp_point():

    D = 6

    for dims in range(1, D + 1):

        print("\nINTERP_POINT: Entering dims =", dims, "\n")

        o = np.random.uniform(0.0, 0.01, dims)
        d = np.random.uniform(0.1, 5.0, dims)
        n = np.random.randint(50, 60, dims)
        p = np.random.uniform(0.02, 5.0, dims)

        g = iwopy.tools.LightRegGrid(o, d, n)

        pts, c = g.interpolation_coeffs_point(p)

        q = np.einsum("pd,p->d", pts, c)

        print("pts\n", pts)
        print("c\n", c)
        print("p\n", p)
        print("q\n", q)

        assert np.all(np.abs(p - q) < 1e-10)

def test_interp_points():

    D = 6
    N = 100

    for dims in range(1, D + 1):

        print("\nINTERP_POINTS: Entering dims =", dims, "\n")

        o = np.random.uniform(0.0, 0.05, dims)
        d = np.random.uniform(0.1, 0.2, dims)
        n = np.random.randint(51, 60, dims)
        pts = np.random.uniform(0.1, 5.0, (N, dims))

        g = iwopy.tools.LightRegGrid(o, d, n)

        gpts, c = g.interpolation_coeffs_points(pts)

        print("RESULTS", gpts.shape, c.shape)

        qts = np.einsum("pdx,pd->px", gpts, c)

        print("QTS", qts.shape, "PTS", pts.shape)

        assert np.all(np.abs(pts - qts) < 1e-10)

def test_deriv():

    dnl = (
        (0.01, 2, 2, 0.001),
        (0.01, 2, 1, 0.01),
        (0.01, 1, 1, 0.01),
        (0.001, 2, 2, 1.35e-6),
        (0.001, 2, 1, 0.001),
        (0.001, 1, 1, 0.001),
    )

    def f(x):
        return x + 0.5 * np.sin(2*x)
    
    def g(x):
        return 1 + np.cos(2*x)

    for step, order, orderb, lim in dnl:

        print("\nENTERING", (step, order, orderb, lim), "\n")

        o = [0.]
        d = [step]
        n = [int(1/step) - 1]
        gpts = np.array([[0.], [0.3], [0.6], [1.0]])

        grid = iwopy.tools.LightRegGrid(o, d, n)
        print("n_dims  :", grid.n_dims)
        print("n_steps :", grid.n_steps)
        print("n_points:", grid.n_points)
        
        inds = grid.gpts2inds(gpts)
        print("\ngpts =", gpts.tolist())
        print("inds =", inds.tolist())

        x = gpts[:, 0]
        print("\nx =", x)
        fv = f(gpts)
        print("f =", fv[:, 0])
        gv = g(gpts)
        print("g =", gv[:, 0])

        cpts, c = grid.deriv_coeffs_gridpoints(inds, 0, order, orderb)
        print(f"\ncpts {cpts.shape} =\n",cpts.tolist())
        print(f"c {c.shape} =\n",c.tolist())
        fc = f(cpts)
        print(f"fc {fc.shape} =\n", fc.tolist())
        
        rg = np.einsum('pgd,pg->pd', fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())

        assert np.all(delta < lim)

def test_grad():

    dnl = (
        (None, 0.0011, 2, 1, [0.001, 0.002, 0.008]),
        (None, 0.011, 2, 2, [0.03, 0.05, 0.1]),
        (None, 0.011, 2, 1, [0.03, 0.05, 0.14]),
        (None, 0.011, 1, 1, [0.03, 0.05, 0.14]),
        ([0, 2], 0.011, 1, 1, [0.03, 0.14]),
        ([0, 2], 0.1, 1, 1, [0.12, 0.41]),
    )

    def f(x, y, z):
        return x*y*z - 2*y*z**2 + (x + z)/y - y**2 + 3*x
    
    def g(x, y, z, vars=None):
        if vars is None:
            vars = [0, 1, 2]
        out = np.zeros(list(x.shape) + [len(vars)])
        for vi, v in enumerate(vars):
            if v == 0:
                out[..., vi] = y*z + 1/y + 3
            if v == 1:
                out[..., vi] = x*z - 2*z**2 - (x + z)/y**2 - 2*y
            if v == 2:
                out[..., vi] = x*y - 4*y*z + 1/y
        return out

    for vars, step, order, orderb, lim in dnl:

        print("\nENTERING", (vars, step, order, orderb, lim), "\n")

        o = [0., 1., 0.]
        d = [step, step, step]
        n = [int(1/step) - 1, int(1/step) - 1, int(1/step) - 1]
        gpts = np.array([[0., 1., 0.], [0.5, 1.5, 0.5], [0.8, 1.1, 0.7], [0., 1.3, 1.], [1.0, 2.0, 1.0]])

        grid = iwopy.tools.LightRegGrid(o, d, n)
        print("n_dims  :", grid.n_dims)
        print("n_steps :", grid.n_steps)
        print("n_points:", grid.n_points)
        
        inds = grid.gpts2inds(gpts)
        print("\ngpts =", gpts.tolist())
        print("inds =", inds.tolist())

        fv = f(gpts[:, 0], gpts[:, 1], gpts[:, 2])
        print(f"\nf {fv.shape} =", fv)
        gv = g(gpts[:, 0], gpts[:, 1], gpts[:, 2], vars)
        print(f"g {gv.shape} =", gv)

        cpts, c = grid.grad_coeffs_gridpoints(inds, vars, order, orderb)
        print(f"\ncpts {cpts.shape} =\n",cpts.tolist())
        print(f"c {c.shape} =\n",c.tolist())
        fc = f(cpts[..., 0], cpts[..., 1], cpts[..., 2])
        print(f"fc {fc.shape} =\n", fc.tolist())
        
        rg = np.einsum('pvg,pvg->pv', fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())
        print(np.max(delta, axis=0))

        assert np.all(delta < np.array(lim)[None, :])

if __name__ == "__main__":
    
    #test_interp_point()
    #test_interp_points()
    #test_deriv()
    test_grad()
