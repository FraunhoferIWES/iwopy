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

        g = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        g.print_info()

        pts, c = g.interpolation_coeffs_point(p)

        q = np.einsum("pd,p->d", pts, c)

        print("pts\n", pts.tolist())
        print("c\n", c.tolist())
        print("p\n", p.tolist())
        print("q\n", q.tolist())

        d = np.abs(p - q)
        print("d\n", d.tolist())

        assert np.all(d < 1e-11)


def test_interp_points():

    D = 6
    N = 100

    for dims in range(1, D + 1):

        print("\nINTERP_POINTS: Entering dims =", dims, "\n")

        o = np.random.uniform(0.0, 0.05, dims)
        d = np.random.uniform(0.1, 0.2, dims)
        n = np.random.randint(51, 60, dims)
        pts = np.random.uniform(0.1, 5.0, (N, dims))

        g = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        g.print_info()

        gpts, c = g.interpolation_coeffs_points(pts)
        print("RESULTS", gpts.shape, c.shape)
        # print("GPTS\n",gpts.tolist())
        # print("COEFFS\n",c.tolist())

        qts = np.einsum("gx,pg->px", gpts, c)
        print("QTS", qts.shape, "PTS", pts.shape)

        d = np.abs(pts - qts)
        print("MAX DELTA", d.shape, np.max(d))

        assert np.all(d < 1e-12)


def test_deriv_gp():

    dnl = (
        (0.01, 2, 2, 0.00015),
        (0.01, 2, 1, 0.01),
        (0.01, 1, 1, 0.01),
        (0.001, 2, 2, 1.35e-6),
        (0.001, 2, 1, 0.001),
        (0.001, 1, 1, 0.001),
    )

    def f(x):
        return x + 0.5 * np.sin(2 * x)

    def g(x):
        return 1 + np.cos(2 * x)

    for step, order, orderb, lim in dnl:

        print("\nENTERING", (step, order, orderb, lim), "\n")

        o = [0.0]
        d = [step]
        n = [int(1 / step)]
        gpts = np.array([[0.0], [0.3], [0.6], [1.0]])

        grid = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        grid.print_info()

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
        print(f"\ncpts {cpts.shape} =\n", cpts.tolist())
        print(f"c {c.shape} =\n", c.tolist())
        fc = f(cpts)
        print(f"fc {fc.shape} =\n", fc.tolist())

        rg = np.einsum("gd,pg->pd", fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())
        print("max delta =", np.max(delta))

        assert np.all(delta < lim)


def test_deriv():

    dnl = (
        (0.01, 2, 2, 0.00015),
        (0.01, 2, 1, 0.01),
        (0.01, 1, 1, 0.01),
        (0.001, 2, 2, 1.35e-06),
        (0.001, 2, 1, 0.001),
        (0.001, 1, 1, 0.001),
    )

    def f(x):
        return x + 0.5 * np.sin(2 * x)

    def g(x):
        return 1 + np.cos(2 * x)

    for step, order, orderb, lim in dnl:

        print("\nENTERING", (step, order, orderb, lim), "\n")

        o = [0.0]
        d = [step]
        n = [int(1 / step)]
        pts = np.array([[0.0], [0.3], [0.1 * np.pi], [0.60000123124], [1.0]])

        grid = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        grid.print_info()

        print("\ngpts =", pts.tolist())

        x = pts[:, 0]
        print("\nx =", x)
        fv = f(pts)
        print("f =", fv[:, 0])
        gv = g(pts)
        print("g =", gv[:, 0])

        cpts, c = grid.deriv_coeffs(pts, 0, order, orderb)
        print(f"\ncpts {cpts.shape} =\n", cpts.tolist())
        print(f"c {c.shape} =\n", c.tolist())
        fc = f(cpts)
        print(f"fc {fc.shape} =\n", fc.tolist())

        rg = np.einsum("gd,pg->pd", fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())
        print("max delta =", np.max(delta))

        assert np.all(delta < lim)


def test_grad_gp():

    dnl = (
        (None, 0.001, 2, 1, [0.001, 0.005, 0.005]),
        (None, 0.01, 2, 2, [0.0001, 0.0002, 0.001]),
        (None, 0.01, 2, 1, [0.0001, 0.02, 0.05]),
        (None, 0.01, 1, 1, [0.0001, 0.02, 0.05]),
        ([0, 2], 0.01, 1, 1, [0.0001, 0.05]),
        ([0, 2], 0.1, 1, 1, [0.001, 0.5]),
    )

    def f(x, y, z):
        return x * y * z - 2 * y * z**2 + (x + z) / y - y**2 + 3 * x

    def g(x, y, z, vars=None):
        if vars is None:
            vars = [0, 1, 2]
        out = np.zeros(list(x.shape) + [len(vars)])
        for vi, v in enumerate(vars):
            if v == 0:
                out[..., vi] = y * z + 1 / y + 3
            if v == 1:
                out[..., vi] = x * z - 2 * z**2 - (x + z) / y**2 - 2 * y
            if v == 2:
                out[..., vi] = x * y - 4 * y * z + 1 / y
        return out

    for vars, step, order, orderb, lim in dnl:

        print("\nENTERING", (vars, step, order, orderb, lim), "\n")

        o = [0.0, 1.0, 0.0]
        d = [step, step, step]
        n = [int(1 / step), int(1 / step), int(1 / step)]
        gpts = np.array(
            [
                [0.0, 1.0, 0.0],
                [0.5, 1.5, 0.5],
                [0.8, 1.1, 0.7],
                [0.0, 1.3, 1.0],
                [1.0, 2.0, 1.0],
            ]
        )

        grid = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        grid.print_info()

        inds = grid.gpts2inds(gpts)
        print("\ngpts =", gpts.tolist())
        print("inds =", inds.tolist())

        fv = f(gpts[:, 0], gpts[:, 1], gpts[:, 2])
        print(f"\nf {fv.shape} =", fv)
        gv = g(gpts[:, 0], gpts[:, 1], gpts[:, 2], vars)
        print(f"g {gv.shape} =", gv)

        cpts, c = grid.grad_coeffs_gridpoints(inds, vars, order, orderb)
        print(f"\ncpts {cpts.shape} =\n", cpts.tolist())
        print(f"c {c.shape} =\n", c.tolist())
        fc = f(cpts[:, 0], cpts[:, 1], cpts[:, 2])
        print(f"fc {fc.shape} =\n", fc.tolist())

        rg = np.einsum("g,pvg->pv", fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())
        print("max delta =", np.max(delta, axis=0))

        assert np.all(delta < np.array(lim)[None, :])


def test_grad():

    dnl = (
        (None, 0.001, 2, 1, [0.001, 0.0008, 0.005]),
        (None, 0.01, 2, 2, [0.001, 0.001, 0.001]),
        (None, 0.01, 2, 1, [0.001, 0.008, 0.05]),
        (None, 0.01, 1, 1, [0.001, 0.008, 0.05]),
        ([0, 2], 0.01, 1, 1, [0.001, 0.05]),
        ([0, 2], 0.1, 1, 1, [0.001, 0.5]),
    )

    def f(x, y, z):
        return x * y * z - 2 * y * z**2 + (x + z) / y - y**2 + 3 * x

    def g(x, y, z, vars=None):
        if vars is None:
            vars = [0, 1, 2]
        out = np.zeros(list(x.shape) + [len(vars)])
        for vi, v in enumerate(vars):
            if v == 0:
                out[..., vi] = y * z + 1 / y + 3
            if v == 1:
                out[..., vi] = x * z - 2 * z**2 - (x + z) / y**2 - 2 * y
            if v == 2:
                out[..., vi] = x * y - 4 * y * z + 1 / y
        return out

    for vars, step, order, orderb, lim in dnl:

        print("\nENTERING", (vars, step, order, orderb, lim), "\n")

        o = [0.0, 1.0, 0.0]
        d = [step, step, step]
        n = [int(1 / step), int(1 / step), int(1 / step)]
        pts = np.array(
            [
                [0.0, 1.0, 0.54],
                [0.512345, 1.5, 0.7875],
                [0.899, 1.1, 0.7],
                [0.0999, 1.3999, 0.999],
                [1.0, 2.0, 1.0],
            ]
        )

        grid = iwopy.utils.RegularDiscretizationGrid(o, d, n)
        grid.print_info()

        print("\npts =", pts.tolist())

        fv = f(pts[:, 0], pts[:, 1], pts[:, 2])
        print(f"\nf {fv.shape} =", fv)
        gv = g(pts[:, 0], pts[:, 1], pts[:, 2], vars)
        print(f"g {gv.shape} =", gv)

        cpts, c = grid.grad_coeffs(pts, vars, order, orderb)
        print(f"\ncpts {cpts.shape} =\n", cpts.tolist())
        print(f"c {c.shape} =\n", c.tolist())
        fc = f(cpts[:, 0], cpts[:, 1], cpts[:, 2])
        print(f"fc {fc.shape} =\n", fc.tolist())

        rg = np.einsum("g,pvg->pv", fc, c)
        print(f"\nrg {rg.shape} =\n", rg.tolist())

        delta = np.abs(rg - gv)
        print("delta =\n", delta.tolist())
        print("max delta =", np.max(delta, axis=0))

        assert np.all(delta < np.array(lim)[None, :])


if __name__ == "__main__":
    np.random.seed(42)
    test_interp_point()
    test_interp_points()
    test_deriv_gp()
    test_deriv()
    test_grad_gp()
    test_grad()
