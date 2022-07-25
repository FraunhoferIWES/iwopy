import numpy as np

import iwopy


def test_interp_point():

    D = 6

    for dims in range(1, D + 1):

        print("\nINTERP_POINT: Entering dims =", dims, "\n")

        o = np.random.uniform(-10.0, 10.0, dims)
        d = np.random.uniform(0.1, 5.0, dims)
        n = np.random.randint(1, 20, dims)
        p = np.random.uniform(-50.0, 50.0, dims)

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


if __name__ == "__main__":
    test_interp_point()
    test_interp_points()
