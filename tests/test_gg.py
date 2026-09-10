import numpy as np

from iwopy.optimizers import GG


def test_zero_gradient_produces_zero_step():
    optimizer = GG.__new__(GG)

    step = optimizer._grad2deltax(
        np.zeros(3, dtype=np.float64),
        np.ones(3, dtype=np.float64),
    )

    np.testing.assert_array_equal(step, np.zeros(3))
