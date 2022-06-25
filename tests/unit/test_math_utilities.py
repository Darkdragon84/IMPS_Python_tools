import pytest
import numpy as np

from src.math_utilities import matrix_dot


@pytest.mark.parametrize("dims, seed", [
    ((3, 3), 42),
    ((10, 20), 84),
])
def test_matrix_dot(dims, seed):
    np.random.seed(seed)
    x = np.random.randn(*dims)
    y = np.random.randn(*dims)
    assert matrix_dot(x, y) == matrix_dot(y, x)
    assert matrix_dot(x, None) == matrix_dot(None, x) == np.trace(x)
    assert matrix_dot(None, y) == matrix_dot(y, None) == np.trace(y)
