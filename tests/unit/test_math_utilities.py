import pytest
import numpy as np

from src.constants import RDTYPES, CDTYPES
from src.math_utilities import matrix_dot, qr_pos
from src.utilities import dtype_precision


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


@pytest.mark.parametrize("dtype", RDTYPES)
@pytest.mark.parametrize("dims, seed", [
    ((20, 10), 42),
    ((15, 18), 69),
    ((50, 50), 84),
])
def test_qr_pos(dims, seed, dtype):
    np.random.seed(seed)
    mat = np.random.randn(*dims).astype(dtype)
    q, r = np.linalg.qr(mat)
    qpos, rpos = qr_pos(mat)
    assert np.array_equal(np.abs(np.diag(r)), np.diag(rpos))
    assert np.allclose(qpos @ rpos, mat, atol=10 * dtype_precision(dtype))


@pytest.mark.parametrize("dtype", CDTYPES)
def test_qr_pos_fails(dtype):
    np.random.seed(42)
    mat = np.random.randn(20, 20).astype(dtype)
    with pytest.raises(NotImplementedError):
        qr_pos(mat)
