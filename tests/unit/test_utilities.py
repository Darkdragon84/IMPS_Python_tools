import pytest
import numpy as np

from src.utilities import dtype_precision


@pytest.mark.parametrize("dt, expected", [
    (np.float128, 1e-32),
    (np.float64, 1e-16),
    (np.float32, 1e-8),
    (np.float16, 1e-4),
    (np.complex256, 1e-32),
    (np.complex128, 1e-16),
    (np.complex64, 1e-8),
])
def test_dtype_precision(dt, expected):
    assert dtype_precision(dt) == expected
