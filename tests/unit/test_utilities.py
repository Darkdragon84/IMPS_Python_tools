import pytest
import numpy as np

from src.utilities import dtype_precision


@pytest.mark.parametrize("dt, expected", [
    (np.dtype("float128"), 1e-32),
    (np.dtype("float64"), 1e-16),
    (np.dtype("float32"), 1e-8),
    (np.dtype("float16"), 1e-4),
    (np.dtype("complex256"), 1e-32),
    (np.dtype("complex128"), 1e-16),
    (np.dtype("complex64"), 1e-8),
])
def test_dtype_precision(dt, expected):
    assert dtype_precision(dt) == expected
