import numpy as np
import pytest

from src.mps import IMPS

TEST_MATRIX_PARAMETERS = [
    (2, (3, 4), None, 42),
    (3, (10, 5), None, 69),
    (4, (3, 8), np.float64, 12),
    (4, (3, 8), np.float32, 0),
    (4, (3, 8), np.float16, 84),
]


class TestIMPS:

    @pytest.mark.parametrize("dim_phys, dims, dtype, seed", TEST_MATRIX_PARAMETERS)
    def test_to_from_full_matrix(self, dim_phys, dims, dtype, seed):
        mps = IMPS.random_mps(dim_phys, dims, dtype, seed)
        assert mps == IMPS.from_full_matrix(mps.to_full_matrix(0), dim_phys, 0)
        assert mps == IMPS.from_full_matrix(mps.to_full_matrix(1), dim_phys, 1)

    @pytest.mark.parametrize("scalar", [2, 3.4])
    @pytest.mark.parametrize("dim_phys, dims, dtype, seed", TEST_MATRIX_PARAMETERS)
    def test_mult_div_with_scalar(self, dim_phys, dims, dtype, seed, scalar):
        np.random.seed(seed)
        mps_array = np.random.randn(dim_phys, *dims).astype(dtype)
        mps = IMPS(mps_array)
        assert (scalar * mps) == IMPS(mps_array * scalar)
        assert (mps * scalar) == IMPS(mps_array * scalar)
        assert (mps / scalar) == IMPS(mps_array / scalar)
