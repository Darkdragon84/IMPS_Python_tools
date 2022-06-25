from itertools import product, chain

import numpy as np
import pytest

from src.constants import DTYPES, RDTYPES, CDTYPES
from src.mps import IMPS

TEST_MATRIX_PARAMETERS = [
    (2, (3, 3), 42),
    (4, (4, 2), 69),
    (5, (3, 8), 84),
]


class TestIMPS:

    @pytest.mark.parametrize("dim_phys, dims, seed", TEST_MATRIX_PARAMETERS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_to_from_full_matrix(self, dim_phys, dims, dtype, seed):
        mps = IMPS.random_mps(dim_phys, dims, dtype, seed)
        assert mps == IMPS.from_full_matrix(mps.to_full_matrix(0), dim_phys, 0)
        assert mps == IMPS.from_full_matrix(mps.to_full_matrix(1), dim_phys, 1)

    @pytest.mark.parametrize("scalar", [2, 3.4])
    @pytest.mark.parametrize("dim_phys, dims, seed", TEST_MATRIX_PARAMETERS)
    @pytest.mark.parametrize("dtype", DTYPES)
    def test_mult_div_with_scalar(self, dim_phys, dims, dtype, seed, scalar):
        np.random.seed(seed)
        mps_array = np.random.randn(dim_phys, *dims).astype(dtype)
        mps = IMPS(mps_array)
        assert (scalar * mps) == IMPS(mps_array * scalar)
        assert (mps * scalar) == IMPS(mps_array * scalar)
        assert (mps / scalar) == IMPS(mps_array / scalar)

    @pytest.mark.parametrize("dim_phys, dims, seed", TEST_MATRIX_PARAMETERS)
    @pytest.mark.parametrize("dtype_mat, dtype_mps", chain(product(RDTYPES, CDTYPES), product(CDTYPES, RDTYPES)))
    def test_mat_mult(self, dim_phys, dims, dtype_mat, dtype_mps, seed):
        matl = np.random.randn(18, dims[0]).astype(dtype_mat)
        matr = np.random.randn(dims[-1], 37).astype(dtype_mat)

        mps = IMPS.random_mps(dim_phys, dims, dtype_mps, seed)

        mpsl = matl @ mps
        mpsr = mps @ matr

        assert mpsl.dims == (18, dims[-1])
        assert mpsr.dims == (dims[0], 37)
        assert mpsl == IMPS([matl @ mat for mat in mps.matrices])
        assert mpsr == IMPS([mat @ matr for mat in mps.matrices])
