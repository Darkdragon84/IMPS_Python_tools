import numbers
from functools import lru_cache
from typing import Tuple

import numpy as np
import pytest

from src.mps import MPSMat
from src.transfer_operator import TransferOperator
from src.utilities import MatType
from tests.unit.test_data import TRANSOP_TEST_SETTINGS, TRANSOP_EXPECTED_LEFT, TRANSOP_EXPECTED_RIGHT


@lru_cache
def create_test_matrix(dims: Tuple[int, int], dtype: np.dtype = np.dtype(np.float64)) -> MatType:
    mat = np.asarray(range(np.prod(dims))).reshape(*dims)
    if not issubclass(dtype.type, numbers.Real):
        mat = mat + 1j * mat
    return mat.astype(dtype)


@lru_cache
def create_test_mps(dim_phys: int, dims: Tuple[int, int], dtype: np.dtype = np.dtype(np.float64)) -> MPSMat:
    dim1, dim2 = dims
    full_mat = create_test_matrix((dim_phys * dim1, dim2), dtype)
    return MPSMat.from_full_matrix(full_mat, dim_phys, 0)


class TestTransferOperator:
    @pytest.mark.parametrize("dim_phys, dims, dtype, expected_empty, expected_mat", [
        settings + expected for settings, expected in zip(TRANSOP_TEST_SETTINGS, TRANSOP_EXPECTED_LEFT)
    ])
    def test_mult_left(self, dim_phys, dims, dtype, expected_empty, expected_mat):
        mps = create_test_mps(dim_phys, dims, dtype)
        mat = create_test_matrix(dims, dtype)

        tm = TransferOperator(mps)
        assert not tm.is_mixed
        assert tm.is_square
        assert np.array_equal(tm.mult_left(), expected_empty)
        assert np.array_equal(tm.mult_left(mat), expected_mat)

    @pytest.mark.parametrize("dim_phys, dims, dtype, expected_empty, expected_mat", [
        settings + expected for settings, expected in zip(TRANSOP_TEST_SETTINGS, TRANSOP_EXPECTED_RIGHT)
    ])
    def test_mult_right(self, dim_phys, dims, dtype, expected_empty, expected_mat):
        mps = create_test_mps(dim_phys, dims, dtype)
        mat = create_test_matrix(dims, dtype)

        tm = TransferOperator(mps)
        assert not tm.is_mixed
        assert tm.is_square
        assert np.array_equal(tm.mult_right(), expected_empty)
        assert np.array_equal(tm.mult_right(mat), expected_mat)

    @pytest.mark.parametrize("dim11, dim12, dim21, dim22", [
        (10, 10, 20, 20),
        (10, 12, 20, 22),
        (10, 10, 20, 22),
        (10, 12, 20, 20),
    ])
    def test_dims(self, dim11, dim12, dim21, dim22):
        dims1 = (dim11, dim12)
        dims2 = (dim21, dim22)
        mps1 = create_test_mps(2, dims1)
        mps2 = create_test_mps(2, dims2)
        tm = TransferOperator(mps1, mps2)
        assert tm.is_mixed
        assert tm.is_square == (dim11 == dim12 and dim21 == dim22)
        assert tm.dims == (dim11 * dim21, dim12 * dim22)
        assert tm.argdims == ((dim21, dim11), (dim12, dim22))
