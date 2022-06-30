from collections.abc import Sequence
from functools import singledispatchmethod
from typing import TypeVar, Optional

import numpy as np

from src.math_utilities import qr_pos
from src.operators import Operator
from src.utilities import DimsType, MatType


class IMPS(Sequence[MatType]):
    # as per https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 10000

    def __init__(self, matrices: Sequence[MatType]):
        matrices = list(matrices)

        self._dim_phys = len(matrices)
        self._dims = matrices[0].shape
        self._dtype = matrices[0].dtype

        assert all(mat.shape == self._dims and mat.dtype == self._dtype for mat in matrices)

        self._matrices = matrices

    def __repr__(self):
        return f"{self.__class__.__name__}(dim_phys={self.dim_phys}, dims={self.dims}, dtype={self.dtype})"

    def __eq__(self, other: "IMPS"):
        return all([
            type(self) == type(other),
            self.dim_phys == other.dim_phys,
            self.dims == other.dims,
            np.array_equal(self.matrices, other.matrices)  # let np.array_equal handle different dtypes
        ])

    def __len__(self) -> int:
        return self._dim_phys

    def __getitem__(self, ind) -> MatType:
        return self._matrices[ind]

    @property
    def matrices(self):
        return self._matrices

    @property
    def dim_phys(self) -> int:
        return len(self)

    @property
    def dims(self) -> DimsType:
        return self._dims

    @property
    def ndims(self):
        return len(self.dims)

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    def to_full_matrix(self, axis: int) -> MatType:
        return np.concatenate(self.matrices, axis)

    @classmethod
    def from_full_matrix(cls, mat: MatType, dim_phys: int, axis: int):
        assert mat.shape[axis] % dim_phys == 0
        return cls(np.split(mat, dim_phys, axis))

    @classmethod
    def random_mps(cls, dim_phys: int, dims: DimsType, dtype: Optional[np.dtype] = None, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        nrm_fac = np.sqrt(dim_phys*np.sqrt(np.prod(dims))).astype(dtype)

        dtype = dtype or np.float
        return cls([np.random.randn(*dims).astype(dtype) / nrm_fac for _ in range(dim_phys)])

    @classmethod
    def random_left_ortho_mps(cls, dim_phys: int, dims: DimsType, dtype: Optional[np.dtype] = None,
                              seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        dtype = dtype or np.float
        q, _ = qr_pos(np.random.randn(dim_phys * dims[0], *dims[1:]).astype(dtype))
        return cls.from_full_matrix(q, dim_phys, 0)

    @classmethod
    def random_right_ortho_mps(cls, dim_phys: int, dims: DimsType, dtype: Optional[np.dtype] = None,
                               seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
        dtype = dtype or np.float
        q, _ = qr_pos(np.random.randn(dim_phys * dims[-1], *dims[:-1]).astype(dtype))
        return cls.from_full_matrix(q.T, dim_phys, 1)

    @singledispatchmethod
    def __mul__(self, scalar: np.ScalarType):
        return self.mult_with_scalar(scalar)

    @__mul__.register(Operator)
    def __rmul__(self, operator: Operator):
        pass

    def __truediv__(self, scalar: np.ScalarType):
        return self.div_by_scalar(scalar)

    def __matmul__(self, other: MatType):
        if isinstance(other, np.ndarray):
            return self.mult_right_with_matrix(other)
        else:
            raise NotImplementedError(f"unsupported operand type(s) for @: {self.__class__.__name__} "
                                      f"and {other.__class__.__name__}")

    def __rmatmul__(self, other: MatType):
        if isinstance(other, np.ndarray):
            return self.mult_left_with_matrix(other)
        else:
            raise NotImplementedError(f"unsupported operand type(s) for @: {other.__class__.__name__} "
                                      f"and {self.__class__.__name__}")

    def mult_with_scalar(self, scalar: np.ScalarType):
        assert np.isscalar(scalar)
        return self.__class__([mat * scalar for mat in self._matrices])

    def div_by_scalar(self, scalar: np.ScalarType):
        assert np.isscalar(scalar)
        return self.__class__([mat / scalar for mat in self._matrices])

    def mult_right_with_matrix(self, x: Optional[MatType] = None):
        if x is None:
            # this is useful for applying the transfer operator to unity (i.e. a None matrix)
            return self

        if x.shape[0] != self.dims[1]:
            raise ValueError(f"x has wrong shape[0] (={x.shape[0]}, should be {self.dims[1]})")

        return self.__class__([mat @ x for mat in self._matrices])

    def mult_left_with_matrix(self, x: Optional[MatType] = None):
        if x is None:
            # this is useful for applying the transfer operator to unity (i.e. a None matrix)
            return self

        if x.shape[1] != self.dims[0]:
            raise ValueError(f"x has wrong shape[0] (={x.shape[1]}, should be {self.dims[0]})")

        return self.__class__([x @ mat for mat in self._matrices])


MpsType = TypeVar("MpsType", bound=IMPS)
