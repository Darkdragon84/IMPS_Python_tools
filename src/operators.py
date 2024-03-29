from dataclasses import dataclass
from functools import singledispatch
from itertools import product
from typing import Iterable, TypeVar, Tuple, Iterator

import numpy as np
from scipy.sparse import csr_array

from src.mps import MPSMat
from src.utilities import tuple_to_index, HasMatOps, dtype_precision

VT = TypeVar("VT")
T = TypeVar("T")


@dataclass
class Operator(HasMatOps):
    n_sites: int
    dim_phys: int
    dtype: VT = np.float64
    matrix: csr_array = None

    def __post_init__(self):
        if self.matrix is None:
            self.matrix = csr_array((self.dim, self.dim), dtype=self.dtype)
        else:
            self.dtype = self.matrix.dtype

    @classmethod
    def from_data(cls, data: Iterable[VT], indices: Iterable[Tuple[int, int]], n_sites: int, dim_phys: int):
        dim = dim_phys ** n_sites
        matrix = csr_array((data, zip(*indices)), shape=(dim, dim))
        return cls(n_sites, dim_phys, matrix.dtype, matrix)

    @property
    def dim(self):
        return self.dim_phys ** self.n_sites

    @property
    def dense_matrix(self):
        return self.matrix.todense()

    def __eq__(self, other: "Operator") -> bool:
        return all([
            isinstance(other, Operator),
            self.n_sites == other.n_sites,
            self.dim_phys == other.dim_phys,
            (self.matrix != other.matrix).nnz == 0
        ])

    def __getitem__(self, inds) -> VT:
        return self.matrix[inds]

    def row_iter(self, row: int) -> Iterator[Tuple[int, VT]]:
        yield from zip(self.matrix.indices[self.matrix.indptr[row]: self.matrix.indptr[row + 1]],
                       self.matrix.data[self.matrix.indptr[row]: self.matrix.indptr[row + 1]])

    def __neg__(self) -> "Operator":
        return self.__class__(self.n_sites, self.dim_phys, matrix=-self.matrix)  # type: ignore

    def __add__(self, other: "Operator") -> "Operator":
        assert self.dim_phys == other.dim_phys
        assert self.n_sites == other.n_sites
        return self.__class__(self.n_sites, self.dim_phys, matrix=self.matrix + other.matrix)  # type: ignore

    def __sub__(self, other: "Operator") -> "Operator":
        assert self.dim_phys == other.dim_phys
        assert self.n_sites == other.n_sites
        return self.__class__(self.n_sites, self.dim_phys, matrix=self.matrix - other.matrix)  # type: ignore

    def __matmul__(self, other: "Operator") -> "Operator":
        assert self.dim_phys == other.dim_phys
        assert self.n_sites == other.n_sites
        return self.__class__(self.n_sites, self.dim_phys, matrix=self.matrix @ other.matrix)  # type: ignore

    def __mul__(self, other: T):
        # singledispatchmethod can't be used here bc we can't register "Operator" yet.
        return op_mult(other, self)

    def __truediv__(self, scalar: VT) -> "Operator":
        return self.__class__(self.n_sites, self.dim_phys, matrix=self.matrix / scalar)  # type: ignore

    def close_to(self, other: "Operator", tol: float = 0) -> bool:
        return all_close(self, other, tol)


@singledispatch
def op_mult(right: T, left: Operator) -> T:
    raise NotImplementedError


@op_mult.register(MPSMat)
def _mult_op_mps(right: MPSMat, left: Operator) -> MPSMat:
    assert right.dim_phys == left.dim_phys, f"mps must be of same physical dimension ({left.dim_phys}), but has {right.dim_phys}"
    matrices = [
        sum([val * right[j] for j, val in left.row_iter(i)])
        for i in range(left.dim_phys)
    ]
    return right.__class__(matrices)


@op_mult.register(Operator)
def _mult_op_op(right: Operator, left: Operator) -> Operator:
    assert left.dim_phys == right.dim_phys
    data, indices = zip(*[
        (left[i1, j1] * right[i2, j2], (tuple_to_index((i1, i2), left.dim_phys),
                                        tuple_to_index((j1, j2), left.dim_phys)))
        for (i1, j1), (i2, j2) in product(zip(*left.matrix.nonzero()), zip(*right.matrix.nonzero()))
    ])
    return left.__class__.from_data(data, indices, left.n_sites + right.n_sites, left.dim_phys)


def all_close(op1: Operator, op2: Operator, tol: float = 0) -> bool:
    diff = op1 - op2
    tol = tol or 10 * dtype_precision(diff.dtype)
    return all(abs(diff[i, j]) < tol for i, j in zip(*diff.matrix.nonzero()))
