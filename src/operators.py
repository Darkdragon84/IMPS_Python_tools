from typing import Type, Iterable, TypeVar

import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import dataclass, field

VT = TypeVar("VT")

@dataclass
class Operator:
    n_sites: int
    dim_phys: int
    dtype: VT = np.float64
    matrix: csr_matrix = field(init=False)

    def __post_init__(self):
        self.matrix = csr_matrix((self.dim, self.dim), dtype=self.dtype)

    @property
    def dim(self):
        return self.dim_phys ** self.n_sites

    def __getitem__(self, indices):
        return self.matrix[indices]

    @classmethod
    def from_ijk(cls, row_inds: Iterable[int], col_inds: Iterable[int], data: Iterable[VT]):
        pass
