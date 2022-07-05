from dataclasses import dataclass, field
from itertools import pairwise
from math import sqrt

from src.operators import Operator


@dataclass
class SU2Generators:
    dim_phys: int

    spin: float = field(init=False)
    sx: Operator = field(init=False)
    isy: Operator = field(init=False)
    sz: Operator = field(init=False)

    sp: Operator = field(init=False)
    sm: Operator = field(init=False)

    def __post_init__(self):

        self.spin = (self.dim_phys - 1) / 2.

        off_diag_inds = list(pairwise(range(self.dim_phys)))

        mvec = [m - self.spin for m in range(self.dim_phys)]
        xy_offdiag = [sqrt(self.spin*(self.spin + 1) - m1*m2) for m1, m2 in pairwise(mvec)]

        data, inds = zip(*[(m, (i, i)) for i, m in enumerate(reversed(mvec)) if m != 0])
        self.sz = Operator.from_data(data, inds, 1, self.dim_phys)
        self.sp = Operator.from_data(xy_offdiag, off_diag_inds, 1, self.dim_phys)
        self.sm = Operator(1, self.dim_phys, matrix=self.sp.matrix.T)

        self.sx = (self.sp + self.sm) / 2
        self.isy = (self.sp - self.sm) / 2
