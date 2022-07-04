from dataclasses import dataclass, field
from itertools import pairwise
from math import sqrt

import numpy as np

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

        diag_inds = list(zip(range(self.dim_phys), range(self.dim_phys)))
        upper_off_diag_inds = list(pairwise(range(self.dim_phys)))
        lower_off_diag_inds = [(i2, i1) for i1, i2 in upper_off_diag_inds]

        mvec = [m - self.spin for m in range(self.dim_phys)]
        xy_offdiag = np.asarray([sqrt(self.spin*(self.spin + 1) - m1*m2) / 2. for m1, m2 in pairwise(mvec)])

        self.sz = Operator.from_data(list(reversed(mvec)), diag_inds, 1, self.dim_phys)
        self.sx = Operator.from_data(
            np.concatenate((xy_offdiag, xy_offdiag)), upper_off_diag_inds + lower_off_diag_inds, 1, self.dim_phys
        )
        self.isy = Operator.from_data(
            np.concatenate((xy_offdiag, -xy_offdiag)), upper_off_diag_inds + lower_off_diag_inds, 1, self.dim_phys
        )

        self.sp = self.sx + self.isy
        self.sm = self.sx - self.isy