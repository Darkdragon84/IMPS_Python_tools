import numpy

from src.math_utilities import matrix_dot
from src.mps import IMPS
from src.transfer_operator import transop_dominant_eigs, TransferOperator, transop_geometric_sum, Direction
from numpy.linalg import norm


def main():
    d = 3
    m = 200
    dtype = numpy.float64
    # dtype = numpy.float32

    AL = IMPS.random_left_ortho_mps(d, (m, m), dtype=dtype)
    TMAL = TransferOperator(AL)

    eR, R = transop_dominant_eigs(TMAL, Direction.RIGHT)
    L = None

    x = numpy.random.randn(m, m)
    yl, info = transop_geometric_sum(x, TMAL, Direction.LEFT, L, R, chk=True)

    print(f"Tr(yl*R)/|yl||R|={matrix_dot(yl, R) / (norm(yl) * norm(R)):2.6e}")

    BR = IMPS.random_right_ortho_mps(d, (m, m), dtype=dtype)
    TMBR = TransferOperator(BR)

    eL, L = transop_dominant_eigs(TMBR, Direction.LEFT)
    R = None

    x = numpy.random.randn(m, m)
    yr, info = transop_geometric_sum(x, TMBR, Direction.RIGHT, L, R, chk=True)

    print(f"Tr(L*yr)/|L||yr|={matrix_dot(L, yr) / (norm(L) * norm(yr)):2.6e}")


if __name__ == '__main__':
    main()
