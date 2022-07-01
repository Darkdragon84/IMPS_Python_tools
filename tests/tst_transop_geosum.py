import numpy
import numpy as np

from src.math_utilities import matrix_dot
from src.mps import MPSMat
from src.transfer_operator import transfer_op_dominant_eigs, TransferOperator, transfer_op_geometric_sum
from src.utilities import Direction
from numpy.linalg import norm


def main():
    reltol = 1e-10
    d = 3
    m = 200
    one = np.eye(m)
    dtype = numpy.float64
    # dtype = numpy.float32

    AL = MPSMat.random_left_ortho_mps(d, (m, m), dtype=dtype)
    TMAL = TransferOperator(AL)

    eR, R = transfer_op_dominant_eigs(TMAL, Direction.RIGHT)
    L = None

    x = numpy.random.randn(m, m)
    yl = transfer_op_geometric_sum(x, TMAL, Direction.LEFT, L, R, chk=True, reltol=reltol)

    x_projl = x - matrix_dot(x, R) * one
    res_l = yl - TMAL.mult_left(yl) + matrix_dot(yl, R) * one - x_projl
    print(f"Tr(yl*R)/|yl||R|={matrix_dot(yl, R) / (norm(yl) * norm(R)):2.6e}")
    print(f"system of eqns. fulfilled to rel. prec {norm(res_l) / norm(x_projl):2.6e}")

    BR = MPSMat.random_right_ortho_mps(d, (m, m), dtype=dtype)
    TMBR = TransferOperator(BR)

    eL, L = transfer_op_dominant_eigs(TMBR, Direction.LEFT)
    R = None

    x = numpy.random.randn(m, m)
    yr = transfer_op_geometric_sum(x, TMBR, Direction.RIGHT, L, R, chk=True, reltol=reltol)
    x_projr = x - matrix_dot(L, x) * one
    res_r = yr - TMBR.mult_right(yr) + matrix_dot(L, yr) * one - x_projr

    print(f"Tr(L*yr)/|L||yr|={matrix_dot(L, yr) / (norm(L) * norm(yr)):2.6e}")
    print(f"system of eqns. fulfilled to rel. prec {norm(res_r) / norm(x_projr):2.6e}")


if __name__ == '__main__':
    main()
