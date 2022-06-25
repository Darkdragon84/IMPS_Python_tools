import numpy

from src.math_utilities import matrix_dot, inf_norm
from src.mps import IMPS
from src.transfer_operator import transop_dominant_eigs, TransferOperator, transop_geometric_sum


def main():
    d = 3
    m = 200
    dtype = numpy.float64
    # dtype = numpy.float32

    AL = IMPS.get_random_left_ortho_mps(d, m, dtype=dtype)
    TMAL = TransferOperator(AL)

    eR, R = transop_dominant_eigs(TMAL, 'right')
    L = None
    # print(1 - eR, matrix_dot(L, R))
    # print("left:", inf_norm(TMAL.mult_left(L) - numpy.eye(m)))
    # print("right:", inf_norm(TMAL.mult_right(R) - R))

    x = numpy.random.randn(m, m)
    y, info = transop_geometric_sum(x, TMAL, 'left', L, R, chk=True)

    print("Tr(y*R)=", matrix_dot(y, R))
    print("done")


if __name__ == '__main__':
    main()
