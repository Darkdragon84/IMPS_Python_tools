import numpy

from helpers.mps_matrix import MPSMatrix
from helpers.transfer_operator import transop_dominant_eigs, TransferOperator


def main():
    d = 3
    m = 50
    dtype = numpy.float64
    # dtype = numpy.float32

    AL = MPSMatrix.get_random_left_ortho_mps(d, m, dtype=dtype)
    TMAL = TransferOperator(AL)

    eR, R = transop_dominant_eigs(TMAL, 'right')
    print(eR, R.trace())


if __name__ == '__main__':
    main()
