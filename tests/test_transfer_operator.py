import numpy

from helpers.mps_matrix import MPSMatrix
from helpers.transfer_operator import TransferOperator, eigs_transop


def main():
    d = 3
    m = 10

    A = MPSMatrix.get_random_left_ortho_mps(d, m)
    B = MPSMatrix.get_random_right_ortho_mps(d, m)

    TMA = TransferOperator(A)
    TMB = TransferOperator(B)

    numpy.set_printoptions(precision=5, suppress=True)
    y1 = TMA.mult_left()
    y2 = TMB.mult_right()
    print(numpy.max(numpy.abs(y1 - numpy.eye(m))))
    print(numpy.max(numpy.abs(y2 - numpy.eye(m))))

    eA, rhoA = eigs_transop(TMA, 'right', 1)
    eB, rhoB = eigs_transop(TMB, 'left', 1)
    print(eA)
    print(eB)


if __name__ == '__main__':
    main()
