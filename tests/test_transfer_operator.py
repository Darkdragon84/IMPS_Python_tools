import numpy

from helpers.mps_matrix import MPSMatrix
from helpers.transfer_operator import TransferOperator, transop_dominant_eigs


def main():
    d = 3
    mA = 500
    mB = 80
    # dtype = numpy.complex128
    dtype = numpy.float64

    A = MPSMatrix.get_random_left_ortho_mps(d, mA, dtype=dtype)
    B = MPSMatrix.get_random_right_ortho_mps(d, mB, dtype=dtype)

    TMA = TransferOperator(A)
    TMB = TransferOperator(B)

    numpy.set_printoptions(precision=5, suppress=True)
    y1 = TMA.mult_left()
    y2 = TMB.mult_right()
    print(numpy.max(numpy.abs(y1 - numpy.eye(mA))))
    print(numpy.max(numpy.abs(y2 - numpy.eye(mB))))

    eA, rhoA = transop_dominant_eigs(TMA, 'right')
    eB, rhoB = transop_dominant_eigs(TMB, 'left')
    print(eA, eA.dtype, rhoA.dtype)
    print(eB, eB.dtype, rhoB.dtype)
    print(numpy.max(numpy.abs(rhoA - eA*TMA.mult_right(rhoA))))
    print(numpy.max(numpy.abs(rhoB - eB*TMB.mult_left(rhoB))))

    M = MPSMatrix.get_random_mps(d, mA)
    TM = TransferOperator(M)
    eL, VL = transop_dominant_eigs(TM, 'left')
    eR, VR = transop_dominant_eigs(TM, 'right')
    print(abs(eL - eR))
    # nrm = (eL + eR)/2


if __name__ == '__main__':
    main()
