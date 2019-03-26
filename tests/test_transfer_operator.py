import numpy

from helpers.mps_matrix import MPSMatrix
from helpers.transfer_operator import TransferOperator, transop_dominant_eigs


def main():
    d = 3
    mA = 400
    mB = 80
    # dtype = numpy.complex128
    # dtype = numpy.float64
    dtype = numpy.float32

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
    print()
    print(eA, eA.dtype, rhoA.dtype)
    print(eB, eB.dtype, rhoB.dtype)
    print(numpy.max(numpy.abs(TMA.mult_right(rhoA) - eA*rhoA)))
    print(numpy.max(numpy.abs(TMB.mult_left(rhoB) - eB*rhoB)))

    M = MPSMatrix.get_random_mps(d, mA, dtype=dtype)
    TM = TransferOperator(M)
    eL, VL = transop_dominant_eigs(TM, 'left')
    eR, VR = transop_dominant_eigs(TM, 'right')
    print()
    print(eL, eL.dtype, VL.dtype)
    print(eR, eR.dtype, VR.dtype)
    print(abs(eL - eR))
    nrm = (eL + eR)/2

    M = M/numpy.sqrt(nrm)
    TM = TransferOperator(M)
    print()
    print(numpy.max(numpy.abs(VL - TM.mult_left(VL))))
    print(numpy.max(numpy.abs(VR - TM.mult_right(VR))))

    print(numpy.max(numpy.abs(VR - TransferOperator(M @ VR, M).mult_right())))
    print(numpy.max(numpy.abs(VL - TransferOperator(VL @ M, M).mult_left())))



if __name__ == '__main__':
    main()
