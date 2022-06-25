import numpy

from src.math_utilities import inf_norm
from src.mps import IMPS
from src.transfer_operator import TransferOperator, transop_dominant_eigs, transop_eigs


def main():
    d = 3
    mA = 80
    mB = 80
    # dtype = numpy.complex128
    # dtype = numpy.float64
    dtype = numpy.float32

    A = IMPS.get_random_left_ortho_mps(d, mA, dtype=dtype)
    B = IMPS.get_random_right_ortho_mps(d, mB, dtype=dtype)

    TMA = TransferOperator(A)
    TMB = TransferOperator(B)

    numpy.set_printoptions(precision=5, suppress=True)
    y1 = TMA.mult_left()
    y2 = TMB.mult_right()
    print(inf_norm(y1 - numpy.eye(mA)))
    print(inf_norm(y2 - numpy.eye(mB)))

    eA, RA = transop_dominant_eigs(TMA, 'right')
    eB, LB = transop_dominant_eigs(TMB, 'left')
    print()
    print("AL and BR separately")
    print(eA, eA.dtype, RA.dtype)
    print(eB, eB.dtype, LB.dtype)
    print(inf_norm(TMA.mult_right(RA) - eA*RA))
    print(inf_norm(TMB.mult_left(LB) - eB*LB))

    M = IMPS.get_random_mps(d, mA, dtype=dtype)
    TM = TransferOperator(M)
    eL, LM = transop_dominant_eigs(TM, 'left')
    eR, RM = transop_dominant_eigs(TM, 'right')
    print()
    print("general unnormalized M")
    print(eL, eL.dtype, LM.dtype)
    print(eR, eR.dtype, RM.dtype)
    print(abs(eL - eR))
    nrm = (eL + eR)/2

    M = M/numpy.sqrt(nrm)
    TM = TransferOperator(M)
    print()
    print("check after normalization")
    print(inf_norm(LM - TM.mult_left(LM)))
    print(inf_norm(RM - TM.mult_right(RM)))

    print(inf_norm(RM - TransferOperator(M @ RM, M).mult_right()))
    print(inf_norm(LM - TransferOperator(LM @ M, M).mult_left()))

    print()
    print("mixed AL BR")
    TMAB = TransferOperator(A, B)
    eABL, LAB = transop_eigs(TMAB, 'left', nev=4, which='LM', sort=True)
    eABR, RAB = transop_eigs(TMAB, 'right', nev=4, which='LM', sort=True)

    print("left:")
    for e, L in zip(eABL, LAB):
        chk = inf_norm(TMAB.mult_left(L) - e*L)
        print("|{}|={}: {}".format(e, numpy.abs(e), chk))
    print("right:")
    for e, R in zip(eABR, RAB):
        chk = inf_norm(TMAB.mult_right(R) - e*R)
        print("|{}|={}: {}".format(e, numpy.abs(e), chk))



if __name__ == '__main__':
    main()
