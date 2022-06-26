import numpy

from src.math_utilities import inf_norm
from src.mps import IMPS
from src.transfer_operator import TransferOperator, transfer_op_dominant_eigs, transfer_op_eigs
from src.utilities import Direction, Which


def main():
    d = 2
    mA = 20
    mB = 20
    # dtype = numpy.complex128
    # dtype = numpy.float64
    dtype = numpy.float32

    A = IMPS.random_left_ortho_mps(d, (mA, mA), dtype=dtype)
    B = IMPS.random_right_ortho_mps(d, (mB, mB), dtype=dtype)

    TMA = TransferOperator(A)
    TMB = TransferOperator(B)

    numpy.set_printoptions(precision=5, suppress=True)
    y1 = TMA.mult_left()
    y2 = TMB.mult_right()
    print(f"AL left ortho: {inf_norm(y1 - numpy.eye(mA))}")
    print(f"BR right ortho: {inf_norm(y2 - numpy.eye(mB))}")

    eA, RA = transfer_op_dominant_eigs(TMA, Direction.RIGHT)
    eB, LB = transfer_op_dominant_eigs(TMB, Direction.LEFT)
    print()
    print("AL separately")
    print(f"dominant EV={eA}, EV.dtype={eA.dtype}, RA.dtype={RA.dtype}")
    print(f"RA right eigenmat: {inf_norm(TMA.mult_right(RA) - eA * RA)}")
    print("BR separately")
    print(f"dominant EV={eB}, EV.dtype={eB.dtype}, LB.dtype={LB.dtype}")
    print(f"LB left eigenmat: {inf_norm(TMB.mult_left(LB) - eB * LB)}")

    M = IMPS.random_mps(d, (mA, mA), dtype=dtype)
    TM = TransferOperator(M)
    eL, LM = transfer_op_dominant_eigs(TM, Direction.LEFT)
    eR, RM = transfer_op_dominant_eigs(TM, Direction.RIGHT)
    print()
    print("general unnormalized M")
    print(f"left EV={eL}, EV.dtype={eL.dtype}, LM.dtype={LM.dtype}")
    print(f"right EV={eR}, EV.dtype={eR.dtype}, RM.dtype={RM.dtype}")
    nrm = (eL + eR) / 2
    print(f"rel. difference: {abs(eL - eR) / nrm}")

    M = M / numpy.sqrt(nrm)
    TM = TransferOperator(M)
    print()
    print("check EV eqn after normalization")
    print(inf_norm(LM - TM.mult_left(LM)))
    print(inf_norm(RM - TM.mult_right(RM)))

    print(inf_norm(RM - TransferOperator(M @ RM, M).mult_right()))
    print(inf_norm(LM - TransferOperator(LM @ M, M).mult_left()))

    print()
    print("mixed AL BR")
    TMAB = TransferOperator(A, B)
    eABL, LAB = transfer_op_eigs(TMAB, Direction.LEFT, nev=4, which=Which.LM, sort=True)
    eABR, RAB = transfer_op_eigs(TMAB, Direction.RIGHT, nev=4, which=Which.LM, sort=True)

    print("left:")
    for e, L in zip(eABL, LAB):
        chk = inf_norm(TMAB.mult_left(L) - e * L)
        print("|{}|={}: {}".format(e, numpy.abs(e), chk))
    print("right:")
    for e, R in zip(eABR, RAB):
        chk = inf_norm(TMAB.mult_right(R) - e * R)
        print("|{}|={}: {}".format(e, numpy.abs(e), chk))


if __name__ == '__main__':
    main()
