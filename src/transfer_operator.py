import enum
import warnings
from functools import cached_property
from typing import Optional, Tuple

import numpy as np
from numpy._typing import DTypeLike
from scipy.sparse.linalg import eigs, LinearOperator, gmres

from src.math_utilities import inf_norm, matrix_dot, add_scalar_times_matrix
from src.mps import MpsType, MatType

DIR_TO_AXIS = {'left': 0,
               'right': 1}


class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1


class Which(enum.Enum):
    # possible values for the `which` flag for scipy.sparse.linalg.eigs
    LM = "LM"
    SM = "SM"
    LR = "LR"
    SR = "SR"
    LI = "LI"
    SI = "SI"


class TransferOperator:
    def __init__(self, mps_bottom: MpsType, mps_top: Optional[MpsType] = None):
        self._mps_bottom = mps_bottom
        self._mps_top = mps_top or mps_bottom
        assert self._mps_bottom.dim_phys == self._mps_top.dim_phys
        self._dims = (self._mps_bottom.dims[0] * self._mps_top.dims[0],
                      self._mps_bottom.dims[1] * self._mps_top.dims[1])
        self._dtype = np.dtype(np.result_type(self._mps_bottom.dtype, self._mps_top.dtype))

    @property
    def mps_bottom(self) -> MpsType:
        return self._mps_bottom

    @property
    def mps_top(self) -> MpsType:
        return self._mps_top

    @cached_property
    def is_mixed(self) -> bool:
        return self.mps_top != self.mps_bottom

    @property
    def is_square(self):
        return self.dims[0] == self.dims[1]

    @property
    def dtype(self) -> DTypeLike:
        return self._dtype

    @property
    def dims(self) -> Tuple[int, ...]:
        return self._dims

    @property
    def argdims(self) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        return (
            (self._mps_top.dims[0], self._mps_bottom.dims[0]),
            (self._mps_bottom.dims[1], self._mps_top.dims[1])
        )

    def mult_left(self, x: Optional[MatType] = None) -> MatType:
        m, n = self.argdims[0]

        if x is None:
            assert m == n
            y = np.zeros((m, n), dtype=self.dtype)
            for a, b in zip(self._mps_bottom, self._mps_top):
                y += b.T @ a
        else:
            assert x.shape == (m, n)
            y = np.zeros((m, n), dtype=np.result_type(self.dtype, x.dtype))
            for a, b in zip(self._mps_bottom, self._mps_top):
                y += (b.T @ x) @ a

        return y

    def mult_right(self, x: Optional[MatType] = None) -> MatType:
        m, n = self.argdims[1]

        if x is None:
            assert m == n
            y = np.zeros((m, n), dtype=self.dtype)
            for a, b in zip(self._mps_bottom, self._mps_top):
                y += a @ b.T
        else:
            assert x.shape == (m, n)
            y = np.zeros((m, n), dtype=np.result_type(self.dtype, x.dtype))
            for a, b in zip(self._mps_bottom, self._mps_top):
                y += (a @ x) @ b.T

        return y


def transop_dominant_eigs(
    transfer_op: TransferOperator,
    direction: Direction,
    tol: float = 0,
    which: Which = Which.LR,
    v0: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None
) -> Tuple[np.floating, np.ndarray]:
    """
    this function is only meant for non-mixed TM, for which the dominant eigenvalue is guaranteed to be positive real
    :param transfer_op:
    :param direction:
    :param tol:
    :param which:
    :param v0:
    :param maxiter:
    :param ncv:
    :return:
    """
    if transfer_op.is_mixed:
        raise ValueError("`transfer_op` must not be mixed for dominant eigs.")

    E, Vm = transop_eigs(transfer_op, direction, 1, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)

    V = Vm[0]
    E: np.floating = np.real(E[0])
    # make hermitian
    # due to TM = \sum_i A[i]* \otimes A[i], both V and V' are eigenmatrices (check by transposing EV equation)
    # this should also remove the arbitrary complex phase, as the diagonal is then real by definition
    V += V.conj().T

    # remove potential remaining sign of eigenmatrix (should already be zero from hermitization) and normalize
    V /= V.trace()

    # we want the result to be contiguous in memory, so we copy once here
    V = np.real(V).copy()
    return E, V


def transop_eigs(
    transfer_op: TransferOperator,
    direction: Direction,
    nev: int,
    which: Which = Which.LR,
    tol: float = 0,
    v0: Optional[np.ndarray] = None,
    maxiter: Optional[int] = None,
    ncv: Optional[int] = None,
    sort: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    if not transfer_op.is_square:
        raise ValueError(f"`transfer_op` needs to be square, i.e. dims[0] == dims[1]")
    dim = transfer_op.dims[0]

    m, n = transfer_op.argdims[direction.value]
    if direction == Direction.LEFT:
        func = transfer_op.mult_left
    elif direction == Direction.RIGHT:
        func = transfer_op.mult_right
    else:
        raise ValueError(f"direction '{direction}' not recognized, must be one of {tuple(Direction)}")

    def matvec(xv):
        x = xv.reshape(m, n)
        y = func(x)
        return y.ravel()

    TMop = LinearOperator(shape=(dim, dim), matvec=matvec, dtype=transfer_op.dtype)  # type: ignore
    E, Vm = eigs(TMop, nev, which=which.value, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)  # type: ignore

    # V is in fortran order (F_CONTIGUOUS, column major),
    # so V.T uses same memory, but is in C order (C_CONTIGUOUS, row major)
    V = Vm.T.reshape((-1, m, n))
    if sort:
        inds = np.argsort(np.abs(E))[::-1]
        E = E[inds]
        V = V[inds]
    return E, V


def transop_geometric_sum(
    x: MatType,
    transfer_op: TransferOperator,
    direction: Direction,
    L: Optional[MatType] = None,
    R: Optional[MatType] = None,
    reltol: float = 1e-6,
    x0: Optional[np.ndarray] = None,
    maxiter: int = None,
    chk: bool = False
) -> MatType:
    if not transfer_op.is_square:
        raise ValueError(f"`transfer_op` needs to be square, i.e. dims[0] == dims[1]")
    dim = transfer_op.dims[0]
    m, n = transfer_op.argdims[direction.value]

    if x.shape != (m, n):
        raise ValueError(f"inhomogeneity has dimensions {x.shape}, but should be {(m, n)}.")

    assert dim == m * n  # sanity check for dimensions

    if chk:
        ltmp = transfer_op.mult_left(L)
        rtmp = transfer_op.mult_right(R)
        ltmp -= np.eye(*transfer_op.argdims[0]) if L is None else L
        rtmp -= np.eye(*transfer_op.argdims[1]) if R is None else R

        lchk = inf_norm(ltmp)
        rchk = inf_norm(rtmp)
        if lchk > reltol * (1 if L is None else inf_norm(L)):
            warnings.warn(f"L is not a good left dominant eigenmatrix to reltol={reltol:2.4} (lchk={lchk:2.4})")
        if rchk > reltol * (1 if R is None else inf_norm(R)):
            warnings.warn(f"R is not a good right dominant eigenmatrix to reltol={reltol:2.4} (rchk={rchk:2.4})")
    # properly normalize L and R
    lrnrm = matrix_dot(L, R)
    if L is not None:
        L /= lrnrm
    else:
        R /= lrnrm

    if x0 is None:
        # TODO how to generate complex x0
        x0 = np.random.randn(m, n).astype(transfer_op.dtype)

    if direction == Direction.LEFT:
        # project out dominante eigenspace
        # x -= trace(x*R)*L
        # one iteration:
        # y = x - [Tm(x) - tr(x*R)*L] = x - Tm(x) + tr(x*R)*L

        mult_fun = transfer_op.mult_left
        proj_out, mult_with = R, L
    elif direction == Direction.RIGHT:
        # project out dominante eigenspace
        # x -= trace(L*x)*R
        # one iteration:
        # y = x - [Tm(x) - tr(L*x)*R] = x - Tm(x) + tr(L*x)*R

        mult_fun = transfer_op.mult_right
        proj_out, mult_with = L, R
    else:
        raise ValueError(f"direction '{direction}' not recognized, must be one of {tuple(Direction)}")

    add_scalar_times_matrix(x, mult_with, -matrix_dot(proj_out, x))
    add_scalar_times_matrix(x0, mult_with, -matrix_dot(proj_out, x0))

    def matvec(xv):
        xm = xv.reshape(m, n)
        Txm = mult_fun(xm)
        add_scalar_times_matrix(xm, mult_with, -matrix_dot(proj_out, xm))
        ym = xm - Txm
        return ym.ravel()

    TMop = LinearOperator(shape=(dim, dim), matvec=matvec, dtype=transfer_op.dtype)  # type: ignore
    yv, info = gmres(TMop, x.ravel(), x0=x0.ravel(), tol=reltol, maxiter=maxiter)
    if info > 0:
        warnings.warn(f"gmres: convergence to reltol={reltol} not achieved")

    y = yv.reshape(m, n)
    return y
