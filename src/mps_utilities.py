from typing import Tuple, Type, Optional
import warnings
import numpy as np

from src.math_utilities import qr_pos, inf_norm
from src.mps import MpsType, MPSMat
from src.transfer_operator import TransferOperator, transfer_op_dominant_eigs
from src.utilities import Direction, MatType, DirectionError, dtype_precision, DimsType


def mps_qr(mps: MpsType, direction: Direction) -> Tuple[MpsType, MatType]:
    mps_type: Type[MpsType] = mps.__class__

    full_mat = mps.to_full_matrix(direction.value)
    q, r = qr_pos(full_mat, direction)

    mps_q = mps_type.from_full_matrix(q, mps.dim_phys, direction.value)
    return mps_q, r


def is_gauged(mps: MpsType, reduced_dm: MatType, direction: Direction):
    tm = TransferOperator(mps)
    one = np.eye(*mps.dims[:2])
    tol = 50 * dtype_precision(mps.dtype)
    if direction == Direction.LEFT:
        one_mult = tm.mult_left
        dm_mult = tm.mult_right
    elif direction == Direction.RIGHT:
        one_mult = tm.mult_right
        dm_mult = tm.mult_left
    else:
        raise DirectionError()
    return inf_norm(one_mult() - one) < tol and inf_norm(dm_mult(reduced_dm) - reduced_dm) < tol


def random_left_right_ortho_mps_pair(dim_phys: int, dims: DimsType, dtype: Optional[np.dtype] = None,
                                     seed: Optional[int] = None):
    mps_l = MPSMat.random_left_ortho_mps(dim_phys, dims, dtype, seed)
    _, r = transfer_op_dominant_eigs(TransferOperator(mps_l), Direction.RIGHT)

    # decompose R = C*C.T, which is achieved exactly by a cholesky decomposition
    c = np.linalg.cholesky(r)
    mps_r, _ = mps_qr(mps_l @ c, Direction.RIGHT)
    if not is_gauged(mps_l, r, Direction.LEFT):
        warnings.warn("created random IMPS is not left orthogonal")
    if not is_gauged(mps_r, c.T@c, Direction.RIGHT):
        warnings.warn("created random IMPS is not right orthogonal")
    return mps_l, mps_r, c

