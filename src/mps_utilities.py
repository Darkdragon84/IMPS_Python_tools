from typing import Tuple, Type

from src.math_utilities import qr_pos
from src.mps import MpsType
from src.utilities import Direction, MatType, DirectionError


def mps_qr(mps: MpsType, direction: Direction) -> Tuple[MpsType, MatType]:
    mps_type: Type[MpsType] = mps.__class__

    full_mat = mps.to_full_matrix(direction.value)
    if direction == Direction.LEFT:
        q, r = qr_pos(full_mat)
    elif direction == Direction.RIGHT:
        q, r = qr_pos(full_mat.T)
        q = q.T
    else:
        raise DirectionError()
    mps_q = mps_type.from_full_matrix(q, mps.dim_phys, direction.value)
    return mps_q, r
