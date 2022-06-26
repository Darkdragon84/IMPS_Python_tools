import enum
from typing import TypeVar, Tuple, Optional

import numpy as np
from numpy.typing import DTypeLike

VT = TypeVar("VT", bound=np.floating)
DimsType = Tuple[int, ...]
MatType = np.ndarray[Tuple[int, int], VT]


class Direction(enum.Enum):
    LEFT = 0
    RIGHT = 1


class DirectionError(Exception):
    def __init__(self, *args):
        args = (f"direction not recognized, must be one of {tuple(Direction)}",) + args
        super().__init__(*args)


class Which(enum.Enum):
    # possible values for the `which` flag for scipy.sparse.linalg.eigs
    LM = "LM"
    SM = "SM"
    LR = "LR"
    SR = "SR"
    LI = "LI"
    SI = "SI"


def dtype_precision(dt: DTypeLike) -> Optional[float]:
    if issubclass(dt, np.floating):
        return float(f"1e-{2*dt().itemsize}")
    if issubclass(dt, np.complexfloating):
        return float(f"1e-{dt().itemsize}")
    return None
