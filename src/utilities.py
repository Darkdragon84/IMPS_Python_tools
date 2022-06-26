import enum
from typing import TypeVar, Tuple, Optional

import numpy as np
from numpy.typing import DTypeLike

T = TypeVar("T", bound=np.floating)
DimsType = Tuple[int, ...]
MatType = np.ndarray[Tuple[int, int], T]


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


def dtype_precision(dt: DTypeLike) -> Optional[float]:
    if issubclass(dt, np.floating):
        return float(f"1e-{2*dt().itemsize}")
    if issubclass(dt, np.complexfloating):
        return float(f"1e-{dt().itemsize}")
    return None
