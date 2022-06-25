import enum
from typing import TypeVar, Tuple

import numpy as np

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
