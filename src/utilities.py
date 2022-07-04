import enum
import numbers
from typing import TypeVar, Tuple, Optional

import numpy as np

VT = TypeVar("VT", bound=np.floating)
OT = TypeVar("OT")

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


def dtype_precision(dt: np.dtype) -> Optional[float]:
    if issubclass(dt.type, numbers.Real):
        return float(f"1e-{2*dt.itemsize}")
    if issubclass(dt.type, numbers.Complex):
        return float(f"1e-{dt.itemsize}")
    return None


def index_to_tuple(index: int, base: int, n_digits: int) -> Tuple[int, ...]:
    repr = f"{np.base_repr(index, base):>0{n_digits}}" if base > 2 else np.binary_repr(index, n_digits)
    if len(repr) > n_digits:
        raise ValueError(f"integer {index} cannot be represented by a length-{n_digits} base-{base} tuple.")
    return tuple(int(x) for x in repr)


def tuple_to_index(indices: Tuple[int, ...], base: int):
    return int("".join(map(str, indices)), base)


def commutator(o1: OT, o2: OT) -> OT:
    return o1 @ o2 - o2 @ o1
