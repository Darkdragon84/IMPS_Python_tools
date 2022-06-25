from typing import Tuple, Optional

import numpy
import numpy as np
from numpy.linalg import qr

from src.utilities import MatType


def inf_norm(x) -> float:
    """
    slightly altered infinity norm, as the maximum absolute value entry of the matrix
    :param x:   ndarray
    :return:
    """
    return numpy.max(numpy.abs(x))


def matrix_dot(x: Optional[np.ndarray], y: Optional[np.ndarray]) -> np.ScalarType:
    """
    This version of matrix dot product does *not* complex conjugate x!
    None values are interpreted as identity matrices:
    e.g. matrix_dot(None, y) = trace(Id*y) = trace(y)

    :param x:   ndarray
    :param y:   ndarray
    :return:    scalar product between matrices \sum_ij x_ij * y_ij
    """
    if x is None:
        return y.trace()
    if y is None:
        return x.trace()
    assert x.shape == y.shape
    return numpy.dot(x.ravel(), y.ravel())


def add_scalar_times_matrix(x: np.ndarray, y: np.ndarray, a: np.ScalarType):
    """
    adds scalar `a` time matrix `Y` to matrix `X`:
    X += a*Y
    If Y is none, it is interpreted as identity, so `a` is added to the diagonal elements of a `X`.
    This is equivalent to `X += a*eye(X.shape)`.
    This function modifies `X` and returns nothing!

    :param x:  ndarray
    :param y:  ndarray
    :param a:  scalar
    :return:
    """
    if y is None:
        numpy.fill_diagonal(x, x.diagonal() + a)
    else:
        x += a * y


def qr_pos(x: MatType) -> Tuple[np.ndarray, np.ndarray]:
    """
    computes a unique QR factorization, where the diagonals of R are positive
    :param x:
    :return:
    """
    if np.iscomplexobj(x):
        raise NotImplementedError
    q, r = qr(x)
    # TODO make this work for complex numbers
    transf = np.diag(np.sign(np.diag(r)))
    return q @ transf, transf @ r
