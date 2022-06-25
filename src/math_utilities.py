import numpy


def inf_norm(X):
    """
    slightly altered infinity norm, as the maximum absolute value entry of the matrix
    :param X:   ndarray
    :return:
    """
    return numpy.max(numpy.abs(X))


def matrix_dot(X, Y):
    """
    This version of matrix dot product does *not* complex conjugate X!
    None values are interpreted as identity matrices:
    e.g. matrix_dot(None, Y) = trace(Id*Y) = trace(Y)

    :param X:   ndarray
    :param Y:   ndarray
    :return:    scalar product between matrices \sum_ij X_ij * Y_ij
    """
    if X is None:
        return Y.trace()
    if Y is None:
        return X.trace()
    assert X.shape == Y.shape
    return numpy.dot(X.ravel(), Y.ravel())


def add_scalar_times_matrix(X, Y, a):
    """
    adds scalar `a` time matrix `Y` to matrix `X`:
    X += a*Y
    If Y is none, it is interpreted as identity, so `a` is added to the diagonal elements of a `X`.
    This is equivalent to `X += a*eye(X.shape)`.
    This function modifies `X` and returns nothing!

    :param X:  ndarray
    :param Y:  ndarray
    :param a:  scalar
    :return:
    """
    if Y is None:
        numpy.fill_diagonal(X, X.diagonal() + a)
    else:
        X += a*Y
