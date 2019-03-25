import numpy


def apply_transop_left(A, B=None, x=None):
    B = B or A
    assert len(A) == len(B)
    m, n = A.shape[0], B.shape[0]

    y = 0
    if x is None:
        assert m == n
        for a, b in zip(A, B):
            y += b.T @ a
    else:
        assert x.shape == (m, n)
        for a, b in zip(A, B):
            y += (b.T @ x) @ a

    return y


def apply_transop_right(A, B=None, x=None):
    B = B or A
    assert len(A) == len(B)
    m, n = A.shape[1], B.shape[1]

    y = 0
    if x is None:
        assert m == n
        for a, b in zip(A, B):
            y += a @ b.T
    else:
        assert x.shape == (m, n)
        for a, b in zip(A, B):
            y += (a @ x) @ b.T

    return y

