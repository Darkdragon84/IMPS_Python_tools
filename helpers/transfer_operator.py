from scipy.sparse.linalg import eigs, LinearOperator
import numpy


class TransferOperator(object):
    def __init__(self, A, B=None):
        self._A = A
        self._B = B or A
        assert self._A.d == self._B.d
        self._dims = (self._A.dims[0] * self._B.dims[0], self._A.dims[1] * self._B.dims[1])
        self._dtype = numpy.result_type(self._A.dtype, self._B.dtype)

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def dtype(self):
        return self._dtype

    @property
    def dims(self):
        return self._dims

    @property
    def argdims(self):
        return (self._B.dims[0], self._A.dims[0]), (self._A.dims[1], self._B.dims[1])

    def mult_left(self, x=None):
        m, n = self._B.dims[0], self._A.dims[0]

        y = 0
        if x is None:
            assert m == n
            for a, b in zip(self._A, self._B):
                y += b.T @ a
        else:
            assert x.shape == (m, n)
            for a, b in zip(self._A, self._B):
                y += (b.T @ x) @ a

        return y

    def mult_right(self, x=None):
        m, n = self._A.dims[1], self._B.dims[1]
        y = 0
        if x is None:
            assert m == n
            for a, b in zip(self._A, self._B):
                y += a @ b.T
        else:
            assert x.shape == (m, n)
            for a, b in zip(self._A, self._B):
                y += (a @ x) @ b.T

        return y

    # def aslinearoperator(self, direction):
    #
    #     dim = self.shape[0]
    #     assert dim == self.shape[1]
    #
    #     if direction == 'left':
    #         m, n = self.argshapes[0]
    #
    #         def matvec(xv):
    #             x = xv.reshape(m, n)
    #             y = self.mult_left(x)
    #             return y.ravel()
    #     elif direction == 'right':
    #         m, n = self.argshapes[1]
    #
    #         def matvec(xv):
    #             x = xv.reshape(m, n)
    #             y = self.mult_right(x)
    #             return y.ravel()
    #     return LinearOperator(shape=(dim, dim), dtype=self.dtype, matvec=matvec)


def transop_dominant_eigs(transop, direction, tol=0, v0=None, maxiter=None, ncv=None):
    E, Vm = transop_eigs(transop, direction, 1, 'LR', tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)
    return numpy.real(E[0]), numpy.real(Vm[0])


def transop_eigs(transop, direction, nev, which='LR', tol=0, v0=None, maxiter=None, ncv=None):

    dim = transop.dims[0]
    assert dim == transop.dims[1]

    if direction == 'left':
        m, n = transop.argdims[0]

        def matvec(xv):
            x = xv.reshape(m, n)
            y = transop.mult_left(x)
            return y.ravel()
    elif direction == 'right':
        m, n = transop.argdims[1]

        def matvec(xv):
            x = xv.reshape(m, n)
            y = transop.mult_right(x)
            return y.ravel()
    else:
        raise ValueError("direction {} not recognized, must be one of ['left', 'right']")

    # the next two calls are fine, even though PyCharm complains
    TMop = LinearOperator(shape=(dim, dim), matvec=matvec, dtype=transop.dtype)
    E, V = eigs(TMop, nev, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)

    Vm = V.reshape((-1, m, n))
    return E, Vm



