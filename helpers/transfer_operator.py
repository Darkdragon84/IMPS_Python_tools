from scipy.sparse.linalg import eigs, LinearOperator


class TransferOperator(object):
    def __init__(self, A, B=None):
        self._A = A
        self._B = B or A
        assert len(self._A) == len(self._B)
        self._shape = (self._A.shape[0] * self._B.shape[0], self._A.shape[1] * self._B.shape[1])

    @property
    def A(self):
        return self._A

    @property
    def B(self):
        return self._B

    @property
    def shape(self):
        return self._shape

    @property
    def argshapes(self):
        return (self._B.shape[0], self._A.shape[0]), (self._A.shape[1], self._B.shape[1])

    def mult_left(self, x=None):
        m, n = self._B.shape[0], self._A.shape[0]

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
        m, n = self._A.shape[1], self._B.shape[1]
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


def eigs_transop(transop, dir, nev, which='LR', tol=0, v0=None, maxiter=None, ncv=None):

    dim = transop.shape[0]
    assert dim == transop.shape[1]

    if dir == 'left':
        m, n = transop.argshapes[0]

        def matvec(xv):
            x = xv.reshape(m, n)
            y = transop.mult_left(x)
            return y.ravel()
    elif dir == 'right':
        m, n = transop.argshapes[1]

        def matvec(xv):
            x = xv.reshape(m, n)
            y = transop.mult_right(x)
            return y.ravel()

    # TODO can we fix the argument complaints?
    Ax = LinearOperator(shape=(dim, dim), matvec=matvec)
    E, V = eigs(Ax, nev, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)

    Vm = [V[:, i].reshape(m, n) for i in range(nev)]
    return E, Vm



