import warnings

from scipy.sparse.linalg import eigs, LinearOperator, gmres
import numpy

from helpers.math_utilities import inf_norm, matrix_dot,add_scalar_times_matrix

DIR_TO_AXIS = {'left': 0,
               'right': 1}


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


def transop_dominant_eigs(transop, direction, tol=0, which='LR', v0=None, maxiter=None, ncv=None):
    """
    this function is only meant for non-mixed TM, for which the dominant eigenvalue is guaranteed to be positive real
    :param transop:
    :param direction:
    :param tol:
    :param which:
    :param v0:
    :param maxiter:
    :param ncv:
    :return:
    """
    assert transop.A == transop.B
    E, Vm = transop_eigs(transop, direction, 1, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)

    V = Vm[0]
    # make hermitian
    # due to TM = \sum_i A[i]* \otimes A[i], both V and V' are eigenmatrices (check by transposing EV equation)
    V += V.conj().T

    # # remove arbitrary complex phase of eigenmatrix (should already be zero from hermitization) and normalize
    # V /= V.trace()

    # we want the result to be contiguous in memory, so we copy once here
    V = numpy.real(V).copy()
    return numpy.real(E[0]), V


def transop_eigs(transop, direction, nev, which='LR', tol=0, v0=None, maxiter=None, ncv=None, sorted=False):

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
    E, Vm = eigs(TMop, nev, which=which, tol=tol, v0=v0, maxiter=maxiter, ncv=ncv)

    # V is in fortran order (F_CONTIGUOUS, column major),
    # so V.T uses same memory, but is in C order (C_CONTIGUOUS, row major)
    V = Vm.T.reshape((-1, m, n))
    if sorted:
        inds = numpy.argsort(numpy.abs(E))[::-1]
        E = E[inds]
        V = V[inds]
    return E, V


def transop_geometric_sum(x, transop, direction, L=None, R=None, reltol=1e-6, x0=None, maxiter=None,
                          verbose=False, chk=False):
    axis = DIR_TO_AXIS[direction] if isinstance(direction, str) else direction

    dim = transop.dim[0]
    m, n = transop.argdims[axis]

    assert x.shape == (m, n)  # inhomogeneity must have correct dimensions
    assert transop.dim[1] == dim  # transop must be square
    assert dim == m*n  # sanity check for dimensions

    if chk:
        ltmp = transop.mult_left(L)
        rtmp = transop.mult_right(R)
        ltmp -= numpy.eye(transop.argdims[0]) if L is None else L
        rtmp -= numpy.eye(transop.argdims[1]) if R is None else R

        lchk = inf_norm(ltmp)
        rchk = inf_norm(rtmp)
        if lchk > reltol * (1 if L is None else inf_norm(L)):
            warnings.warn("L is not a good left dominant eigenmatrix to reltol={:2.4} (lchk={:2.4})".format(reltol,
                                                                                                            lchk))
        if rchk > reltol * (1 if R is None else inf_norm(R)):
            warnings.warn("R is not a good right dominant eigenmatrix to reltol={:2.4} (rchk={:2.4})".format(reltol,
                                                                                                             rchk))
    # properly normalize L and R
    lrnrm = matrix_dot(L, R)
    if L is not None:
        L /= lrnrm
    else:
        R /= lrnrm

    if x0 is None:
        # TODO how to generate complex x0
        x0 = numpy.random.randn(m, n).astype(transop.dtype)

    if direction == 'left':
        # project out dominante eigenspace
        # x -= trace(x*R)*L
        add_scalar_times_matrix(x, L, -matrix_dot(x, R))
        add_scalar_times_matrix(x0, L, -matrix_dot(x0, R))

        def matvec(xv):
            xm = xv.reshape(m, n)
            Txm = transop.mult_left(xm)
            add_scalar_times_matrix(xm, L, matrix_dot(xm, R))
            # y = x - [Tm(x) - tr(x*R)*L] = x - Tm(x) + tr(x*R)*L
            ym = xm - Txm
            return ym.ravel()

    elif direction == 'right':
        # project out dominante eigenspace
        # x -= trace(L*x)*R
        add_scalar_times_matrix(x, R, -matrix_dot(L, x))
        add_scalar_times_matrix(x0, R, -matrix_dot(L, x0))

        def matvec(xv):
            xm = xv.reshape(m, n)
            Txm = transop.mult_right(xm)
            add_scalar_times_matrix(xm, R, matrix_dot(L, xm))
            # y = x - [Tm(x) - tr(L*x)*R] = x - Tm(x) + tr(L*x)*R
            ym = xm - Txm
            return ym.ravel()
    else:
        raise ValueError("direction {} not recognized, must be one of ['left', 'right']")

    TMop = LinearOperator(shape=(dim, dim), matvec=matvec, dtype=transop.dtype)
    yv, info = gmres(TMop, x.ravel(), x0=x0.ravel(), tol=reltol, maxiter=maxiter)






