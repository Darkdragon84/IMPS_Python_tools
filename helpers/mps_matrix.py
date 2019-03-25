import numpy


class MPSMatrix(object):

    # as per https://stackoverflow.com/questions/40252765/overriding-other-rmul-with-your-classs-mul
    __array_priority__ = 10000

    def __init__(self, matrices):
        assert isinstance(matrices, (list, tuple))

        self._d = len(matrices)
        self._dims = matrices[0].shape
        self._dtype = matrices[0].dtype

        for i in range(1, self._d):
            assert matrices[i].shape == self._dims
            assert matrices[i].dtype == self._dtype

        self._matrices = matrices

    @property
    def d(self):
        return self._d

    def __getitem__(self, item):
        return self._matrices[item]

    def __setitem__(self, index, matrix):
        self._matrices[index] = matrix

    def __iter__(self):
        return iter(self._matrices)

    @property
    def dims(self):
        return self._dims

    @property
    def dtype(self):
        return self._dtype

    def to_full_matrix(self, axis):
        return numpy.stack(self._matrices, axis)

    @classmethod
    def from_full_matrix(cls, mat, d, axis):
        if mat.shape[axis] % d != 0:
            raise ValueError("shape[{}] (={}) of mat is not divisible by {}".format(axis, mat.shape[axis], d))

        stride = mat.shape[axis] // d
        split_inds = range(stride, d*stride, stride)
        return cls(numpy.split(mat, split_inds, axis))

    @classmethod
    def get_random_mps(cls, d, m, n=None, dtype=numpy.float64):
        n = n or m
        return cls([numpy.random.randn(m, n).astype(dtype) for _ in range(d)])

    @classmethod
    def get_random_left_ortho_mps(cls, d, m, n=None, dtype=numpy.float64):
        n = n or m
        q, _ = numpy.linalg.qr(numpy.random.randn(d*m, n).astype(dtype))
        return cls.from_full_matrix(q, d, 0)

    @classmethod
    def get_random_right_ortho_mps(cls, d, m, n=None, dtype=numpy.float64):
        n = n or m
        q, _ = numpy.linalg.qr(numpy.random.randn(d*n, m).astype(dtype))
        return cls.from_full_matrix(q.T, d, 1)

    def __mul__(self, scalar):
        return self.mult_with_scalar(scalar)

    def __rmul__(self, scalar):
        return self.mult_with_scalar(scalar)

    def __truediv__(self, scalar):
        return self.mult_with_scalar(1./scalar)

    def __matmul__(self, other):
        if isinstance(other, numpy.ndarray):
            return self.mult_right_with_matrix(other)
        else:
            raise NotImplementedError("unsupported operand type(s) for *: {} and {}".format(other.__class__.__name__,
                                                                                            self.__class__.__name__))

    def __rmatmul__(self, other):
        if isinstance(other, numpy.ndarray):
            return self.mult_left_with_matrix(other)
        else:
            raise NotImplementedError("unsupported operand type(s) for *: {} and {}".format(self.__class__.__name__,
                                                                                            other.__class__.__name__))

    def mult_with_scalar(self, scalar):
        assert numpy.isscalar(scalar)
        return self.__class__([mat * scalar for mat in self._matrices])

    def mult_right_with_matrix(self, x=None):
        if x is None:
            return self

        if x.shape[0] != self.dims[1]:
            raise ValueError("x has wrong shape[0] (={}, should be {})".format(x.shape[0], self.dims[1]))

        return self.__class__([mat @ x for mat in self._matrices])

    def mult_left_with_matrix(self, x=None):
        if x is None:
            return self

        if x.shape[1] != self.dims[0]:
            raise ValueError("x has wrong shape[0] (={}, should be {})".format(x.shape[1], self.dims[0]))

        return self.__class__([x @ mat for mat in self._matrices])


