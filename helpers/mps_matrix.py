import numpy


class MPSMatrix(list):
    def __init__(self, matrices):
        if not isinstance(matrices, (list, tuple)):
            raise ValueError("matrices must be list or tuple")

        self._d = len(matrices)
        self._shape = matrices[0].shape
        for i in range(1, self._d):
            if matrices[i].shape != self._shape:
                raise ValueError("matrix {} has wrong shape".format(i))

        super().__init__(matrices)

    @property
    def shape(self):
        return self._shape

    def to_full_matrix(self, axis):
        return numpy.stack(self, axis)

    @classmethod
    def from_full_matrix(cls, mat, d, axis):
        if mat.shape[axis] % d != 0:
            raise ValueError("shape[{}] (={}) of mat is not divisible by {}".format(axis, mat.shape[axis], d))

        stride = mat.shape[axis] // d
        split_inds = range(stride, d*stride, stride)
        return cls(numpy.split(mat, split_inds, axis))

    @classmethod
    def get_random_mps(cls, d, m, n=None):
        n = n or m
        return cls([numpy.random.randn(m, n) for _ in range(d)])

    @classmethod
    def get_random_left_ortho_mps(cls, d, m, n=None):
        n = n or m
        q, _ = numpy.linalg.qr(numpy.random.randn(d*m, n))
        return cls.from_full_matrix(q, d, 0)

    @classmethod
    def get_random_right_ortho_mps(cls, d, m, n=None):
        n = n or m
        q, _ = numpy.linalg.qr(numpy.random.randn(d*n, m))
        return cls.from_full_matrix(q.T, d, 1)

    def mult_right_with_matrix(self, x=None):
        if x is None:
            return self

        if x.shape[0] != self.shape[1]:
            raise ValueError("x has wrong shape[0] (={}, should be {})".format(x.shape[0], self.shape[1]))

        return self.__class__([mat @ x for mat in self])

    def mult_left_with_matrix(self, x=None):
        if x is None:
            return self

        if x.shape[1] != self.shape[0]:
            raise ValueError("x has wrong shape[0] (={}, should be {})".format(x.shape[1], self.shape[0]))

        return self.__class__([x @ mat for mat in self])


