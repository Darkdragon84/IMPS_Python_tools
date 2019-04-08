from collections import defaultdict
from itertools import product

import numpy


class BlockTensor(object):
    def __init__(self, rank):
        self._rank = rank
        self._dims = tuple(dict() for _ in range(rank))
        self._tensors = dict()
        self._qn_to_tensors = tuple(defaultdict(dict) for _ in range(rank))

    @property
    def rank(self):
        return self._rank

    @property
    def dims(self):
        return self._dims

    @property
    def total_dims(self):
        return tuple(sum(dims.values()) for dims in self._dims)

    @property
    def qn_to_tensors(self):
        return self._qn_to_tensors

    @property
    def tensors(self):
        return self._tensors

    def __contains__(self, qns):
        return qns in self._tensors

    def __getitem__(self, qns):
        return self._tensors[qns]

    def __setitem__(self, qns, tensor):
        assert isinstance(qns, tuple)
        assert len(tensor.shape) == len(qns)

        if qns in self._tensors:
            assert self._tensors[qns].shape == tensor.shape
        else:
            for ax, (dim, qn) in enumerate(zip(tensor.shape, qns)):
                old_dim = self._dims[ax].get(qn, None)
                if old_dim is not None:
                    assert dim == old_dim
                else:
                    self._dims[ax][qn] = dim

        self._tensors[qns] = tensor

        for ax, qn in enumerate(qns):
            self._qn_to_tensors[ax][qn][qns] = tensor.view()

    @classmethod
    def from_tensordict(cls, qns_to_tensor, rank=None):
        rank = rank or len(next(iter(qns_to_tensor.keys())))

        block_tensor = cls(rank)
        for qn, tensor in qns_to_tensor.items():
            block_tensor[qn] = tensor

        return block_tensor

    def tensordot(self, other, ax, ax_other):

        rank = self.rank + other.rank - 2
        result = self.__class__(rank)

        # get all qn to tensor mapping from both BlockTensors for their respective axes
        tensors = self.qn_to_tensors[ax]
        tensors_other = other.qn_to_tensors[ax_other]

        # determine common central qn which can be contracted over
        qns_contract = tensors.keys() & tensors_other.keys()

        for qn_contract in qns_contract:
            for (qn, tensor), (qn_other, tensor_other) in product(tensors[qn_contract].items(),
                                                                  tensors_other[qn_contract].items()):
                new_qn = qn[:ax] + qn[ax+1:] + qn_other[:ax_other] + qn_other[ax_other+1:]
                res = numpy.tensordot(tensor, tensor_other, [ax, ax_other])
                if new_qn in result:
                    result[new_qn] += res
                else:
                    result[new_qn] = res

        return result
