from typing import Dict, Any, Tuple, Union

IdxType = Union[Tuple[Any], Any]

class SparseTensor:
    def __init__(self, idx_to_value: Dict[IdxType, Any] = None, rank: int = None):
        self._rank = rank
        self._indices = tuple()
        self._idx_to_value = idx_to_value or dict()

        for idxs in self._idx_to_value:
            idxs = idxs if isinstance(idxs, tuple) else tuple(idxs)
            if self._rank:
                if len(idxs) != self._rank:
                    raise ValueError(f"{idxs} is not of length {self._rank}")
            else:
                self._init_rank_indices(idxs)

            for i, idx in enumerate(idxs):
                self._indices[i].add(idx)

    def _init_rank_indices(self, idxs: IdxType):
        self._rank = len(idxs)
        self._indices = tuple(set() for _ in range(self._rank))

    def __getitem__(self, idxs):
        pass

    def __setitem__(self, idxs, value):
        pass

    @property
    def rank(self):
        return self._rank

    @property
    def indices(self) -> Tuple:
        return self._indices

    @property
    def nnz(self) -> int:
        """
        number of non-zero entries
        """
        return len(self._idx_to_value)
