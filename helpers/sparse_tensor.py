from collections import defaultdict
from typing import Dict, Any, Tuple, Union, Hashable

IdxType = Union[Tuple[Hashable], Hashable]


class SparseTensor:
    def __init__(self, idx_to_value: Dict[IdxType, Any] = None, rank: int = None) -> None:
        self._rank = rank
        self._init_indices_from_rank()
        self._idx_to_value = dict()
        self.update(idx_to_value)

    def _init_indices_from_rank(self) -> None:
        self._indices = tuple(defaultdict(set) for _ in range(self._rank)) if self._rank else None

    def _check_set_rank(self, idxs: IdxType) -> None:
        if self._rank:
            if len(idxs) != self._rank:
                raise ValueError(f"{idxs} is not of length {self._rank}")
        else:
            self._rank = len(idxs)
            self._init_indices_from_rank()

    def _check_add_indices(self, idxs: IdxType) -> None:
        idxs = (idxs,) if self._rank == 1 or not isinstance(idxs, tuple) else idxs
        self._check_set_rank(idxs)

        for i, idx in enumerate(idxs):
            self._indices[i][idx].add(idxs)

    def __getitem__(self, idxs: IdxType) -> Any:
        return self._idx_to_value[idxs]

    def __setitem__(self, idxs, value) -> None:
        self._check_add_indices(idxs)
        self._idx_to_value[idxs] = value

    def update(self, idx_to_value: Dict[IdxType, Any]) -> None:
        for idxs in idx_to_value:
            self._check_add_indices(idxs)

        self._idx_to_value.update(idx_to_value)

    @property
    def rank(self) -> int:
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
