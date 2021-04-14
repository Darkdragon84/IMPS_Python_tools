from collections import defaultdict
from itertools import product
from typing import Dict, Any, Tuple, Union, Hashable

IdxType = Union[Tuple[Hashable], Hashable]


class SparseTensor:
    def __init__(self, idxs_to_value: Dict[IdxType, Any] = None, rank: int = None) -> None:
        self._rank = rank
        self._init_indices_from_rank()
        self._idxs_to_value = dict()
        self.update(idxs_to_value)

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
            self._indices[i][idx].add(idxs[:i] + idxs[i+1:])

    def __getitem__(self, idxs: IdxType) -> Any:
        return self._idxs_to_value[idxs]

    def __setitem__(self, idxs, value) -> None:
        self._check_add_indices(idxs)
        self._idxs_to_value[idxs] = value

    def update(self, idx_to_value: Dict[IdxType, Any]) -> None:
        for idxs in idx_to_value:
            self._check_add_indices(idxs)

        self._idxs_to_value.update(idx_to_value)

    @property
    def rank(self) -> int:
        return self._rank

    @property
    def indices(self) -> Tuple:
        return self._indices

    @property
    def idxs_to_value(self):
        return self._idxs_to_value

    @property
    def nnz(self) -> int:
        """
        number of non-zero entries
        """
        return len(self._idxs_to_value)

    def tensordot(self, other: "SparseTensor", idx_left: int, idx_right: int) -> "SparseTensor":
        if not 0 <= idx_left < self.rank:
            raise ValueError(f"idx_left must be in [0, {self.rank - 1}]")
        if not 0 <= idx_right < other.rank:
            raise ValueError(f"idx_right must be in [0, {other.rank - 1}]")
        result_idxs_to_value = dict()
        sum_idxs = self.indices[idx_left].keys() & other.indices[idx_right].keys()

        for idxs in sum_idxs:
            for left_partial, right_partial in product(self.indices[idx_left][idxs], other.indices[idx_right][idxs]):
                out_idx = left_partial + right_partial
                left_full = left_partial[:idx_left] + (idxs,) + left_partial[idx_left:]
                right_full = right_partial[:idx_right] + (idxs,) + right_partial[idx_right:]
                res = self.idxs_to_value[left_full] * other.idxs_to_value[right_full]
                if out_idx in result_idxs_to_value:
                    result_idxs_to_value[out_idx] += res
                else:
                    result_idxs_to_value[out_idx] = res

        return SparseTensor(idxs_to_value=result_idxs_to_value)
