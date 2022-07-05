import pytest

from src.operators import all_close
from src.spin_operators import SU2Generators
from src.utilities import commutator


class TestOperator:

    @pytest.mark.parametrize("dim_phys", [2, 3, 4, 5])
    def test_op_mult(self, dim_phys: int):
        gen = SU2Generators(dim_phys)
        assert all_close(commutator(gen.sx, gen.isy), -gen.sz)
