import pytest

from src.operators import Operator
from src.spin_operators import SU2Generators
from src.utilities import commutator


class TestOperator:

    @pytest.mark.parametrize("dim_phys", [2, 3, 4])
    def test_op_mult(self, dim_phys: int):
        gen = SU2Generators(dim_phys)
        assert commutator(gen.sx, gen.isy) == -gen.sz