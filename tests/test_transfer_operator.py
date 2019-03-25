import numpy

from helpers.mps_matrix import MPSMatrix
from helpers.transfer_operator import apply_transop_left, apply_transop_right


def main():
    d = 3
    m = 10

    A = MPSMatrix.get_random_left_ortho_mps(d, m)
    B = MPSMatrix.get_random_right_ortho_mps(d, m)

    numpy.set_printoptions(precision=5, suppress=True)
    y1 = apply_transop_left(A)
    y2 = apply_transop_right(B)
    print(numpy.max(numpy.abs(y1 - numpy.eye(m))))
    print(numpy.max(numpy.abs(y2 - numpy.eye(m))))
    # A = [numpy.random.randn(M, M) for _ in range(D)]


if __name__ == '__main__':
    main()
