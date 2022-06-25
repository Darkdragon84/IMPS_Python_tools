import numpy

from src.block_tensor import BlockTensor


def main():
    # rank = 4
    t1 = numpy.random.randn(2, 4, 4, 6)
    t2 = numpy.random.randn(4, 3, 4, 6)
    bt = BlockTensor(4)
    # bt.add_tensor(t1, (2, 1, 0, 1))
    # bt.add_tensor(t2, (-1, -2, 0, 1))
    bt[(2, 1, 0, 1)] = t1
    bt[(-1, -2, 0, 1)] = t2

    print(bt.dims)
    print(bt.total_dims)

    # rank = 2
    m1 = numpy.random.randn(2, 3)
    m2 = numpy.random.randn(4, 3)
    m3 = numpy.random.randn(4, 2)
    m3_new = numpy.random.randn(4, 2)
    mt = BlockTensor(2)
    # mt.add_tensor(m1, (1, 0))
    # mt.add_tensor(m2, (-1, 0))
    # mt.add_tensor(m3, (-1, -2))
    mt[1, 0] = m1
    mt[-1, 0] = m2
    mt[-1, -2] = m3

    print(mt.dims)
    print(mt.total_dims)

    mt[(-1, -2)] = m3_new

    print(mt.dims)
    print(mt.total_dims)

    mt2 = BlockTensor.from_tensordict({(1, 0): m1, (-1, 0): m2, (-1, -2): m3_new})

    # print(mt2[-1, 0])
    # mt2[(-1, 0)] += m2
    #
    # print(mt2[-1, 0])
    # print(numpy.max(numpy.abs(mt2[-1, 0] - m2)))
    mtT = BlockTensor(2)
    mtT[0, 1] = m1.T
    mtT[0, -1] = m2.T
    mtT[-2, -1] = m3.T

    res = mt.tensordot(mtT, 1, 0)
    print(res.dims)
    print(res.total_dims)
    print("done")


if __name__ == '__main__':
    main()
