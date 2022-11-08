from typing import Union
from argparse import ArgumentParser


def get_receptive_field_size(kernel_sizes: Union[list, tuple], strides: Union[list, tuple]):
    """
    extract receptive field.
    Note: the stride of the last layer is not taken into account becasue we only care about one point of it.
    formula is taken from https://distill.pub/2019/computing-receptive-fields/#return-from-solving-receptive-field-size
    :param kernel_sizes:
    :param strides:
    :return:
    >>> get_receptive_field_size([1, 1, 1,1], [2, 2,2,2])
    1
    >>> get_receptive_field_size([5, 2], [3, 1])
    8
    >>> get_receptive_field_size([2, 4, 3], [1, 2, 1])
    9
    >>> get_receptive_field_size([250, 48, 7, 7, 7, 7, 7, 7, 7, 32, 1, 1], [160, 2, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    31130
    """
    assert len(kernel_sizes) == len(strides), "The lengths of the input parameters do no match!"
    receptive_field = 0
    length = len(kernel_sizes)
    for l in range(length):
        s = 1
        for i in range(l):
            s *= strides[i]
        receptive_field += ((kernel_sizes[l] - 1) * s)
    return receptive_field + 1


def get_receptive_field_start_end_indices(kernel_sizes, strides, pooling=None):
    pooling = [0 for _ in kernel_sizes] if pooling is None else pooling
    assert len(kernel_sizes) == len(strides) == len(pooling), "The lengths of the input parameters do no match!"
    u = v = 0
    length = len(kernel_sizes)
    for l in range(length - 1, -1, -1):
        u = -pooling[l] + u * strides[l]
        v = -pooling[l] + v * strides[l] + kernel_sizes[l] - 1
    return u, v


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument("--kernel_sizes", nargs="+", type=int,
                        help="pass a sequence of kernels starting from the first layer. ex: 7, 3, 3")
    parser.add_argument("--strides", nargs="+", type=int,
                        help="pass a sequence of strides starting from the first layer. ex: 2, 1, 2")
    args = parser.parse_args()

    receptive_field = get_receptive_field_size(args.kernel_sizes, args.strides)
    print(f"[INFO] receptive field for a network with kernel-sizes: {args.kernel_sizes} & strides: {args.strides} is: {receptive_field}")
