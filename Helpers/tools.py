import numpy as np
from itertools import product


def get_window(iterable, size, offset=0):
    """
    Returns an iterator by splitting an iterable into windows of given size
    The offset determines by how much the windows overlap

    >>> [i for i in get_window('123456', size=3, offset=1)]
    ['123', '345']

    >>> [i for i in get_window('ABCDEF', size=2)]
    ['AB', 'CD', 'EF']
    """

    assert size < len(iterable), 'Use a window size that is smaller than iterable length'
    assert offset < size, 'Use an offset that is smaller than the window size'

    step = size - offset
    return (iterable[pos: pos + size] for pos in range(0, len(iterable), step) if pos + size <= len(iterable))


def ceil_odd(x):
    """
    Nearest odd number ceiling

    >>> ceil_odd(3.2)
    5

    >>> ceil_odd(4.7)
    5

    """
    ceil_x = int(np.ceil(x))
    return ceil_x if ceil_x % 2 != 0 else ceil_x + 1


def ceil_even(x):
    """
    Nearest even number ceiling

    >>> ceil_even(2.1)
    4

    >>> ceil_even(3.1)
    4

    """
    ceil_x = int(np.ceil(x))
    return ceil_x if ceil_x % 2 == 0 else ceil_x + 1


def odd_number_gen(start=1, stop=1000, step=2):
    return (i for i in range(start, stop, step))


def even_number_gen(start=2, stop=1000, step=2):
    return (i for i in range(start, stop, step))


def is_valid_idx(matrix_shape, *args):
    for idx, dim_size in zip(args, matrix_shape):
        if 0 <= idx < dim_size:
            continue
        else:
            return False
    return True


def get_neighbouring_indices(row_idx, col_idx, matrix_shape):
    """
    Gets the indices of all adjacent cells in a matrix
    Currently only works for a two dimensional matrix

    :param row_idx:
    :param col_idx:
    :param matrix_shape:
    :return:
    """

    assert len(matrix_shape) == 2, 'This function only works for a two dimensional matrix'

    row_indices, col_indices = [], []

    for x, y in product([0, 1, -1], [0, 1, -1]):
        if x == 0 and y == 0:
            continue
        if is_valid_idx(matrix_shape, row_idx + y, col_idx + x):
            row_indices.append(row_idx + y)
            col_indices.append(col_idx + x)

    return row_indices, col_indices


if __name__ == '__main__':
    import doctest
    doctest.testmod()