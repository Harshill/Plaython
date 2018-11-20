import numpy as np


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


if __name__ == '__main__':
    import doctest
    doctest.testmod()