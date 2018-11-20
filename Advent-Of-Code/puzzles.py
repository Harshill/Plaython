import numpy as np
from itertools import chain
from Helpers.tools import get_window, ceil_odd


def sum_similar(digit_seq, offset=1):
    """
    See: https://adventofcode.com/2017/day/1

    Calculates the sum of digits that match the next digit in the sequence.
    For the last digit the first digit is used to check for a match.

    :param digit_seq: A sequence of digits
    :return: Total of the sum where the digit matches the next digit in the sequence

    >>> sum_similar('1122')
    3

    >>> sum_similar('1212', offset=2)
    6

    >>> sum_similar('912121319')
    9

    >>> sum_similar('1234')
    0
    """
    iterator = zip(digit_seq, chain(digit_seq[offset:], digit_seq[:offset]))
    return sum(int(digit_1) for digit_1, digit_2 in iterator if digit_1 == digit_2)


def sum_of_row_wise_max_min_diff(matrix):
    """
    See: https://adventofcode.com/2017/day/2

    Calculate the difference between max and min for each row then sum all differences

    :param matrix: list of lists of numbers
    :return: sum

    >>> sum_of_row_wise_max_min_diff([[5, 1, 9, 5], [7, 5, 3], [2, 4, 6, 8]])
    18

    """

    return sum(max(row) - min(row) for row in matrix)


def sum_answer_if_perfect_divisible(filepath, sep='\t'):

    """
    Given a file with numbers, on each row find two numbers that perfectly divide without remainders
    Return the sum from the result of the divisions across all rows

    :param filepath:
    :param sep:
    :return:
    """

    def get_sum(line):
        row = np.array([int(item) for item in line.strip().split(sep)])

        # Divide every number from the other
        n_items = row.shape[0]
        division = row.reshape(n_items, 1) / row.reshape(1, n_items)
        np.fill_diagonal(division, 0.5)

        # Filter
        result = division[np.isclose(division - np.round(division), 0)]
        return result.sum()

    with open(filepath) as infile:
        return sum(get_sum(line) for line in infile)


def get_distance(n):
    """
    See: https://adventofcode.com/2017/day/3

    Imagine a data structure that stores the subsequent value in a spiral pattern from the center of the array
    For the n'th value find the manhattan distance to the center array

    My approach is:

     1. Find the spiral index where the n'th number would be, this is the distance in the first dimension
     2. Find the distance to the center term on the edge of the spiral the n'th term appears in,
        this is the distance in the 2nd dimension

    The sum of these two distances is the total distance to the center

    >>> get_distance(1)
    0

    >>> get_distance(277678)
    475

    >>> get_distance(10)
    3

    >>> get_distance(1024)
    31

    """
    if n == 1:
        return 0

    nearest_odd = ceil_odd(np.sqrt(n))

    # Index of the spiral the n'th term appears
    first_dimension_dist = nearest_odd // 2

    # Get distance from the center term along the spiral
    max_n_in_nth_spiral = nearest_odd ** 2
    min_n_in_nth_spiral = (nearest_odd - 2) ** 2 + 1
    all_n_in_nth_spiral = [max_n_in_nth_spiral] + list(range(min_n_in_nth_spiral, max_n_in_nth_spiral + 1))

    for column in get_window(all_n_in_nth_spiral, int(nearest_odd), 1):
        try:
            index = column.index(n)
            second_dimension_dist = abs(index - (nearest_odd // 2))
        except ValueError:
            continue

    return int(first_dimension_dist + second_dimension_dist)


if __name__ == '__main__':
    import doctest
    doctest.testmod()