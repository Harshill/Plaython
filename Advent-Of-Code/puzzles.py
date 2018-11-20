import numpy as np
from itertools import chain, cycle
from Helpers.tools import get_window, ceil_odd, odd_number_gen, even_number_gen, get_neighbouring_indices


def sum_similar(digit_seq, offset=1):
    """
    See: https://adventofcode.com/2017/day/1

    Calculates the sum of digits that match the next digit in the sequence.
    For the last digit the first digit is used to check for a match.

    :param digit_seq: A sequence of digits
    :param offset: The distance on the iterable where to check for a match
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

    spiral_size = ceil_odd(np.sqrt(n))

    # Index of the spiral the n'th term appears
    first_dimension_dist = spiral_size // 2

    # Get distance from the center term along the spiral
    max_n_in_nth_spiral = spiral_size ** 2
    min_n_in_nth_spiral = (spiral_size - 2) ** 2 + 1
    all_n_in_nth_spiral = [max_n_in_nth_spiral] + list(range(min_n_in_nth_spiral, max_n_in_nth_spiral + 1))

    for column in get_window(all_n_in_nth_spiral, int(spiral_size), 1):
        try:
            index = column.index(n)
            second_dimension_dist = abs(index - (spiral_size // 2))
        except ValueError:
            continue

    return int(first_dimension_dist + second_dimension_dist)


class Spiral():

    def __init__(self, n):
        self.n = n
        self.spiral_size = ceil_odd(np.sqrt(self.n))
        self.center_idx = self.spiral_size // 2
        self.spiral = self.make_spiral()

    def make_spiral(self):
        return np.zeros(shape=(self.spiral_size, self.spiral_size))

    def get_next_index(self, dimension='row'):
        """
                The spiral starts at the center of the matrix and moves first right then up then left then down
                The amount to move in each direction is a sequence:
                    if Right or Up: the sequence of odd numbers
                    if Left or Down: the sequence of even numbers

                To determine the next index, we just find the next direction movement.

                If we are interested in the row index only Up and Down movements matter:
                    if Up: Add -1 from the current index
                    if Down: Add 1 to the current index

                If we are interested in column index only Right and Left movements matter:
                    if Right: Add 1 to the current index
                    if Left: Add -1 to the current index


                17  16  15  14  13
                18   5   4   3  12
                19   6   1   2  11
                20   7   8   9  10
                21  22  23---> ...

                :param center_idx:
                :param dimension:
                :return:
                """

        # Create the odd and even number generators to determine step size
        right, left = odd_number_gen(), even_number_gen()
        up, down = odd_number_gen(), even_number_gen()

        # Make dictionary to get the next direction value and how much to move in that direction
        key = ['right', 'up', 'left', 'down']
        gen = [right, up, left, down]
        value = [0, -1, 0, 1] if dimension == 'row' else [1, 0, -1, 0]
        steps_dict = dict(zip(key, zip(gen, value)))

        # Establish order of directions to make spiral Right, Up, Left, Down
        step_order = cycle(key)

        # Get next index
        curr_index = self.center_idx
        yield curr_index
        while True:
            next_step = next(step_order)
            n_steps_gen, step_dir = steps_dict[next_step]
            n_steps = next(n_steps_gen)

            for _ in range(n_steps):
                curr_index += step_dir
                yield curr_index

    def get_max_sum(self, target_sum):
        # Get next index of spiral to fill
        row_index_gen = self.get_next_index(dimension='row')
        col_index_gen = self.get_next_index(dimension='column')

        max_sum = self.spiral.max()

        while max_sum < target_sum:
            row_index, col_index = next(row_index_gen), next(col_index_gen)

            if row_index == self.center_idx and col_index == self.center_idx:
                self.spiral[row_index, col_index] = 1
            else:
                row_indices, col_indices = get_neighbouring_indices(row_index, col_index, self.spiral.shape)
                self.spiral[row_index, col_index] = self.spiral[row_indices, col_indices].sum()
            max_sum = self.spiral.max()

        return int(max_sum)


if __name__ == '__main__':
    import doctest
    doctest.testmod()