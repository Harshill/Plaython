# Convert an integer to binary then return the number of 1's in the binary
from math import floor


def checkio(number):

    bin = []
    while int(number) is not 0:
        bin.append(str(floor(number % 2)))
        number /= 2

    return ''.join(bin)[::-1].count('1')

print(checkio(4))
