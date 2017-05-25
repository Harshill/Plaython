# Hamming Distance = sum of the bit differences between two integers in binary
from math import floor


def count(num1, num2):
    return len([char1 for char1, char2 in zip(num1,num2) if char1 is not char2])


def hamming(n, m):

    num1 = str(bin(n))[2:]
    num2 = str(bin(m))[2:]
    diff = len(num1)-len(num2)

    def add_zero(num, numzeros):
        num = (num[::-1] + numzeros*'0')[::-1]
        return num

    num2 = add_zero(num2, diff) if len(num1) > len(num2) else num2 = add_zero(num1, diff)
    if len(num1) > len(num2):
        num2 = add_zero(num2, diff)
    else:
        num1 = add_zero(num1, -diff)
    return count(list(num1), list(num2))

print(hamming(16, 15))