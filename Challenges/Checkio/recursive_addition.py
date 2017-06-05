"""
Add numbers in a list without using loops, importing, or using the word sum
"""


def recursive_add(data, total=0):
    if len(data) == 0:
        return total
    else:
        total += data.pop()
        return recursive_add(data, total)

if __name__ == '__main__':
    print(recursive_add([1, 2, 3]))
