
def recall_password(cipher_grille, ciphered_password):
    rotate = lambda grid: list(zip(*grid[::-1]))
    letter = ''

    def passmaker(grille, passs):
        for index, line in enumerate(grille):
            for idx, char in enumerate(line):
                if char == 'X':
                    yield passs[index][idx]


    for rotations in range(4):
        letter += ''.join(passmaker(cipher_grille, ciphered_password))
        cipher_grille = rotate(cipher_grille)

    return letter


print(recall_password((
        'X...',
         '..X.',
         'X..X',
         '....'),
        ('itdf',
         'gdce',
         'aton',
         'qrdi')))

#Other solutions


def recall_password2(chipher_grid, cipher_key):

    password = []

    for _ in range(len(chipher_grid)):
        password += [cipher_key[row][column] for row in range(len(chipher_grid)) for column in range(len(chipher_grid[0])) if chipher_grid[row][column] == 'X']
        chipher_grid = tuple(zip(*chipher_grid[::-1]))

    return ''.join(password)

print(recall_password2((
        'X...',
        '..X.',
        'X..X',
        '....'),
        ('itdf',
         'gdce',
         'aton',
         'qrdi')))

#Another



def recall_password3(gril, ciph):

    pwd, j = '', ''.join

    for _ in range(4):
        pwd += j(c for g, c in zip(j(gril), j(ciph)) if g == 'X')
        gril = list(map(j, zip(*gril[::-1])))

    return pwd

print(recall_password3((
        'X...',
        '..X.',
        'X..X',
        '....'),
        ('itdf',
         'gdce',
         'aton',
         'qrdi')))