#Hidden message in rare characters of a string
import pandas
mess = [line.rstrip('\n') for line in open('rarecharacters.csv')]

characters = []
for char in mess[0]:
    if char not in characters:
        characters.append(char)

print(''.join(character for character in characters if mess[0].count(character) < 5))

# other solutions

OCCURRENCES = {}
for c in mess[0]: OCCURRENCES[c] = OCCURRENCES.get(c, 0) + 1
print(''.join([c for c in mess[0] if OCCURRENCES[c] < 5]))
