#For a list of pawns from the white side, see how many pawns are covered by another pawn


def safe_pawns(pawns):

    asc1 = [chr(ord(pawn[0])+1) + str(int(pawn[1])-1) for pawn in pawns]
    asc2 = [chr(ord(pawn[0])-1) + str(int(pawn[1])-1) for pawn in pawns]

    safes = 0
    for square1, square2 in list(zip(asc1, asc2)):
        if square1 in pawns or square2 in pawns:
            safes += 1
    return safes

print(safe_pawns({"b4", "d4", "f4", "c3", "e3", "g5", "d2"}))