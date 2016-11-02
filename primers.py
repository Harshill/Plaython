Ref = 'AATTGCTA'
seq1 = 'AATT'
seq2 = 'TAGC'

seq2reverse = seq2[len(Ref)-len(seq1)-1::-1]
complement = {'A':'T', 'T':'A', 'G':'C', 'C':'G'}

final = seq1 + ''.join(complement.get(base, base) for base in seq2reverse)
print('Your protein sequence is:', final)

if Ref != final:
    print('There is a mismatch')
else:
    print('There is no mismatch')

if 'N' in Ref+seq1+seq2:
    print("One of the sequences has an 'N'")
