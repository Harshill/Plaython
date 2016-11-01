#converts text to a coded alphabet

import string

text = 'mapdasdafgewac .0(.56d]fsdfasfsdf'

original = string.ascii_lowercase                                   #alphabet
new = string.ascii_lowercase[2:]+string.ascii_lowercase[:2]         #alphabet shifted by 2

table = str.maketrans(original,new)
print(text.translate(table))




#other solutions
message = text.translate(str.maketrans("".join([chr(x) for x in range(97,123)]),"".join([chr(97+(x+2)%26) for x in range(26)])))
print(message)

#another
cypher = dict(zip(original,new))
print ("".join(cypher.get(c,c) for c in text))
