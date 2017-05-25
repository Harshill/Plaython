#Check to see if an expression has properly closed its brackets, return True if true

"""
When you find an opening racket add it to the stack
When you find a closed bracket check if the last item in the stack was its opening bracket
If this is true, pop the stack and continue, if this is false return false

"""

def checkio(expression: str):

    brackets = {'{': '}', '(': ')', '[': ']'}
    stack = []


    for char in expression:
        if char in brackets.keys():
            stack.append(char)
        elif char in brackets.values():
            if len(stack) == 0:
                return False
            elif char == brackets[stack[-1]]:
                stack.pop()
                continue
            else:
                return False
    if len(stack) is not 0:
        return False
    else:
        return True

#### Does not work because the inner brackets are not checked first

### maybe change this to split at every bracket and compare brackets????
print(checkio('str({[]})'))

print(checkio("(3+{1-1)}"))

print(checkio("(({[(((1)-2)+3)-3]/3}-3)"))

print(checkio("(((1+(1+1))))]"))