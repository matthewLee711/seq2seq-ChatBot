# bobby
# order 1
"""
given b
    b is next 1/3
    o is next 1/3
    y is next 1/3
given o
    b is next 1
given y
    terminates

{'b': {'y': 1, 'b': 1, 'o': 1}, 'o': {'b': 1}}
"""

from random import choice
import sys

# Split words by white spaces and store into dictionary
# Replace characters for full words. Store model into mongodb
def generateModel(text, order):
    model = {}
    for i in range(0, len(text) - order):
        fragment = text[i:i+order]
        next_letter = text[i+order]
        if fragment not in model:
            model[fragment] = {}
        if next_letter not in model[fragment]:
            model[fragment][next_letter] = 1
        else:
            model[fragment][next_letter] += 1
    return model

# Replace with full words instead of characters
def getNextCharacter(model, fragment):
    letters = []
    for letter in model[fragment].keys():
        for times in range(0, model[fragment][letter]):
            letters.append(letter)
    return choice(letters)

def generateText(text, order, length):
    model = generateModel(text, order)
    currentFragment = text[0:order]
    output = ""
    for i in range(0, length-order):
        newCharacter = getNextCharacter(model, currentFragment)
        output += newCharacter
        currentFragment = currentFragment[1:] + newCharacter
    print output


text = "some sample text the world as we no it is going to be there"
if __name__ == "__main__":
    generateText(text, int(sys.argv[1]), int(sys.argv[2]))
