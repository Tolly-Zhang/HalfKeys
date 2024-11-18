import nltk
nltk.download('words')
from nltk.corpus import words
word_list = words.words()

key_mapping = {
    'q': ['q', 'p'],
    'w': ['w', 'o'],
    'e': ['e', 'i'],
    'r': ['r', 'u'],
    't': ['t', 'y'],
    'a': ['a', 'a'],
    's': ['s', 'l'],
    'd': ['d', 'k'],
    'f': ['f', 'j'],
    'g': ['g', 'h'],
    'z': ['z', 'z'],
    'x': ['x', 'x'],
    'c': ['c', 'c'],
    'v': ['v', 'm'],
    'b': ['b', 'n'],
}

def permute_helper(string, index, arr):
    if index == len(string):
        return arr

    newArr = []
    
    if string[index] not in key_mapping:
        print(f"Invalid character: {string[index]}")    
        return

    char1 = key_mapping[string[index]][0]
    char2 = key_mapping[string[index]][1]
    
    if len(arr) != 0:
        for i in arr:
            newArr.append(i + char1)
            newArr.append(i + char2)

    elif len(arr) == 0:
        newArr.append(char1)
        newArr.append(char2)
    # print(f"Index: {index} Arr: {newArr}")
    return permute_helper(string, index + 1, newArr)

def permuteText(text):
    return permute_helper(text, 0, [])

def searchDictionary(word):
    if word in word_list:
        return True
    return False


input_word = input()
input_word = input_word.lower()

print(f"Input: {input_word}")
permutations = permuteText(input_word)
for word in permutations:
    if searchDictionary(word):
        print(f"Found: {word}")
print("Done")
