import nltk
from nltk.util import ngrams
from collections import defaultdict, Counter

# Step 1: Download necessary corpora and tokenizer
nltk.download('reuters')
nltk.download('punkt')  # Download Punkt tokenizer
nltk.download('punkt_tab')  # Download punkt_tab tokenizer

# Step 2: Prepare training data
from nltk.corpus import reuters

# Directly use the words from the Reuters corpus
words = [word.lower() for word in reuters.words()]

# Step 3: Build a trigram model with counts
trigrams = list(ngrams(words, 3))
trigram_freq = defaultdict(Counter)
for w1, w2, w3 in trigrams:
    trigram_freq[(w1, w2)][w3] += 1

# Step 4: Define key mappings for ambiguous keys
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



# Step 5: Define the prediction function using trigrams
def predict_char(prev_chars, current_key):
    # Get possible characters for the current key
    possible_chars = key_mapping.get(current_key, [current_key])
    
    # If there aren't enough previous characters, just return the first possible character
    if len(prev_chars) < 2:
        return possible_chars[0]
    
    # Use the last two typed characters as context
    w1, w2 = prev_chars[-2], prev_chars[-1]
    
    # Score each possible character based on trigram frequency
    scores = {char: trigram_freq[(w1, w2)][char] for char in possible_chars}
    
    # Choose the character with the highest score or default to the first if scores are equal
    return max(scores, key=scores.get, default=current_key)

# Step 6: Define a function to simulate typing with disambiguation
def type_with_disambiguation(text):
    typed_text = []
    for char in text:
        predicted_char = predict_char(typed_text, char)
        typed_text.append(predicted_char)
    return ''.join(typed_text)

# Example Usage
input_text = "cav twr fesq vewetg swve gwbevwrd"  # Example input (assuming you're typing with a mirrored half-keyboard)
output_text = type_with_disambiguation(input_text)
print("Input Text:", input_text)
print("Predicted Output Text:", output_text)
