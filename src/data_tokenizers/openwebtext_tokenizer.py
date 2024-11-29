from datasets import load_dataset
import os

import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # Download tokenizer model
import re
import unicodedata

dataset = load_dataset("openwebtext")

text = dataset["train"][0]["text"]
tokens = word_tokenize(text)
print(tokens)

class OpenWebTextTokenizer:
    def __init__(self, input_path=None, output_path=None):
        import os
        # Optional paths can be provided; otherwise, default paths are used
        self.input_path = input_path or r'F:\Projects\HalfKeys\data\raw\openwebtext'  # Replace with the actual input dataset path
        self.output_path = output_path or r'F:\Projects\HalfKeys\data\processed\openwebtext'
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)

    def tokenize_text(self):
        import os
        import re
        import unicodedata
        # Process all text files in the input directory
        for root, _, files in os.walk(self.input_path):
            for file in files:
                if file.endswith('.txt'):  # Assuming text files
                    input_file = os.path.join(root, file)
                    output_file = os.path.join(self.output_path, file)

                    # Load the Input Text
                    with open(input_file, 'r', encoding='utf-8') as f:
                        text = f.read()

                    # Convert to Lowercase
                    text = text.lower()

                    # Remove Punctuation (Except Hyphens)
                    text = re.sub(r'[^\w\s\-]', '', text)

                    # Normalize Accented Characters
                    text = unicodedata.normalize('NFKD', text)
                    text = ''.join(c for c in text if not unicodedata.combining(c))

                    # Collapse Extra Whitespace
                    text = re.sub(r'\s+', ' ', text)

                    # Split Text into Tokens
                    tokens = text.split()

                    # Discard Alphanumeric Words with Mixed Letters and Numbers
                    tokens = [token for token in tokens if token.isalpha()]

                    # Join tokens back into cleaned text
                    cleaned_text = ' '.join(tokens)

                    # Write the Cleaned Text to the Output File
                    with open(output_file, 'w', encoding='utf-8') as f:
                        f.write(cleaned_text)