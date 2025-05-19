import os
import re
import unicodedata
from data_tokenizers.openwebtext_tokenizer import OpenWebTextTokenizer

class TestTokenizer:
    def __init__(self):
        # Define path for test output
        self.output_path = r'F:\Projects\HalfKeys\tests\output_TEMP'
        # Ensure the test output directory exists
        os.makedirs(self.output_path, exist_ok=True)

    def run_test(self):
        # Create an instance of the tokenizer with test output path
        tokenizer = OpenWebTextTokenizer(output_path=self.output_path)

        # Run a preliminary test before tokenizing
        self.preliminary_test(tokenizer)

        # Test the tokenization with a small sample (without user prompt)
        sample_text = "This is a test sample."
        with open(os.path.join(self.output_path, 'text_0.txt'), 'w', encoding='utf-8') as f:
            f.write(tokenizer._process_text(sample_text))

        print("\nTest output:")
        print("------------")
        # Read and print the output for verification
        output_file = os.path.join(self.output_path, 'text_0.txt')
        with open(output_file, 'r', encoding='utf-8') as f:
            cleaned_text = f.read()
            print(cleaned_text)
            print("------------")

        # Clear test files
        self.clear_test_output()

    def clear_test_output(self):
        """Clear all files in the test output directory"""
        if os.path.exists(self.output_path):
            for file in os.listdir(self.output_path):
                file_path = os.path.join(self.output_path, file)
                try:
                    if os.path.isfile(file_path):
                        os.unlink(file_path)
                except Exception as e:
                    print(f"Error: {e}")

    def preliminary_test(self, tokenizer):
        # Load a small sample from the dataset
        sample_text = "This is a sample text for preliminary testing."

        # Convert to Lowercase
        sample_text = sample_text.lower()

        # Remove Punctuation (Except Hyphens)
        sample_text = re.sub(r'[^\w\s\-]', '', sample_text)

        # Normalize Accented Characters
        sample_text = unicodedata.normalize('NFKD', sample_text)
        sample_text = ''.join(c for c in sample_text if not unicodedata.combining(c))

        # Collapse Extra Whitespace
        sample_text = re.sub(r'\s+', ' ', sample_text)

        # Split Text into Tokens
        tokens = sample_text.split()

        # Discard Alphanumeric Words with Mixed Letters and Numbers
        tokens = [token for token in tokens if token.isalpha()]

        # Join tokens back into cleaned text
        cleaned_text = ' '.join(tokens)

        # Print the cleaned text for preliminary verification
        print("Preliminary Test Output:", cleaned_text)
