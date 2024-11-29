import os
from data_tokenizers.openwebtext_tokenizer import OpenWebTextTokenizer

class TestTokenizer:
    def __init__(self):
        # Define paths for test input and output
        self.input_path = r'F:\Projects\HalfKeys\tests'
        self.output_path = r'F:\Projects\HalfKeys\tests\output'
        # Ensure the test output directory exists
        os.makedirs(self.output_path, exist_ok=True)

    def run_test(self):
        # Create an instance of the tokenizer with test paths
        tokenizer = OpenWebTextTokenizer(input_path=self.input_path, output_path=self.output_path)
        # Run the tokenizer
        tokenizer.tokenize_text()
        # Read and print the output for verification
        output_file = os.path.join(self.output_path, 'test_document.txt')
        with open(output_file, 'r', encoding='utf-8') as f:
            cleaned_text = f.read()
            print(cleaned_text)