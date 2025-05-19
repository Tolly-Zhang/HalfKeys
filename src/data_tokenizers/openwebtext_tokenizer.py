from datasets import load_dataset
import os
import nltk
from nltk.tokenize import word_tokenize
nltk.download('punkt')  # Download tokenizer model
import re
import unicodedata
from tqdm import tqdm
import hashlib
import gzip
import json
from datetime import datetime

class OpenWebTextTokenizer:
    def __init__(self, output_path=None, files_per_group=100000):
        # Update default path to use consistent structure
        if output_path is None:
            base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
            output_path = os.path.join(base_dir, 'data', 'processed', 'openwebtext')
        self.output_path = output_path
        print(f"Tokenizer initialized. Output path: {self.output_path}")
        # Ensure output directory exists
        os.makedirs(self.output_path, exist_ok=True)
        self.files_per_group = files_per_group
        self.current_group = []
        self.group_count = 0
        self.tokenizer_version = "1.0"  # Add version tracking

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words

        Args:
            text: Input text to tokenize

        Returns:
            List of tokens
        """
        # Normalize text (lowercase, remove accents)
        text = text.lower()
        text = unicodedata.normalize('NFKD', text)
        text = ''.join([c for c in text if not unicodedata.combining(c)])

        # Tokenize using NLTK
        tokens = word_tokenize(text)

        # Filter tokens
        valid_tokens = []
        for token in tokens:
            # Only keep word tokens (no numbers or special chars)
            if re.match(r'^[a-z]+(-[a-z]+)*$', token):
                valid_tokens.append(token)

        return valid_tokens

    def _save_group(self):
        if not self.current_group:
            return

        # Change file extension to .gz
        output_file = os.path.join(self.output_path, f'group_{self.group_count:04d}.txt.gz')
        # Use gzip compression to write the file
        with gzip.open(output_file, 'wt', encoding='utf-8') as f:
            f.write('\n'.join(self.current_group))

        self.current_group = []
        self.group_count += 1

    def _process_text(self, text):
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
        return ' '.join(tokens)

    def verify_processed_files(self):
        """Verify processed files by calculating and comparing their hash."""
        print(f"\nVerifying processed files in: {self.output_path}")

        if not os.path.exists(self.output_path):
            print("Error: Output directory doesn't exist!")
            return False

        # Calculate hash of all processed files
        hash_func = hashlib.sha256()
        file_list = []

        # Update to look for .gz files
        for file in os.listdir(self.output_path):
            if file.startswith('group_') and file.endswith('.txt.gz'):
                file_list.append(os.path.join(self.output_path, file))

        if not file_list:
            print("No processed files found!")
            return False

        # Sort files to ensure consistent order
        file_list.sort()

        # Calculate hash from compressed file contents
        for filepath in tqdm(file_list, desc="Verifying processed files"):
            try:
                with gzip.open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        hash_func.update(chunk)
            except Exception as e:
                print(f"Error reading file {filepath}: {e}")
                return False

        current_hash = hash_func.hexdigest()

        # Update to use JSON metadata
        metadata_file = os.path.join(self.output_path, "processed_files_metadata.json")

        if os.path.exists(metadata_file):
            # Verify against existing metadata
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)

            if current_hash == metadata["file_hash"]:
                print("Verification successful! Processed files are intact.")
                # Update verification date
                metadata["date_last_verified"] = datetime.now().isoformat()
                with open(metadata_file, 'w') as f:
                    json.dump(metadata, f, indent=4)
                return True
            else:
                print("Verification failed! Files may be corrupted or modified.")
                return False
        else:
            # Create new metadata
            metadata = {
                "file_hash": current_hash,
                "date_tokenized": datetime.now().isoformat(),
                "date_last_verified": datetime.now().isoformat(),
                "tokenizer_version": self.tokenizer_version,
                "compression": "gzip",
                "files_format": "txt.gz",
                "processing_details": {
                    "case": "lowercase",
                    "special_chars": "removed except hyphens",
                    "normalization": "NFKD unicode",
                    "token_filters": ["alpha only", "no mixed alphanumeric"]
                }
            }

            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=4)
            print(f"Initial metadata created and saved.")
            return True

    def tokenize_text(self):
        print(f"\nStarting tokenization process. Files will be saved to: {self.output_path}")

        # Get user confirmation
        while True:
            response = input("\nDo you want to proceed with tokenization? (y/n): ").lower()
            if response in ['y', 'n']:
                break
            print("Please enter 'y' or 'n'")

        if response != 'y':
            print("Tokenization cancelled.")
            return

        # Load the openwebtext dataset
        dataset = load_dataset("openwebtext", split='train')

        # Initialize progress bar
        pbar = tqdm(total=len(dataset), desc="Tokenizing texts")

        # Process each text entry in the dataset
        for i, entry in enumerate(dataset):
            text = entry["text"]
            cleaned_text = self._process_text(text)

            # Add processed text to current group
            self.current_group.append(cleaned_text)

            # If group is full, save it
            if len(self.current_group) >= self.files_per_group:
                self._save_group()

            # Update progress bar
            pbar.update(1)

        # Save any remaining texts in the last group
        if self.current_group:
            self._save_group()

        # Close progress bar
        pbar.close()

        # Print completion message
        print(f"\nTokenization complete! {self.group_count} group files created in {self.output_path}")
