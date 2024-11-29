import sys
import os

# Add the project root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from data_loaders import datasets_loader
from tests.test_tokenizer import TestTokenizer  # Updated import statement

def main():
    """Main function to fetch status and verify the dataset."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Fetch and print status of all datasets
    datasets_loader.DatasetManager.fetch_status()

    # Verify the dataset files for a specific dataset
    dataset_name = "openwebtext"
    datasets_loader.DatasetManager.verify_files(dataset_name)

    # Create an instance of TestTokenizer and run the test
    test = TestTokenizer()
    test.run_test()

if __name__ == "__main__":
    main()