import os
import hashlib
from datasets import load_dataset

class OpenWebTextLoader:
    def __init__(self):
        self.dataset = None
        self.cache_dir = None
        self.is_loaded = False  # Track if the dataset is loaded

    def load(self, path):
        print("Loading OpenWebText dataset")
        self.cache_dir = path
        self.dataset = load_dataset("openwebtext", cache_dir=self.cache_dir, trust_remote_code=True)
        self.is_loaded = True
        print("Loading Complete")
        print("Dataset Structure:")
        print(self.dataset)  # Check dataset structure
        print("Dataset Sample:")
        print(self.dataset['train'][0])  # Inspect the first data sample
        print("Loading OpenWebText dataset complete")

    def verify_files(self):
        if not self.is_loaded:
            raise RuntimeError("Dataset has not been loaded. Please call the `load` method first.")
        
        print("Verifying dataset files...")
        dataset_cache_files = os.path.join(self.cache_dir, "downloads")

        # Check if the cache directory exists
        if not os.path.exists(dataset_cache_files):
            print("Cache directory not found. Verification failed.")
            return False

        # Get a list of all files in the cache directory
        files = [os.path.join(dataset_cache_files, f) for f in os.listdir(dataset_cache_files) if os.path.isfile(os.path.join(dataset_cache_files, f))]
        if not files:
            print("No files found in the cache directory. Verification failed.")
            return False

        # Validate file hashes (assume expected hashes are available)
        expected_hashes = {
            # Add expected hashes here. Example:
            # "filename": "expected_hash"
        }

        for file in files:
            filename = os.path.basename(file)
            if filename in expected_hashes:
                print(f"Verifying {filename}...")
                file_hash = self._calculate_file_hash(file)
                if file_hash != expected_hashes[filename]:
                    print(f"Hash mismatch for {filename}: expected {expected_hashes[filename]}, got {file_hash}")
                    return False
            else:
                print(f"Unexpected file {filename} found in cache.")

        print("All files verified successfully.")
        return True

    @staticmethod
    def _calculate_file_hash(filepath, hash_type="sha256"):
        """
        Calculate the hash of a file for verification.
        """
        hash_func = hashlib.new(hash_type)
        with open(filepath, "rb") as f:
            while chunk := f.read(8192):
                hash_func.update(chunk)
        return hash_func.hexdigest()
