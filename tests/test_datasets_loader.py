import os
import sys
import shutil
import logging

# Add the src directory to the system path
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.abspath(os.path.join(script_dir, '..', 'src'))
sys.path.append(src_dir)

from data_loaders.datasets_loader import DatasetManager

# Configure logging
logging.basicConfig(level=logging.INFO)

def test_dataset_manager():
    dataset_name = "test_dataset"
    dataset_dir = "test_datasets/test_dataset_dir"

    # Ensure the test directory is clean
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.exists(DatasetManager.METADATA_FILE):
        os.remove(DatasetManager.METADATA_FILE)

    # Initial load
    DatasetManager.initial_load(dataset_name, dataset_dir)

    # Verify files
    success = DatasetManager.verify_files(dataset_name, auto_reload=False)
    if not success:
        logging.error("Verification failed after initial load.")
        return

    # Reload dataset
    DatasetManager.reload(dataset_name, dataset_dir, overwrite=True)

    # Verify files after reload
    success = DatasetManager.verify_files(dataset_name, auto_reload=False)
    if not success:
        logging.error("Verification failed after reload.")
        return

    # Delete dataset
    DatasetManager.delete_dataset(dataset_name)

    # Clean up
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    if os.path.exists(DatasetManager.METADATA_FILE):
        os.remove(DatasetManager.METADATA_FILE)

    logging.info("All tests completed successfully.")

if __name__ == "__main__":
    test_dataset_manager()