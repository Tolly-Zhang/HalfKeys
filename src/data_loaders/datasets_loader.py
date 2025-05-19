import os
import hashlib
import json
import shutil
import logging
from datetime import datetime, timedelta
from tqdm import tqdm
import datasets
import tempfile
import atexit

class DatasetManager:
    # Class variables
    _verified_datasets = set()  # Track verified datasets for the session

    # Update paths structure
    DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))
    METADATA_DIR = os.path.join(DATA_DIR, 'metadata')
    RAW_DIR = os.path.join(DATA_DIR, 'raw')
    PROCESSED_DIR = os.path.join(DATA_DIR, 'processed')

    # Update metadata file paths
    METADATA_FILE = os.path.join(METADATA_DIR, "dataset_metadata.json")
    DATASET_INFO_FILE = os.path.join(METADATA_DIR, "dataset_info.json")
    STATE_FILE = os.path.join(METADATA_DIR, "state.json")

    @staticmethod
    def initial_load(dataset_name, dataset_dir):
        """
        Initial load of the dataset. Downloads the dataset files to the specified directory.
        Computes and saves a reference hash for verification.
        """
        # Add this line at the start to see where temp files go
        logging.info(f"Temporary directory location: {tempfile.gettempdir()}")

        if DatasetManager._is_directory_populated(dataset_dir):
            raise RuntimeError(f"Directory '{dataset_dir}' is not empty. Please provide an empty directory.")

        logging.info(f"Downloading '{dataset_name}' dataset into '{dataset_dir}'...")
        try:
            # Create temporary directory for dataset download
            with tempfile.TemporaryDirectory() as temp_dir:
                # Load the dataset using temporary directory
                dataset = datasets.load_dataset(dataset_name, cache_dir=temp_dir)
                # If dataset is a DatasetDict, combine all splits
                if isinstance(dataset, datasets.DatasetDict):
                    combined_dataset = datasets.concatenate_datasets([split for split in dataset.values()])
                    dataset = combined_dataset
                # Save the dataset directly to raw directory
                dataset.save_to_disk(dataset_dir)

                # Rest of the function remains the same
                reference_hash = DatasetManager._calculate_directory_hash(dataset_dir)
                logging.info(f"Reference hash for '{dataset_name}': {reference_hash}")

                new_entry = {
                    "name": dataset_name,
                    "path": dataset_dir,
                    "reference_hash": reference_hash,
                    "date_loaded": datetime.now().isoformat(),
                    "date_last_verified": None
                }
                metadata = DatasetManager._load_metadata()
                metadata = [entry for entry in metadata if entry["name"] != dataset_name]
                metadata.append(new_entry)
                DatasetManager._save_metadata(metadata)

                logging.info("Dataset downloaded and metadata saved.")
        except Exception as e:
            logging.error(f"Error during dataset loading: {e}")
            raise

    @staticmethod
    def reload(dataset_name, dataset_dir, overwrite=False):
        """
        Reloads the dataset files. If `overwrite=True`, clears the directory before downloading.
        Otherwise, verifies existing files and updates as necessary.
        """
        if overwrite:
            logging.info(f"Overwriting existing dataset in '{dataset_dir}'...")
            if os.path.exists(dataset_dir):
                shutil.rmtree(dataset_dir)
            os.makedirs(dataset_dir, exist_ok=True)
        elif DatasetManager._is_directory_populated(dataset_dir):
            logging.info(f"Directory '{dataset_dir}' contains files. Comparing with the reference dataset...")

        logging.info(f"Reloading dataset '{dataset_name}' into '{dataset_dir}'...")
        try:
            # Create temporary directory for dataset download
            with tempfile.TemporaryDirectory() as temp_dir:
                dataset = datasets.load_dataset(dataset_name, cache_dir=temp_dir)
                if isinstance(dataset, datasets.DatasetDict):
                    combined_dataset = datasets.concatenate_datasets([split for split in dataset.values()])
                    dataset = combined_dataset
                dataset.save_to_disk(dataset_dir)

                # Rest of the function remains the same...
                reference_hash = DatasetManager._calculate_directory_hash(dataset_dir)
                logging.info(f"New reference hash for '{dataset_name}': {reference_hash}")

                new_entry = {
                    "name": dataset_name,
                    "path": dataset_dir,
                    "reference_hash": reference_hash,
                    "date_loaded": datetime.now().isoformat(),
                    "date_last_verified": None
                }

                metadata = DatasetManager._load_metadata()
                metadata = [entry for entry in metadata if entry["name"] != dataset_name]
                metadata.append(new_entry)
                DatasetManager._save_metadata(metadata)

                logging.info("Dataset reloaded and metadata updated.")
        except Exception as e:
            logging.error(f"Error during dataset reloading: {e}")
            raise

    @staticmethod
    def fetch_status(dataset_name=None):
        """
        Fetches and prints the status report of the specified dataset.
        If no dataset_name is provided, fetches the status of all datasets.
        """
        try:
            metadata = DatasetManager._load_metadata()
            if (dataset_name):
                # Fetch status for the specified dataset
                entry = next((e for e in metadata if e["name"] == dataset_name), None)
                if (entry):
                    logging.info(f"Dataset '{dataset_name}' status:")
                    logging.info(f"  Path: {entry['path']}")
                    logging.info(f"  Date Loaded: {entry['date_loaded']}")
                    logging.info(f"  Date Last Verified: {entry.get('date_last_verified', 'Never')}")
                else:
                    logging.info(f"No metadata found for dataset '{dataset_name}'.")
            else:
                # Fetch status for all datasets
                if (metadata):
                    logging.info("Status of all loaded datasets:")
                    for entry in metadata:
                        logging.info(f"- Dataset '{entry['name']}':")
                        logging.info(f"    Path: {entry['path']}")
                        logging.info(f"    Date Loaded: {entry['date_loaded']}")
                        logging.info(f"    Date Last Verified: {entry.get('date_last_verified', 'Never')}")
                else:
                    logging.info("No datasets have been loaded.")
        except Exception as e:
            logging.error(f"An error occurred while fetching status: {e}")

    @classmethod
    def verify_files(cls, dataset_name: str) -> bool:
        """Verify dataset files and update verification timestamp"""
        # Skip if already verified this session
        if dataset_name in cls._verified_datasets:
            logging.info(f"Dataset {dataset_name} already verified this session.")
            return True

        try:
            metadata = cls._load_metadata()
            entry = next((e for e in metadata if e["name"] == dataset_name), None)
            if not entry:
                raise RuntimeError(f"No metadata found for dataset '{dataset_name}'.")

            dataset_dir = entry["path"]
            reference_hash = entry["reference_hash"]

            if not os.path.exists(dataset_dir):
                logging.error(f"Dataset directory not found: {dataset_dir}")
                return False

            # Calculate current hash
            logging.info("Dataset verification initiated.")
            current_hash = cls._calculate_directory_hash(dataset_dir)

            logging.info(f"Reference hash: {reference_hash}")
            logging.info(f"Current hash: {current_hash}")

            if current_hash == reference_hash:
                logging.info("Dataset verification successful.")
                entry["date_last_verified"] = datetime.now().isoformat()
                cls._save_metadata(metadata)
                cls._verified_datasets.add(dataset_name)
                return True
            else:
                logging.error("Dataset verification failed! Files may be corrupted.")
                return False

        except Exception as e:
            logging.error(f"Error during verification: {e}")
            return False

    @staticmethod
    def delete_dataset(dataset_name):
        """
        Deletes the dataset files and removes its metadata entry.
        """
        metadata = DatasetManager._load_metadata()
        entry = next((e for e in metadata if e["name"] == dataset_name), None)

        if not entry:
            raise RuntimeError(f"No metadata found for dataset '{dataset_name}'.")

        dataset_dir = entry["path"]

        if os.path.exists(dataset_dir):
            logging.info(f"Deleting dataset directory '{dataset_dir}'...")
            shutil.rmtree(dataset_dir)

        metadata = [e for e in metadata if e["name"] != dataset_name]
        DatasetManager._save_metadata(metadata)
        logging.info(f"Dataset '{dataset_name}' and its metadata have been deleted.")

    @classmethod
    def _calculate_directory_hash(cls, directory: str) -> str:
        """Calculate hash of all files in directory"""
        hash_func = hashlib.sha256()
        file_list = []

        for file in os.listdir(directory):
            if file.endswith('.arrow'):
                file_list.append(os.path.join(directory, file))

        file_list.sort()

        for filepath in tqdm(file_list, desc="Verifying files"):
            with open(filepath, "rb") as f:
                while chunk := f.read(8192):
                    hash_func.update(chunk)

        return hash_func.hexdigest()

    @staticmethod
    def _is_directory_populated(directory):
        """Check if the specified directory contains files or subdirectories."""
        return os.path.exists(directory) and any(os.scandir(directory))

    @classmethod
    def _load_metadata(cls) -> list:
        """Load dataset metadata"""
        os.makedirs(cls.METADATA_DIR, exist_ok=True)
        if not os.path.exists(cls.METADATA_FILE):
            return []
        with open(cls.METADATA_FILE, "r") as f:
            return json.load(f)

    @classmethod
    def _save_metadata(cls, metadata: list):
        """Save dataset metadata"""
        with open(cls.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
