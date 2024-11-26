import os
import hashlib
import json
import shutil
import logging
from datetime import datetime
from tqdm import tqdm
import datasets

class DatasetManager:
    METADATA_FILE = "dataset_metadata.json"

    @staticmethod
    def initial_load(dataset_name, dataset_dir):
        """
        Initial load of the dataset. Downloads the dataset files to the specified directory.
        Computes and saves a reference hash for verification.
        """
        # Ensure directory is empty
        if DatasetManager._is_directory_populated(dataset_dir):
            raise RuntimeError(f"Directory '{dataset_dir}' is not empty. Please provide an empty directory.")

        logging.info(f"Downloading '{dataset_name}' dataset into '{dataset_dir}'...")
        try:
            # Use the datasets library to download the dataset
            dataset = datasets.load_dataset(dataset_name, split='train')
            # Save the dataset to the specified directory
            dataset.save_to_disk(dataset_dir)

            # Compute reference hash
            reference_hash = DatasetManager._calculate_directory_hash(dataset_dir)
            logging.info(f"Reference hash for '{dataset_name}': {reference_hash}")

            # Save metadata
            new_entry = {
                "name": dataset_name,
                "path": dataset_dir,
                "reference_hash": reference_hash,
                "date_loaded": datetime.now().isoformat(),
                "date_last_verified": None
            }
            metadata = DatasetManager._load_metadata()
            metadata = [entry for entry in metadata if entry["name"] != dataset_name]  # Remove old entries for the same dataset
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
            shutil.rmtree(dataset_dir)
            os.makedirs(dataset_dir, exist_ok=True)
        elif DatasetManager._is_directory_populated(dataset_dir):
            logging.info(f"Directory '{dataset_dir}' contains files. Comparing with the reference dataset...")

        # Simulate download/update logic
        logging.info(f"Reloading dataset '{dataset_name}' into '{dataset_dir}'...")
        # Use the datasets library to reload the dataset
        dataset = datasets.load_dataset(dataset_name, split='train')
        dataset.save_to_disk(dataset_dir)

        # Update reference hash
        reference_hash = DatasetManager._calculate_directory_hash(dataset_dir)
        logging.info(f"New reference hash for '{dataset_name}': {reference_hash}")

        # Update metadata
        metadata = DatasetManager._load_metadata()
        for entry in metadata:
            if entry["name"] == dataset_name:
                entry["reference_hash"] = reference_hash
                entry["date_loaded"] = datetime.now().isoformat()
        DatasetManager._save_metadata(metadata)

        logging.info("Dataset reloaded and metadata updated.")

    @staticmethod
    def verify_files(dataset_name, auto_reload=False):
        """
        Verifies the current files against the reference hash stored during the initial load.
        If `auto_reload=True`, automatically reloads the dataset upon verification failure.
        Updates the 'date_last_verified' in the metadata if verification is successful.
        """
        metadata = DatasetManager._load_metadata()
        entry = next((e for e in metadata if e["name"] == dataset_name), None)

        if not entry:
            raise RuntimeError(f"No metadata found for dataset '{dataset_name}'. Please load it first.")

        dataset_dir = entry["path"]
        reference_hash = entry["reference_hash"]

        if not os.path.exists(dataset_dir):
            logging.error(f"Dataset directory '{dataset_dir}' does not exist. Verification failed.")
            return False

        logging.info("Verifying dataset files...")
        current_hash = DatasetManager._calculate_directory_hash(dataset_dir)
        logging.info(f"Reference hash: {reference_hash}")
        logging.info(f"Current hash: {current_hash}")

        if current_hash != reference_hash:
            logging.warning("Hash mismatch! The dataset files have been altered.")
            if auto_reload:
                DatasetManager.reload(dataset_name, dataset_dir, overwrite=True)
            else:
                raise RuntimeError("Dataset verification failed. Please reload the dataset to proceed.")
        else:
            logging.info("Dataset verification successful.")
            entry["date_last_verified"] = datetime.now().isoformat()
            DatasetManager._save_metadata(metadata)
        return True

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

    @staticmethod
    def _calculate_directory_hash(directory, hash_type="sha256"):
        """
        Computes a hash for the entire directory by hashing all files within it.
        """
        hash_func = hashlib.new(hash_type)
        for root, _, files in os.walk(directory):
            for file in sorted(files):  # Sort to ensure consistent order
                filepath = os.path.join(root, file)
                with open(filepath, "rb") as f:
                    while chunk := f.read(8192):
                        hash_func.update(chunk)
        return hash_func.hexdigest()

    @staticmethod
    def _is_directory_populated(directory):
        """Check if the specified directory contains files or subdirectories."""
        return os.path.exists(directory) and any(os.scandir(directory))

    @staticmethod
    def _load_metadata():
        """Load dataset metadata from the disk."""
        if not os.path.exists(DatasetManager.METADATA_FILE):
            return []
        with open(DatasetManager.METADATA_FILE, "r") as f:
            return json.load(f)

    @staticmethod
    def _save_metadata(metadata):
        """Save dataset metadata to the disk."""
        with open(DatasetManager.METADATA_FILE, "w") as f:
            json.dump(metadata, f, indent=4)
