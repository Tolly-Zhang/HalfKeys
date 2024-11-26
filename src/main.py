import logging
from data_loaders import datasets_loader
from datetime import datetime, timedelta

def main():
    """Main function to load and verify the dataset."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    dataset_name = "openwebtext"
    dataset_dir = "./data/raw/openwebtext"

    try:
        # Load metadata
        metadata = datasets_loader.DatasetManager._load_metadata()
        entry = next((e for e in metadata if e["name"] == dataset_name), None)

        if entry and entry["date_last_verified"]:
            last_verified = datetime.fromisoformat(entry["date_last_verified"])
            time_since_verification = datetime.now() - last_verified

            if time_since_verification < timedelta(hours=1):
                logging.info("Dataset was verified less than 1 hour ago. Skipping verification.")
                return
            else:
                logging.info("More than 1 hour since last verification. Proceeding with verification.")
        else:
            logging.info("No previous verification record found. Proceeding with verification.")

        # Verify the dataset files
        datasets_loader.DatasetManager.verify_files(
            dataset_name=dataset_name,
            auto_reload=False
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()