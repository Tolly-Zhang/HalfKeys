import logging
from data_loaders import datasets_loader

def main():
    """Main function to load and verify the dataset."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    dataset_name = "openwebtext"
    dataset_dir = "./data/raw/openwebtext"

    try:
        # Initial load of the dataset
        datasets_loader.DatasetManager.initial_load(
            dataset_name=dataset_name,
            dataset_dir=dataset_dir
        )

        # Verify the dataset files
        datasets_loader.DatasetManager.verify_files(
            dataset_name=dataset_name,
            auto_reload=False
        )

    except Exception as e:
        logging.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()