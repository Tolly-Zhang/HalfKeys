import logging
from data_loaders import datasets_loader

def main():
    """Main function to fetch status and verify the dataset."""
    # Configure logging
    logging.basicConfig(level=logging.INFO)

    # Fetch and print status of all datasets
    datasets_loader.DatasetManager.fetch_status()

    # Verify the dataset files for a specific dataset
    dataset_name = "openwebtext"
    datasets_loader.DatasetManager.verify_files(dataset_name)

if __name__ == "__main__":
    main()