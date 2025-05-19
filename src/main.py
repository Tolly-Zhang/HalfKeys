import sys
import os
import logging
from pathlib import Path
from data_loaders.datasets_loader import DatasetManager
from trainer import ModelTrainer

def setup_logging():
    """Configure logging with more detailed format"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

def main():
    """Main function to verify dataset and train model."""
    setup_logging()
    logger = logging.getLogger(__name__)

    # Verify dataset is available
    dataset_name = "openwebtext"
    DatasetManager.verify_files(dataset_name)

    # Train model with quick testing settings
    trainer = ModelTrainer()
    model, metrics = trainer.train_ngram_model(
        n=3,                     # Trigram model
        max_samples=1000,        # Quick test with 1000 samples
        splits={'train': 0.8, 'valid': 0.1, 'test': 0.1}
    )

    # Test the model
    logger.info("\nTesting model predictions:")
    test_contexts = [
        ["the", "quick"],
        ["i", "am"],
        ["to", "be"],
    ]

    for context in test_contexts:
        predictions = model.predict_next(context, k=3)
        logger.info(f"\nContext: {' '.join(context)}")
        for word, prob in predictions:
            logger.info(f"  → {word}: {prob:.4f}")

if __name__ == "__main__":
    main()
