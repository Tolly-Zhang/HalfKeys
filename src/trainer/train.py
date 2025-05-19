import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from math import log, exp
from data_tokenizers.openwebtext_tokenizer import OpenWebTextTokenizer
from .ngram_model import NgramModel
from datasets import load_dataset
import random
from tqdm import tqdm

class ModelTrainer:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.tokenizer = OpenWebTextTokenizer()

    def evaluate_model(self, model: NgramModel, texts: List[str]) -> float:
        """Calculate perplexity score on evaluation data"""
        total_log_prob = 0
        total_tokens = 0

        for text in texts:
            tokens = self.tokenizer.tokenize(text)
            if len(tokens) < model.n:
                continue

            for i in range(len(tokens) - model.n + 1):
                context = tuple(tokens[i:i + model.n - 1])
                next_word = tokens[i + model.n - 1]
                prob = model.get_probability(context, next_word)
                if prob > 0:  # Avoid log(0)
                    total_log_prob += log(prob)
                total_tokens += 1

        if total_tokens == 0:
            return float('inf')
        return exp(-total_log_prob / total_tokens)

    def train_ngram_model(
        self,
        n: int = 3,
        max_samples: int = 1000,
        splits: Dict[str, float] = None,
        output_path: Optional[str] = None
    ) -> Tuple[NgramModel, Dict[str, float]]:
        """Train an N-gram model on the OpenWebText dataset

        Args:
            n: Length of n-grams
            max_samples: Maximum samples to train on
            splits: Dict with train/valid/test split ratios
            output_path: Path to save model to

        Returns:
            Tuple of (trained model, metrics dict)
        """
        if splits is None:
            splits = {'train': 0.8, 'valid': 0.1, 'test': 0.1}

        # Initialize model and load dataset
        model = NgramModel(n=n)
        dataset = load_dataset("openwebtext", streaming=True)

        # Collect samples
        self.logger.info("Collecting samples...")
        samples = []
        for i, sample in enumerate(dataset["train"]):
            if i >= max_samples:
                break
            samples.append(sample["text"])

        # Split samples
        random.shuffle(samples)
        train_size = int(len(samples) * splits['train'])
        valid_size = int(len(samples) * splits['valid'])

        train_samples = samples[:train_size]
        valid_samples = samples[train_size:train_size + valid_size]
        test_samples = samples[train_size + valid_size:]

        # Train model
        self.logger.info("Training model...")
        for text in tqdm(train_samples, desc="Training"):
            tokens = self.tokenizer.tokenize(text)
            model.train(tokens)

        # Evaluate
        metrics = {
            'train_perplexity': self.evaluate_model(model, train_samples[:100]),  # Sample of train set
            'valid_perplexity': self.evaluate_model(model, valid_samples),
            'test_perplexity': self.evaluate_model(model, test_samples)
        }

        self.logger.info(f"Training metrics: {metrics}")

        # Save model
        if output_path is None:
            output_dir = Path("models")
            output_dir.mkdir(exist_ok=True)
            output_path = str(output_dir / f"ngram_{n}.pkl")

        model.save(output_path)
        self.logger.info(f"Model saved to {output_path}")

        return model, metrics
