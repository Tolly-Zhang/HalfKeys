from collections import defaultdict
import pickle
from typing import List, Dict, Tuple
import logging

class NgramModel:
    def __init__(self, n: int):
        """Initialize an N-gram language model

        Args:
            n: The length of n-grams to use
        """
        self.n = n
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.logger = logging.getLogger(__name__)

    def train(self, tokens: List[str]):
        """Train the model on a list of tokens

        Args:
            tokens: List of tokens to train on
        """
        for i in range(len(tokens) - self.n + 1):
            ngram = tuple(tokens[i:i + self.n])
            context = tuple(tokens[i:i + self.n - 1])
            self.ngram_counts[ngram] += 1
            self.context_counts[context] += 1

    def get_probability(self, context: Tuple[str], next_word: str) -> float:
        """Get probability of next_word given context

        Args:
            context: Tuple of previous words
            next_word: The word to predict

        Returns:
            Probability between 0 and 1
        """
        ngram = context + (next_word,)
        if self.context_counts[context] == 0:
            return 0.0
        return self.ngram_counts[ngram] / self.context_counts[context]

    def predict_next(self, context: List[str], k: int = 5) -> List[Tuple[str, float]]:
        """Predict the k most likely next words given a context

        Args:
            context: List of previous words
            k: Number of predictions to return

        Returns:
            List of (word, probability) tuples, sorted by probability
        """
        if len(context) != self.n - 1:
            context = context[-(self.n-1):]

        context_tuple = tuple(context)
        candidates = []

        for ngram in self.ngram_counts:
            if ngram[:-1] == context_tuple:
                prob = self.get_probability(context_tuple, ngram[-1])
                candidates.append((ngram[-1], prob))

        return sorted(candidates, key=lambda x: x[1], reverse=True)[:k]

    def save(self, filepath: str):
        """Save model to file

        Args:
            filepath: Path to save the model to
        """
        with open(filepath, 'wb') as f:
            pickle.dump((self.ngram_counts, self.context_counts), f)
        self.logger.info(f"Model saved to {filepath}")

    def load(self, filepath: str):
        """Load model from file

        Args:
            filepath: Path to load the model from
        """
        with open(filepath, 'rb') as f:
            self.ngram_counts, self.context_counts = pickle.load(f)
        self.logger.info(f"Model loaded from {filepath}")
