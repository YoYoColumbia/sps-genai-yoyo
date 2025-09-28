from __future__ import annotations

from collections import defaultdict, Counter
import random
import re
from typing import Dict, Iterable, List, Tuple, Optional


# -------------------------
# Module 2 notebook-style functions (as provided)
# -------------------------

def simple_tokenizer(text: str, frequency_threshold: Optional[int] = 5) -> List[str]:
    """Simple tokenizer that splits text into words."""
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not frequency_threshold:
        return tokens
    word_counts = Counter(tokens)
    filtered_tokens = [t for t in tokens if word_counts[t] >= frequency_threshold]
    return filtered_tokens


def analyze_bigrams(text: str, frequency_threshold: Optional[int] = None):
    """Analyze text to compute bigram probabilities."""
    words = simple_tokenizer(text, frequency_threshold)
    bigrams = list(zip(words[:-1], words[1:]))
    bigram_counts = Counter(bigrams)
    unigram_counts = Counter(words)

    bigram_probs = defaultdict(dict)
    for (word1, word2), count in bigram_counts.items():
        bigram_probs[word1][word2] = count / unigram_counts[word1]

    # Return vocabulary as it would appear in the notebook
    return list(unigram_counts.keys()), bigram_probs


def generate_text_from_probs(bigram_probs: Dict[str, Dict[str, float]],
                             start_word: str,
                             num_words: int = 20) -> str:
    """Generate text based on bigram probabilities."""
    current_word = start_word.lower()
    generated_words = [current_word]

    for _ in range(max(0, num_words - 1)):
        next_words = bigram_probs.get(current_word)
        if not next_words:
            break
        next_word = random.choices(list(next_words.keys()), weights=next_words.values())[0]
        generated_words.append(next_word)
        current_word = next_word

    return " ".join(generated_words)


def print_bigram_probs_matrix_python(vocab: List[str],
                                     bigram_probs: Dict[str, Dict[str, float]]) -> None:
    """Pretty-print bigram probabilities in a matrix format (console output)."""
    print(f"{'':<15}", end="")
    for word in vocab:
        print(f"{word:<15}", end="")
    print("\n" + "-" * (15 * (len(vocab) + 1)))
    for word1 in vocab:
        print(f"{word1:<15}", end="")
        for word2 in vocab:
            prob = bigram_probs.get(word1, {}).get(word2, 0)
            print(f"{prob:<15.2f}", end="")
        print()


# -------------------------
# API-friendly class wrapper
# -------------------------

class BigramModel:
    """
    Thin wrapper around the Module 2 helpers so they can be used in an API.

    Parameters
    ----------
    corpus : Iterable[str] | str
        Either a list of strings or a single string of text.
    frequency_threshold : Optional[int]
        Passed to `simple_tokenizer` (Module 2 logic). If None/0, keeps all tokens.
    seed : Optional[int]
        RNG seed for reproducible `generate_text`.
    """

    def __init__(self,
                 corpus: Iterable[str] | str,
                 frequency_threshold: Optional[int] = None,
                 seed: Optional[int] = 42) -> None:
        if isinstance(corpus, str):
            text = corpus
        else:
            text = " ".join(corpus)

        if seed is not None:
            random.seed(seed)

        # Build notebook-style structures
        self.vocab, self.bigram_probs = analyze_bigrams(text, frequency_threshold)
        # For optional introspection/debugging:
        self.frequency_threshold = frequency_threshold

    # Notebook-parity helpers exposed via class
    def vocabulary(self) -> List[str]:
        return list(self.vocab)

    def next_word_probabilities(self, word: str) -> Dict[str, float]:
        return dict(self.bigram_probs.get(word.lower(), {}))

    def generate_text(self, start_word: str, length: int = 20) -> str:
        return generate_text_from_probs(self.bigram_probs, start_word, num_words=length)
