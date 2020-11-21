"""
    This file defines the protocol for all words to embedding model.
"""

def load() -> None:
    """
        Load word to vec models.
    """
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: load.")

def word2vec(*args) -> list:
    """
        Given words, convert to vector representation.
    """
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: word2vec.")

def most_similar(*args) -> list:
    """
        Given words, find similar words.
    """
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implments method: word2vec.")