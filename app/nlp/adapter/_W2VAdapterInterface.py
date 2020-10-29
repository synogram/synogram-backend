import numpy as np

def load() -> None:
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: load.")


def word2vec(*args) -> list:
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: word2vec.")


def most_similar(*args) -> list:
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implments method: word2vec.")