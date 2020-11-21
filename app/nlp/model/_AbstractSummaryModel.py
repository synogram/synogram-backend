"""
    This file defines the protocol for all Summary Model.
"""

def load() -> None:
    """
        This method should load the model. 
    """
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: load.")

def summarize_article(*args) -> list:
    """
        Given article, summarizes the text.
    """
    raise NotImplementedError("All derived modules of W2VAdapterInterface must implements method: word2vec.")