"""
    Word2VecModelUser is entry point for user to interact with NLP package regarding word embedding tasks.
"""
from ..model import Wiki2VecModel
from itertools import cycle, islice

def load():
    """
        Load Wiki2Vec model.
    """
    Wiki2VecModel.load()

def word2vec(*words, dim=2):
    """
        Convert words into embedding than reduce dimension.
        :param words: words to be converted to vectors.
        :param dim: dimensionality of reduced vector.

    """
    embeddings = Wiki2VecModel.word2vec(*words)    
    return Wiki2VecModel.reduce_dim(*embeddings, dim=dim)

def most_similar(*words, topn=5):
    """
        Find most closely related words.
        :param words: words to find similar words
        :param topn: top nth closest words
        :return: return list of tuples (word, -log cosine distance)
    """
    return Wiki2VecModel.most_similar(*words, topn=topn)

def sematic_field(*words, topn=5, edge_n=10, dim=2):
    """
        Find most closely related words and also provide vector representation of the words.
        :param word: query word. 
        :param topn: top nth closet words.
        :param dim: vector representation of the closely related words.
        :return: return list of tuples (word, -log consine distance, vector representation).
    """
    if topn <= dim:
        raise ValueError(f'topn: {topn} value must be greater than dim: {dim}')

    # Find most similar words & distance from the query word. 
    sim_words, dists = most_similar(*words, topn=topn)

    # If by very low probability, most_similar return numbers of words less than topn, cycle.
    if len(sim_words) <= dim:
        idx = list(range(len(sim_words)))
        rp_idx = list(islice(cycle(idx), topn))
        words = [sim_words[ri] for ri in rp_idx]
        dists = [dists[ri] for ri in rp_idx]

    # Provide reduced representation of vector    
    embs = word2vec(*sim_words, dim=dim)

    return list(zip(sim_words, dists, embs))