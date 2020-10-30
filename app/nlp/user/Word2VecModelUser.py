"""
    Word2VecModelUser is entry point for user to interact with NLP package regarding word embedding tasks.
"""
from ..model import Wiki2VecModel

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

def sematic_field(word, topn=5, dim=2):
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
    neighbour = most_similar(word, topn=topn)

    # Unzip the neighbour to list of words and distances. 
    neighbour = list(zip(*neighbour))
    words, dists = neighbour[0], neighbour[1]

    # Provide reduced representation of vector
    embs = word2vec(*words, dim=dim)

    # TODO: Clean words

    return list(zip(words, dists, embs))