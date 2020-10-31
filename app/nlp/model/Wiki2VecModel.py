"""
    Manage word2vec model trained by wikipedia: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/.
"""
from ..settings  import config
from ..exceptions import ModelReloadException
from sklearn.decomposition import PCA
import gensim.models as gm
import numpy as np
import logging


########################################################################################################################################
# Config & Load
########################################################################################################################################
path = config['model']['embedding']['wiki2vec']['path']
edge_n = config['model']['embedding']['wiki2vec']['edge_n']
max_score = config['model']['embedding']['wiki2vec']['max_score']
_wiki2vec = None
n_vocab = None
n_embed = None

def load():
    """
        Load wiki2vec model from misc files.
    """
    global _wiki2vec, n_vocab, n_embed, max_n, min_n
    if _wiki2vec is not None:
        raise ModelReloadException(f'Try to load model from {path} when there already exists Wiki2Vec Model.')

    logging.info(f'Loading Wiki2Vec from {path}...')
    _wiki2vec = gm.KeyedVectors.load_word2vec_format(path, binary=False)   
    logging.info('Wik2Vec Model is loaded.')

    n_vocab = len(_wiki2vec.wv.vocab)
    n_embed = _wiki2vec.vector_size


########################################################################################################################################
# Helper
########################################################################################################################################
lower = lambda words: [w.lower() for w in words]

def as_input_words(*words):
    # TODO: Better lemmenization and word cleaning.
    return lower(words)

def as_output_words(*words):
    o_words = []
    for w in words:
        o_w = w.replace('ENTITY/', '')
        o_w = o_w.replace('_', ' ')
        o_w = o_w.title()
        o_words.append(o_w)

    return o_words

########################################################################################################################################
# Functionality
########################################################################################################################################
def word2vec(*words) -> list:
    """
        Convert words to vectors.
        :param *words: words to convert to vector representation.
        :return: list of embedding.
    """
    words = as_input_words(*words)
    return _wiki2vec[words]

def most_similar(*words, topn=5) -> list:
    """
        Find most similar words.
        :param *words: words to query.
        :param topn: number of expected return words.
        :param edge_n: the method will look for topn+edge_n similar word in case some words are duplicated.
        :return: list of tuple (word, -log of cosine similiarity). This function does not guarantee to return topn words. 
    """
    # Get most similar topn + edge_n words.
    zipped = _wiki2vec.most_similar(positive=words, topn=topn+edge_n)
    zipped = np.array(zipped)

    # Process the outputs words.
    o_words = np.array(as_output_words(*zipped[:, 0]))
    unique_o_words = np.unique(o_words)
    dists =  zipped[:, 1].astype(float)
    uw_dists = []   

    # If there are duplicated words i.e. ENTITY/Obama & obama, get maximum score.
    for uw in unique_o_words:
        mask = o_words == uw
        uw_dists.append(-np.log(dists[mask].max()))
    
    unique_o_words = unique_o_words.tolist()

    # Append the query words with max score.
    query_words = as_output_words(*words)
    for q in query_words:
        if q not in unique_o_words:
            unique_o_words.insert(0, q)
            uw_dists.insert(0, max_score)

    return unique_o_words[:topn], uw_dists[:topn]

def reduce_dim(*embeddings, dim=2):
    """
        Reducme dimension of word embeddings. 
        :param embeddings: embedding vectors.
        :param dim: dimension to reduce to.
        :return: return lists of reduced dimension embeddings.
    """
    if len(embeddings) == 1:
        embeddings = np.array(embeddings).reshape(1, -1)
    else:
        embeddings = np.array(embeddings)
    
    assert len(embeddings.shape) == 2
    # TODO: implement PCA rather than using sklearn.
    return PCA(dim).fit_transform(embeddings).tolist()

