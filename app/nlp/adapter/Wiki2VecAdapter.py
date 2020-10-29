from ._W2VAdapterInterface import *
from ..model.pretrained import load_gensim
from ..settings  import config
from ..exceptions import ModelReloadException
from ..lib import text_util as txt

import numpy as np

path = config['model']['embedding']['wiki2vec']['path']

_wiki2vec = None

n_vocab = None
n_embed = None


def load():
    global _wiki2vec, n_vocab, n_embed, max_n, min_n
    if _wiki2vec is not None:
        raise ModelReloadException(f'Try to load model from {path} when there already exists Wiki2Vec Model.')

    _wiki2vec = load_gensim(path)    
    n_vocab = len(_wiki2vec.wv.vocab)
    n_embed = _wiki2vec.vector_size


def word2vec(*words) -> list:
    words = txt.clean_words(*words)
    return _wiki2vec[words]


def most_similar(*words, topn=5) -> list:
    zipped = _wiki2vec.most_similar(positive=words, topn=topn)
    words, dists = list(zip(*zipped))
    dists = -np.log(dists)
    return list(zip(words, dists.tolist()))

