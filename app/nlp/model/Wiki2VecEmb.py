"""
    Manage word2vec model trained by wikipedia: https://wikipedia2vec.github.io/wikipedia2vec/pretrained/.
"""

from ..exceptions import ModelFailedLoadException
from sklearn.decomposition import PCA
import gensim.models as gm
import numpy as np
import logging
from .Model import AbstractWordEmbed

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


class Wiki2VecEmbed(AbstractWordEmbed):
    def __init__(self, path='', binary=False):
        super().__init__('Wiki2VecEmbed', f'Word2Vec model pretrained on wikipedia @ {path}')
        self.path = path
        self.binary = binary
        self.n_vocab = -1
        self.n_emb = -1
    
    def load(self):
        try:
            logging.info(f'Loading Wiki2VecEmb at {self.path}')
            self.model = gm.KeyedVectors.load_word2vec_format(self.path, binary=self.binary)
            self.n_vocab = len(self.model.wv.vocab)
            self.n_emb =self.model.vector_size
        except Exception as e:
            raise ModelFailedLoadException(f'Failed to Wiki2VecEmbed at {self.path}')
    
    def w2v(self, *word):
        words = as_input_words(*word)
        return self.model[words]

    # TODO this should be done by reduce dimension lib.
    # def w2rv(self, *word, dim=2):
    #     """
    #         Reducme dimension of word embeddings. 
    #         :param embeddings: embedding vectors.
    #         :param dim: dimension to reduce to.
    #         :return: return lists of reduced dimension embeddings.
    #     """
    #     embeddings = self.w2v(*word)
    #     if len(embeddings) == 1:
    #         embeddings = np.array(embeddings).reshape(1, -1)
    #     else:
    #         embeddings = np.array(embeddings)
        
    #     assert len(embeddings.shape) == 2
    #     return PCA(dim).fit_transform(embeddings).tolist()

    def most_similar(self, *word, topn=10):
        """
            Find most similar words.
            :param *words: words to query.
            :param topn: number of expected return words.
            :return: list of tuple (word, -log of cosine similiarity). This function does not guarantee to return topn words. 
        """
        # Get most similar topn + edge_n words.
        zipped = self.model.most_similar(positive=word, topn=topn*2)
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
        query_words = as_output_words(*word)
        for q in query_words:
            if q not in unique_o_words:
                unique_o_words.insert(0, q)
                uw_dists.insert(0, 1.0)

        return unique_o_words[:topn], uw_dists[:topn]
