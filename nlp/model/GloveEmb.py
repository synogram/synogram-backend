from ..exceptions import ModelFailedLoadException
import pandas as pd
import csv
import numpy as np
import logging
from .Model import AbstractWordEmbed


lower = lambda words: [w.lower() for w in words]


class GloveEmbed(AbstractWordEmbed):
    def __init__(self, path=''):
        super().__init__('GloveEmbed', f'Glove Embed trained by standord')
        self.path = path
        self.vocab = [] 
        self.n_vocab = -1
        self.n_emb = -1  
    
    def load(self):
        try:
            logging.info(f'Loading Glove model at {self.path}')
            self.model = pd.read_table(self.path, sep=" ", index_col=0, header=None, quoting=csv.QUOTE_NONE)
            self.vocab = np.array(self.model.index.tolist())
            self.n_vocab = self.model.shape[0]
            self.n_emb = self.model.shape[1]
        except Exception as e:
            raise ModelFailedLoadException(f'Failed to Glove at {self.path} with {e}')
    
    def w2v(self, *word):
        return self.model.loc[list(word)].to_numpy()

    def cosine_sim(self, u, v):
        if isinstance(u, str) or (isinstance(u, list) and isinstance(u[0], str)):
            u = self.w2v(u)
        if isinstance(v, str):
            v = self.w2v(v)

        axis_u = int(len(u.shape) > 1)
        dot = np.dot(u, v)
        norm = np.sqrt(np.sum(u**2, axis=axis_u)) * np.sqrt(np.sum(v**2))
        return dot/norm

    def most_similar(self, *word, topn=10):
        if len(word) > 0:
            w = np.sum(self.w2v(*word), axis=0)
        else:
            w = self.w2v(*word)

        sim = self.cosine_sim(self.model.to_numpy(), w)
        idx = np.argpartition(sim, -topn)[-topn:]
        
        words = self.vocab[idx]
        sim = sim[idx]
        idx = np.argsort(sim)[::-1]
        return words[idx].tolist(), sim[idx].tolist()