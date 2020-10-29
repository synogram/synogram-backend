import nlp
import logging

# logging.basicConfig(filename='log/test.log', level=logging.INFO)

nlp.Word2VecModelUser.load('Wiki2Vec')
nlp.Word2VecModelUser.load('BERT2Vec')

import numpy as np

wiki2vec_adapter = nlp.adapter.Wiki2VecAdapter
print(wiki2vec_adapter.n_vocab, wiki2vec_adapter.n_embed)
print(wiki2vec_adapter.most_similar('and', 'of'))


bert2vec_adapter = nlp.adapter.BERT2VecAdapter

import numpy as np
print(bert2vec_adapter.n_vocab, bert2vec_adapter.n_emb)
print(np.array(bert2vec_adapter.sent2vec('My name is john')).shape)
print(np.array(bert2vec_adapter.word2vec('test', 'wOrd', 'cpu')).shape)



