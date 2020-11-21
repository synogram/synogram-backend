"""
    Word2VecModelUser is entry point for user to interact with NLP package regarding word embedding tasks.
"""
from .. import model as md_pool
from ..settings import config
from itertools import cycle, islice

sota = config['user']['sota']['embedding']
_model = None

def load():
    global _model
    if sota == 'wiki2vec':
        model_path = config['model']['embedding']['wiki2vec']['path']
        binary = config['model']['embedding']['wiki2vec']['binary']
        _model = md_pool.Wiki2VecEmbed(path=model_path, binary=binary)
        _model.load()
    elif sota == "glove":
        model_path = config['model']['embedding']['glove']['path']
        _model = md_pool.GloveEmbed(path=model_path)
        _model.load()
    else:
        raise Exception()

def w2v(*word):
    try:
        return _model.w2v(*word)
    except Exception as e:
        raise e
        return None

def most_similar(*word):
    try:
        return _model.most_similar(*word)
    except Exception as e:
        # TODO:
        raise e
        return None

# def sematic_field(*word, topn=5, dim=2):
#     """
#         Find most closely related words and also provide vector representation of the words.
#         :param word: query word. 
#         :param topn: top nth closet words.
#         :param dim: vector representation of the closely related words.
#         :return: return list of tuples (word, -log consine distance, vector representation).
#     """
#     if topn <= dim:
#         raise ValueError(f'topn: {topn} value must be greater than dim: {dim}')

#     # Find most similar words & distance from the query word. 
#     sim_words, dists = _model.most_similar(*words, topn=topn)

#     # If by very low probability, most_similar return numbers of words less than topn, cycle.
#     if len(sim_words) <= dim:
#         idx = list(range(len(sim_words)))
#         rp_idx = list(islice(cycle(idx), topn))
#         words = [sim_words[ri] for ri in rp_idx]
#         dists = [dists[ri] for ri in rp_idx]

#     # Provide reduced representation of vector    
#     embs = _word2vec(*sim_words, dim=dim)

#     return list(zip(sim_words, dists, embs))