"""
Wikipedia2Vec are Word2Vec model that is trained by Wikipedia. This files load the wiki2vec.txt and allows and relevant methods. 
https://wikipedia2vec.github.io/wikipedia2vec/
"""
import time
import logging
import gensim.models as gm
import numpy as np
from ..config import config
from sklearn.decomposition import PCA


# Module Attributes.
_model = None
_logger = logging.getLogger(__name__)

# Configuration and hyperparameters.
path = config['Wikipedia2Vec']['path']
topn = config['Wikipedia2Vec']['topn']


def init():
    """
        Load and initialize module.
    """
    global _model
    if _model:
        _logger.error('Model is already loaded')
    else:
        _logger.info(f'Loading model:{path}...')
        try:
            timestamp = time.time()
            _model = gm.KeyedVectors.load_word2vec_format(path, binary=False)
            _logger.info(f'Finished loading model:{path} ({(timestamp - time.time()):.0f}s)')
        except Exception as e:
            _logger.error(f'Exception occured while loading Wikipedia2Vec\nError message: {e}')


def embed_words(words):
    """
        Get word embedding from Wikipedia2Vec model.
        Args:
            words: str of list of str words
        Returns:
            numpy array of word(s) embedding/

    """
    return _model[words]


def get_vocabs():
    """
        All vocaburaty of the Wikipeida2Vec model
        Returns:
            set of string.
    """
    return set(_model.wv.vocab.keys())


def _sematic_field(query, level=2, topn=topn):
    """
        Find neighbours and consine of distances between all association between neighbours.
        Args:
            query: query word.
            level: level of the search. Default 2.
            topn: topn per each target.

        Returns:
            2d matrix graph of numpy array of level'th neighbours, and dictionary of words and index mapping. 
    """
    graph = {}
    words = [query]
    targets = [(query, 1)]
    for lev in range(level):
        next_targets = []
        for target in targets:
            # For each target in level, find neighbours.
            nbs =  _model.most_similar(query, topn=topn)

            # Store the consine distance of word embedding.
            for w, d in nbs:
                graph[(target[0], w)] = d
                words.append(w)

            # Update targets
            next_targets.extend(nbs.copy())            
        targets = next_targets
    
    # Build word -> index mapping. 
    V = {}
    for i, w in enumerate(set(words)):
        V[w] = i

    # Convert graph dictionary to numpy matrix.         
    matrix_graph = np.zeros(shape=(len(V), len(V)))
    for (w_x, w_y), d in graph.items():
        matrix_graph[V[w_x], V[w_y]] = d
        
    return matrix_graph, V


def build_web(query, dim=2):
    """
        Build web (dimensionality reducted points of word embedding of neighbour words)
        Args:
            query: query word
            dim: dimension of the coordinates.
        Returns:
            dictionary of information need to build the web.
    """
    # Format sematic field.
    graph, V_map = _sematic_field(query)
    graph = (graph - graph.min()) / (graph.max() - graph.min())
    V, idx = list(zip(*(V_map.items())))

    # Get word embedding and dimensionality reduction using PCA. 
    emb = _model[V]
    emb_dim = PCA(dim).fit_transform(emb) # TODO: Build own PCA

    # Target words are not added in neigbours for convienience.
    target = emb_dim[V_map[query]]
    mask = np.ones(shape=emb_dim.shape[0]).astype(bool)
    mask[V_map[query]] = False    

    # Format coordinates of the reduced embedding.
    coord = {}
    for v, idx in V_map.items():
        if v != query:
            coord[v] = emb_dim[idx].tolist()

    # Build web as dictionary. 
    web = {
        query: target.tolist(),
        'neighbors': coord,
        'web': graph.tolist(),
        'dictionary': V_map
    }

    return web