from ._W2VAdapterInterface import *
from ..model.pretrained import load_bert
from ..settings import config
from ..exceptions import ModelReloadException
from ..lib import text_util as txt
import torch


model_name = config['model']['embedding']['bert']['modelName']
n_emb = config['model']['embedding']['bert']['hidden']
n_vocab = None

_bert = None
_tokenizer = None


def load():
    global _bert, _tokenizer, n_vocab, n_emb
    if (_bert is not None) or (_tokenizer is not None):
        raise ModelReloadException(f'Try to load model: {model_name} when there already exists BERT Model.')
    _bert, _tokenizer = load_bert(model_name)

def _tokenize_to_tensor(sentence):
    tokens = _tokenizer.tokenize(sentence)
    tk_ids = _tokenizer.convert_tokens_to_ids(tokens)
    tk_ids = _tokenizer.build_inputs_with_special_tokens(tk_ids)

    tk_tensor = torch.tensor([tk_ids])
    return tk_tensor


def word2vec(*words):
    tk_tensor = _tokenize_to_tensor(' '.join(words))
    embs, _ = _bert(tk_tensor)
    return embs.squeeze().tolist()


def sent2vec(sentence):
    tk_tensor = _tokenize_to_tensor(sentence)
    _, emb = _bert(tk_tensor)
    return emb.squeeze().tolist()


def most_similar(*args):
    pass