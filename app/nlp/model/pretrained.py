"""
This files is a libraries to load different embedding pretrained model. 
Wiki2Vec: https://wikipedia2vec.github.io/wikipedia2vec/
BERT: https://github.com/huggingface/transformers
"""
import torch
import torch.nn as nn
import gensim.models as gm


def load_gensim(path):
    return gm.KeyedVectors.load_word2vec_format(path, binary=False)


def load_genism_as_torch(path):
    model = gm.KeyedVectors.load_word2vec_format(path, binary=False)
    weight = torch.FloatTensor(model.vectors)
    return nn.Embedding.from_pretrained(weight)


def load_bert(model_name):
    tokenizer = torch.hub.load('huggingface/pytorch-transformers', 'tokenizer', model_name) 
    model = torch.hub.load('huggingface/pytorch-transformers', 'model', model_name)  
    return model, tokenizer