from ..adapter import *

ADAPTERS = ['Wiki2Vec', 'BERT2Vec']

def load(*args):
    for m in args:            
        if m == 'Wiki2Vec':
            Wiki2VecAdapter.load()
        elif m == 'BERT2Vec':
            BERT2VecAdapter.load()


        

