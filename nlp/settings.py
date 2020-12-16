"""
    This file contains configuration of NLP modules including path of stored model and hyperparamer
"""
import os
import datetime

ROOT = os.path.dirname(__file__)
MODEL_DIR = os.path.join(ROOT, 'misc')

config = {
    'user': {
        'sota': {
            'embedding': 'glove',
            'summarize': 'bert_extractive'
        }
    },
    'model': {    
        'embedding':{
            'wiki2vec':{
                'path': os.path.join(MODEL_DIR, 'quick_model.txt'), # os.path.join(ROOT,'misc', 'enwiki_20180420_100d.txt')
                'binary': False
            },
            'glove': {
                'url': r'http://nlp.stanford.edu/data/glove.6B.zip',
                'path': os.path.join(MODEL_DIR, 'glove.6B.50d.txt')
            }
        },
    }
}