"""
    This file contains configuration of NLP modules including path of stored model and hyperparamer
"""
import os
import datetime

ROOT = os.path.dirname(__file__)

config = {
    'adapter': {},
    'model': {    
        'embedding':{
            'wiki2vec':{
                'path': os.path.join(ROOT, 'misc', 'quick_model.txt'), # os.path.join(ROOT,'misc', 'enwiki_20180420_100d.txt')
                'edge_n': 10,
                'max_score': 1000
            },
        },
    },
}