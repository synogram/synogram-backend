"""
This file contains configuration of NLP modules including path of stored model and hyperparamer
"""
import os
import datetime

ROOT = os.path.dirname(__file__)

config = {
    'Wikipedia2Vec': {
        'path': os.path.join(ROOT, 'misc', 'quick_model.txt'), # os.path.join('misc', 'enwiki_20180420_100d.txt')
        'topn': 15
    }
}