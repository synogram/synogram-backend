"""
    SummaryModelUser is the entry point for users to interacts with NLP package regarding summarization task.
"""
from .. import model as md_pool
from ..settings import config

sota = config['user']['sota']['summarize']
_model = None

def load():
    """  
        Load BERT Extractive Model. If the model is not cached, it will download new model ~ 1.5 GB
    """
    global _model
    if sota == 'bert_extractive':
        _model = md_pool.BERTExtractiveSummarizer()
        _model.load()

def summarize(article):
    """
        Extractive Summary (Pull most important sentences from given article).
        :param article: string of text to be summarized.
    """
    return _model.summarize(article)
        

