"""
    Load BERT Summarization model using: https://pypi.org/project/bert-extractive-summarizer/
    Paper could be found: https://arxiv.org/abs/1906.04165.

    summarizer uses hugggingface's summarization models.
"""
from . import _AbstractSummaryModel
from ..exceptions import ModelReloadException
from summarizer import Summarizer
import logging


_summarizer = None

def load():
    """
        Load bert - extract summarizer.
    """
    global _summarizer
    if _summarizer is not None:
        raise ModelReloadException(f'Try to load BERT summerizer model when there already exists loaded model.')
    
    logging.info('Loading BERT Summarizer...')
    _summarizer = Summarizer()

    logging.info('BERT Extractive Summarizer is loadded.')

def summarize_article(article):
    """
        Summarize an article.
        :param article: article to be summarized.
    """
    return _summarizer(article)