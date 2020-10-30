"""
    SummaryModelUser is the entry point for users to interacts with NLP package regarding summarization task.
"""
from ..model import BERTExtractiveModel

def load():
    """  
        Load BERT Extractive Model. If the model is not cached, it will download new model ~ 1.5 GB
    """
    BERTExtractiveModel.load()

def summarize(article):
    """
        Extractive Summary (Pull most important sentences from given article).
        :param article: string of text to be summarized.
    """
    return BERTExtractiveModel.summarize_article(article)
        

