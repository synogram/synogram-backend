"""
    Load BERT Summarization model using: https://pypi.org/project/bert-extractive-summarizer/
    Paper could be found: https://arxiv.org/abs/1906.04165.

    summarizer uses hugggingface's summarization models.
"""
from ..exceptions import ModelFailedLoadException
from summarizer import Summarizer
from .Model import AbstractSummarizer
from abc import ABC
import logging


class BERTExtractiveSummarizer(AbstractSummarizer):
    def __init__(self):
        super().__init__('BERTExtractiveSummarizer', 'Extractive Summarizer based on BERT pretrained model using summarizer library.')

    def load(self):
        try:
            logging.info('Loading BERTExtractiveSummarizer...')
            BERTExtractiveSummarizer.model = Summarizer()
        except Exception as e:
            raise ModelFailedLoadException(f'Failed to load BERTExtractiveSummarizer with {str(e)}')
            
    def summarize(self, corpus):
        return BERTExtractiveSummarizer.model(corpus)
