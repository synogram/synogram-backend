from abc import ABC
import logging

model_instances = {}

def register_instance(title, description, model):
    if title in model_instances.keys():
        raise ValueError(f'Model with title {title} already exists.')
    model_instances[title] = {
        'description': description, 
        'model': model
    }
    logging.info(f'Registered instance {title}')

class Model(ABC):
    def __init__(self, title='', description=''):
        self.model = None
        if not title:
            title = f'Untitled Model at {id(self)}'
        register_instance(title, description, self)

        self.title = title
        self.description = description

    def load(self):
        raise NotImplementedError()


class AbstractWordEmbed(Model):
    def __init__(self, title='', description=''):
        if not description:
            description = 'Model for word to vector embedding.'
        super().__init__(title, description)
        self.n_vocab = -1
        self.n_emb = -1
    
    def load():
        raise NotImplementedError('')
    
    def w2v(self, word):
        raise NotImplementedError('')

    def most_similar(self, word):
        """
            return {
                'words': [],
                'scores': [],    
            }
        """
        raise NotImplementedError('')


class AbstractSummarizer(Model):
    def __init__(self, title='', description=''):
        if not description:
            description = 'Model to summarize article.'
        super().__init__(title, description)

    def load():
        raise NotImplementedError('')
    
    def summarize(self, corpus):
        raise NotImplementedError('')






