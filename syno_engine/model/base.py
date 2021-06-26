from abc import ABC, abstractmethod, abstractproperty

class BaseModel(ABC):
    def __init__(self) -> None:
        super().__init__()
    
    @abstractproperty
    def model(self):
        return None
    
    @abstractproperty
    def load_model(self):
        return self

    @property
    def is_loaded(self):
        return self.model is not None

    def __call__(self, *args, **kwds):
        assert self.is_loaded, "Model is not loaded"
        return super().__call__(*args, **kwds)

class BaseNER(BaseModel):
    def __init__(self):
        super().__init__()

    @abstractmethod    
    def _extract_entities(self, sent):
        pass

    def __call__(self, sent):
        return self._extract_entities(sent)

    
class BaseRE(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _extract_relations(self, sent, ent_pair):
        pass

    def __call__(self, sent, ent_pair):
        return self._extract_relations(sent, ent_pair)


class BaseCOREF(BaseModel):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def _resolve_coref(self, text):
        pass

    def __call__(self, text):
        return self._resolve_coref(text)


    
