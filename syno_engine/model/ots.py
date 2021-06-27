from .base import BaseCOREF, BaseNER, BaseRE

class SpacyNER(BaseNER):
    def __init__(self, model_name='en_core_web_sm'):
        super().__init__()
        self.model_name = model_name
        self._model = None

    def load_model(self):
        import spacy
        self._model = spacy.load(self.model_name)
        return self

    @property
    def model(self):
        return self._model

    def _extract_entities(self, sent):
        """
        Args
            sent [str, list]: source sentences.
        Return
            [{
                'name': entity text, 
                'start': entity index start,
                'end': entity index end,
                'label': entity pos label.
            }]
        """
        sents = []
        ents = []
        if isinstance(sent, str):
            sents = [sent]

        if isinstance(sent, list):
            sents = sent

        for _sent in sents:
            doc = self._model(_sent)

            _ents = []
            for _e in doc.ents:
                _ents.append({
                    'name': _e.text,
                    'start': _e.start,
                    'end': _e.end,
                    'label': _e.label_
                })
            
            ents.append(_ents)

        return ents
    

class OpenRE(BaseRE):
    def __init__(self, model_name='wiki80_cnn_softmax') -> None:
        super().__init__()
        self.model_name = model_name
        self._model = None

    def load_model(self):
        import opennre
        self._model = opennre.get_model(self.model_name)
        return self

    @property
    def model(self):
        return self._model
    
    def _extract_relations(self, sent, ent_pair):
        """
        Args:
            sent [str, list]: source sentences. 
            ent_pair [tuple, list[tuple]]: tuples of two entity (head, tail)
        Returns: 
            relations [list]
            [
                {
                    'sentence': sentence,
                    'relations': [
                        # Per sentences -> n relations
                        {
                            'head': head entity, 
                            'tail': tail entity,
                            'relation': string of relation text,
                            'score': score of the relation.
                        },...
                    ]
                }, ...
            ]
        """
        sents = []
        ent_pairs = []
        relations = []
        if isinstance(sent, str):
            assert isinstance(ent_pair, tuple(dict))
            sents = [sent]
            ent_pairs = [ent_pair]

        if isinstance(sent, list):
            assert isinstance(ent_pair, list)
            sents = sent
            ent_pairs = ent_pair

        assert len(sent) == len(ent_pair)
        
        for _sent, _ent_pairs in zip(sents, ent_pairs):
            _relations = []
            for pair in _ent_pairs:
                item = {
                    'text': _sent,
                    'h': { 'pos': (pair[0]['start'], pair[0]['end']) },
                    't': { 'pos': (pair[1]['start'], pair[1]['end']) }
                }
                _rel = self._model.infer(item)
                _relations.append({
                    'head': pair[0],
                    'tail': pair[1],
                    'relation': _rel[0],
                    'score': _rel[1]
                })
            relations.append({
                'sentence': _sent, 
                'relations': _relations
            })

        return relations

class AllenCOREF(BaseCOREF):
    def __init__(self, model_url="https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2020.02.27.tar.gz") -> None:
        super().__init__()
        self.model_url = model_url
        self._model = None

    @property
    def model(self):
        return self._model

    def load_model(self):
        from allennlp.predictors.predictor import Predictor
        self._model = Predictor.from_path(self.model_url)
        return self

    def _resolve_coref(self, text):
        return self._model.coref_resolved(text)

    @property
    def model(self):
        return self._model



