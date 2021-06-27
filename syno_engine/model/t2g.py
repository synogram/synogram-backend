from .base import BaseT2G, BaseRE, BaseCOREF, BaseNER
from itertools import combinations

ENTITYE_TYPES = {
    "PERSON": "People, including fictional",
    "NORP": "Nationalities or religious or political groups",
    "FACILITY": "Buildings, airports, highways, bridges, etc.",
    "FAC": "Buildings, airports, highways, bridges, etc.",
    "ORG": "Companies, agencies, institutions, etc.",
    "GPE": "Countries, cities, states",
    "LOC": "Non-GPE locations, mountain ranges, bodies of water",
    "PRODUCT": "Objects, vehicles, foods, etc. (not services)",
    "EVENT": "Named hurricanes, battles, wars, sports events, etc.",
    "WORK_OF_ART": "Titles of books, songs, etc.",
    "LAW": "Named documents made into laws.",
    "LANGUAGE": "Any named language",
    "DATE": "Absolute or relative dates or periods",
    "TIME": "Times smaller than a day",
    "PERCENT": 'Percentage, including "%"',
    "MONEY": "Monetary values, including unit",
    "QUANTITY": "Measurements, as of weight or distance",
    "ORDINAL": '"first", "second", etc.',
    "CARDINAL": "Numerals that do not fall under another type",
    # Named Entity Recognition
    # Wikipedia
    # http://www.sciencedirect.com/science/article/pii/S0004370212000276
    # https://pdfs.semanticscholar.org/5744/578cc243d92287f47448870bb426c66cc941.pdf
    "PER": "Named person or family.",
    "MISC": "Miscellaneous entities, e.g. events, nationalities, products or works of art",
    # https://github.com/ltgoslo/norne
    "EVT": "Festivals, cultural events, sports events, weather phenomena, wars, etc.",
    "PROD": "Product, i.e. artificially produced entities including speeches, radio shows, programming languages, contracts, laws and ideas",
    "DRV": "Words (and phrases?) that are dervied from a name, but not a name in themselves, e.g. 'Oslo-mannen' ('the man from Oslo')",
    "GPE_LOC": "Geo-political entity, with a locative sense, e.g. 'John lives in Spain'",
    "GPE_ORG": "Geo-political entity, with an organisation sense, e.g. 'Spain declined to meet with Belgium'"
}


class NaiveT2G(BaseT2G):
    def __init__(self, ner: BaseNER,
                 rext: BaseRE,
                 coref: BaseCOREF,
                 sent_split='. ',
                 exclude_ents=None):
        super().__init__()
        self.ner = ner
        self.rext = rext
        self.coref = coref
        self.sent_split = sent_split
        self.exclude_ents = [] if exclude_ents is None else exclude_ents

    @property
    def model(self):
        return {
            'ner': self.ner,
            'rext': self.rext,
            'coref': self.coref
        }
    
    @property
    def is_loaded(self):
        return all([(self.ner is not None), (self.rext is not None), (self.coref is not None)])

    def load_model(self):
        if not self.ner.is_loaded:
            self.ner.load_model()

        if not self.rext.is_loaded:
            self.rext.load_model()

        if not self.coref.is_loaded:
            self.coref.load_model()

        return self

    def _text2graph(self, text):
        sents = self.coref(text).split(self.sent_split)
        ents = self.ner(sents)

        valid_sents = []
        ents_comb = []

        for sent, _ents in zip(sents, ents):
            _ents_node = [e for e in _ents if e['label']
                          not in self.exclude_ents]
            if len(_ents_node) > 1:
                valid_sents.append(sent)
                ents_comb.append(list(combinations(_ents_node, 2)))

        rels = self.rext(valid_sents, ents_comb)

        return rels