from pipeline.process import t2g_clean_text
from itertools import combinations

rext = None
ner = None
coref = None

ENTITYE_TYPES =     {
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

def load():   
    global rext, ner, coref
    from model.ots import SpacyNER, OpenRE, AllenCOREF
    ner = SpacyNER().load_model()
    rext = OpenRE().load_model()
    coref = AllenCOREF().load_model()


def text2gaph(text):
    sents = t2g_clean_text(text)
    sents = coref(". ".join(sents)).split('. ')
    ents = ner(sents)

    print(ents)

    valid_sents = []
    ents_comb = []
    for sent, _ents in zip(sents, ents):
        _ents_node = [e for e in _ents if e['label'] not in (
            # filter
        )]
        if len(_ents_node) > 1:
            valid_sents.append(sent)
            ents_comb.append(list(combinations(_ents_node, 2)))
    
    rels = rext(valid_sents, ents_comb)

    return rels 



if __name__ == '__main__':
    text = """
    Barack Hussein Obama II (/bəˈrɑːk huːˈseɪn oʊˈbɑːmə/ (About this soundlisten) bə-RAHK hoo-SAYN oh-BAH-mə;[1] born August 4, 1961) is an American politician and attorney who served as the 44th president of the United States from 2009 to 2017. 
    A member of the Democratic Party, Obama was the first African-American president of the United States. 
    He previously served as a U.S. senator from Illinois from 2005 to 2008 and as an Illinois state senator from 1997 to 2004.

    Obama was born in Honolulu, Hawaii. After graduating from Columbia University in 1983, he worked as a community organizer in Chicago.
    In 1988, he enrolled in Harvard Law School, where he was the first black president of the Harvard Law Review. 
    After graduating, he became a civil rights attorney and an academic, teaching constitutional law at the University of Chicago Law School from 1992 to 2004. 
    Turning to elective politics, he represented the 13th district in the Illinois Senate from 1997 until 2004, when he ran for the U.S. Senate. 
    Obama received national attention in 2004 with his March Senate primary win, his well-received July Democratic National Convention keynote address, 
    and his landslide November election to the Senate. In 2008, he was nominated by the Democratic Party for president a year after beginning his campaign, 
    and after a close primary campaign against Hillary Clinton. Obama was elected over Republican nominee John McCain in the general election and was inaugurated 
    alongside his running mate, Joe Biden, on January 20, 2009. Nine months later, he was named the 2009 Nobel Peace Prize laureate.

    Obama signed many landmark bills into law during his first two years in office. 
    The main reforms that were passed include the Affordable Care Act (commonly referred to as ACA or "Obamacare"), 
    although without a public health insurance option, the Dodd–Frank Wall Street Reform and Consumer Protection Act, 
    and the Don't Ask, Don't Tell Repeal Act of 2010. The American Recovery and Reinvestment Act of 2009 and Tax Relief,
    Unemployment Insurance Reauthorization, and Job Creation Act of 2010 served as economic stimuli amidst the Great Recession.

    After a lengthy debate over the national debt limit, he signed the Budget Control and the American Taxpayer Relief Acts. 
    In foreign policy, he increased U.S. troop levels in Afghanistan, reduced nuclear weapons with the United States–Russia New START treaty, 
    and ended military involvement in the Iraq War. He ordered military involvement in Libya for the implementation of the UN Security Council Resolution 1973, 
    contributing to the overthrow of Muammar Gaddafi. He also ordered the military operation that resulted in the killing of Osama bin Laden.

    After winning re-election by defeating Republican opponent Mitt Romney, 
    Obama was sworn in for a second term in 2013. During this term, he promoted inclusion for LGBT Americans. 
    His administration filed briefs that urged the Supreme Court to strike down same-sex marriage bans as unconstitutional (United States v. Windsor and Obergefell v. Hodges); same-sex marriage was legalized nationwide in 2015 after the Court ruled so in Obergefell. He advocated for gun control in response to the Sandy Hook Elementary School shooting, indicating support for a ban on assault weapons, and issued wide-ranging executive actions concerning global warming and immigration. In foreign policy, he ordered successful military interventions in Iraq and Syria in response to gains made by ISIL after the 2011 withdrawal from Iraq, continued the process of ending U.S. combat operations in Afghanistan in 2016, promoted discussions that led to the 2015 Paris Agreement on global climate change, initiated sanctions against Russia following the invasion in Ukraine and again after interference in the 2016 U.S. elections, brokered the JCPOA nuclear deal with Iran, and normalized U.S. relations with Cuba. Obama nominated three justices to the Supreme Court: Sonia Sotomayor and Elena Kagan were confirmed as justices, while Merrick Garland faced partisan obstruction from the Republican-led Senate led by Mitch McConnell, which never held hearings or a vote on the nomination. Obama left office in January 2017 and continues to reside in Washington, D.C.[2][3]

    During Obama's terms in office, the United States' reputation abroad, as well as the American economy, significantly improved.[4] 
    Obama's presidency has generally been regarded favorably, and evaluations of his presidency among historians, political scientists, 
    and the general public frequently place him among the upper tier of American presidents.
    """
    print('Loading...')
    load()
    print('Creating graph...')
    relations = text2gaph(text)
    for rel in relations:
        print(rel['sentence'])
        for r in rel['relations']:
            if r['score'] > 0.5:
                print('\t', r['head']['name'], '->', r['relation'], '->', r['tail']['name'], r['score'])

    import json

    with open('example.json', 'w') as f:
        json.dump(relations, f)

