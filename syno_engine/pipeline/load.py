from json import load
from neo4j import GraphDatabase

url = "bolt://192.168.99.100:7687/"

def load_knowledge_graph(kgraph):
    with GraphDatabase.driver(url, auth=("neo4j", "test1234")) as driver:
        sess = driver.session()

        for relation in kgraph:
            sess.run("CREATE (n:Sentence { sentence: $s })", s=relation['sentence'])

            for rel in relation['relations']:
                sess.run("MERGE (n:Entity { name: $name, label: $label })", name=rel['head']['name'], label=rel['head']['label'])
                sess.run("MERGE (n:Entity { name: $name, label: $label })", name=rel['tail']['name'], label=rel['tail']['label'])
                sess.run("""
                    MATCH 
                        (h:Entity),
                        (t:Entity)
                    WHERE h.name=$h AND t.name=$t
                    MERGE (h) - [r:Relation {relation:$rel, score:$score}] -> (t)
                """, 
                h=rel['head']['name'], 
                t=rel['tail']['name'],
                rel=rel['relation'],
                score=rel['score'])


if __name__ == '__main__':
    import json 

    with open('example.json') as f:
        kgraph = json.load(f)

    load_knowledge_graph(kgraph)

