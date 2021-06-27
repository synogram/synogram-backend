from neo4j import GraphDatabase

def load_graph_neo4j(graph, url, auth):
    with GraphDatabase.driver(url, auth=auth) as driver:
        sess = driver.session()
        for relation in graph:
            for rel in relation['relations']:
                sess.run("MERGE (n:Entity { name: $name, label: $label })", name=rel['head']['name'], label=rel['head']['label'])
                sess.run("MERGE (n:Entity { name: $name, label: $label })", name=rel['tail']['name'], label=rel['tail']['label'])
                sess.run("""
                    MATCH 
                        (h:Entity),
                        (t:Entity)
                    WHERE h.name=$h AND t.name=$t
                    MERGE (h) - [r:Relation {relation:$rel, score:$score, context: $ctx}] -> (t)
                """, 
                h=rel['head']['name'], 
                t=rel['tail']['name'],
                rel=rel['relation'],
                score=rel['score'],
                ctx=relation['sentence'])
            

    

