eval "$(docker-machine env default)"

ROOT=$HOME
CONTAINER_NAME="synogram_db"

echo Removing $CONTAINER_NAME if exists..
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME

echo $ROOT

docker run \
    --name $CONTAINER_NAME \
    --publish 7474:7474 \
    --publish 7687:7687 \
    neo4j
    # -v $ROOT/neo4j/data:/data \
    # -v $ROOT/neo4j/logs:/logs \
    # -v $ROOT/neo4j/import:/var/lib/neo4j/import \
    # -v $ROOT/neo4j/plugins:/plugins \
    # --env NEO4J_AUTH=neo4j/test \
    