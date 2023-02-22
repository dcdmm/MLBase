### docker单机部署

```shell
docker pull docker.elastic.co/elasticsearch/elasticsearch:7.11.2 
docker pull docker.elastic.co/kibana/kibana:7.11.2

docker run -d -p 9200:9200 -p 9300:9300 -e "discovery.type=single-node" docker.elastic.co/elasticsearch/elasticsearch:7.11.2

docker run --link 1428f0fc63ce(即easticsearch容器id):elasticsearch -d -p 5601:5601 docker.elastic.co/kibana/kibana:7.11.2
```