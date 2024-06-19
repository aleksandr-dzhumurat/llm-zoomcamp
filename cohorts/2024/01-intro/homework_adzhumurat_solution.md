https://github.com/aleksandr-dzhumurat/llm-zoomcamp/blob/main/01-intro/rag-intro.ipynb
https://www.youtube.com/watch?v=1lgbR5wMvsI


```shell
make prepare-dirs
```

```shell
docker-compose up
```

```shell
jupyter notebook . --ip 0.0.0.0 --port 8889 --NotebookApp.token='' --NotebookApp.password='' --allow-root --no-browser 
```

```shell
docker-compose up
```

```shell
pyenv virtualenv 3.10 llmops-env
```

```shell
source ~/.pyenv/versions/llmops-env/bin/activate
```

```shell
pip install --upgrade pip && pip install -r cohorts/2024/01-intro/requirements.txt
```

Drop index
```shell
curl -X DELETE "localhost:9200/course-questions"
```

Q1
version.build_hash
```json
{
  "name" : "763be84f2c79",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "RRT76JpwSNidf8kdx-5E8w",
  "version" : {
    "number" : "8.4.3",
    "build_flavor" : "default",
    "build_type" : "docker",
    "build_hash" : "42f05b9372a9a4a470db3b52817899b99a76ee73",
    "build_date" : "2022-10-04T07:17:24.662462378Z",
    "build_snapshot" : false,
    "lucene_version" : "9.3.0",
    "minimum_wire_compatibility_version" : "7.17.0",
    "minimum_index_compatibility_version" : "7.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

Q2
Which function do you use for adding your data to elastic? Index