import os
import sys
import json

import requests
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

embedder = None

def get_ground_truth(local_path):
    if not os.path.exists(local_path):
        base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
        relative_url = '03-vector-search/eval/ground-truth-data.csv'
        ground_truth_url = f'{base_url}/{relative_url}?raw=1'

        df_ground_truth = pd.read_csv(ground_truth_url)
        df_ground_truth = df_ground_truth[df_ground_truth.course == 'machine-learning-zoomcamp']
        df_ground_truth.to_csv(local_path, index=False)
    else:
        df_ground_truth = pd.read_csv(local_path)
    ground_truth = df_ground_truth.to_dict(orient='records')
    return ground_truth

def get_corpus(data_dir):
    base_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main'
    relative_url = '03-vector-search/eval/documents-with-ids.json'
    docs_url = f'{base_url}/{relative_url}?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    json_path = os.path.join(data_dir, 'documents.json')
    if not os.path.exists(json_path):
        documents = []

        for course in documents_raw:
            if course['course'] == 'machine-learning-zoomcamp':
                documents.append(course)
        with open(json_path, 'w') as json_file:
            json.dump(documents, json_file)
        print('File saved, num rows %d' % len(documents))
    else:
        with open(json_path, 'r') as json_file:
            documents = json.load(json_file)
    return documents

def get_or_create_embedder(root_dir):
    global embedder
    if embedder is None:
        embedder = get_pytorch_model(root_dir)
    return embedder

def train_embeds(corpus_texts, root_dir, embeds_file_name, overwrite=False):
    embedder = get_or_create_embedder(root_dir)
    sentence_embedding_path = os.path.join(root_dir, 'models', embeds_file_name)
    if os.path.exists(sentence_embedding_path) and not overwrite:
        print('corpus loading from %s' % sentence_embedding_path)
        passage_embeddings = np.load(sentence_embedding_path)
    else:
        passage_embeddings = embedder.encode(corpus_texts, show_progress_bar=True)
        passage_embeddings = np.array([embedding for embedding in passage_embeddings]).astype("float32")
        with open(sentence_embedding_path, 'wb') as f:
            np.save(f, passage_embeddings)
        print('corpus saved to %s' % sentence_embedding_path)
    return passage_embeddings

def get_pytorch_model(root_dir, model_name='multi-qa-distilbert-cos-v1'):
  from sentence_transformers import SentenceTransformer

  models_dir = os.path.join(root_dir, 'models')
  if not os.path.exists(models_dir):
      os.mkdir(models_dir)
  model_path = os.path.join(models_dir, model_name)

  if not os.path.exists(model_path):
      print('huggingface model loading...')
      embedder = SentenceTransformer(model_name)
      embedder.save(model_path)
  else:
      print('pretrained model loading...')
      embedder = SentenceTransformer(model_name_or_path=model_path)
  print('model loadind done')

  return embedder

def top_similar(docs, sims, top=10):
  top_similar_idx = [int(i) for i in np.argsort(-sims)][:top]
  return [{'score': sims[i], 'text': docs[i]['text']} for i in top_similar_idx]

class VectorSearchEngine:
    def __init__(self, documents, embeddings):
        self.documents = documents
        self.embeddings = embeddings

    def search(self, v_query, num_results=10):
        scores = self.embeddings.dot(v_query)
        idx = np.argsort(-scores)[:num_results]
        return [{'score': scores[i], 'doc': self.documents[i]} for i in idx]

class ElasticSearchEngine:
    def __init__(self, es_client):
        self.es_client = es_client

    def search(self, v_query, num_results=10):
        query = {
            "field": "text_vector",
            "query_vector": v_query,
            "k": 5,
            "num_candidates": 10000, 
        }
        index_name = "course-questions"

        res = self.es_client.search(index=index_name, knn=query, source=["text", "section", "question", "course", "id"])
        res = [{'score': hit['_score'], 'doc': hit['_source']} for hit in res["hits"]["hits"][:num_results]]

        return res

def eval_search(search_index, ground_truth, ground_truth_embeds, num_results=5):
    hits = []
    for ind, doc in enumerate(ground_truth):
        q  = ground_truth_embeds[ind,:]
        truth = doc['document']
        hits.append(1 if len([i['doc'] for i in search_index.search(q, num_results) if i['doc']['id'] == truth]) > 0 else 0)
    hit_rate = sum(hits) / len(hits)
    return hit_rate

def insert_docs(es_client, documents, index_name):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "section": {"type": "text"},
                "question": {"type": "text"},
                "course": {"type": "keyword"} ,
                "id": {"type": "text"},
                "text_vector": {"type": "dense_vector", "dims": 768, "index": True, "similarity": "cosine"},
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_settings)

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)
    print('Index created')

def build_es_index(es_client, docs, index_name):
    print(es_client.info())

    requests.delete(f'http://{elastic_host}:9200/course-questions').json()
    print('Fields: %s' % docs[0].keys())
    insert_docs(es_client, docs, index_name)

def prepare_operations(docs, root_dir, X):
    json_path = os.path.join(root_dir, 'documents_vectors.json')
    if not os.path.exists(json_path):
        print('preparing vectors')
        operations = []
        for ind, doc in enumerate(docs):
            # Transforming the title into an embedding using the model
            doc["text_vector"] = X[ind,:].tolist()
            operations.append(doc)
        with open(json_path, 'w') as json_file:
            json.dump(operations, json_file)
    else:
        print('loading from json')
        with open(json_path, 'r') as json_file:
            operations = json.load(json_file)
    return operations


if __name__ == '__main__':
    root_dir = sys.argv[1]
    elastic_host = os.getenv('ELASTIC_HOST', 'localhost')
    index_name = "course-questions"

    es_client = Elasticsearch(f'http://{elastic_host}:9200')

    queries = [
        "I just discovered the course. Can I still join it?",
    ]

    embedder = get_or_create_embedder(root_dir)
    v = embedder.encode(queries, show_progress_bar=True)[0]
    print(f'Q1 First coord: {v[0]:.4f}')

    docs = get_corpus(root_dir)
    print(f'num docs after filtering: {len(docs)}')

    corpus_texts = [f"{i['question']} {i['text']}" for i in docs]
    embeds_path = os.path.join(root_dir, )
    X = train_embeds(corpus_texts, root_dir, embeds_file_name='embeds.npy', overwrite=False)
    print(f'Q2: shape of embeddings {X.shape}, Num docs {len(docs)}')
    scores = X.dot(v)
    top_similar_content = top_similar(docs, scores, top=2)[0]
    print('Q3: highest score for dot product %.4f' % top_similar_content['score'])

    # Hit rate = 0.9399
    search_index = VectorSearchEngine(documents=docs, embeddings=X)
    ground_truth = get_ground_truth(os.path.join(root_dir, 'models', 'ground_truth.csv'))
    corpus_texts = [i['question'] for i in ground_truth]
    ground_truth_embeds = train_embeds(corpus_texts, root_dir, embeds_file_name='ground_truth.npy', overwrite=False)
    hit_rate = eval_search(search_index, ground_truth, ground_truth_embeds)
    print('Q4: hit rate dot product %.4f' % hit_rate)

    operations = prepare_operations(docs, root_dir, X)
    build_es_index(es_client, operations, index_name)
    search_es_index = ElasticSearchEngine(es_client)
    search_term = queries[0]
    vector_search_term = embedder.encode(search_term)
    # SCORE: 0.825
    res = search_es_index.search(vector_search_term)[0]
    print('Q5: highest score %.4f, doc_id: %s' % (res['score'], res['doc']['id']))

    hit_rate = eval_search(search_es_index, ground_truth, ground_truth_embeds)
    print('Q6: hit rate elastic %.4f' % hit_rate)