import os
import sys
import json

import requests
import numpy as np
import pandas as pd
from elasticsearch import Elasticsearch
from tqdm.auto import tqdm

embedder = None

def get_ground_truth(ground_truth_url, local_path):
    if not os.path.exists(local_path):

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

def get_or_create_embedder(root_dir, model_name):
    global embedder
    if embedder is None:
        embedder = get_pytorch_model(root_dir, model_name)
    return embedder
def normed_vector(v):
    norm = np.sqrt((v * v).sum())
    v_norm = v / norm
    return v_norm

def normalize_embeds(embeds_input):
    embeds_normed = np.vstack([normed_vector(embeds_input[i]) for i in range(embeds_input.shape[0])])
    return embeds_normed
