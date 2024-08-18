from datetime import datetime
from typing import Dict, List, Tuple, Union

import numpy as np
from elasticsearch import Elasticsearch

if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter

from mage_ai.data_preparation.variable_manager import set_global_variable

@data_exporter
def elasticsearch(
    documents: List[Dict[str, Union[Dict, List[int], np.ndarray, str]]], *args, **kwargs,
):
    """
    Exports document data to an Elasticsearch database.
    """

    connection_string = kwargs.get('connection_string', 'http://elasticsearch:9200')
    current_time = datetime.now().strftime("%Y%m%d_%M%S")
    index_name_prefix = kwargs.get('index_name', 'documents')
    index_name = f"{index_name_prefix}_{current_time}"
    print("index name:", index_name)
    set_global_variable('Abyssal_enigma', 'index_name', index_name)
    number_of_shards = kwargs.get('number_of_shards', 1)
    number_of_replicas = kwargs.get('number_of_replicas', 0)
    vector_column_name = kwargs.get('vector_column_name', 'embedding')

    dimensions = kwargs.get('dimensions')
    if dimensions is None and len(documents) > 0:
        document = documents[0]
        dimensions = len(document.get(vector_column_name) or [])

    es_client = Elasticsearch(connection_string)

    print(f'Connecting to Elasticsearch at {connection_string}')

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
                "course": {"type": "keyword"},
                "document_id": {"type": "keyword"}
            }
        }
    }
    es_client.indices.delete(index=index_name, ignore_unavailable=True)
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name)
        print('Index created with properties:', index_settings)
        print('Embedding dimensions:', dimensions)

    print(f'Indexing {len(documents)} documents to Elasticsearch index {index_name}')
    current_doc = None
    for document in documents:
        current_doc = document
        print(f'Indexing document {document["document_id"]}')

        #if isinstance(document[vector_column_name], np.ndarray):
        #    document[vector_column_name] = document[vector_column_name].tolist()

        es_client.index(index=index_name, document=document)
    print('Last document: %s' % current_doc)