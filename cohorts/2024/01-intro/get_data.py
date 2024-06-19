import requests
import json
import os

from elasticsearch import Elasticsearch, BadRequestError
from tqdm.auto import tqdm
from openai import OpenAI
import tiktoken


client = OpenAI()

def search(query):
    hits = elastic_search(es_client, query, filter=1)
    return [i['_source'] for i in hits]

def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.
    
    QUESTION: {question}
    
    CONTEXT:
    {context}
    """.strip()

    context_template = """
    Q: {question}
    A: {text}
    """.strip()

    context = ''
    for doc in search_results:
        # context = context + f"section: {doc['section']}\nquestion: {doc['question']}\nanswer: {doc['text']}\n\n"
        context = context + context_template.format(question=doc["question"], text=doc["text"]) + '\n\n'
    
    prompt = prompt_template.format(question=query, context=context).strip()
    print('Result prompt len %d, context len %d, num search results %d' % (len(prompt), len(context), len(search_results)))

    encoding = tiktoken.encoding_for_model("gpt-4o")
    tokens = encoding.encode(prompt)
    print('num tokens %d' % len(tokens))
    
    return prompt


def llm(prompt):
    response = client.chat.completions.create(
        model='gpt-4o',
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content

def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = ''
    # answer = llm(prompt)
    print('Answer len: %d' % len(answer))
    return answer

def elastic_search(es_client, query, limit=3, index_name="course-questions", filter=None):
    search_query = {
        "size": 5,
        "query": {
            "bool": {
                    "must": [
                        {
                        "multi_match": {
                            "query": query,
                            "fields": ["question^4", "text"],
                            "type": "best_fields"
                        }
                    },
                ],
                # "filter": [
                #     {
                #         "term": {
                #             "course": "data-engineering-zoomcamp"
                #         }
                #     }
                # ]
            }
        }
    }

    if filter is not None:
        search_query['query']['bool'].update({'filter': {"term": {"course": "machine-learning-zoomcamp"}}})

    response = es_client.search(index=index_name, body=search_query)
    
    return response['hits']['hits'][:limit]

def pretty(search_results):
    result_docs = []
    
    for hit in search_results:
        result_docs.append({hit['_score']: hit['_source']['question']})
    return result_docs

def get_data():
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    json_path = './data/documents.json'
    if not os.path.exists(json_path):
        documents = []

        for course in documents_raw:
            course_name = course['course']

            for doc in course['documents']:
                doc['course'] = course_name
                documents.append(doc)

        with open(json_path, 'w') as json_file:
            json.dump(documents, json_file)
        print('File saved, num rows %d' % len(documents))
    else:
        with open(json_path, 'r') as json_file:
            documents = json.load(json_file)
    print(len(documents))
    #print(documents[:2])
    return documents

def build_es_index(es_client, documents, index_name):
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
                "course": {"type": "keyword"} 
            }
        }
    }

    es_client.indices.create(index=index_name, body=index_settings)

    for doc in tqdm(documents):
        es_client.index(index=index_name, document=doc)
    print('Index created')


if __name__ == '__main__':
    es_client = Elasticsearch('http://localhost:9200')
    # index_name = "course-questions"
    # docs = get_data()
    # try:
    #     build_es_index(es_client, docs, index_name)
    # except BadRequestError as e:
    #     print(e)
    #print(es_client.info())



    query = 'How do I execute a command in a running docker container?'
    print(rag(query))
