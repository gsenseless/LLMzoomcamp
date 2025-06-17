# docker run -it \
#     --rm \
#     --name elasticsearch \
#     -m 4GB \
#     -p 9200:9200 \
#     -p 9300:9300 \
#     -e "discovery.type=single-node" \
#     -e "xpack.security.enabled=false" \
#     docker.elastic.co/elasticsearch/elasticsearch:8.4.3

import requests 
import json
from tqdm.auto import tqdm
from elasticsearch import Elasticsearch
import tiktoken


def fetch_documents():
    docs_url = 'https://github.com/DataTalksClub/llm-zoomcamp/blob/main/01-intro/documents.json?raw=1'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()
    
    documents = []
    for course in documents_raw:
        course_name = course['course']
        documents.extend([
            {**doc, 'course': course_name} 
            for doc in course['documents']
        ])
    
    return documents

def create_elasticsearch_index(documents):
    index_settings = {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0
        },
        "mappings": {
            "properties": {
                "text": {"type": "text"},
                "course": {"type": "keyword"} 
            }
        }
    }

    es_client = Elasticsearch('http://localhost:9200')
    
    if not es_client.indices.exists(index=index_name):
        es_client.indices.create(index=index_name, body=index_settings)
    
    for doc in tqdm(documents, desc="Indexing documents"):
        es_client.index(index=index_name, document=doc)
    
    return es_client

def elastic_search(query, size=5, course="data-engineering-zoomcamp"):
    search_query = {
        "size": size,
        "query": {
            "bool": {
                "must": {
                    "multi_match": {
                        "query": query,
                        "fields": ["question^4", "text"],
                        "type": "best_fields"
                    }
                },
                "filter": {
                    "term": {
                        "course": course
                    }
                }
            }
        }
    }

    response = es_client.search(index=index_name, body=search_query)
    
    result_docs = []
    
    for hit in response['hits']['hits']:
        doc = hit['_source']
        doc['score'] = hit['_score']
        result_docs.append(doc)
    
    return result_docs

def build_prompt(query, search_results):
    prompt_template = """
    You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
    Use only the facts from the CONTEXT when answering the QUESTION.

    QUESTION: {question}

    CONTEXT:
    {context}
    """.strip()

    context = ""
    
    for doc in search_results:
        context = context + f"text: {doc['text']}\n\nquestion: {doc['question']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt


documents = fetch_documents()
index_name = "course-questions"
es_client = create_elasticsearch_index(documents)


query = "How do execute a command on a Kubernetes pod?"

search_results = elastic_search(query)
scores = [result.get('score') for result in search_results]
print(sorted(scores, reverse=True))

query = "How do copy a file to a Docker container?"
search_results = elastic_search(query, size = 3, course="machine-learning-zoomcamp")
search_results[2]['question']

prompt = build_prompt(query, search_results)
len(prompt)    

encoding = tiktoken.encoding_for_model("gpt-4")
token_count = len(encoding.encode(prompt))
print(f"Number of tokens in prompt: {token_count}")


###########

import os
from google import genai

#import dotenv
#load_dotenv()
client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))


def llm(prompt):
    response = client.models.generate_content(
        model="gemini-2.0-flash", contents=prompt
    )
    return response.text

def rag(query):
    search_results = elastic_search(query)
    print(search_results)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer

rag(query)


