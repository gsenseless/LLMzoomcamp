from fastembed import TextEmbedding
from qdrant_client import QdrantClient, models
import numpy as np
import requests 


CLIENT_URL = "http://localhost:6333"
QUERY = 'I just discovered the course. Can I join now?'
MODEL_HANDLE = 'jinaai/jina-embeddings-v2-small-en'
DOC = 'Can I still join the course after the start date?'

client = QdrantClient(CLIENT_URL)
embedding_model = TextEmbedding(MODEL_HANDLE)
query_embeddings = list(embedding_model.embed(QUERY))

# Q1
min(query_embeddings[0])
# np.linalg.norm(query_embeddings[0])
# query_embeddings[0].dot(query_embeddings[0])

doc_embeddings = list(embedding_model.embed(DOC))

# Q2
query_embeddings[0].dot(doc_embeddings[0])

documents = [{'text': "Yes, even if you don't register, you're still eligible to submit the homeworks.\nBe aware, however, that there will be deadlines for turning in the final projects. So don't leave everything for the last minute.",
  'section': 'General course-related questions',
  'question': 'Course - Can I still join the course after the start date?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Yes, we will keep all the materials after the course finishes, so you can follow the course at your own pace after it finishes.\nYou can also continue looking at the homeworks and continue preparing for the next cohort. I guess you can also start working on your final capstone project.',
  'section': 'General course-related questions',
  'question': 'Course - Can I follow the course after it finishes?',
  'course': 'data-engineering-zoomcamp'},
 {'text': "The purpose of this document is to capture frequently asked technical questions\nThe exact day and hour of the course will be 15th Jan 2024 at 17h00. The course will start with the first  “Office Hours'' live.1\nSubscribe to course public Google Calendar (it works from Desktop only).\nRegister before the course starts using this link.\nJoin the course Telegram channel with announcements.\nDon’t forget to register in DataTalks.Club's Slack and join the channel.",
  'section': 'General course-related questions',
  'question': 'Course - When will the course start?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'You can start by installing and setting up all the dependencies and requirements:\nGoogle cloud account\nGoogle Cloud SDK\nPython 3 (installed with Anaconda)\nTerraform\nGit\nLook over the prerequisites and syllabus to see if you are comfortable with these subjects.',
  'section': 'General course-related questions',
  'question': 'Course - What can I do before the course starts?',
  'course': 'data-engineering-zoomcamp'},
 {'text': 'Star the repo! Share it with friends if you find it useful ❣️\nCreate a PR if you see you can improve the text or the structure of the repository.',
  'section': 'General course-related questions',
  'question': 'How can we contribute to the course?',
  'course': 'data-engineering-zoomcamp'}]

def process_documents_and_calculate_similarities(documents, query_embeddings, embedding_model, include_question=True):
    document_embeddings = []
    for doc in documents:
        if include_question:
            full_text = doc['question'] + ' ' + doc['text']
        else:
            full_text = doc['text']
        doc_embedding = list(embedding_model.embed(full_text))
        document_embeddings.append(doc_embedding[0])
    
    cosine_similarities = []
    for doc_embedding in document_embeddings:
        dot_product = query_embeddings[0].dot(doc_embedding)
        query_norm = np.linalg.norm(query_embeddings[0])
        doc_norm = np.linalg.norm(doc_embedding)
        cosine_sim = dot_product / (query_norm * doc_norm)
        cosine_similarities.append(cosine_sim)

    print("Cosine similarities between query and documents:")
    for i, similarity in enumerate(cosine_similarities):
        print(f"Document {i} ({documents[i]['question']}): {similarity:.4f}")

    max_index = np.argmax(cosine_similarities)
    print("\nDocument with highest similarity:")
    print(f"Document {max_index} ({documents[max_index]}): {cosine_similarities[max_index]:.4f}")
    
# Q3
process_documents_and_calculate_similarities(documents, query_embeddings, embedding_model, include_question=False)

# Q4
process_documents_and_calculate_similarities(documents, query_embeddings, embedding_model, include_question=True)

# Q5
supported_models = TextEmbedding.list_supported_models()
min_dim_model = min(supported_models, key=lambda x: x['dim'])
print(f"\nModel with smallest dimensionality: {min_dim_model['model']} ({min_dim_model['dim']} dimensions)")


def load_machine_learning_zoomcamp_documents():
    docs_url = 'https://github.com/alexeygrigorev/llm-rag-workshop/raw/main/notebooks/documents.json'
    docs_response = requests.get(docs_url)
    documents_raw = docs_response.json()

    documents = []
    for course in documents_raw:
        course_name = course['course']
        if course_name != 'machine-learning-zoomcamp':
            continue

        for doc in course['documents']:
            doc['course'] = course_name
            documents.append(doc)
    return documents

# Q6
MODEL_HANDLE = 'BAAI/bge-small-en'
embedding_model = TextEmbedding(MODEL_HANDLE)
query_embeddings = list(embedding_model.embed(QUERY))

documents = load_machine_learning_zoomcamp_documents()
process_documents_and_calculate_similarities(documents, query_embeddings, embedding_model, include_question=True)


