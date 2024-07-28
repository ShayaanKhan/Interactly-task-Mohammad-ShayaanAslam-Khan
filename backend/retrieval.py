from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
import numpy as np

# Elasticsearch setup
es = Elasticsearch()

# Initialize a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve_candidates(job_description, top_k=3):
    query = {
        "query": {
            "match": {
                "Profile": job_description
            }
        }
    }
    res = es.search(index="candidates", body=query, size=top_k)
    return res['hits']['hits']

def generate_response(retrieved_candidates):
    responses = []
    for candidate in retrieved_candidates:
        response = f"Candidate: {candidate['_source']['Name']}, Skills: {candidate['_source']['Job Skills']}, Experience: {candidate['_source']['Experience']}"
        responses.append(response)
    return responses
