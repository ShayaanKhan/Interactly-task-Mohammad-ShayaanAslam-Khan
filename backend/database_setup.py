import pandas as pd
from pymongo import MongoClient
from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = '../data/RecruterPilot candidate sample input dataset - Sheet1.csv'
candidates_df = pd.read_csv(file_path)

# Combine relevant columns to create candidate profiles
candidates_df['Profile'] = candidates_df.apply(
    lambda row: f"{row['Job Skills']} {row['Experience']} {row['Projects']} {row['Comments']}", axis=1
)

# MongoDB setup
client = MongoClient("mongodb://localhost:27017/")
db = client["candidateDB"]
collection = db["profiles"]

# Insert candidate data into MongoDB
candidates = candidates_df.to_dict(orient='records')
collection.insert_many(candidates)

# Elasticsearch setup
es = Elasticsearch()

# Initialize a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the candidate profiles
candidate_embeddings = model.encode(candidates_df['Profile'].tolist(), convert_to_tensor=True)

# Index embeddings in Elasticsearch
for i, candidate in enumerate(candidates):
    candidate['embedding'] = candidate_embeddings[i].tolist()
    es.index(index="candidates", id=candidate["Contact Details"], body=candidate)

print("Database and indexing setup complete.")
