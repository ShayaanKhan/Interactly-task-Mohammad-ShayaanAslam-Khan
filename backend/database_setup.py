import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# Load the dataset
file_path = '../data/RecruterPilot candidate sample input dataset - Sheet1.csv'
candidates_df = pd.read_csv(file_path)

# Combine relevant columns to create candidate profiles
candidates_df['Profile'] = candidates_df.apply(
    lambda row: f"{row['Job Skills']} {row['Experience']} {row['Projects']} {row['Comments']}", axis=1
)

# Initialize a sentence transformer model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the candidate profiles
candidate_profiles = candidates_df['Profile'].tolist()
candidate_embeddings = model.encode(candidate_profiles, convert_to_tensor=False)

# Convert embeddings to numpy array
candidate_embeddings = np.array(candidate_embeddings)

# Initialize FAISS index
d = candidate_embeddings.shape[1]
index = faiss.IndexFlatL2(d)

# Add embeddings to the index
index.add(candidate_embeddings)

# Save the index and candidate data
faiss.write_index(index, "candidate_index.faiss")
candidates_df.to_pickle("candidates_df.pkl")

print("Database and indexing setup complete.")
