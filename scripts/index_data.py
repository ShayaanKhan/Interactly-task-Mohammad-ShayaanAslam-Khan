import numpy as np
import faiss
from preprocess_data import preprocess_data, load_data

def index_data(profile_embeddings):
    dimension = profile_embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(np.array(profile_embeddings))
    return faiss_index

if __name__ == "__main__":
    candidate_df, _ = load_data()
    candidate_df, profile_embeddings = preprocess_data(candidate_df)
    faiss_index = index_data(profile_embeddings)
    print(faiss_index.is_trained)
    print(faiss_index.ntotal)
