from sentence_transformers import SentenceTransformer
from load_data import load_data

def preprocess_candidate(row):
    return f"{row['Name']} {row['Contact Details']} {row['Location']} {row['Experience']} {' '.join(row['Job Skills'].split(','))} {' '.join(row['Projects'].split(','))} {row['Comments']}"

def preprocess_data(candidate_df):
    candidate_df['profile_text'] = candidate_df.apply(preprocess_candidate, axis=1)
    model = SentenceTransformer('all-MiniLM-L6-v2')
    profile_embeddings = model.encode(candidate_df['profile_text'].tolist())
    return candidate_df, profile_embeddings

if __name__ == "__main__":
    candidate_df, _ = load_data()
    candidate_df, profile_embeddings = preprocess_data(candidate_df)
    print(candidate_df.head())
    print(profile_embeddings.shape)
