from rag_framework import retrieve_candidates, generate_response, load_model
from preprocess_data import load_data, preprocess_data
from index_data import index_data

def main():
    candidate_df, _ = load_data()
    candidate_df, profile_embeddings = preprocess_data(candidate_df)
    faiss_index = index_data(profile_embeddings)
    tokenizer, model = load_model()
    
    print("Enter job description:")
    job_description = input().strip()
    
    print("Retrieving candidate profiles...")
    candidates = retrieve_candidates(job_description, faiss_index, candidate_df)
    
    print("Generating responses...")
    for candidate in candidates:
        candidate_text = candidate['profile_text']
        response = generate_response(f"Job Description: {job_description}\nCandidate Profile: {candidate_text}", tokenizer, model)
        print(f"\nCandidate Name: {candidate['Name']}")
        print(f"Response: {response}\n")

if __name__ == "__main__":
    main()
