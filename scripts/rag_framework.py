import numpy as np
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from sentence_transformers import SentenceTransformer
from preprocess_data import preprocess_data, load_data
from index_data import index_data

def load_model():
    tokenizer = GPT2Tokenizer.from_pretrained('./models/fine_tuned_model')
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token
    model = GPT2LMHeadModel.from_pretrained('./models/fine_tuned_model')
    return tokenizer, model

def retrieve_candidates(query, faiss_index, candidate_df, num_candidates=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')
    query_embedding = model.encode([query])
    _, indices = faiss_index.search(np.array(query_embedding), num_candidates)
    return candidate_df.iloc[indices[0]].to_dict(orient='records')

def generate_response(query, tokenizer, model):
    inputs = tokenizer.encode_plus(query, return_tensors='pt', padding=True, truncation=True)
    outputs = model.generate(
        input_ids=inputs['input_ids'], 
        attention_mask=inputs['attention_mask'],  # Pass the attention mask
        max_length=150,
        pad_token_id=tokenizer.eos_token_id  # Set pad token ID to eos token ID
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

if __name__ == "__main__":
    candidate_df, _ = load_data()
    candidate_df, profile_embeddings = preprocess_data(candidate_df)
    faiss_index = index_data(profile_embeddings)
    tokenizer, model = load_model()
    print("RAG framework loaded successfully.")
