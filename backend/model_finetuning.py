import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
import torch

# Load the dataset
file_path = 'data/resumes.csv'
resumes_df = pd.read_csv(file_path)

# Combine relevant columns to create candidate profiles
resumes_df['Profile'] = resumes_df.apply(
    lambda row: f"{row['Job Skills']} {row['Experience']} {row['Projects']} {row['Comments']}", axis=1
)

# Display the first few rows of the processed dataset
print(resumes_df[['Job Description', 'Profile', 'Match']].head())

# Load tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Prepare texts and labels
texts = resumes_df['Job Description'] + " [SEP] " + resumes_df['Profile']
labels = resumes_df['Match'].tolist()

# Tokenize the texts
encodings = tokenizer(texts.tolist(), truncation=True, padding=True, max_length=512)

# Convert encodings and labels to PyTorch tensors
input_ids = torch.tensor(encodings['input_ids'])
attention_masks = torch.tensor(encodings['attention_mask'])
labels = torch.tensor(labels)

# Create a TensorDataset
dataset = torch.utils.data.TensorDataset(input_ids, attention_masks, labels)

# Display the size of the dataset
print(f"Dataset size: {len(dataset)}")

# Load the model
model = BertForSequenceClassification.from_pretrained('bert-base-uncased')

# Define training arguments
training_args = TrainingArguments(
    output_dir='./results',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
)

# Create a Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    eval_dataset=dataset,  # In practice, use a separate validation set
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained('./fine-tuned-model')
tokenizer.save_pretrained('./fine-tuned-model')

print("Model fine-tuning complete.")
