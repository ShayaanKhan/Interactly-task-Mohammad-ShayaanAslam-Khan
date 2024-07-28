import pandas as pd
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel, TextDataset, DataCollatorForLanguageModeling, Trainer, TrainingArguments

def create_training_text_file(csv_path, text_file_path):
    training_df = pd.read_csv(csv_path)
    training_data = "\n".join(training_df['Resume_str'].tolist())
    with open(text_file_path, 'w', encoding='utf-8') as f:
        f.write(training_data)

def fine_tune_llm(training_text_path):
    tokenizer = GPT2Tokenizer.from_pretrained('distilgpt2')
    tokenizer.pad_token = tokenizer.eos_token  # Set the padding token

    model = GPT2LMHeadModel.from_pretrained('distilgpt2')
    model.resize_token_embeddings(len(tokenizer))

    def load_dataset(file_path, tokenizer, block_size=128):
        def read_text():
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        text = read_text()
        examples = tokenizer(text, return_tensors='pt', max_length=block_size, truncation=True, padding='max_length').input_ids
        return examples

    train_dataset = load_dataset(training_text_path, tokenizer)
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    training_args = TrainingArguments(
        output_dir='./models/fine_tuned_model',
        overwrite_output_dir=True,
        num_train_epochs=100,
        per_device_train_batch_size=4,  # Reduced for faster training
        save_steps=1000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=train_dataset,
    )

    trainer.train()
    model.save_pretrained('./models/fine_tuned_model')
    tokenizer.save_pretrained('./models/fine_tuned_model')  # Ensure tokenizer is saved

if __name__ == "__main__":
    csv_path = 'data/resumes.csv'
    text_file_path = 'data/training_data.txt'
    create_training_text_file(csv_path, text_file_path)
    fine_tune_llm(text_file_path)
