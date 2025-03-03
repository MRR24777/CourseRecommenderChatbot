from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, DataCollatorForLanguageModeling
import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

class CourseDataset(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings

    def __getitem__(self, idx):
        return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    def __len__(self):
        return len(self.encodings['input_ids'])

def train_gpt2(input_csv, output_model):
    print("Cargando datos para entrenamiento de GPT-2...")
    df = pd.read_csv(input_csv)
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    df['Tokens'] = df['Description'].apply(lambda x: tokenizer(x, truncation=True, padding="max_length", max_length=50))
    input_ids = pad_sequence([torch.tensor(d['input_ids']) for d in df['Tokens']], batch_first=True, padding_value=tokenizer.pad_token_id)
    attention_mask = pad_sequence([torch.tensor(d['attention_mask']) for d in df['Tokens']], batch_first=True, padding_value=0)
    
    encodings = {'input_ids': input_ids, 'attention_mask': attention_mask}
    dataset = CourseDataset(encodings)
    
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
    training_args = TrainingArguments(
        output_dir=output_model,
        num_train_epochs=3,
        per_device_train_batch_size=4,
        save_steps=10_000,
        save_total_limit=2,
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )
    
    trainer.train()
    model.save_pretrained(output_model)
    tokenizer.save_pretrained(output_model)
    print("Modelo guardado en:", output_model)

if __name__ == "__main__":
    train_gpt2('data/indexed_courses.csv', 'data/trained_gpt2')