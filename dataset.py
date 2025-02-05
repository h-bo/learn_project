import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import T5Tokenizer

class QADataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512, max_samples=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_samples is not None and i >= max_samples:
                    break
                item = json.loads(line.strip())
                self.data.append(item)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        
        # Format input text
        input_text = f"问题：{question} 文章：{context}"
        
        # Tokenize input and target
        inputs = self.tokenizer(
            input_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        targets = self.tokenizer(
            answer,
            max_length=64,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': targets['input_ids'].squeeze(),
            'target_attention_mask': targets['attention_mask'].squeeze()
        }

def create_dataloaders(train_path, dev_path, tokenizer, batch_size=8, max_length=512, max_samples=None):
    train_dataset = QADataset(train_path, tokenizer, max_length=max_length, max_samples=max_samples)
    dev_dataset = QADataset(dev_path, tokenizer, max_length=max_length, max_samples=max_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False
    )
    
    return train_loader, dev_loader
