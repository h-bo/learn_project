import json
import torch
from torch.utils.data import Dataset, DataLoader
from modelscope.preprocessors import TextGenerationT5Preprocessor

class QADataset(Dataset):
    def __init__(self, data_path, preprocessor, max_length=512, max_samples=None):
        self.data = []
        self.preprocessor = preprocessor
        self.max_length = max_length
        
        # 读取数据
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            if max_samples:
                data = data[:max_samples]
            self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        
        # 构造输入
        text = f"问题：{question}\n上下文：{context}\n答案："
        
        # 使用preprocessor处理数据
        inputs = self.preprocessor({'text': text, 'answer': answer})
        inputs['answer'] = answer  # 保留原始答案用于评估
        
        return inputs

def create_dataloaders(train_path, dev_path, preprocessor, batch_size=8, 
                      max_length=512, max_samples=None):
    # 创建训练集
    train_dataset = QADataset(
        train_path, 
        preprocessor,
        max_length=max_length,
        max_samples=max_samples
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: x[0]  # ModelScope preprocessor已经处理了batch
    )
    
    # 创建验证集
    dev_dataset = QADataset(
        dev_path, 
        preprocessor,
        max_length=max_length,
        max_samples=max_samples
    )
    dev_loader = DataLoader(
        dev_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: x[0]  # ModelScope preprocessor已经处理了batch
    )
    
    return train_loader, dev_loader
