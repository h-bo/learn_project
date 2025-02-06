import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer, max_length=512, max_samples=None):
        self.data = []
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # 读取数据（JSON Lines 格式）
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)
                if max_samples and len(self.data) >= max_samples:
                    break
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        context = item['context']
        question = item['question']
        answer = item['answer']
        
        # 构造输入
        source_text = f"问题：{question}\n上下文：{context}\n答案："
        target_text = answer
        
        try:
            # 使用tokenizer处理数据
            model_inputs = self.tokenizer(
                source_text,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            # 处理目标文本
            with self.tokenizer.as_target_tokenizer():
                labels = self.tokenizer(
                    target_text,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
            
            # 移除batch维度
            input_ids = model_inputs['input_ids'].squeeze(0)
            attention_mask = model_inputs['attention_mask'].squeeze(0)
            labels = labels['input_ids'].squeeze(0)
            
            # 将padding token的label设为-100，这样它们在计算损失时会被忽略
            labels[labels == self.tokenizer.pad_token_id] = -100
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels,
                'answer': answer  # 保留原始答案用于评估
            }
        except Exception as e:
            print(f"Error processing item {idx}:")
            print(f"Source text: {source_text}")
            print(f"Target text: {target_text}")
            print(f"Error: {str(e)}")
            raise

def create_dataloaders(train_path, dev_path, train_tokenizer, eval_tokenizer, 
                      batch_size=8, max_length=512, max_samples=None):
    """
    创建训练和验证数据的DataLoader
    
    Args:
        train_path: 训练数据路径
        dev_path: 验证数据路径
        train_tokenizer: 用于训练数据的tokenizer
        eval_tokenizer: 用于验证数据的tokenizer
        batch_size: 批次大小
        max_length: 最大序列长度
        max_samples: 每个数据集的最大样本数，用于快速测试
        
    Returns:
        train_loader: 训练数据的DataLoader
        dev_loader: 验证数据的DataLoader
    """
    train_dataset = QADataset(train_path, train_tokenizer, max_length, max_samples)
    dev_dataset = QADataset(dev_path, eval_tokenizer, max_length, max_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, dev_loader
