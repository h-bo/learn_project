import os
import torch
import logging
import argparse
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from dataset import create_dataloaders
from metrics import MetricsCalculator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='T5 Question Answering Model Training')
    
    # 数据相关参数
    parser.add_argument('--train_path', type=str, default='data/DuReaderQG/train.json',
                        help='训练数据路径')
    parser.add_argument('--dev_path', type=str, default='data/DuReaderQG/dev.json',
                        help='验证数据路径')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='每个数据集最大样本数，用于快速测试')
    
    # 模型相关参数
    parser.add_argument('--model_id', type=str, default='langboat/mengzi-t5-base',
                        help='Hugging Face模型ID')
    parser.add_argument('--model_dir', type=str, default='pretrained_models',
                        help='预训练模型本地保存目录')
    parser.add_argument('--output_dir', type=str, default='qa_model',
                        help='模型保存路径')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='从某个检查点继续训练')
    parser.add_argument('--save_steps', type=int, default=100,
                        help='每多少步保存一次检查点')
    
    # wandb相关参数
    parser.add_argument('--wandb_project', type=str, default='t5-qa',
                        help='Weights & Biases项目名称')
    parser.add_argument('--wandb_name', type=str, default=None,
                        help='Weights & Biases运行名称')
    parser.add_argument('--wandb_entity', type=str, default=None,
                        help='Weights & Biases实体名称')
    
    # 训练相关参数
    parser.add_argument('--num_epochs', type=int, default=5,
                        help='训练轮数')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--learning_rate', type=float, default=5e-5,
                        help='学习率')
    parser.add_argument('--warmup_steps', type=int, default=100,
                        help='warmup步数')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    
    # 快速测试模式
    parser.add_argument('--fast_test', action='store_true',
                        help='是否使用快速测试模式')
    
    args = parser.parse_args()
    
    # 如果使用快速测试模式，覆盖相关参数
    if args.fast_test:
        args.num_epochs = 2
        args.batch_size = 4
        args.max_samples = 10
        args.learning_rate = 5e-4
        args.warmup_steps = 10
        args.save_steps = 5
    
    return args

def train(model, train_loader, dev_loader, tokenizer, device, 
          num_epochs=5, learning_rate=5e-5, warmup_steps=0,
          output_dir='qa_model', save_steps=100, start_epoch=0):
    
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    # 设置进度条
    total_steps = len(train_loader) * num_epochs
    progress_bar = tqdm(total=total_steps, desc="Training")
    
    global_step = 0
    best_eval_loss = float('inf')
    
    for epoch in range(start_epoch, num_epochs):
        epoch_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            epoch_loss += loss.item()
            global_step += 1
            progress_bar.update(1)
            
            # 记录到wandb
            wandb.log({
                'train_loss': loss.item(),
                'epoch': epoch,
                'global_step': global_step
            })
            
            # 定期保存和评估
            if global_step % save_steps == 0:
                # 保存模型
                save_path = os.path.join(output_dir, f'checkpoint-{global_step}')
                os.makedirs(save_path, exist_ok=True)
                model.save_pretrained(save_path)
                tokenizer.save_pretrained(save_path)
                
                # 评估
                eval_loss = evaluate(model, dev_loader, tokenizer, device)
                model.train()  # 切回训练模式
                
                wandb.log({
                    'eval_loss': eval_loss,
                    'epoch': epoch,
                    'global_step': global_step
                })
                
                # 保存最佳模型
                if eval_loss < best_eval_loss:
                    best_eval_loss = eval_loss
                    best_model_path = os.path.join(output_dir, 'best_model')
                    os.makedirs(best_model_path, exist_ok=True)
                    model.save_pretrained(best_model_path)
                    tokenizer.save_pretrained(best_model_path)
        
        # 打印每个epoch的平均损失
        avg_loss = epoch_loss / len(train_loader)
        logger.info(f'Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.4f}')
    
    progress_bar.close()
    return global_step

def evaluate(model, dev_loader, tokenizer, device):
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
    
    avg_loss = total_loss / len(dev_loader)
    logger.info(f'Evaluation Loss: {avg_loss:.4f}')
    return avg_loss

def main():
    args = parse_args()
    
    # 初始化wandb
    wandb.init(project=args.wandb_project)
    wandb.config.update(args)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 加载tokenizer和model
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSeq2SeqLM.from_pretrained(args.model_id)
    model = model.to(device)
    
    # 创建数据加载器
    train_loader, dev_loader = create_dataloaders(
        args.train_path, 
        args.dev_path,
        tokenizer,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    # 如果指定了检查点，从检查点恢复
    start_epoch = 0
    if args.resume_from:
        checkpoint = torch.load(os.path.join(args.resume_from, 'pytorch_model.bin'))
        model.load_state_dict(checkpoint)
        # 从检查点路径中提取epoch数
        try:
            start_epoch = int(args.resume_from.split('-')[-1])
        except:
            start_epoch = 0
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 训练模型
    global_step = train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        tokenizer=tokenizer,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        start_epoch=start_epoch
    )
    
    # 保存最终模型
    final_model_path = os.path.join(args.output_dir, f'final-model')
    os.makedirs(final_model_path, exist_ok=True)
    model.save_pretrained(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    
    wandb.finish()

if __name__ == "__main__":
    main()
