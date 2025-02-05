import os
import torch
import logging
import argparse
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt
from modelscope.models import Model
from modelscope.preprocessors import TextGenerationT5Preprocessor
from modelscope.trainers.nlp.text_generation_trainer import TextGenerationTrainer
from modelscope.utils.constant import Tasks
from modelscope.pipelines import pipeline
from modelscope.utils.config import Config
from modelscope.metrics.base import Metric
from modelscope.metrics.builder import METRICS
from modelscope.utils.registry import default_group
from modelscope.metrics import METRICS
from modelscope.msdatasets import MsDataset
from modelscope.utils.hub import snapshot_download
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
                        help='ModelScope模型ID')
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

def train(model, train_loader, dev_loader, preprocessor, device, 
          num_epochs=5, learning_rate=5e-5, warmup_steps=0,
          output_dir='qa_model', save_steps=100, start_epoch=0):
    
    # 创建优化器和学习率调度器
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    total_steps = len(train_loader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=total_steps,
    )
    
    train_losses = []
    dev_metrics_history = []
    global_step = start_epoch * len(train_loader)
    
    for epoch in range(start_epoch, num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, batch in enumerate(progress_bar):
            # 预处理数据
            inputs = preprocessor(batch)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # 前向传播
            outputs = model(**inputs)
            loss = outputs['loss']
            total_loss += loss.item()
            
            # 反向传播
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.set_postfix({'loss': loss.item()})
            
            # 记录到wandb
            wandb.log({
                'train/loss': loss.item(),
                'train/learning_rate': scheduler.get_last_lr()[0],
                'train/epoch': epoch + batch_idx / len(train_loader),
                'train/global_step': global_step,
            })
            
            global_step += 1
            # 保存检查点
            if global_step % save_steps == 0:
                checkpoint_dir = os.path.join(output_dir, f'checkpoint-{global_step}')
                os.makedirs(checkpoint_dir, exist_ok=True)
                model.save_pretrained(checkpoint_dir)
                # 保存优化器和调度器状态
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'train_losses': train_losses,
                    'dev_metrics_history': dev_metrics_history
                }, os.path.join(checkpoint_dir, 'training_state.pt'))
                logger.info(f'Saved checkpoint to {checkpoint_dir}')
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Evaluation
        metrics = evaluate(model, dev_loader, preprocessor, device)
        dev_metrics_history.append(metrics)
        
        logger.info(f'Epoch {epoch+1}:')
        logger.info(f'Average training loss: {avg_train_loss:.4f}')
        for metric_name, value in metrics.items():
            logger.info(f'{metric_name}: {value:.4f}')
            
        # 记录评估指标到wandb
        wandb.log({
            'eval/loss': avg_train_loss,
            'eval/epoch': epoch + 1,
            **{f'eval/{k}': v for k, v in metrics.items()}
        })
        
        # 保存每个epoch结束后的模型
        epoch_dir = os.path.join(output_dir, f'epoch-{epoch+1}')
        os.makedirs(epoch_dir, exist_ok=True)
        model.save_pretrained(epoch_dir)
        torch.save({
            'epoch': epoch + 1,
            'global_step': global_step,
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_losses': train_losses,
            'dev_metrics_history': dev_metrics_history
        }, os.path.join(epoch_dir, 'training_state.pt'))
        logger.info(f'Saved epoch checkpoint to {epoch_dir}')
    
    # Plot training curve
    plot_training_curve(train_losses, dev_metrics_history)
    
    return train_losses, dev_metrics_history

def evaluate(model, dev_loader, preprocessor, device):
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    with torch.no_grad():
        for batch in tqdm(dev_loader, desc='Evaluating'):
            # 预处理数据
            inputs = preprocessor(batch)
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            # 生成答案
            outputs = model.generate(**inputs)
            predictions = preprocessor.decode(outputs)
            references = batch['answer']
            
            metrics_calculator.update(references, predictions)
    
    return metrics_calculator.get_metrics()

def main():
    # 解析命令行参数
    args = parse_args()
    
    # 初始化wandb
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        entity=args.wandb_entity,
        config=vars(args)
    )
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # 创建预处理器
    preprocessor = TextGenerationT5Preprocessor(model_dir=args.model_id)
    
    # Load model
    if args.resume_from:
        logger.info(f'Loading model from checkpoint: {args.resume_from}')
        model = Model.from_pretrained(args.resume_from, task=Tasks.text_generation)
        # 加载训练状态
        training_state = torch.load(os.path.join(args.resume_from, 'training_state.pt'))
        start_epoch = training_state['epoch']
        logger.info(f'Resuming from epoch {start_epoch}')
    else:
        model = Model.from_pretrained(args.model_id, task=Tasks.text_generation)
        start_epoch = 0
    
    model.to(device)
    
    # Create dataloaders
    train_loader, dev_loader = create_dataloaders(
        args.train_path, 
        args.dev_path, 
        preprocessor,
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    # Train the model
    train(
        model=model,
        train_loader=train_loader,
        dev_loader=dev_loader,
        preprocessor=preprocessor,
        device=device,
        num_epochs=args.num_epochs,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        output_dir=args.output_dir,
        save_steps=args.save_steps,
        start_epoch=start_epoch
    )
    
    # Save the final model
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_pretrained(args.output_dir)
    logger.info(f"Model saved to {args.output_dir}")
    
    # 结束wandb运行
    wandb.finish()

if __name__ == "__main__":
    main()
