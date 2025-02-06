import argparse
import torch
import logging
from dataset import create_dataloaders
from metrics import MetricsCalculator
from model_utils import load_model_and_tokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description='T5 Question Answering Model Testing')
    
    parser.add_argument('--model_path', type=str, required=True,
                        help='模型路径，可以是检查点目录')
    parser.add_argument('--model_source', type=str, default='huggingface',
                        choices=['huggingface', 'modelscope'],
                        help='模型来源')
    parser.add_argument('--test_path', type=str, default='data/DuReaderQG/dev.json',
                        help='测试数据路径')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='批次大小')
    parser.add_argument('--max_length', type=int, default=512,
                        help='最大序列长度')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='测试样本数量')
    
    return parser.parse_args()

def test(model, test_loader, tokenizer, device):
    model.eval()
    metrics_calculator = MetricsCalculator()
    
    logger.info("Starting evaluation...")
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            generated_ids = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )
            
            # Decode predictions
            predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            
            # Replace -100 in labels with pad_token_id for proper decoding
            labels_for_decode = labels.clone()
            labels_for_decode[labels_for_decode == -100] = tokenizer.pad_token_id
            references = tokenizer.batch_decode(labels_for_decode, skip_special_tokens=True)
            
            # Print some examples
            for pred, ref in zip(predictions[:2], references[:2]):
                logger.info(f"预测: {pred}")
                logger.info(f"参考: {ref}")
                logger.info("-" * 50)
            
            metrics_calculator.update(references, predictions)
    
    metrics = metrics_calculator.get_metrics()
    for metric_name, value in metrics.items():
        logger.info(f'{metric_name}: {value:.4f}')
    
    return metrics

def main():
    args = parse_args()
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f'Using device: {device}')
    
    # Load model and tokenizer
    logger.info(f'Loading model from {args.model_path}')
    model, tokenizer = load_model_and_tokenizer(args.model_path, args.model_source == 'modelscope')
    model.to(device)
    
    # Create test dataloader
    test_loader, _ = create_dataloaders(
        args.test_path,
        args.test_path,  # dev_path 参数不会被使用
        tokenizer,
        tokenizer,  # 使用同一个tokenizer作为eval_tokenizer
        batch_size=args.batch_size,
        max_length=args.max_length,
        max_samples=args.max_samples
    )
    
    # Test the model
    test(model, test_loader, tokenizer, device)

if __name__ == "__main__":
    main()
