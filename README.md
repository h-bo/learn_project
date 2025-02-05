# T5 问答模型

基于 T5 的生成式问答模型，支持训练、继续训练和测试功能。

## 项目结构

```
.
├── README.md           # 项目文档
├── requirements.txt    # 项目依赖
├── train.py           # 训练脚本
├── test.py            # 测试脚本
├── dataset.py         # 数据集处理
├── metrics.py         # 评估指标计算
└── data/             # 数据目录
```

## 环境配置

### 1. 创建虚拟环境

```bash
# 创建虚拟环境
conda create --prefix /home/featurize/work/t5qa_env python=3.11

# 激活虚拟环境
## Windows
t5qa_env\Scripts\activate
## macOS/Linux
source t5qa_env/bin/activate

# 退出虚拟环境
deactivate
```

### 2. 安装依赖

激活虚拟环境后安装依赖：
```bash
pip install -r requirements.txt
```

### 3. 配置 Weights & Biases

```bash
# 登录到 Weights & Biases
wandb login

# 设置环境变量（可选）
export WANDB_PROJECT="t5-qa"  # 项目名称
export WANDB_ENTITY="your-username"  # 你的用户名或组织名
```

## 使用方法

### 1. 训练模型

基本训练：
```bash
python train.py \
    --train_path data/train.json \
    --dev_path data/dev.json \
    --model_id langboat/mengzi-t5-base \
    --output_dir qa_model \
    --num_epochs 5 \
    --batch_size 8 \
    --learning_rate 5e-5 \
    --wandb_project t5-qa \
    --wandb_name "baseline-run"
```

快速测试模式：
```bash
python train.py --fast_test --wandb_name "fast-test-run"
```

从检查点继续训练：
```bash
python train.py --resume_from qa_model/checkpoint-1000
```

主要参数说明：
- `--train_path`: 训练数据路径
- `--dev_path`: 验证数据路径
- `--model_id`: ModelScope 模型ID
- `--output_dir`: 模型保存路径
- `--resume_from`: 从某个检查点继续训练
- `--save_steps`: 每多少步保存一次检查点
- `--num_epochs`: 训练轮数
- `--batch_size`: 批次大小
- `--learning_rate`: 学习率
- `--max_length`: 最大序列长度
- `--max_samples`: 每个数据集最大样本数（用于快速测试）
- `--fast_test`: 快速测试模式
# Weights & Biases 相关参数
- `--wandb_project`: W&B项目名称
- `--wandb_name`: 本次运行的名称
- `--wandb_entity`: W&B用户名或组织名

### 2. 测试模型

```bash
python test.py \
    --model_path qa_model/checkpoint-1000 \
    --test_path data/test.json \
    --batch_size 8
```

参数说明：
- `--model_path`: 模型路径，可以是检查点目录
- `--test_path`: 测试数据路径
- `--batch_size`: 批次大小
- `--max_length`: 最大序列长度
- `--max_samples`: 测试样本数量

### 3. 数据格式

训练数据应为 JSON 格式，每行包含一个样本：
```json
{
    "context": "上下文文本",
    "question": "问题文本",
    "answer": "答案文本"
}
```

## 训练过程

1. 模型会在以下情况保存检查点：
   - 每隔指定步数（由 `save_steps` 参数控制）
   - 每个 epoch 结束时
   - 训练完成时

2. 检查点保存位置：
   - 步数检查点：`qa_model/checkpoint-{step_number}/`
   - Epoch 检查点：`qa_model/epoch-{epoch_number}/`
   - 最终模型：`qa_model/`

3. 检查点内容：
   - 模型权重
   - 分词器配置
   - 训练状态（包括优化器状态、学习率调度器状态等）
   - 训练历史（损失和评估指标）

4. 训练监控：
   - 通过 Weights & Biases 实时监控训练过程
   - 记录的指标包括：
     - 训练损失
     - 评估损失
     - BLEU 分数
     - 学习率变化
     - 训练速度
   - 可以在 W&B 界面上查看训练曲线和指标变化

5. 训练过程会生成训练曲线图（`training_curves.png`），展示：
   - 训练损失变化
   - BLEU 分数变化

## 评估指标

模型使用以下指标进行评估：
- BLEU-1
- BLEU-2
- BLEU-3
- BLEU-4

## 注意事项

1. 建议在使用大规模数据集之前，先用 `--fast_test` 模式测试代码是否正常运行。

2. 如果训练中断，可以使用 `--resume_from` 参数从最近的检查点继续训练。

3. 模型和数据都比较大，请确保有足够的磁盘空间和内存。

4. 请确保在激活的虚拟环境中运行所有命令。
