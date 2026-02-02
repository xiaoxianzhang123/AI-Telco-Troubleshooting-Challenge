# GRPO Reinforcement Learning for 5G Network Analysis

A comprehensive reinforcement learning training pipeline using GRPO (Group Relative Policy Optimization) for fine-tuning large language models on 5G network analysis tasks. This project leverages Huawei Ascend NPUs with verl-ascend and vllm-ascend for efficient distributed training.

## 🖥️ Hardware Environment

- **NPU**: (8 × Huawei Ascend 910B2)  * 4
- **Driver Version**: 25.3.rc1
- **Firmware Version**: 7.8.0.2.212
- **Framework**: verl-ascend with vllm-ascend inference engine

## 📋 Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Training](#training)
- [Reward Function](#reward-function)
- [Configuration](#configuration)
- [Monitoring](#monitoring)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This project implements a reinforcement learning pipeline for training language models to solve 5G network analysis problems. The system uses:

- **Algorithm**: GRPO (Group Relative Policy Optimization)
- **Base Model**: Qwen3-32B
- **Task**: Multiple-choice question answering for 5G network scenarios
- **Reward**: Binary reward (1.0 for correct, 0.0 for incorrect)

## 📁 Project Structure

```
.
├── start.bash              # Main training launcher
├── transfer.py             # Data preprocessing script
├── reward_function.py      # Custom reward computation
├── train.parquet          # Training data (generated)
└── test.parquet           # Validation data (generated)
```


## 📊 Data Preparation

### Input Data Format

Your source dataset should be in JSON format with the following structure:

```json
{
  "question": "5G network question with multiple choice options",
  "answer": "C1"
}
```

### Data Conversion

Convert your dataset to the required format:

```bash
python transfer.py \
  --dataset_file path/to/your/dataset.json \
  --save_dir ./processed_data
```

This script:
- Extracts questions and answers from your JSON dataset
- Formats data for reinforcement learning with GRPO
- Extracts answer choices (C1, C2, C3, etc.) from various formats
- Saves processed data as Parquet files

### Output Format

The processed data will have:
- `prompt`: User question in chat format
- `ability`: Task type ("5g_network_analysis")
- `reward_model.ground_truth`: Correct answer (e.g., "C1")
- `extra_info`: Metadata for debugging

## 🚀 Training

### Quick Start

```bash
bash start.bash
```

### Training Configuration

Key parameters in `start.bash`:

#### Data Configuration
```bash
data.train_files=./train.parquet
data.val_files=./test.parquet
data.train_batch_size=128
data.max_prompt_length=4096
data.max_response_length=8192
```

#### Model Configuration
```bash
actor_rollout_ref.model.path=/models/Qwen3-32B
actor_rollout_ref.actor.optim.lr=1e-6
actor_rollout_ref.model.use_remove_padding=True
```

#### Distributed Training
```bash
trainer.n_gpus_per_node=8        # NPUs per node
trainer.nnodes=4                  # Number of nodes
actor_rollout_ref.actor.ulysses_sequence_parallel_size=4
actor_rollout_ref.rollout.tensor_model_parallel_size=4
```

#### Memory Optimization
```bash
# Mixed precision training
+actor_rollout_ref.actor.fsdp_config.mixed_precision.param_dtype=bf16
+actor_rollout_ref.actor.fsdp_config.mixed_precision.reduce_dtype=bf16
+actor_rollout_ref.actor.fsdp_config.mixed_precision.buffer_dtype=fp32

# Offloading
actor_rollout_ref.actor.fsdp_config.param_offload=True
actor_rollout_ref.actor.fsdp_config.optimizer_offload=False

# Gradient checkpointing
actor_rollout_ref.model.enable_gradient_checkpointing=True
```

#### GRPO Algorithm Settings
```bash
algorithm.adv_estimator=grpo
algorithm.use_kl_in_reward=False
actor_rollout_ref.actor.kl_loss_coef=0.001
actor_rollout_ref.actor.kl_loss_type=low_var_kl
```

#### Rollout Configuration
```bash
actor_rollout_ref.rollout.name=vllm               # Use vLLM engine
actor_rollout_ref.rollout.n=4                      # Generate 4 responses per prompt
actor_rollout_ref.rollout.gpu_memory_utilization=0.7
actor_rollout_ref.rollout.enable_chunked_prefill=True
actor_rollout_ref.rollout.max_num_batched_tokens=32768
```

### Training Loop

```bash
trainer.total_epochs=50           # Total training epochs
trainer.save_freq=500             # Save checkpoint every 500 steps
trainer.test_freq=50              # Evaluate every 50 steps
```

## 🎁 Reward Function

The reward function (`reward_function.py`) evaluates model outputs using exact match:

### How It Works

1. **Answer Extraction**: Extracts choice from model output (C1, C2, etc.)
   - Searches for `\boxed{...}` format first
   - Falls back to finding C+digit pattern in text
   - Case-insensitive matching

2. **Scoring**:
   - `1.0`: Correct answer
   - `0.0`: Incorrect or unparseable answer

### Testing Reward Function

```bash
python reward_function.py
```

This runs built-in test cases covering various answer formats.

### Custom Reward Function

The reward function is loaded via:
```bash
custom_reward_function.path=./reward_function.py
custom_reward_function.name=compute_score
```

## ⚙️ Configuration

### Checkpoint Management

```bash
trainer.resume_from_path=checkpoints/              # Resume from checkpoint
trainer.default_local_dir="./Qwen3-32B-grpo-telelog"  # Save directory
```

### Logging

```bash
trainer.logger=['console','tensorboard']
trainer.project_name='GRPO-Qwen3'
trainer.experiment_name='GRPO-Qwen3-32b-npu'
```

View TensorBoard logs:
```bash
tensorboard --logdir=./Qwen3-32B-grpo-telelog
```

### Batch Size Tuning

Adjust based on available memory:
```bash
actor_rollout_ref.actor.ppo_mini_batch_size=32
actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4
actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4
```

## 📈 Monitoring

### During Training

Monitor the following metrics:
- **Reward**: Average reward per episode
- **KL Divergence**: Policy deviation from reference
- **Loss**: Actor loss, critic loss
- **Learning Rate**: Current learning rate

### TensorBoard

```bash
tensorboard --logdir=./Qwen3-32B-grpo-telelog --port=6006
```

Access at `http://localhost:6006`

### Console Output

Training progress is logged to console with:
- Current epoch/step
- Batch rewards
- Loss values
- Timing information

## 🔍 Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)

**Solutions**:
- Reduce batch size: `data.train_batch_size`
- Reduce micro batch size: `ppo_micro_batch_size_per_gpu`
- Enable parameter offloading: `param_offload=True`
- Lower GPU memory utilization: `gpu_memory_utilization=0.6`

#### 2. Data Loading Errors

**Check**:
- Data files exist at specified paths
- Parquet files are properly formatted
- Ground truth answers are in correct format (C1, C2, etc.)

#### 3. Slow Training

**Optimize**:
- Enable chunked prefill: `enable_chunked_prefill=True`
- Adjust tensor parallel size
- Check NPU utilization with `npu-smi info`

#### 4. Reward Always Zero

**Debug**:
```bash
# Test reward function independently
python reward_function.py

# Check data format
python -c "import pandas as pd; print(pd.read_parquet('test.parquet').head())"
```

### NPU Monitoring

```bash
# Check NPU status
npu-smi info

# Monitor NPU usage in real-time
watch -n 1 npu-smi info
```

## 📝 Example Workflow

Complete workflow from data to trained model:

```bash
# 1. Prepare your dataset
python transfer.py \
  --dataset_file data/5g_questions.json \
  --save_dir ./processed_data

# 2. Verify data format
python -c "import pandas as pd; df = pd.read_parquet('processed_data/test.parquet'); print(df.head())"

# 3. Test reward function
python reward_function.py

# 4. Start training
bash start.bash

# 5. Monitor training (in another terminal)
tensorboard --logdir=./Qwen3-32B-grpo-telelog

# 6. Resume from checkpoint if needed
bash start.bash trainer.resume_from_path=checkpoints/epoch_10/
```

## 📚 Additional Resources

- **verl-ascend Documentation**: [https://verl.readthedocs.io/en/latest/ascend_tutorial/ascend_quick_start.html]
- **vllm-ascend Documentation**: [https://docs.vllm.ai/projects/vllm-ascend-cn/zh-cn/latest/]


## 🤝 Support

For issues related to:
- **Ascend NPUs**: Check Huawei Ascend documentation
- **verl-ascend**: Refer to verl-ascend GitHub repository
- **Training configuration**: See examples in `start.bash`

