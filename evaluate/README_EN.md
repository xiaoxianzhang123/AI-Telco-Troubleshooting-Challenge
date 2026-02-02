# Model Inference Service and Batch Evaluation Tool

This project provides a vLLM-based model inference service startup script and batch evaluation tool with support for multi-server load balancing and concurrent processing.

## 📋 Table of Contents

- [Hardware Environment](#hardware-environment)
- [Software Environment](#software-environment)
- [Quick Start](#quick-start)
- [Inference Service](#inference-service)
- [Batch Evaluation](#batch-evaluation)
- [Configuration Guide](#configuration-guide)
- [FAQ](#faq)

## 🖥️ Hardware Environment

- **NPU**: 8 × Huawei Ascend 910B2
- **Driver Version**: 25.3.rc1
- **Firmware Version**: 7.8.0.2.212

## 🔧 Software Environment

- **Inference Engine**: vllm-ascend 0.11.0
- **Python**: 3.8+
- **Dependencies**: 
  - openai
  - csv (standard library)
  - concurrent.futures (standard library)

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install openai
```

### 2. Start Inference Service

```bash
bash start-infer.sh
```

### 3. Run Batch Evaluation

```bash
python model_batch_processor.py
```

Or run in background using nohup:

```bash
nohup python model_batch_processor.py > output.log 2>&1 &
```

## 🔌 Inference Service

### Startup Script Overview

`start-infer.sh` is used to launch the vLLM inference service.

#### Script Content

```bash
vllm serve /models-infer/qwen3-32b-after-rl/ \
  --served-model-name "Qwen3-32B" \
  --tensor-parallel-size 8 \
  --gpu-memory-utilization 0.95 \
  --max_model_len 32768 \
  --trust-remote-code \
  --port 1025
```

#### Parameter Description

| Parameter | Value | Description |
|-----------|-------|-------------|
| Model Path | `/models-infer/qwen3-32b-after-rl/` | Path to model files |
| `--served-model-name` | `Qwen3-32B` | Model name in service |
| `--tensor-parallel-size` | `8` | Tensor parallelism (matches 8 NPU cards) |
| `--gpu-memory-utilization` | `0.95` | NPU memory utilization ratio |
| `--max_model_len` | `32768` | Maximum context length |
| `--trust-remote-code` | - | Allow execution of custom code in model |
| `--port` | `1025` | Service port number |

#### Service Endpoint

After startup, the service will be available at:
```
http://localhost:1025/v1
```

### Modifying Configuration

To adjust configuration, edit the `start-infer.sh` file:

- **Change Port**: Modify `--port` parameter
- **Adjust Memory**: Modify `--gpu-memory-utilization` (between 0-1)
- **Change Model Path**: Modify the first parameter path
- **Adjust Max Length**: Modify `--max_model_len` parameter

## 📊 Batch Evaluation

### Key Features

✅ **Multi-Server Support**: Configure multiple inference servers with automatic load balancing  
✅ **Concurrent Processing**: Multi-threaded concurrent calls for improved processing speed  
✅ **Multiple Generations**: Support generating multiple results for the same question (for RL training data preparation)  
✅ **CSV Format Checking**: Automatic detection and repair of CSV format issues  
✅ **Progress Tracking**: Real-time display of processing progress and statistics  
✅ **Error Handling**: Comprehensive exception handling and retry mechanism

### Input File Format

CSV file should contain the following fields:
- `ID`: Unique identifier for the question
- `question`: Question content

Example:
```csv
ID,question
1,什么是人工智能？
2,如何训练大语言模型？
```

### Output File Format

Output CSV contains the following fields:
- `ID`: Format is `original_ID_iteration_number` (e.g., `1_1`, `1_2`)
- Model columns: Contains generated answers

Example:
```csv
ID,Qwen3-32B,Qwen2.5-7B-Instruct,Qwen2.5-1.5B-Instruct
1_1,答案1,placeholder,placeholder
1_2,答案2,placeholder,placeholder
```

### Configuration Parameters

Configure in the `if __name__ == "__main__":` section at the bottom of the script:

#### Basic Configuration

```python
# File paths
INPUT_FILE = "phase_2_test.csv"      # Input file
OUTPUT_FILE = "output-RL.csv" # Output file

# Models to call
MODELS_TO_CALL = [
    "Qwen3-32B"
]

# System prompt (optional)
SYSTEM_PROMPT = """
Your system prompt here
"""
```

#### Multi-Server Configuration

```python
API_CONFIGS = [
    {
        "api_key": "your-api-key-1",
        "base_url": "http://IP:1025/v1"
    },
    {
        "api_key": "your-api-key-2",
        "base_url": "http://IP:1025/v1"
    }
]
```

#### Performance Configuration

```python
MAX_WORKERS = 90          # Concurrent thread count
MAX_TOKENS = 8192         # Maximum generation tokens
TEMPERATURE = 0.3         # Sampling temperature
NUM_ITERATIONS = 4        # Number of generations per question
MAX_ROWS = None           # Row limit (None = all rows)
```

### Usage Examples

#### Example 1: Single Server Evaluation

```python
API_CONFIGS = [
    {
        "api_key": "dummy-key",
        "base_url": "http://localhost:1025/v1"
    }
]

processor = ModelBatchProcessor(
    api_configs=API_CONFIGS,
    max_workers=10
)

processor.process_csv(
    input_file="test.csv",
    output_file="results.csv",
    models=["Qwen3-32B"],
    num_iterations=1
)
```

#### Example 2: Multi-Server Concurrent Evaluation

```python
API_CONFIGS = [
    {"api_key": "key1", "base_url": "http://server1:1025/v1"},
    {"api_key": "key2", "base_url": "http://server2:1025/v1"},
    {"api_key": "key3", "base_url": "http://server3:1025/v1"}
]

processor = ModelBatchProcessor(
    api_configs=API_CONFIGS,
    max_workers=90
)

processor.process_csv(
    input_file="questions.csv",
    output_file="answers.csv",
    models=["Qwen3-32B"],
    temperature=0.7,
    num_iterations=4,
    max_rows=100  # Process only first 100 rows
)
```

### Understanding Output Information

During execution, you'll see:

```
============================================================
任务信息:
  总任务数: 400 个
  问题数: 100 个
  模型数: 1 个
  每问题生成次数: 4 次
  并发线程数: 90 个
  服务器数: 2 台
  最大输出token数: 8192
  温度参数: 0.3
============================================================

进度: |███████████████---| 150/400 (37.5%)

============================================================
处理完成！
============================================================
总耗时: 245.32秒
平均每个任务: 0.61秒
输出行数: 400 行（100个问题 × 4次）
结果已保存到: output.csv

服务器负载统计:
------------------------------------------------------------
  Server-1 (http://server1:1025/v1):
    成功: 198 次
    失败: 2 次
    总计: 200 次
    平均耗时: 0.58秒
  Server-2 (http://server2:1025/v1):
    成功: 200 次
    失败: 0 次
    总计: 200 次
    平均耗时: 0.64秒
============================================================
```

## ⚙️ Configuration Guide

### Performance Tuning Recommendations

#### Concurrent Thread Count (MAX_WORKERS)

- **Single Server**: Recommended 10-30
- **Multiple Servers**: Can be increased appropriately, suggested = number of servers × 30
- **Note**: Too high may cause server overload

#### Temperature Parameter (TEMPERATURE)

- **0.0-0.3**: Deterministic output, suitable for scenarios requiring consistency
- **0.7-1.0**: Diverse output, suitable for creative tasks
- **>1.0**: Highly random, may produce incoherent content

#### Maximum Tokens (MAX_TOKENS)

Adjust based on task type:
- Short answers: 512-1024
- Medium length: 2048-4096
- Long text generation: 4096-8192

### CSV Format Checker

The tool includes built-in CSV format checking and automatic repair:

**Check Items**:
- Unpaired quotation marks
- Unescaped special characters
- Newline character handling
- CSV standard format validation

**Automatic Repair**:
- Clean newline characters in fields
- Standardize quotation mark usage
- Remove excess whitespace
- Ensure CSV format compliance

## ❓ FAQ

### Q1: How to check if the inference service is running properly?

```bash
# Check service port
netstat -tulnp | grep 1025

# Test API connection
curl http://localhost:1025/v1/models
```

### Q2: What to do about memory overflow?

Adjust the following parameters:
- Lower `--gpu-memory-utilization` (e.g., 0.90)
- Reduce `--max_model_len`
- Lower `MAX_WORKERS` concurrency

### Q3: What to do about CSV file format errors?

The tool will automatically detect and repair. If issues persist:
```python
from model_batch_processor import CSVFixer

fixer = CSVFixer()
fixer.check_csv("your_file.csv")
fixer.fix_csv("your_file.csv", "fixed_file.csv")
```

### Q4: How to monitor background running tasks?

```bash
# View log in real-time
tail -f output.log

# Check process
ps aux | grep python

# Stop task
kill <PID>
```
