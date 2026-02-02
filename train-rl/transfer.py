import argparse
import os
import re
import datasets


def extract_solution(answer_str):
    """
    从答案字符串中提取选项编号（如 C1, C2 等）
    支持多种格式：
    - 直接是 "C1"
    - 在 \\boxed{C1} 中
    - 在其他标记中
    """
    # 如果答案本身就是简单的选项格式（如 "C1"）
    if re.match(r'^C\d+$', answer_str.strip()):
        return answer_str.strip()
    
    # 尝试从 \boxed{} 中提取
    box_pattern = r'\\boxed\{([^}]*)\}'
    box_matches = re.findall(box_pattern, answer_str)
    if box_matches:
        # 从匹配结果中提取 C+数字 格式
        for match in box_matches:
            option = re.search(r'C\d+', match)
            if option:
                return option.group()
    
    # 直接在整个字符串中查找 C+数字 格式
    option = re.search(r'C\d+', answer_str)
    if option:
        return option.group()
    
    # 如果都找不到，返回原始答案
    return answer_str.strip()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='处理5G网络分析数据集，转换为强化学习格式')
    parser.add_argument("--dataset_file", required=True, help="输入的JSON数据集文件路径")
    parser.add_argument("--save_dir", required=True, help="保存处理后数据的目录")
    args = parser.parse_args()
    
    data_source = args.dataset_file
    local_dir = args.save_dir
    
    # 创建保存目录
    os.makedirs(local_dir, exist_ok=True)
    
    # 加载数据集
    dataset = datasets.load_dataset("json", data_files=data_source)
    train_dataset = dataset["train"]
    
    def make_map_fn(split):
        def process_fn(example, idx):
            # 提取问题和答案
            question = example.get("question", "")
            answer_raw = example.get("answer", "")
            
            # 提取标准答案（如 C1, C2 等）
            ground_truth = extract_solution(answer_raw)
            
            # 构建新的数据格式
            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ],
                "ability": "5g_network_analysis",  # 标记为5G网络分析能力
                "reward_model": {
                    "style": "rule",  # 基于规则的奖励：答案匹配则为1，否则为0
                    "ground_truth": ground_truth
                },
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "question": question,
                    "answer_raw": answer_raw,
                },
            }
            return data
        
        return process_fn
    
    # 处理数据集
    print(f"正在处理数据集: {data_source}")
    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
    
    # 保存为 Parquet 格式
    output_path = os.path.join(local_dir, "test.parquet")
    train_dataset.to_parquet(output_path)
    print(f"数据集已保存到: {output_path}")
    print(f"共处理 {len(train_dataset)} 条数据")
