import re


def extract_choice(text):
    """
    从文本中提取选项编号（如 C1, C2, C3 等）
    支持多种格式：
    - C1, C2, ... C8
    - 也可以扩展到 C1-C99
    """
    # 匹配 C 后面跟1-2位数字
    choice_pattern = r'C\d{1,2}'
    matches = re.findall(choice_pattern, text)
    
    if matches:
        # 返回最后一个匹配（通常是最终答案）
        return matches[-1]
    return None


def extract_answer(solution_str):
    """
    从模型生成的solution中提取答案
    优先级：
    1. \boxed{} 中的内容
    2. 直接在文本中的 C1, C2 等格式
    """
    # 首先尝试从 \boxed{} 中提取
    box_pattern = r'\\boxed\{([^}]*)\}'
    box_matches = re.findall(box_pattern, solution_str, re.DOTALL)
    
    if box_matches:
        # 从 boxed 内容中提取选项
        for match in reversed(box_matches):  # 从后往前找，取最后一个
            choice = extract_choice(match)
            if choice:
                return choice
    
    # 如果 boxed 中没找到，直接在整个文本中查找
    choice = extract_choice(solution_str)
    return choice


def compute_score(data_source, solution_str, ground_truth, extra_info=None):
    """
    计算强化学习奖励分数
    
    Args:
        data_source: 数据来源
        solution_str: 模型生成的答案字符串
        ground_truth: 正确答案（如 "C1", "C2" 等）
        extra_info: 额外信息（可选）
    
    Returns:
        float: 奖励分数
            - 1.0: 答案完全正确
            - 0.0: 答案错误或无法提取答案
    """
    # 从模型输出中提取答案
    predicted_answer = extract_answer(solution_str)
    
    # 如果无法提取答案，返回0分
    if predicted_answer is None:
        return 0.0
    
    # 标准化 ground_truth（去除空格，转大写）
    gt = extract_choice(ground_truth)
    if gt is None:
        gt = ground_truth.strip().upper()
    
    # 比较答案（不区分大小写）
    if predicted_answer.upper() == gt.upper():
        return 1.0
    else:
        return 0.0


# 测试函数
def test_reward_function():
    """测试奖励函数的各种情况"""
    test_cases = [
        # (solution_str, ground_truth, expected_score, description)
        ("The answer is \\boxed{C1}", "C1", 1.0, "标准格式-正确"),
        ("After analysis, \\boxed{C2}", "C1", 0.0, "标准格式-错误"),
        ("C3 is the correct answer", "C3", 1.0, "无boxed-正确"),
        ("I think the answer should be C4", "C5", 0.0, "无boxed-错误"),
        ("\\boxed{The answer is C1}", "C1", 1.0, "boxed内有文字"),
        ("No clear answer here", "C1", 0.0, "无法提取答案"),
        ("Multiple: C1, C2, final \\boxed{C3}", "C3", 1.0, "多个选项-取最后"),
        ("", "C1", 0.0, "空字符串"),
        ("\\boxed{c1}", "C1", 1.0, "大小写不敏感"),
    ]
    
    print("=" * 80)
    print("奖励函数测试")
    print("=" * 80)
    
    for i, (solution, gt, expected, desc) in enumerate(test_cases, 1):
        score = compute_score("test_source", solution, gt)
        status = "✓" if score == expected else "✗"
        print(f"{status} 测试 {i}: {desc}")
        print(f"   输入: {solution[:50]}...")
        print(f"   正确答案: {gt}, 得分: {score}, 期望: {expected}")
        if score != expected:
            print(f"   ⚠️  测试失败！")
        print()


if __name__ == "__main__":
    # 运行测试
    test_reward_function()
