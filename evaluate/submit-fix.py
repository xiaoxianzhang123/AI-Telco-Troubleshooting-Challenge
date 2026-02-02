import csv
import os
import re
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import time
import itertools
import sys


class CSVFixer:
    """CSV文件格式修复工具"""
    
    def __init__(self):
        self.issues_found = []

    def check_csv(self, input_file):
        """
        检查CSV文件中的问题
        :param input_file: 输入CSV文件路径
        :return: 问题列表
        """
        print(f"\n正在检查CSV文件格式: {input_file}")
        self.issues_found = []

        try:
            with open(input_file, 'r', encoding='utf-8') as f:
                content = f.read()

            # 检查1: 未配对的引号
            quote_count = content.count('"')
            if quote_count % 2 != 0:
                self.issues_found.append(f"发现未配对的引号，总数: {quote_count}")

            # 检查2: 逐行检查
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if not line.strip():
                    continue

                # 检查每行的引号数量
                line_quotes = line.count('"')
                if line_quotes % 2 != 0:
                    preview = line[:100] + '...' if len(line) > 100 else line
                    self.issues_found.append(f"第 {i} 行: 引号未配对 - {preview}")

                # 检查是否有未转义的引号
                if re.search(r'(?<!^)(?<!")\"(?!")(?!,)(?!$)', line):
                    preview = line[:100] + '...' if len(line) > 100 else line
                    self.issues_found.append(f"第 {i} 行: 可能存在未转义的引号 - {preview}")

            # 检查3: 尝试用csv模块读取
            try:
                with open(input_file, 'r', encoding='utf-8') as f:
                    reader = csv.DictReader(f)
                    row_count = 0
                    for row in reader:
                        row_count += 1
                print(f"✓ CSV标准解析成功，共 {row_count} 行数据")
            except Exception as e:
                self.issues_found.append(f"CSV解析错误: {str(e)}")

        except Exception as e:
            self.issues_found.append(f"文件读取错误: {str(e)}")

        # 输出检查结果
        if self.issues_found:
            print("=" * 60)
            print("发现CSV格式问题，准备自动修复:")
            print("=" * 60)
            for issue in self.issues_found[:5]:  # 只显示前5个问题
                print(f"⚠ {issue}")
            if len(self.issues_found) > 5:
                print(f"⚠ ... 还有 {len(self.issues_found) - 5} 个问题")
            print("=" * 60)
            return False
        else:
            print("✓ CSV文件格式正确！")
            return True

    def clean_field(self, text):
        """
        清理字段内容
        :param text: 原始文本
        :return: 清理后的文本
        """
        if not isinstance(text, str):
            return text

        # 替换可能导致问题的字符
        text = text.replace('\r\n', ' ')  # 替换Windows换行符
        text = text.replace('\n', ' ')  # 替换Unix换行符
        text = text.replace('\r', ' ')  # 替换Mac换行符

        # 移除多余的空格
        text = ' '.join(text.split())

        return text

    def fix_csv(self, input_file, output_file):
        """
        修复CSV文件
        :param input_file: 输入CSV文件路径
        :param output_file: 输出CSV文件路径
        """
        print(f"\n开始修复CSV文件...")

        try:
            # 读取原始文件
            with open(input_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames
                rows = []

                for row in reader:
                    # 清理每个字段的内容
                    cleaned_row = {}
                    for key, value in row.items():
                        if value:
                            # 移除字段中多余的引号和特殊字符
                            cleaned_value = self.clean_field(value)
                            cleaned_row[key] = cleaned_value
                        else:
                            cleaned_row[key] = value
                    rows.append(cleaned_row)

            # 写入修复后的文件，使用正确的CSV格式
            with open(output_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_MINIMAL,  # 只在必要时使用引号
                    escapechar='\\'  # 使用反斜杠转义
                )
                writer.writeheader()
                writer.writerows(rows)

            print(f"✓ CSV格式修复完成！共处理 {len(rows)} 行数据")
            print(f"✓ 修复后的文件已保存到: {output_file}")

            # 验证修复后的文件
            print("\n验证修复后的文件...")
            is_valid = self.check_csv(output_file)
            
            return is_valid

        except Exception as e:
            print(f"✗ CSV修复失败: {str(e)}")
            import traceback
            traceback.print_exc()
            return False


class ModelBatchProcessor:
    def __init__(self, api_configs=None, max_workers=5):
        """
        初始化处理器 - 支持多服务器
        :param api_configs: API配置列表，每个配置包含 api_key 和 base_url
        :param max_workers: 最大并发线程数
        """
        if api_configs is None or len(api_configs) == 0:
            # 默认配置：使用环境变量和默认URL
            api_configs = [{"api_key": os.getenv("OPENAI_API_KEY"), "base_url": None}]

        # 为每个服务器创建OpenAI客户端
        self.clients = []
        for i, config in enumerate(api_configs):
            client = OpenAI(
                api_key=config.get("api_key") or os.getenv("OPENAI_API_KEY"),
                base_url=config.get("base_url")
            )
            self.clients.append({
                "client": client,
                "name": f"Server-{i + 1}",
                "url": config.get("base_url", "默认服务器")
            })

        # 创建循环迭代器用于轮询服务器
        self.client_cycle = itertools.cycle(range(len(self.clients)))
        self.max_workers = max_workers
        self.lock = Lock()  # 用于线程安全
        self.server_stats = {i: {"success": 0, "error": 0, "total_time": 0.0}
                             for i in range(len(self.clients))}
        
        # 创建CSV修复器
        self.csv_fixer = CSVFixer()

        print(f"初始化完成：共 {len(self.clients)} 台服务器")
        for i, client_info in enumerate(self.clients):
            print(f"  {client_info['name']}: {client_info['url']}")
        print()

    def print_progress_bar(self, current, total, bar_length=50, prefix='进度'):
        """
        打印进度条（适合nohup日志）
        """
        percent = float(current) / total
        filled_length = int(bar_length * percent)
        bar = '█' * filled_length + '-' * (bar_length - filled_length)

        progress_text = f'\r{prefix}: |{bar}| {current}/{total} ({percent * 100:.1f}%)\n'
        sys.stdout.write(progress_text)
        sys.stdout.flush()

    def get_next_client(self):
        """
        获取下一个客户端（轮询策略）
        """
        with self.lock:
            client_index = next(self.client_cycle)
        return client_index, self.clients[client_index]

    def call_model(self, model_name, question, system_prompt="", max_tokens=None, temperature=1.0):
        """
        调用单个模型 - 自动选择服务器
        """
        client_index, client_info = self.get_next_client()
        client = client_info["client"]
        server_name = client_info["name"]

        start_time = time.time()
        try:
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": question})

            # 构建API调用参数
            api_params = {
                "model": model_name,
                "messages": messages,
                "temperature": temperature
            }

            if max_tokens is not None:
                api_params["max_tokens"] = max_tokens

            response = client.chat.completions.create(**api_params)
            answer = response.choices[0].message.content

            # 记录成功统计
            elapsed = time.time() - start_time
            with self.lock:
                self.server_stats[client_index]["success"] += 1
                self.server_stats[client_index]["total_time"] += elapsed

            return answer, client_index

        except Exception as e:
            elapsed = time.time() - start_time
            with self.lock:
                print(f"调用模型 {model_name} 时出错 (使用 {server_name}): {str(e)}")
                self.server_stats[client_index]["error"] += 1
                self.server_stats[client_index]["total_time"] += elapsed
            return f"ERROR: {str(e)}", client_index

    def process_single_task(self, task_info):
        """
        处理单个任务（一个ID的一个模型的一次生成）
        """
        question_id = task_info['id']
        question = task_info['question']
        model = task_info['model']
        system_prompt = task_info['system_prompt']
        max_tokens = task_info.get('max_tokens')
        temperature = task_info.get('temperature', 1.0)
        iteration = task_info['iteration']

        with self.lock:
            print(f"[线程] 开始处理 ID: {question_id}, 模型: {model}, 第{iteration}次")

        task_start = time.time()
        answer, server_index = self.call_model(model, question, system_prompt, max_tokens, temperature)
        elapsed_time = time.time() - task_start

        server_name = self.clients[server_index]["name"]

        with self.lock:
            print(
                f"[线程] 完成 ID: {question_id}, 模型: {model}, 第{iteration}次, 使用: {server_name}, 耗时: {elapsed_time:.2f}秒")

        return {
            'id': question_id,
            'model': model,
            'iteration': iteration,
            'answer': answer,
            'server': server_name
        }

    def write_csv_safely(self, output_file, fieldnames, rows):
        """
        安全地写入CSV文件，确保格式正确
        """
        temp_file = output_file + ".tmp"
        
        try:
            # 先写入临时文件
            with open(temp_file, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(
                    f,
                    fieldnames=fieldnames,
                    quoting=csv.QUOTE_MINIMAL,
                    escapechar='\\'
                )
                writer.writeheader()
                writer.writerows(rows)
            
            # 检查临时文件格式
            print("\n检查输出文件格式...")
            is_valid = self.csv_fixer.check_csv(temp_file)
            
            if is_valid:
                # 格式正确，直接重命名为目标文件
                if os.path.exists(output_file):
                    os.remove(output_file)
                os.rename(temp_file, output_file)
                print(f"✓ 文件格式正确，已保存到: {output_file}")
            else:
                # 格式有问题，自动修复
                print("\n检测到格式问题，正在自动修复...")
                success = self.csv_fixer.fix_csv(temp_file, output_file)
                
                # 清理临时文件
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                
                if success:
                    print(f"✓ 文件已修复并保存到: {output_file}")
                else:
                    print(f"⚠ 文件已保存到: {output_file}，但可能仍存在格式问题，请手动检查")
                    
        except Exception as e:
            print(f"✗ 写入CSV文件时出错: {str(e)}")
            # 清理临时文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
            raise

    def process_csv(self, input_file, output_file, models, system_prompt="", max_tokens=None, temperature=1.0,
                    num_iterations=4, max_rows=None):
        """
        处理CSV文件（多线程版本）- 每个问题生成多次，多服务器负载均衡
        """
        start_total_time = time.time()

        # 读取输入CSV
        with open(input_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        # 如果指定了max_rows，只取前N行
        original_row_count = len(rows)
        if max_rows is not None and max_rows > 0:
            rows = rows[:max_rows]
            print(f"注意: 已设置最大处理行数为 {max_rows}，原始数据共 {original_row_count} 行，实际处理 {len(rows)} 行\n")

        # 定义所有可能的模型列
        all_models = ["Qwen3-32B", "Qwen2.5-7B-Instruct", "Qwen2.5-1.5B-Instruct"]

        # 准备所有任务
        tasks = []
        for row in rows:
            question_id = row['ID']
            question = row['question']

            # 对每个模型，生成num_iterations次
            for model in models:
                for iteration in range(1, num_iterations + 1):
                    tasks.append({
                        'id': question_id,
                        'question': question,
                        'model': model,
                        'system_prompt': system_prompt,
                        'max_tokens': max_tokens,
                        'temperature': temperature,
                        'iteration': iteration
                    })

        print(f"{'=' * 60}")
        print(f"任务信息:")
        print(f"  总任务数: {len(tasks)} 个")
        print(f"  问题数: {len(rows)} 个" + (f" (限制前{max_rows}条)" if max_rows else ""))
        print(f"  模型数: {len(models)} 个")
        print(f"  每问题生成次数: {num_iterations} 次")
        print(f"  并发线程数: {self.max_workers} 个")
        print(f"  服务器数: {len(self.clients)} 台")
        if max_tokens:
            print(f"  最大输出token数: {max_tokens}")
        print(f"  温度参数: {temperature}")
        print(f"{'=' * 60}\n")

        # 使用线程池执行任务
        results_dict = {}
        completed = 0

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(self.process_single_task, task): task
                              for task in tasks}

            # 收集结果
            for future in as_completed(future_to_task):
                result = future.result()
                question_id = result['id']
                model = result['model']
                iteration = result['iteration']
                answer = result['answer']

                # 存储结果 - 使用嵌套字典结构
                if question_id not in results_dict:
                    results_dict[question_id] = {}
                if model not in results_dict[question_id]:
                    results_dict[question_id][model] = {}
                results_dict[question_id][model][iteration] = answer

                completed += 1

                # 更新进度条
                self.print_progress_bar(completed, len(tasks))

        # 构建输出结果 - 每个原始ID扩展为num_iterations行
        output_rows = []
        for row in rows:
            original_id = row['ID']

            # 为每次迭代创建一行
            for iteration in range(1, num_iterations + 1):
                result_row = {'ID': f"{original_id}_{iteration}"}

                # 为每个模型添加对应的结果
                for model in all_models:
                    if model in models and original_id in results_dict and model in results_dict[original_id]:
                        result_row[model] = results_dict[original_id][model].get(iteration, "placeholder")
                    else:
                        result_row[model] = "placeholder"

                output_rows.append(result_row)

        # 使用安全写入方法（自动检查和修复格式）
        fieldnames = ['ID'] + all_models
        self.write_csv_safely(output_file, fieldnames, output_rows)

        total_time = time.time() - start_total_time

        # 打印详细统计信息
        print(f"\n{'=' * 60}")
        print(f"处理完成！")
        print(f"{'=' * 60}")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"平均每个任务: {total_time / len(tasks):.2f}秒")
        print(f"输出行数: {len(output_rows)} 行（{len(rows)}个问题 × {num_iterations}次）")
        print(f"结果已保存到: {output_file}")
        print(f"\n服务器负载统计:")
        print(f"{'-' * 60}")
        for i, client_info in enumerate(self.clients):
            stats = self.server_stats[i]
            total_calls = stats["success"] + stats["error"]
            avg_time = stats["total_time"] / total_calls if total_calls > 0 else 0
            print(f"  {client_info['name']} ({client_info['url']}):")
            print(f"    成功: {stats['success']} 次")
            print(f"    失败: {stats['error']} 次")
            print(f"    总计: {total_calls} 次")
            print(f"    平均耗时: {avg_time:.2f}秒")
        print(f"{'=' * 60}")


# 使用示例
if __name__ == "__main__":
    # 配置参数
    INPUT_FILE = "phase_2_test.csv"  # 输入文件路径
    OUTPUT_FILE = "output-RL.csv"  # 输出文件路径

    # 你想调用的模型列表
    MODELS_TO_CALL = [
        "Qwen3-32B"
    ]

    # 系统提示词（可选）
    SYSTEM_PROMPT = """
    """

    # ============ 多服务器配置 ============
    API_CONFIGS = [
        {
            "api_key": "your-api-key-1",
            "base_url": "http://10.238.190.179:1025/v1"
        },
        {
            "api_key": "your-api-key-2",
            "base_url": "http://10.238.190.180:1025/v1"
        }
    ]
    
    # 多线程配置
    MAX_WORKERS = 90

    # 模型参数配置
    MAX_TOKENS = 8192
    TEMPERATURE = 0.3

    # 生成次数配置
    NUM_ITERATIONS = 4

    # 数据行数限制（可选）
    MAX_ROWS = None  # 设置为None处理全部数据

    # 创建处理器并执行
    processor = ModelBatchProcessor(
        api_configs=API_CONFIGS,
        max_workers=MAX_WORKERS
    )

    processor.process_csv(
        input_file=INPUT_FILE,
        output_file=OUTPUT_FILE,
        models=MODELS_TO_CALL,
        system_prompt=SYSTEM_PROMPT,
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE,
        num_iterations=NUM_ITERATIONS,
        max_rows=MAX_ROWS
    )
