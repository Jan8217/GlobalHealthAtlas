"""
Model inference script without training, designed for benchmarking on multiple languages.
"""

import json
import os
import time
import re
import threading
import logging
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI, APIError, RateLimitError, Timeout
from typing import List, Dict, Optional, Set, Tuple
from tqdm import tqdm

# ================= 配置区域 =================
API_KEY = os.getenv('DASHSCOPE_API_KEY')
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwq-32b"

DATA_DIR = Path(r'/home/ubuntu/Projects/ICML/data_2W')
RESULT_DIR = Path(r'/home/ubuntu/Projects/ICML/output/data2W_qwq_32b')
RESULT_DIR.mkdir(parents=True, exist_ok=True)

DEBUG_MODE = False  # True: 只处理前20题测试; False: 全量运行
MAX_WORKERS = 10  # 最大并发线程数
REQUEST_TIMEOUT = 300.0  # 请求超时时间(秒)
CHECK_REASONING = False  # True: 检查思维链; False: 不检查思维链

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
output_lock = threading.Lock()
failed_ids_lock = threading.Lock()

logging.basicConfig(
    filename='benchmark_processing.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ================= 强约束 Prompt (确保思维链) =================
SYSTEM_PROMPT = (
    "You are a Public Health Expert. You MUST strictly follow these rules:\n"
    "1) Strict Language Alignment: You MUST detect the language of the question and respond EXCLUSIVELY in that same language. Do not use English to explain non-English questions."
    "2) Ultra-brief reasoning: Regardless of question type, your medical/scientific rationale MUST be extremely concise: "
    "at most 2–3 key logical points OR no more than 3 sentences. Do NOT provide long reasoning.\n"
    "3) Question-type routing (VERY IMPORTANT):\n"
    "   - If the question is a single-choice A/B/C/D item (the prompt contains options like A., B., C., D. or similar), "
    "your final output MUST be ONLY one letter: A or B or C or D. Do NOT output any explanation, label, or extra characters.\n"
    "   - If the question is an open-ended Q&A (NOT a single-choice A/B/C/D item), "
    "answer directly in a concise, professional manner (1–3 sentences). No extra formatting.\n"
)

USER_PROMPT_TEMPLATE = (
    "Please answer in the original language of the question.\n"
    "If it is a single-choice question: output ONLY one letter among A/B/C/D.\n"
    "If it is an open-ended Q&A: provide a concise, professional answer in 1–3 sentences with no extra formatting.\n"
    "Also: regardless of question type, keep the medical/scientific rationale extremely brief "
    "(at most 2–3 key points OR no more than 3 sentences).\n\n"
    "{question}"
)


# ================= 核心处理逻辑 =================

def get_response(item: Dict, lang_name: str) -> Tuple[int, Optional[Dict], Optional[str]]:
    """处理单个问题，返回 (id, 结果字典, 错误信息)"""
    q_id = item.get("id")
    if q_id is None:
        return None, None, "No ID found"

    question = item.get("question", "").strip()
    label = item.get("label", "")

    language_instruction = f"IMPORTANT: Both reasoning and answer MUST be in {lang_name}."

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",
                 "content": f"{USER_PROMPT_TEMPLATE.format(question=question)}\n\n{language_instruction}"}
            ],
            temperature=0.5,
            top_p=1.0,
            max_tokens=512,
            presence_penalty=0,
            frequency_penalty=0,
            stream=True,
            extra_body={"enable_thinking": True},
            timeout=REQUEST_TIMEOUT
        )

        content = ""
        reasoning = ""
        for chunk in completion:
            if chunk.choices:
                delta = chunk.choices[0].delta
                if hasattr(delta, 'content') and delta.content:
                    content += delta.content
                if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                    reasoning += delta.reasoning_content

        msg = type('Message', (), {'content': content, 'reasoning_content': reasoning})()

        # 检查思维链（如果开关开启）
        if CHECK_REASONING and not reasoning.strip():
            err_msg = "No reasoning content found"
            print(f"Failed to process ID={q_id} [{lang_name}]: {err_msg}")
            logging.error(f"Failed to process ID={q_id} [{lang_name}]: {err_msg}")
            return q_id, None, err_msg

        # 清理答案
        if label == "Single-Choice":
            match = re.search(r'([A-D])', content.upper())
            clean_ans = match.group(1) if match else content
        else:
            clean_ans = content

        logging.info(
            f"Successfully processed ID={q_id} [{lang_name}]: Answer={clean_ans}, Reasoning={reasoning.strip()}")

        return q_id, {
            "id": q_id,
            "question": question,
            "gold_answer": item.get("answer"),
            "complexCOT": item.get("complexCOT"),
            "model_answer": clean_ans,
            "model_reasoning": reasoning.strip(),
            "meta": {
                "domain": item.get("domain"),
                "difficulty": item.get("difficulty"),
                "language": lang_name
            }
        }, None

    except Exception as e:
        import traceback
        err_type = type(e).__name__
        if err_type == 'Timeout':
            err_msg = f"Timeout after {REQUEST_TIMEOUT}s"
        elif err_type == 'RateLimitError':
            err_msg = "RateLimitError"
        else:
            err_msg = f"Error: {err_type}: {e}"
            stack_trace = traceback.format_exc()
            logging.error(f"Failed to process ID={q_id} [{lang_name}]: {err_msg}{stack_trace}")
        print(f"Failed to process ID={q_id} [{lang_name}]: {err_msg}")
        logging.error(f"Failed to process ID={q_id} [{lang_name}]: {err_msg}")
        return q_id, None, err_msg


def load_existing_ids(output_path: str) -> Set[int]:
    """加载已处理的ID集合，用于断点续传"""
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        ids = {item["id"] for item in data if isinstance(item.get("id"), (int, str))}
        print(f"Loaded {len(ids)} existing IDs from {output_path}")
        return ids
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"文件 {output_path} 可能损坏，将重新处理所有项目")
        return set()
    except Exception as e:
        print(f"Error loading existing IDs: {e}")
        return set()


def save_output(output_map: Dict, path: str):
    """保存输出结果到文件"""
    with output_lock:
        output_list = list(output_map.values())
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(output_list, f, ensure_ascii=False, indent=2)


def process_language(file_path: Path):
    """处理单个语言文件"""
    lang = file_path.stem
    print(f"\n🚀 启动评测 [语言: {lang}]")
    with open(file_path, 'r', encoding='utf-8') as f:
        items = json.load(f)
    valid_items = [item for item in items if item.get("id") is not None]
    print(f"原始数据: {len(items)} 项, 有效数据: {len(valid_items)} 项")
    if DEBUG_MODE:
        valid_items = valid_items[:80]
        print(f"DEBUG模式: 只处理前20项")
    save_path = RESULT_DIR / f"result_{lang}.json"
    existing_ids = load_existing_ids(str(save_path))
    to_process = [item for item in valid_items if item.get("id") not in existing_ids]

    print(f"需要处理: {len(to_process)} 项 (跳过 {len(valid_items) - len(to_process)} 已完成项)")

    if not to_process:
        print(f"✅ {lang} 所有项目已处理完成")
        return

    # 加载已有结果
    if os.path.exists(save_path):
        try:
            with open(save_path, 'r', encoding='utf-8') as f:
                output_data = json.load(f)
            output_map = {item["id"]: item for item in output_data if "id" in item}
            print(f"已加载 {len(output_map)} 个已有结果")
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            print(f"文件 {save_path} 可能损坏，将重新处理所有项目")
            output_map = {}
        except Exception as e:
            print(f"加载已有结果时出错: {e}")
            output_map = {}
    else:
        output_map = {}

    failed_ids = []
    completed = 0
    success_batch = []  # 用于批量记录成功处理的ID

    print(f"Starting concurrent processing with {MAX_WORKERS} workers...")
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_item = {
            executor.submit(get_response, item, lang): item
            for item in to_process
        }
        with tqdm(total=len(to_process), desc=f"Processing {lang[:10]}", unit="item") as pbar:
            for future in as_completed(future_to_item):
                qid, result, error = future.result()
                item = future_to_item[future]

                if result is not None:
                    with output_lock:
                        output_map[qid] = result
                    completed += 1
                    success_batch.append(qid)
                    pbar.set_postfix({"ID": qid})
                    pbar.update(1)

                    # 每25条保存检查点并打印成功ID
                    if completed % 25 == 0 or completed == len(to_process):
                        save_output(output_map, str(save_path))
                        print(f"检查点: 已保存 {len(output_map)} 个项目")
                        print(f"成功处理的ID: {success_batch}")
                        success_batch = []
                else:
                    with failed_ids_lock:
                        failed_ids.append(qid)
                    pbar.set_postfix({"ID": qid, "Error": error[:20]})
                    pbar.update(1)
                    print(f"失败 ID={qid}: {error} (进度: {completed}/{len(to_process)})")
                    logging.error(f"Failed ID={qid} [{lang}]: {error}")
                    time.sleep(1)

    save_output(output_map, str(save_path))
    print(f"\n{'=' * 50}")
    print(f"{lang} 处理完成")
    print(f"本次处理: {len(to_process)}")
    print(f"成功: {completed}")
    print(f"失败: {len(failed_ids)}")
    if failed_ids:
        print(f"失败的ID列表: {failed_ids}")
    print(f"结果保存至: {save_path.name}")


def run_benchmark():
    """运行所有语言的评测"""
    json_files = list(DATA_DIR.glob("*.json"))
    if not json_files:
        print("未找到 JSON 文件，请检查路径。")
        return

    print(f"检测到 {len(json_files)} 个语言文件。模式: {'DEBUG' if DEBUG_MODE else 'FULL'}")
    for file_path in json_files:
        process_language(file_path)


if __name__ == "__main__":
    run_benchmark()
