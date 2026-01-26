"""
data leakage inference module for n-gram continuation detection using DashScope (OpenAI compatible) API.
"""

import os
import json
import re
import random
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from multiprocessing import Pool, cpu_count, Manager
from functools import partial

from openai import OpenAI


# =========================================================
# 0) 配置区（你只需要改这里）
# =========================================================

API_KEY = os.getenv('DASHSCOPE_API_KEY')
BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
MODEL_NAME = "qwen3-max"

# 可选：用单独的 Judge 模型更好（也可以与 MODEL_NAME 一样）
JUDGE_MODEL_NAME = "qwen3-max"

# 数据路径（支持 json / jsonl）
DATA_PATH = r"/home/ubuntu/Projects/ICML/data_2W/Public_Benchmark_10_27511.json"

output_dir = Path("output") / MODEL_NAME
output_dir.mkdir(parents=True, exist_ok=True)

# 输出文件路径 = output_dir + filename
OUT_REPORT_PATH = output_dir / "5gram_leakage_report_full.json"
OUT_SUSPICIOUS_PATH = output_dir / "5gram_leakage_suspicious.json"
OUT_PREDICTIONS_PATH = output_dir / "5gram_leakage_predictions.json" 


# =========================================================
# 1) Leakage 检测参数（prop1）
# =========================================================

# n-gram continuation: 预测后续 N 个词
N_NEXT_WORDS = 5

# 每个 question 随机抽多少个截断点
SAMPLES_PER_QUESTION = 5

# 单题命中率达到多少，标记为 suspicious
LEAK_RATIO_THRESHOLD = 0.5

# 每次请求间隔
SLEEP_SECONDS = 0.25

# 只抽多少条 question 进行检测（防止成本爆炸）
MAX_QUESTIONS_TO_CHECK = 2000

GLOBAL_SEED = 42

# 并行进程数
NUM_WORKERS = 10


# =========================================================
# 2) prop2 参数（对 suspicious subset 的 accuracy）
# =========================================================

# 是否计算 prop2（需要 data 中有 answer 字段）
ENABLE_PROP2 = True

# Judge 模式：
# - "exact": 只做字符串包含/近似匹配（便宜但不准）
# - "llm": 用 LLM-as-a-judge（推荐）
PROP2_MODE = "llm"  # "exact" or "llm"

# prop2 只评估前 K 个 suspicious（避免太贵）
MAX_SUSPICIOUS_FOR_PROP2 = 30

# 模型回答 question 的生成长度（可调）
MODEL_ANSWER_MAX_TOKENS = 512

# judge 评分的随机性
JUDGE_TEMPERATURE = 0.0


# =========================================================
# 3) 工具函数：文本处理
# =========================================================

def normalize_for_compare(text: str) -> str:
    """
    用于 strict compare 的归一化：
    - 小写
    - 多空格压缩
    - 去掉首尾标点
    """
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" \t\n\r\"'“”‘’.,;:!?()[]{}")
    return text

def split_words(text: str) -> List[str]:
    text = re.sub(r"\s+", " ", text.strip())
    return text.split(" ")

def pick_cut_positions(words: List[str], n_next: int, k: int, rng: random.Random) -> List[int]:
    """
    从句子中随机抽截断点：
      prefix = words[:pos]
      gt_next = words[pos:pos+n_next]
    """
    if len(words) <= n_next + 3:
        return []

    min_cut = 3
    max_cut = len(words) - n_next
    if max_cut <= min_cut:
        return []

    all_pos = list(range(min_cut, max_cut))
    rng.shuffle(all_pos)
    return all_pos[:k]


# =========================================================
# 4) 安全加载 JSON / JSONL
# =========================================================

def safe_json_load(path: str) -> List[Dict[str, Any]]:
    p = Path(path)
    print(f"[DEBUG] Loading file: {p}")

    if not p.exists():
        raise FileNotFoundError(f"[ERROR] File not found: {p}")

    size = p.stat().st_size
    print(f"[DEBUG] File size: {size} bytes")
    if size == 0:
        raise ValueError(f"[ERROR] File is empty (0 bytes): {p}")

    # 标准 JSON（支持 BOM）
    try:
        with open(p, "r", encoding="utf-8-sig") as f:
            content = f.read()
        content_strip = content.strip()
        if not content_strip:
            raise ValueError(f"[ERROR] File content is empty after strip(): {p}")
        return json.loads(content_strip)

    except json.JSONDecodeError:
        # JSONL fallback
        data = []
        with open(p, "r", encoding="utf-8-sig") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                data.append(json.loads(line))
        if not data:
            raise ValueError(f"[ERROR] JSONL parse got empty list: {p}")
        return data


# =========================================================
# 5) DashScope (OpenAI compatible) client
# =========================================================

def build_client() -> OpenAI:
    if not API_KEY:
        raise ValueError(
            "Missing DASHSCOPE_API_KEY.\n"
            "Please set environment variable DASHSCOPE_API_KEY first."
        )
    return OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_chat_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
    enable_thinking: bool = False,
) -> str:
    """
    统一封装调用，便于控制 enable_thinking / max_tokens
    """
    extra_body = {"enable_thinking": enable_thinking}

    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        extra_body=extra_body,
    )
    out = resp.choices[0].message.content.strip()
    out = re.sub(r"\s+", " ", out).strip()
    return out


# =========================================================
# 6) Leakage 检测：n-gram continuation (prop1 的基础)
# =========================================================

def ask_next_words(client: OpenAI, prefix: str, n_words: int) -> str:
    system = (
        "You are a text continuation engine. "
        "You must strictly follow the user's instruction and output format."
    )
    user = (
        "Task: Continue the given text with EXACTLY the next "
        f"{n_words} words.\n"
        "Rules:\n"
        f"1) Output ONLY {n_words} words.\n"
        "2) Words must be separated by single spaces.\n"
        "3) Do NOT add quotes, punctuation, numbering, or extra text.\n"
        "4) If uncertain, still guess, but obey the output format.\n\n"
        f"Text:\n{prefix}"
    )
    return call_chat_completion(
        client=client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=64,
        enable_thinking=False,
    )

def evaluate_single_question_leakage(
    client: OpenAI,
    qid: int,
    question: str,
    n_next: int,
    samples: int,
    rng: random.Random,
) -> Dict[str, Any]:
    words = split_words(question)
    cut_positions = pick_cut_positions(words, n_next=n_next, k=samples, rng=rng)

    # 太短则不测
    if not cut_positions:
        return {
            "id": qid,
            "question": question,
            "tested": 0,
            "hits": 0,
            "hit_ratio": 0.0,
            "api_fail": 0,
            "details": [],
            "note": "Question too short to test n-gram continuation."
        }

    hits = 0
    api_fail = 0
    details = []

    for pos in cut_positions:
        prefix_words = words[:pos]
        gt_next_words = words[pos:pos + n_next]

        prefix = " ".join(prefix_words)
        gt_next = " ".join(gt_next_words)

        try:
            pred = ask_next_words(client, prefix=prefix, n_words=n_next)
        except Exception as e:
            api_fail += 1
            details.append({
                "cut_pos": pos,
                "prefix": prefix,
                "gt_next": gt_next,
                "pred_next": "",
                "match": False,
                "error": str(e)
            })
            time.sleep(SLEEP_SECONDS)
            continue

        pred_norm = normalize_for_compare(pred)
        gt_norm = normalize_for_compare(gt_next)
        match = (pred_norm == gt_norm)

        if match:
            hits += 1

        details.append({
            "cut_pos": pos,
            "prefix": prefix,
            "gt_next": gt_next,
            "pred_next": pred,
            "match": match,
        })

        time.sleep(SLEEP_SECONDS)

    tested = len([d for d in details if "error" not in d])
    hit_ratio = hits / tested if tested > 0 else 0.0

    return {
        "id": qid,
        "tested": tested,
        "hits": hits,
        "hit_ratio": round(hit_ratio, 4),
        "api_fail": api_fail,
        "details": details,
    }


# =========================================================
# 7) prop2：在 suspicious subset 上评估模型准确率
# =========================================================

def model_answer_question(client: OpenAI, question: str) -> str:
    """
    让模型正常回答 question
    """
    system = "You are a helpful assistant. Answer the question clearly and concisely."
    user = f"Question:\n{question}\n\nAnswer:"
    return call_chat_completion(
        client=client,
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.0,
        max_tokens=MODEL_ANSWER_MAX_TOKENS,
        enable_thinking=False,
    )

def exact_match_score(pred: str, gold: str) -> bool:
    """
    低成本 baseline：
    - 归一化后，pred 是否包含 gold 或 gold 是否包含 pred
    （开放式QA不太准，但可以当 fallback）
    """
    p = normalize_for_compare(pred)
    g = normalize_for_compare(gold)
    if not p or not g:
        return False
    return (g in p) or (p in g)

def llm_judge_correctness(
    client: OpenAI,
    question: str,
    gold_answer: str,
    model_answer: str,
) -> Dict[str, Any]:
    """
    LLM-as-a-judge：输出严格 JSON
    """
    system = (
        "You are a strict evaluator for question answering.\n"
        "You must output ONLY valid JSON."
    )

    user = f"""
Evaluate whether the model answer is correct given the gold answer.

Rules:
1) If the model answer matches the meaning of the gold answer, mark correct=true.
2) Minor wording differences are OK.
3) If important factual details are missing or wrong, correct=false.
4) Output ONLY JSON in the schema:
{{
  "correct": true/false,
  "reason": "short reason"
}}

QUESTION:
{question}

GOLD ANSWER:
{gold_answer}

MODEL ANSWER:
{model_answer}
""".strip()

    out = call_chat_completion(
        client=client,
        model=JUDGE_MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=JUDGE_TEMPERATURE,
        max_tokens=256,
        enable_thinking=False,
    )

    # 尝试解析 JSON
    try:
        obj = json.loads(out)
        return {
            "correct": bool(obj.get("correct", False)),
            "reason": str(obj.get("reason", "")).strip()[:300],
            "raw": out
        }
    except Exception:
        # fallback
        return {
            "correct": False,
            "reason": "Judge output is not valid JSON.",
            "raw": out
        }

def compute_prop2_on_suspicious(
    client: OpenAI,
    suspicious_items: List[Dict[str, Any]],
    full_data_map: Dict[int, Dict[str, Any]],
    max_items: int = 30,
    mode: str = "llm",
) -> Dict[str, Any]:
    """
    prop2 = accuracy on suspicious subset
    """
    chosen = suspicious_items[:max_items]
    judged = []
    correct_count = 0
    total = 0

    for s in chosen:
        qid = s["id"]
        item = full_data_map.get(qid, {})
        question = item.get("question", "")
        gold = item.get("answer", "")

        if not question or not gold:
            continue

        # 让模型回答
        try:
            pred_ans = model_answer_question(client, question)
        except Exception as e:
            judged.append({
                "id": qid,
                "error": f"model_answer failed: {e}"
            })
            time.sleep(SLEEP_SECONDS)
            continue

        # 判分
        if mode == "exact":
            ok = exact_match_score(pred_ans, gold)
            judged.append({
                "id": qid,
                "correct": ok,
                "judge_mode": "exact",
                "question": question,
                "gold_answer": gold,
                "model_answer": pred_ans,
                "reason": "exact-match heuristic"
            })
        else:
            judge = llm_judge_correctness(
                client=client,
                question=question,
                gold_answer=gold,
                model_answer=pred_ans,
            )
            ok = judge["correct"]
            judged.append({
                "id": qid,
                "correct": ok,
                "judge_mode": "llm",
                "question": question,
                "gold_answer": gold,
                "model_answer": pred_ans,
                "reason": judge["reason"],
                "judge_raw": judge["raw"],
            })

        total += 1
        if ok:
            correct_count += 1

        time.sleep(SLEEP_SECONDS)

    prop2 = correct_count / total if total > 0 else 0.0
    return {
        "prop2_suspicious_accuracy": round(prop2, 6),
        "prop2_total_judged": total,
        "prop2_correct": correct_count,
        "prop2_details": judged
    }


# =========================================================
# 8) Worker 函数（用于多进程并行）
# =========================================================

def worker_evaluate_question(args) -> Dict[str, Any]:
    """
    子进程 worker：每个进程独立创建 client，处理单个 question
    args: (task, counter, lock, total)
    """
    task, counter, lock, total = args
    qid = task["qid"]
    question = task["question"]
    n_next = task["n_next"]
    samples = task["samples"]
    seed = task["seed"]

    # 每个进程独立创建 client 和 rng
    client = build_client()
    rng = random.Random(seed)

    result = evaluate_single_question_leakage(
        client=client,
        qid=qid,
        question=question,
        n_next=n_next,
        samples=samples,
        rng=rng,
    )

    # 更新进度
    with lock:
        counter.value += 1
        print(f"\r[Progress] {counter.value}/{total} questions processed", end="", flush=True)

    return result


# =========================================================
# 9) 主流程：计算 prop1 + prop2（10进程并行）
# =========================================================

def run_leakage_pipeline(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    rng = random.Random(GLOBAL_SEED)

    # 建一个 id -> item 的 map，给 prop2 用
    full_data_map = {}
    for it in data:
        if "id" in it:
            full_data_map[int(it["id"])] = it

    # 抽样检测
    if len(data) > MAX_QUESTIONS_TO_CHECK:
        data_sample = rng.sample(data, MAX_QUESTIONS_TO_CHECK)
    else:
        data_sample = data

    # 构建任务列表（每个任务有独立的 seed）
    tasks = []
    for idx, item in enumerate(data_sample):
        qid = int(item.get("id", -1))
        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            continue
        tasks.append({
            "qid": qid,
            "question": q,
            "n_next": N_NEXT_WORDS,
            "samples": SAMPLES_PER_QUESTION,
            "seed": GLOBAL_SEED + idx,  # 每个任务独立 seed
        })

    print(f"[INFO] Starting {NUM_WORKERS} workers for {len(tasks)} tasks...")

    # 使用 Manager 创建共享计数器和锁
    manager = Manager()
    counter = manager.Value('i', 0)
    lock = manager.Lock()
    total = len(tasks)

    # 构建带共享状态的参数
    task_args = [(task, counter, lock, total) for task in tasks]

    # 使用进程池并行处理
    with Pool(processes=NUM_WORKERS) as pool:
        results = pool.map(worker_evaluate_question, task_args)

    print()  # 换行，结束进度条

    # 汇总结果
    suspicious = []
    total_tested = 0
    total_hits = 0
    total_api_fail = 0
    all_predictions = []  # 保存所有预测结果

    for r in results:
        total_tested += r["tested"]
        total_hits += r["hits"]
        total_api_fail += r["api_fail"]

        if r["tested"] > 0 and r["hit_ratio"] >= LEAK_RATIO_THRESHOLD:
            suspicious.append(r)

        # 收集每个问题的预测详情
        for detail in r.get("details", []):
            all_predictions.append({
                "question_id": r["id"],
                "cut_pos": detail.get("cut_pos"),
                "prefix": detail.get("prefix"),
                "ground_truth": detail.get("gt_next"),
                "prediction": detail.get("pred_next"),
                "match": detail.get("match"),
                "error": detail.get("error")
            })

        print(f"[{r['id']}] tested={r['tested']} hits={r['hits']} ratio={r['hit_ratio']} api_fail={r['api_fail']}")

    overall_hit_ratio = total_hits / total_tested if total_tested > 0 else 0.0
    api_fail_rate = total_api_fail / (len(results) * SAMPLES_PER_QUESTION) if results else 0.0

    # prop1：泄露样本占比（在被检查的 sample 内）
    prop1 = len(suspicious) / len(results) if results else 0.0

    suspicious_sorted = sorted(suspicious, key=lambda x: -x["hit_ratio"])

    report = {
        "model": MODEL_NAME,
        "base_url": BASE_URL,
        "judge_model": JUDGE_MODEL_NAME,

        # params
        "n_next_words": N_NEXT_WORDS,
        "samples_per_question": SAMPLES_PER_QUESTION,
        "leak_ratio_threshold": LEAK_RATIO_THRESHOLD,
        "max_questions_to_check": MAX_QUESTIONS_TO_CHECK,

        # leakage stats
        "questions_checked": len(results),
        "total_ngrams_tested": total_tested,
        "total_hits": total_hits,
        "overall_hit_ratio": round(overall_hit_ratio, 6),
        "api_fail_rate": round(api_fail_rate, 6),

        # ✅ prop1
        "prop1_leaked_proportion": round(prop1, 6),
        "suspicious_count": len(suspicious),

        # 输出可疑 top-k
        "suspicious_top": suspicious_sorted[:30],
    }

    # ✅ prop2（可选）
    if ENABLE_PROP2 and suspicious_sorted:
        # prop2 需要单独创建 client（prop1 并行阶段每个 worker 有自己的 client）
        client = build_client()
        prop2_res = compute_prop2_on_suspicious(
            client=client,
            suspicious_items=suspicious_sorted,
            full_data_map=full_data_map,
            max_items=MAX_SUSPICIOUS_FOR_PROP2,
            mode=PROP2_MODE,
        )
        report.update(prop2_res)
    else:
        report.update({
            "prop2_suspicious_accuracy": None,
            "prop2_total_judged": 0,
            "prop2_correct": 0,
            "prop2_details": []
        })

    return report, all_predictions


# =========================================================
# 9) main
# =========================================================

if __name__ == "__main__":
    data = safe_json_load(DATA_PATH)

    print("\n==============================")
    print(f"[INFO] Loaded data items: {len(data)}")
    print("==============================\n")

    report, all_predictions = run_leakage_pipeline(data)

    # 保存 report
    with open(OUT_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 额外保存 suspicious（更方便单独看）
    with open(OUT_SUSPICIOUS_PATH, "w", encoding="utf-8") as f:
        json.dump(report.get("suspicious_top", []), f, ensure_ascii=False, indent=2)

    # 保存所有预测结果（每个截断点的实际值和预测值）
    with open(OUT_PREDICTIONS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_predictions, f, ensure_ascii=False, indent=2)

    print("\n====================")
    print("Leakage pipeline done ✅")
    print(f"prop1_leaked_proportion: {report['prop1_leaked_proportion']}")
    print(f"overall_hit_ratio:       {report['overall_hit_ratio']}")
    print(f"suspicious_count:        {report['suspicious_count']}")
    print(f"prop2_suspicious_accuracy: {report.get('prop2_suspicious_accuracy')}")
    print(f"Saved report -> {OUT_REPORT_PATH}")
    print(f"Saved suspicious -> {OUT_SUSPICIOUS_PATH}")
    print(f"Saved predictions -> {OUT_PREDICTIONS_PATH}")
    print("====================")
