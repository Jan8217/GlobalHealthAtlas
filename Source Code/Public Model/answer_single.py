import json
import time
import os
import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from transformers import AutoTokenizer

INPUT_JSON_PATH = "./ewai/ewai.json"
OUTPUT_JSON_PATH = "./ewai/ewai_answer_4b.json"
CHECKPOINT_FILE = OUTPUT_JSON_PATH.replace(".json", "_checkpoint.json")
base_model_path = "/root/autodl-tmp/LLM/Qwen3-4B"
lora_path = "/root/autodl-fs/Lora/Qwen3-4B/sft100/"
MAX_MODEL_LEN = 40960
DTYPE = "bfloat16"
GPU_MEMORY_UTILIZATION = 0.9
ENABLE_LORA = True
MAX_LORA_RANK = 64
MAX_LORAS = 1
TEMPERATURE = 0.1
TOP_P = 1.0
MAX_TOKENS = 2500
REPETITION_PENALTY = 1.1
STOP_TOKEN_IDS = [151643, 151645]

PROMPT_TEMPLATE = (
    "Please answer in the original language of the following question.\n\n"
    "Question: {question}\n\n"
    "Follow this exact format:\n"
    "<think>\nYour thinking process\n</think>\n\nAnswer: [The Option Letter OR The Direct Conclusion]\n"
)

def load_input_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_output_data(data, file_path):
    temp_path = file_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, file_path)

def build_prompt(item):
    question = item.get('question', '')
    return PROMPT_TEMPLATE.format(question=question)

def main():
    start_time = time.time()

    gpu_count = torch.cuda.device_count()
    print(f"Detected {gpu_count} GPUs, vLLM will enable Tensor Parallelism.")

    print(f"Loading Tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

    print("Initializing vLLM engine with dynamic LoRA mounting...")
    llm = LLM(
        model=base_model_path,
        tokenizer=base_model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        gpu_memory_utilization=GPU_MEMORY_UTILIZATION,
        max_model_len=MAX_MODEL_LEN,
        dtype=DTYPE,
        enable_prefix_caching=True,
        enable_lora=ENABLE_LORA,
        max_lora_rank=MAX_LORA_RANK,
        max_loras=MAX_LORAS
    )

    if ENABLE_LORA:
        print(f"Preparing to use LoRA weights: {lora_path}")
        lora_request = LoRARequest(
            lora_name="adapter",
            lora_int_id=1,
            lora_path=lora_path
        )
    else:
        lora_request = None
        print("Not mounting LoRA weights, using base model")

    final_stop_token_ids = STOP_TOKEN_IDS.copy()
    if tokenizer.eos_token_id not in final_stop_token_ids:
        final_stop_token_ids.append(tokenizer.eos_token_id)
    sampling_params = SamplingParams(
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_TOKENS,
        stop_token_ids=final_stop_token_ids,
        repetition_penalty=REPETITION_PENALTY
    )

    print(f"Reading input file: {INPUT_JSON_PATH}")
    input_data = load_input_data(INPUT_JSON_PATH)
    total_items = len(input_data)
    final_results = []
    start_index = 0
    if os.path.exists(CHECKPOINT_FILE):
        print(f"Found checkpoint file {CHECKPOINT_FILE}, attempting to load...")
        try:
            with open(CHECKPOINT_FILE, 'r', encoding='utf-8') as f:
                final_results = json.load(f)
            start_index = len(final_results)
            print(f"Successfully loaded {start_index} completed items, resuming from index {start_index}.")
        except Exception as e:
            print(f"Failed to load checkpoint: {e}, starting over.")
            final_results = []

    SAFE_INPUT_LIMIT = MAX_MODEL_LEN - 1024
    for i in range(start_index, total_items):
        item = input_data[i]
        item_id = item.get('id', f'unknown_{i}')

        print(f"\nProcessing item {i+1}/{total_items} (ID: {item_id})...")

        SYSTEM_PROMPT = (
    "You are a Public Health Expert. You MUST strictly follow these rules:\n"
    "1) Strict Language Alignment: Respond EXCLUSIVELY in the same language as the question.\n"
    "2) Output Structure: First output your thinking within <think> blocks, then provide Answer.\n"
    "3) Task-Specific Rules:\n"
    "   - **If Single-Choice (A/B/C/D)**:\n"
    "     * Answer: Output ONLY the letter (A, B, C, or D).\n"
    "   - **If Open-Ended Q&A**:\n"
    "     * Answer: Provide a direct, professional conclusion or solution (1-3 sentences).\n"
    "4) The thinking block should contain your reasoning process. The Reasoning must be under 512 tokens."
)
        user_prompt = build_prompt(item)
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]
        text_input = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        token_ids = tokenizer.encode(text_input)
        if len(token_ids) > SAFE_INPUT_LIMIT:
            print(f"  Warning: [Skip] ID: {item_id} Too Long ({len(token_ids)} tokens)")
            result_item = item.copy()
            result_item['llm_complexCOT'] = ""
            result_item['llm_answer'] = ""
            result_item['error'] = f"Prompt length {len(token_ids)} > limit {SAFE_INPUT_LIMIT}"
            result_item['raw_response'] = ""
            final_results.append(result_item)
            continue

        if ENABLE_LORA:
            outputs = llm.generate([text_input], sampling_params, lora_request=lora_request)
        else:
            outputs = llm.generate([text_input], sampling_params)

        output = outputs[0]
        response = output.outputs[0].text

        try:
            lines = response.strip().split('\n')
            llm_complexCOT = ""
            llm_answer = ""
            answer_idx = -1
            for i, line in enumerate(lines):
                if line.strip().startswith('Answer:'):
                    answer_idx = i
                    break

            if answer_idx != -1:
                thinking_lines = lines[:answer_idx]
                llm_complexCOT = '\n'.join(thinking_lines).strip()
                llm_complexCOT = llm_complexCOT.replace('', '').strip()
                llm_complexCOT = llm_complexCOT.replace('', '').strip()
                llm_answer = lines[answer_idx].strip()[7:].strip()
            else:
                thinking_start = response.find('<think')
                thinking_end = response.find('')

                if thinking_start != -1 and thinking_end != -1:
                    llm_complexCOT = response[thinking_start + 3:thinking_end].strip()
                    after_thinking = response[thinking_end + 4:].strip()
                    answer_idx = after_thinking.find('Answer:')
                    if answer_idx != -1:
                        llm_answer = after_thinking[answer_idx + 7:].strip()
                    else:
                        llm_answer = after_thinking.strip()
                else:
                    llm_complexCOT = response.strip()
                    llm_answer = ""

            result_item = item.copy()
            result_item['llm_complexCOT'] = llm_complexCOT
            result_item['llm_answer'] = llm_answer
            if not llm_complexCOT or not llm_answer:
                print(f"\n{'='*60}")
                print(f"Warning: Parse Failed - ID: {item_id}")
                print(f"llm_complexCOT: {'OK' if llm_complexCOT else 'Missing'}")
                print(f"llm_answer: {'OK' if llm_answer else 'Missing'}")
                print(f"Full Output Block:")
                print(response)
                print(f"{'='*60}\n")

            final_results.append(result_item)

        except Exception as e:
            result_item = item.copy()
            result_item['llm_complexCOT'] = ""
            result_item['llm_answer'] = ""
            result_item['error'] = f"Unexpected error: {str(e)}"
            result_item['raw_response'] = response
            final_results.append(result_item)

        if (i + 1) % 10 == 0:
            print(f"  Saving checkpoint ({len(final_results)} items)...")
            save_output_data(final_results, CHECKPOINT_FILE)

    print(f"\nAll completed! Final save to: {OUTPUT_JSON_PATH}")
    save_output_data(final_results, OUTPUT_JSON_PATH)

    end_time = time.time()
    print(f"Total time: {(end_time - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
