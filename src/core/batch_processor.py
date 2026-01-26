"""
Batch processor for GlobalHealthAtlas scoring
"""
import json
import time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


class BatchProcessor:
    """Handles batch scoring of data using vLLM"""
    
    def __init__(self, llm: LLM, tokenizer: AutoTokenizer, sampling_params: SamplingParams, safe_input_limit: int):
        """
        Initialize the batch processor
        
        Args:
            llm: The vLLM model instance
            tokenizer: The model tokenizer
            sampling_params: Sampling parameters for inference
            safe_input_limit: Maximum token limit for inputs
        """
        self.llm = llm
        self.tokenizer = tokenizer
        self.sampling_params = sampling_params
        self.safe_input_limit = safe_input_limit

    def process_batch(self, batch_items, start_idx):
        """
        Process a batch of items
        
        Args:
            batch_items (list): List of items to process
            start_idx (int): Starting index in the full dataset
            
        Returns:
            list: Results for the batch
        """
        # 1. 构建 Prompt 并过滤超长样本
        valid_prompts = []
        valid_indices = []  # 记录在当前 batch 中的相对索引
        batch_results = [None] * len(batch_items)  # 预占位，用于填入结果

        for idx, item in enumerate(batch_items):
            item_id = item.get('id', f'unknown_{start_idx+idx}')

            # 构建文本
            from src.core.prompt_builder import build_prompt
            user_prompt = build_prompt(item)
            messages = [
                {"role": "system", "content": "You are a Distinguished Professor of Public Health and Infectious Disease Surveillance acting as an expert evaluator (LLM-as-a-Judge).\n"},
                {"role": "user", "content": user_prompt}
            ]
            text_input = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

            # 检查长度
            token_ids = self.tokenizer.encode(text_input)
            if len(token_ids) > self.safe_input_limit:
                batch_results[idx] = {
                    'id': item_id,
                    'scores': None,
                    'error': f"Prompt length {len(token_ids)} > limit {self.safe_input_limit}",
                    'raw_response': ""
                }
            else:
                valid_prompts.append(text_input)
                valid_indices.append(idx)

        # 2. 如果有有效数据，进行推理
        if valid_prompts:
            outputs = self.llm.generate(valid_prompts, self.sampling_params)

            # 3. 填回结果
            for v_idx, output in enumerate(outputs):
                original_idx = valid_indices[v_idx]  # 找回在 batch 中的位置
                item = batch_items[original_idx]
                item_id = item.get('id', f'unknown_{start_idx+original_idx}')
                response = output.outputs[0].text

                try:
                    # 解析 JSON（guided_decoding 已经保证格式正确）
                    score_data = json.loads(response)
                    # 保留原始数据的所有字段,并添加 scores 字段
                    result_item = item.copy()
                    result_item['scores'] = score_data
                    batch_results[original_idx] = result_item

                except json.JSONDecodeError as e:
                    # 保留原始数据的所有字段,并添加错误信息
                    result_item = item.copy()
                    result_item['scores'] = None
                    result_item['error'] = str(e)
                    result_item['raw_response'] = response
                    batch_results[original_idx] = result_item

                except Exception as e:
                    # 其他异常处理
                    result_item = item.copy()
                    result_item['scores'] = None
                    result_item['error'] = f"Unexpected error: {str(e)}"
                    result_item['raw_response'] = response
                    batch_results[original_idx] = result_item

        return batch_results