"""
Model initialization for GlobalHealthAtlas
"""
import torch
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from transformers import AutoTokenizer
from src.config.model_config import JSON_SCHEMA, STOP_TOKEN_IDS


def initialize_model(model_path, base_model_path, max_model_len=40960):
    """
    Initialize the vLLM model and tokenizer
    
    Args:
        model_path (str): Path to the model
        base_model_path (str): Path to the base model
        max_model_len (int): Maximum model length
        
    Returns:
        tuple: (llm, tokenizer)
    """
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 张 GPU，vLLM 将开启 Tensor Parallelism。")
    
    print(f"加载 Tokenizer: {base_model_path}")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("正在初始化 vLLM 引擎...")
    llm = LLM(
        model=model_path,
        tokenizer=base_model_path,
        tensor_parallel_size=gpu_count,
        trust_remote_code=True,
        gpu_memory_utilization=0.9,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enable_prefix_caching=True,
    )
    
    # Ensure stop token IDs are complete
    if tokenizer.eos_token_id not in STOP_TOKEN_IDS:
        STOP_TOKEN_IDS.append(tokenizer.eos_token_id)
    
    return llm, tokenizer


def create_sampling_params(tokenizer):
    """
    Create sampling parameters for model inference
    
    Args:
        tokenizer: The model tokenizer
        
    Returns:
        SamplingParams: Sampling parameters
    """
    # Use GuidedDecodingParams to enforce correct JSON format
    guided_decoding_params = GuidedDecodingParams(json=JSON_SCHEMA)
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=1.0,
        max_tokens=1536,
        stop_token_ids=STOP_TOKEN_IDS,
        repetition_penalty=1.1,
        guided_decoding=guided_decoding_params
    )
    
    return sampling_params