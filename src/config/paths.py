"""
Path configurations for GlobalHealthAtlas
"""
import os

# Default paths
MODEL_PATH = os.getenv("MODEL_PATH", "/home/ubuntu1/.cache/modelscope/hub/models/Qwen3-8B-Merged/")
BASE_MODEL_PATH = os.getenv("BASE_MODEL_PATH", "/home/ubuntu1/.cache/modelscope/hub/models/Qwen/Qwen3-8B/")
GLOBAL_CHECKPOINT_FILE = os.getenv("GLOBAL_CHECKPOINT_FILE", "/home/ubuntu1/.cache/modelscope/hub/models/test_score/global_checkpoint.json")

# Default file pairs for processing
FILE_PAIRS = [
    {
        "input": "/home/ubuntu1/.cache/modelscope/hub/models/27500/waitmedium/result_llama70B00.json",
        "output": "/home/ubuntu1/.cache/modelscope/hub/models/27500/waitmedium/result_llama70B00_score.json"
    }
]