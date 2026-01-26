"""
Main entry point for GlobalHealthAtlas scoring
"""
import time
import torch
from src.config.paths import FILE_PAIRS, MODEL_PATH, BASE_MODEL_PATH
from src.config.model_config import MAX_MODEL_LEN, BATCH_SIZE, SAFE_INPUT_LIMIT
from src.utils.checkpoint_manager import load_global_checkpoint
from src.core.model_initializer import initialize_model, create_sampling_params
from src.core.batch_processor import BatchProcessor
from src.handlers.file_processor import FileProcessor


def main():
    start_time = time.time()
    
    # Initialize model components
    llm, tokenizer = initialize_model(MODEL_PATH, BASE_MODEL_PATH, MAX_MODEL_LEN)
    sampling_params = create_sampling_params(tokenizer)
    
    # Initialize batch processor
    batch_processor = BatchProcessor(llm, tokenizer, sampling_params, SAFE_INPUT_LIMIT)
    
    # Initialize file processor
    file_processor = FileProcessor(batch_processor, BATCH_SIZE)
    
    # 加载全局断点续传信息
    global_checkpoint = load_global_checkpoint("/home/ubuntu1/.cache/modelscope/hub/models/test_score/global_checkpoint.json")

    # 确定起始文件索引
    start_file_index = 0
    start_item_index = 0

    if global_checkpoint is not None:
        # 检查是否有正在处理的文件
        if 'current_file_index' in global_checkpoint:
            start_file_index = global_checkpoint['current_file_index']
            start_item_index = global_checkpoint.get('current_item_index', 0)
            print(f"从断点恢复：文件索引 {start_file_index}，条目索引 {start_item_index}")
    else:
        print("未找到全局断点文件，将从头开始处理")

    # ================= 循环处理每个文件 =================
    for file_idx in range(start_file_index, len(FILE_PAIRS)):
        file_pair = FILE_PAIRS[file_idx]
        input_path = file_pair["input"]
        output_path = file_pair["output"]

        # Process the file
        file_results = file_processor.process_file(input_path, output_path, file_idx, len(FILE_PAIRS))

        # 重置下一文件的起始索引
        start_item_index = 0

    # 全部完成
    print(f"\n{'='*80}")
    print(f"🎉 所有文件处理完成！")
    print(f"{'='*80}")

    # Clean up checkpoint file
    import os
    checkpoint_file = "/home/ubuntu1/.cache/modelscope/hub/models/test_score/global_checkpoint.json"
    if os.path.exists(checkpoint_file):
        os.remove(checkpoint_file)
        print(f"已删除全局断点文件: {checkpoint_file}")

    end_time = time.time()
    print(f"总耗时: {(end_time - start_time)/60:.2f} 分钟")


if __name__ == "__main__":
    main()