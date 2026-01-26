"""
File processor for GlobalHealthAtlas
"""
import json
import os
from src.utils.data_handler import load_input_data, save_output_data
from src.utils.checkpoint_manager import load_global_checkpoint, create_checkpoint, save_global_checkpoint
from src.config.paths import GLOBAL_CHECKPOINT_FILE
from src.core.batch_processor import BatchProcessor


class FileProcessor:
    """Processes individual files for scoring"""
    
    def __init__(self, batch_processor: BatchProcessor, batch_size: int = 4000):
        """
        Initialize the file processor
        
        Args:
            batch_processor: Instance of BatchProcessor
            batch_size: Size of batches to process
        """
        self.batch_processor = batch_processor
        self.batch_size = batch_size
        self.global_checkpoint_file = GLOBAL_CHECKPOINT_FILE

    def process_file(self, input_path: str, output_path: str, file_idx: int = 0, total_files: int = 1):
        """
        Process a single file
        
        Args:
            input_path: Path to input JSON file
            output_path: Path to output JSON file
            file_idx: Index of this file in the processing queue
            total_files: Total number of files to process
            
        Returns:
            list: Results from processing the file
        """
        print(f"\n{'='*80}")
        print(f"处理文件 [{file_idx + 1}/{total_files}]: {input_path}")
        print(f"输出文件: {output_path}")
        print(f"{'='*80}")

        # 读取输入数据
        input_data = load_input_data(input_path)
        total_items = len(input_data)
        print(f"输入文件包含 {total_items} 条数据")

        # 初始化结果列表
        file_results = []

        # ================= 分批处理循环 =================
        for i in range(0, total_items, self.batch_size):
            batch_end = min(i + self.batch_size, total_items)
            current_batch_items = input_data[i:batch_end]

            print(f"\nProcessing Batch: {i} to {batch_end}...")

            # Process the current batch
            batch_results = self.batch_processor.process_batch(current_batch_items, i)

            # Add batch results to file results
            for res in batch_results:
                if res is None:
                    file_results.append({'id': 'unknown', 'error': 'Processing logic error'})
                else:
                    file_results.append(res)

            # 保存当前文件的结果（Checkpoint）
            print(f"  - Saving checkpoint ({len(file_results)} items)...")
            save_output_data(file_results, output_path)

            # 更新全局断点信息
            global_checkpoint = create_checkpoint(
                file_index=file_idx,
                item_index=batch_end,
                total_files=total_files,
                input_file=input_path,
                output_file=output_path,
                batch_progress=f"{batch_end}/{total_items}"
            )
            save_global_checkpoint(global_checkpoint, self.global_checkpoint_file)

        print(f"\n文件 [{file_idx + 1}/{total_files}] 处理完成！结果已保存到: {output_path}")
        
        return file_results