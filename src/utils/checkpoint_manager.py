"""
Checkpoint management utilities for GlobalHealthAtlas
"""
import json
import os
import time


def save_global_checkpoint(checkpoint_data, file_path):
    """
    Save global checkpoint information
    
    Args:
        checkpoint_data: Checkpoint data to save
        file_path (str): Checkpoint file path
    """
    temp_path = file_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, file_path)


def load_global_checkpoint(file_path):
    """
    Load global checkpoint information
    
    Args:
        file_path (str): Checkpoint file path
        
    Returns:
        dict or None: Checkpoint data or None if not found
    """
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"加载全局断点文件失败: {e}，将重新开始。")
        return None


def create_checkpoint(file_index, item_index, total_files, input_file, output_file, batch_progress):
    """
    Create a checkpoint dictionary
    
    Args:
        file_index (int): Current file index
        item_index (int): Current item index
        total_files (int): Total number of files
        input_file (str): Current input file
        output_file (str): Current output file
        batch_progress (str): Batch progress information
        
    Returns:
        dict: Checkpoint data
    """
    return {
        'current_file_index': file_index,
        'current_item_index': item_index,
        'last_update': time.strftime('%Y-%m-%d %H:%M:%S'),
        'total_files': total_files,
        'current_input_file': input_file,
        'current_output_file': output_file,
        'current_batch_progress': batch_progress
    }