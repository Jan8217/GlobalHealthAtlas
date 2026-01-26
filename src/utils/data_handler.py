"""
Data handling utilities for GlobalHealthAtlas
"""
import json
import os


def load_input_data(file_path):
    """
    Load input data from JSON file
    
    Args:
        file_path (str): Path to JSON file
        
    Returns:
        list: Loaded data
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_output_data(data, file_path):
    """
    Save output data to JSON file with atomic write operation
    
    Args:
        data: Data to save
        file_path (str): Output file path
    """
    # 临时写入再重命名，防止写入中断导致文件损坏
    temp_path = file_path + ".tmp"
    with open(temp_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    os.replace(temp_path, file_path)