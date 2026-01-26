"""
Prompt builder for GlobalHealthAtlas evaluator
"""
from src.config.prompts import PROMPT_TEMPLATE


def build_prompt(item):
    """
    Build evaluation prompt from template and data item
    
    Args:
        item (dict): Data item containing fields for prompt
        
    Returns:
        str: Built prompt
    """
    prompt = PROMPT_TEMPLATE
    prompt = prompt.replace('{{domain}}', str(item.get('domain', '')))
    prompt = prompt.replace('{{label}}', str(item.get('label', '')))
    prompt = prompt.replace('{{question}}', str(item.get('question', '')))
    prompt = prompt.replace('{{answer}}', str(item.get('answer', '')))
    prompt = prompt.replace('{{complexCOT}}', str(item.get('complexCOT', '')))
    prompt = prompt.replace('{{llm_complexCOT}}', str(item.get('llm_complexCOT', '')))
    prompt = prompt.replace('{{llm_answer}}', str(item.get('llm_answer', '')))
    return prompt